"""Tests for PRC scoring and perspective_rerank."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..models import CandidatePost
from .perspective import _prc_score, perspective_rerank


def _make_candidate(uri: str, content: str | None = "text", score: float = 1.0) -> CandidatePost:
    return CandidatePost(
        at_uri=uri,
        content=content,
        score=score,
        minilm_l12_embedding=None,
        generator_name="test",
    )


# ---------------------------------------------------------------------------
# _prc_score arithmetic
# ---------------------------------------------------------------------------

class TestPrcScore:
    def test_all_zeros_returns_zero(self):
        attr = {
            "COMPASSION": 0.0, "CURIOSITY": 0.0, "REASONING": 0.0,
            "MORAL_OUTRAGE": 0.0, "SCAPEGOATING": 0.0, "ALIENATION": 0.0,
            "INSULT": 0.0, "IDENTITY_ATTACK": 0.0, "PERSUASION": 0.0,
        }
        assert _prc_score(attr) == pytest.approx(0.0)

    def test_pure_bridging_returns_positive(self):
        attr = {
            "COMPASSION": 1.0, "CURIOSITY": 1.0, "REASONING": 1.0,
            "MORAL_OUTRAGE": 0.0, "SCAPEGOATING": 0.0, "ALIENATION": 0.0,
            "INSULT": 0.0, "IDENTITY_ATTACK": 0.0, "PERSUASION": 0.0,
        }
        # bridging=1.0, toxicity_sub=0.0, persuasion=0.0 → score=1.0
        assert _prc_score(attr) == pytest.approx(1.0)

    def test_pure_toxicity_returns_negative(self):
        attr = {
            "COMPASSION": 0.0, "CURIOSITY": 0.0, "REASONING": 0.0,
            "MORAL_OUTRAGE": 1.0, "SCAPEGOATING": 1.0, "ALIENATION": 1.0,
            "INSULT": 1.0, "IDENTITY_ATTACK": 1.0, "PERSUASION": 0.0,
        }
        # bridging=0, correlated=1.0, toxicity_sub=(1+1+1)/3=1.0
        # score = 0 - 0 - 0.5*1.0 = -0.5
        assert _prc_score(attr) == pytest.approx(-0.5)

    def test_persuasion_penalizes_score(self):
        attr = {
            "COMPASSION": 1.0, "CURIOSITY": 1.0, "REASONING": 1.0,
            "MORAL_OUTRAGE": 0.0, "SCAPEGOATING": 0.0, "ALIENATION": 0.0,
            "INSULT": 0.0, "IDENTITY_ATTACK": 0.0, "PERSUASION": 1.0,
        }
        # bridging=1.0, toxicity_sub=0.0, persuasion=1.0 → 1.0 - 0.5 - 0 = 0.5
        assert _prc_score(attr) == pytest.approx(0.5)

    def test_known_mixed_inputs(self):
        attr = {
            "COMPASSION": 0.6, "CURIOSITY": 0.3, "REASONING": 0.9,
            "MORAL_OUTRAGE": 0.2, "SCAPEGOATING": 0.1, "ALIENATION": 0.3,
            "INSULT": 0.4, "IDENTITY_ATTACK": 0.2, "PERSUASION": 0.5,
        }
        bridging = (0.6 + 0.3 + 0.9) / 3.0
        correlated = (0.2 + 0.1 + 0.3) / 3.0
        toxicity_sub = (0.4 + 0.2 + correlated) / 3.0
        expected = bridging - 0.5 * 0.5 - 0.5 * toxicity_sub
        assert _prc_score(attr) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# perspective_rerank
# ---------------------------------------------------------------------------

def _fake_client(scores: list[float]) -> MagicMock:
    """Build a mock PerspectiveClient whose score() yields values in order."""
    client = MagicMock()
    client.score = AsyncMock(side_effect=scores)
    return client


class TestPerspectiveRerank:
    def test_empty_list_returns_empty(self):
        with patch("app.lib.perspective._get_client") as mock_get:
            import asyncio
            result = asyncio.run(perspective_rerank([]))
        mock_get.assert_not_called()
        assert result == []

    def test_sorts_by_prc_score_descending(self):
        candidates = [
            _make_candidate("at://a/1", content="low quality"),
            _make_candidate("at://a/2", content="medium quality"),
            _make_candidate("at://a/3", content="high quality"),
        ]
        fake = _fake_client([0.1, 0.5, 0.9])

        with patch("app.lib.perspective._get_client", return_value=fake):
            import asyncio
            result = asyncio.run(perspective_rerank(candidates))

        uris = [c.at_uri for c in result]
        assert uris == ["at://a/3", "at://a/2", "at://a/1"]

    def test_none_content_gets_neutral_score(self):
        candidates = [
            _make_candidate("at://a/1", content=None),
            _make_candidate("at://a/2", content="good post"),
        ]
        fake = _fake_client([0.8])  # only called once for the non-None post

        with patch("app.lib.perspective._get_client", return_value=fake):
            import asyncio
            result = asyncio.run(perspective_rerank(candidates))

        # at://a/2 scores 0.8, at://a/1 scores 0.0 (neutral) → a/2 first
        assert result[0].at_uri == "at://a/2"
        assert result[1].at_uri == "at://a/1"

    def test_api_failure_gets_neutral_score(self):
        candidates = [
            _make_candidate("at://a/1", content="some content"),
            _make_candidate("at://a/2", content="other content"),
        ]
        fake = MagicMock()
        fake.score = AsyncMock(side_effect=[RuntimeError("API down"), 0.7])

        with patch("app.lib.perspective._get_client", return_value=fake):
            import asyncio
            result = asyncio.run(perspective_rerank(candidates))

        # at://a/1 fails → 0.0, at://a/2 → 0.7 → a/2 first
        assert result[0].at_uri == "at://a/2"
        assert result[1].at_uri == "at://a/1"

    def test_all_candidates_returned_none_dropped(self):
        candidates = [_make_candidate(f"at://a/{i}", content="text") for i in range(5)]
        fake = _fake_client([0.5] * 5)

        with patch("app.lib.perspective._get_client", return_value=fake):
            import asyncio
            result = asyncio.run(perspective_rerank(candidates))

        assert len(result) == 5
