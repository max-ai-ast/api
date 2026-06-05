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
        attr = {"TOXICITY": 0.0, "SEVERE_TOXICITY": 0.0, "IDENTITY_ATTACK": 0.0, "INSULT": 0.0}
        assert _prc_score(attr) == pytest.approx(0.0)

    def test_full_toxicity_returns_negative_one(self):
        attr = {"TOXICITY": 1.0, "SEVERE_TOXICITY": 1.0, "IDENTITY_ATTACK": 1.0, "INSULT": 1.0}
        # avg=1.0 → score = -1.0
        assert _prc_score(attr) == pytest.approx(-1.0)

    def test_partial_toxicity_proportional(self):
        attr = {"TOXICITY": 0.8, "SEVERE_TOXICITY": 0.0, "IDENTITY_ATTACK": 0.0, "INSULT": 0.0}
        # avg = 0.8/4 = 0.2 → score = -0.2
        assert _prc_score(attr) == pytest.approx(-0.2)

    def test_known_mixed_inputs(self):
        attr = {"TOXICITY": 0.4, "SEVERE_TOXICITY": 0.2, "IDENTITY_ATTACK": 0.6, "INSULT": 0.8}
        expected = -((0.4 + 0.2 + 0.6 + 0.8) / 4.0)
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
