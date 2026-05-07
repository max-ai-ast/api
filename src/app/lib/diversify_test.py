"""Tests for MMR-based feed diversification."""

import pytest

from ..models import CandidatePost
from .diversify import mmr_rerank


def _post(uri: str, score: float, author_did: str | None = None) -> CandidatePost:
    return CandidatePost(at_uri=uri, score=score, author_did=author_did)


def test_empty_input_returns_empty():
    assert mmr_rerank([]) == []


def test_single_candidate_unchanged():
    c = _post("at://x/1", score=1.0, author_did="did:plc:alice")
    result = mmr_rerank([c])
    assert result == [c]


def test_same_author_posts_spread_apart():
    """b1 (lower score, different author) should precede a2 (same author as a1)."""
    a1 = _post("at://alice/1", score=1.0, author_did="did:plc:alice")
    a2 = _post("at://alice/2", score=0.9, author_did="did:plc:alice")
    a3 = _post("at://alice/3", score=0.8, author_did="did:plc:alice")
    b1 = _post("at://bob/1", score=0.5, author_did="did:plc:bob")

    result = mmr_rerank([a1, a2, a3, b1])
    uris = [c.at_uri for c in result]

    assert uris[0] == "at://alice/1"
    assert uris.index("at://bob/1") < uris.index("at://alice/2")


def test_all_different_authors_order_preserved_by_relevance():
    """With no author overlap, MMR reduces to relevance order."""
    posts = [
        _post("at://a/1", score=0.9, author_did="did:plc:a"),
        _post("at://b/1", score=0.7, author_did="did:plc:b"),
        _post("at://c/1", score=0.5, author_did="did:plc:c"),
        _post("at://d/1", score=0.3, author_did="did:plc:d"),
    ]

    result = mmr_rerank(posts)
    uris = [c.at_uri for c in result]
    assert uris == ["at://a/1", "at://b/1", "at://c/1", "at://d/1"]
