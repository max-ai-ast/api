"""Tests for the two-tower ranker configuration behavior."""

import asyncio

import pytest

from ...models import CandidatePost
from .base import RankerError
from .two_tower import TwoTowerRanker


def test_predict_requires_inference_env_vars(monkeypatch):
    monkeypatch.delenv("GE_INFERENCE_BASE_URL", raising=False)
    monkeypatch.delenv("GE_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("GE_INFERENCE_MAX_HISTORY_LEN", raising=False)
    monkeypatch.delenv("GE_INFERENCE_EMBED_DIM", raising=False)

    ranker = TwoTowerRanker()

    with pytest.raises(RankerError, match="GE_INFERENCE_BASE_URL"):
        asyncio.run(
            ranker.predict(
                es=None,
                user_did="did:plc:user1",
                candidates=[CandidatePost(at_uri="at://post/1")],
            )
        )
