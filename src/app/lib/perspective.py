"""Perspective API integration for post-ranking by conversational quality."""

from __future__ import annotations

import asyncio
import logging
import os

from ..models import CandidatePost
from .http_client import get_http_client

logger = logging.getLogger(__name__)

_PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

_REQUESTED_ATTRIBUTES = {
    "COMPASSION": {},
    "CURIOSITY": {},
    "REASONING": {},
    "MORAL_OUTRAGE": {},
    "SCAPEGOATING": {},
    "ALIENATION": {},
    "INSULT": {},
    "IDENTITY_ATTACK": {},
    "PERSUASION": {},
}


def _prc_score(attr: dict[str, float]) -> float:
    """Compute the PRC paper score from a flat dict of Perspective attribute scores.

    Formula (PRC paper tables S13/S14):
        bridging     = avg(COMPASSION, CURIOSITY, REASONING)
        correlated   = avg(MORAL_OUTRAGE, SCAPEGOATING, ALIENATION)
        toxicity_sub = avg(INSULT, IDENTITY_ATTACK, correlated)
        score        = bridging - 0.5 * PERSUASION - 0.5 * toxicity_sub
    """
    bridging = (attr["COMPASSION"] + attr["CURIOSITY"] + attr["REASONING"]) / 3.0
    correlated = (attr["MORAL_OUTRAGE"] + attr["SCAPEGOATING"] + attr["ALIENATION"]) / 3.0
    toxicity_sub = (attr["INSULT"] + attr["IDENTITY_ATTACK"] + correlated) / 3.0
    return bridging - 0.5 * attr["PERSUASION"] - 0.5 * toxicity_sub


class PerspectiveClient:
    """Thin async client for the Perspective API.

    Reads GE_PERSPECTIVE_API_KEY from the environment at instantiation.
    Raises RuntimeError if the key is missing.
    """

    def __init__(self) -> None:
        key = os.environ.get("GE_PERSPECTIVE_API_KEY")
        if not key:
            raise RuntimeError("GE_PERSPECTIVE_API_KEY environment variable is not set")
        self._api_key = key

    async def score(self, content: str) -> float:
        """Return the PRC score for the given text content."""
        payload = {
            "comment": {"text": content},
            "requestedAttributes": _REQUESTED_ATTRIBUTES,
        }
        client = get_http_client()
        response = await client.post(
            _PERSPECTIVE_URL,
            params={"key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        attr_scores = {
            name: data["attributeScores"][name]["summaryScore"]["value"]
            for name in _REQUESTED_ATTRIBUTES
        }
        return _prc_score(attr_scores)


_client: PerspectiveClient | None = None


def _get_client() -> PerspectiveClient:
    global _client
    if _client is None:
        _client = PerspectiveClient()
    return _client


async def perspective_rerank(candidates: list[CandidatePost]) -> list[CandidatePost]:
    """Re-rank candidates by PRC score (descending).

    Posts with content=None or where the API call fails receive a neutral score
    of 0.0.  All candidates are returned — none are dropped.
    """
    if not candidates:
        return []

    client = _get_client()

    async def _score_one(c: CandidatePost) -> float:
        if c.content is None:
            return 0.0
        try:
            return await client.score(c.content)
        except Exception:
            logger.exception("Perspective API scoring failed for post %s", c.at_uri)
            return 0.0

    scores = await asyncio.gather(*(_score_one(c) for c in candidates))
    paired = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
    return [c for _, c in paired]
