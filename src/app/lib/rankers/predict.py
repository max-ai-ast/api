"""Shared ranking pipeline.

Given a `RankPredictRequest`, resolves a named ranker, de-duplicates candidate
posts, and returns the ranker's ordered output.
"""

from ...models import CandidatePost, RankPredictRequest, RankPredictResult
from .base import get_ranker, list_rankers

DEFAULT_RANK_MODEL = "candidate_score"


class RankModelNotFoundError(Exception):
    """Raised when a requested rank model does not exist."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Rank model not found: {name}")


class RankerError(Exception):
    """Raised when ranking cannot be completed for a valid request."""


def dedup_candidates(candidates: list[CandidatePost]) -> list[CandidatePost]:
    """Remove duplicate candidates by `at_uri`, keeping the first occurrence."""
    seen: set[str] = set()
    deduped: list[CandidatePost] = []
    for candidate in candidates:
        if candidate.at_uri is None:
            raise RankerError("All candidates must include at_uri")
        if candidate.at_uri in seen:
            continue
        seen.add(candidate.at_uri)
        deduped.append(candidate)
    return deduped


async def run_predict(
    request: RankPredictRequest,
    es,
    *,
    swallow_errors: bool = False,
) -> RankPredictResult:
    """Rank the supplied candidates using the requested ranker."""
    model_name = request.model or DEFAULT_RANK_MODEL
    ranker = get_ranker(model_name)
    if ranker is None:
        raise RankModelNotFoundError(model_name)

    deduped_request = request.model_copy(update={"candidates": dedup_candidates(request.candidates)})
    result = await ranker.predict(deduped_request)
    return result.result
