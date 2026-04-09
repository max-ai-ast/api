"""Ranking framework for the recommendation system.

Provides an abstraction for named rankers that can be called internally or via
the `/rank` API.
"""

from ...models import RankPredictRequest, RankPredictResult
from .base import (
    Ranker,
    RankerResult,
    get_ranker,
    list_rankers,
    register_ranker,
)
from .predict import (
    DEFAULT_RANK_MODEL,
    RankModelNotFoundError,
    RankerError,
    dedup_candidates,
    run_predict,
)
from .candidate_score import CandidateScoreRanker

_candidate_score = CandidateScoreRanker()
register_ranker(_candidate_score)

__all__ = [
    "DEFAULT_RANK_MODEL",
    "CandidateScoreRanker",
    "Ranker",
    "RankerError",
    "RankerResult",
    "RankModelNotFoundError",
    "RankPredictRequest",
    "RankPredictResult",
    "dedup_candidates",
    "get_ranker",
    "list_rankers",
    "register_ranker",
    "run_predict",
]
