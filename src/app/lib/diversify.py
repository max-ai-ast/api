"""MMR-based feed diversification."""

import math

from .embeddings import decode_float32_b64
from ..models import CandidatePost

BETA = 0.5
AUTHOR_WEIGHT = 0.75

def mmr_rerank(candidates: list[CandidatePost]) -> list[CandidatePost]:
    if len(candidates) <= 1:
        return list(candidates)

    max_score = max((c.score or 0.0) for c in candidates)
    if max_score > 0.0:
        norm_scores = [(c.score or 0.0) / max_score for c in candidates]
    else:
        norm_scores = [0.0] * len(candidates)

    selected: list[int] = []
    remaining: list[int] = list(range(len(candidates)))

    while remaining:
        if not selected:
            best = max(remaining, key=lambda i: norm_scores[i])
        else:
            def mmr_score(i: int) -> float:
                rel = (1 - BETA) * norm_scores[i]
                sim = BETA * max(_similarity(candidates[i], candidates[j]) for j in selected)
                return rel - sim
            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def _similarity(a: CandidatePost, b: CandidatePost) -> float:
    if a.author_did is not None and a.author_did == b.author_did:
        author_match = 1.0
    else:
        author_match = 0.0

    if (
        AUTHOR_WEIGHT < 1.0
        and a.minilm_l12_embedding is not None
        and b.minilm_l12_embedding is not None
    ):
        vec_a = decode_float32_b64(a.minilm_l12_embedding)
        vec_b = decode_float32_b64(b.minilm_l12_embedding)
        cosine = _cosine_similarity(vec_a, vec_b)
    else:
        cosine = 0.0

    return AUTHOR_WEIGHT * author_match + (1 - AUTHOR_WEIGHT) * cosine


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
