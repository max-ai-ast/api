"""Shared Elasticsearch utilities.

Helpers for working with Elasticsearch responses that are used across
routers and candidate generators.
"""

import logging

from elastic_transport import ObjectApiResponse
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# How many recent likes to consider when building the query vector.
DEFAULT_LIKED_POSTS_LIMIT = 50


def unwrap_es_response(resp) -> dict:
    """Unwrap an Elasticsearch response, handling both ObjectApiResponse and dict.

    Raises ``HTTPException`` with 502 if the response type is unexpected.
    """
    if isinstance(resp, ObjectApiResponse):
        return resp.body
    elif isinstance(resp, dict):
        return resp
    else:
        logger.error("Unexpected Elasticsearch response type: %s", type(resp))
        raise HTTPException(status_code=502, detail="Invalid Elasticsearch response")


async def fetch_recent_liked_post_uris(
    es,
    user_did: str,
    limit: int = DEFAULT_LIKED_POSTS_LIMIT,
) -> list[str]:
    """Return the AT URIs of posts the user most recently liked.

    Queries the ``likes`` index for documents where ``author_did`` matches
    *user_did*, sorted by ``created_at`` descending, and extracts the
    ``subject_uri`` field from each hit.
    """
    query = {
        "bool": {
            "filter": [{"term": {"author_did": user_did}}],
        }
    }

    resp = await es.search(
        index="likes",
        query=query,
        size=limit,
        sort=[{"created_at": "desc"}],
        _source=["subject_uri"],
    )

    data = unwrap_es_response(resp)
    uris: list[str] = []
    for hit in data.get("hits", {}).get("hits", []):
        uri = (hit.get("_source") or {}).get("subject_uri")
        if uri:
            uris.append(uri)
    return uris


async def fetch_post_embeddings(
    es,
    at_uris: list[str],
) -> list[list[float]]:
    """Fetch MiniLM L12 embeddings for a list of post AT URIs.

    Returns only the embeddings that were found and non-empty;
    posts without embeddings are silently skipped.
    """
    if not at_uris:
        return []

    query = {"terms": {"at_uri": at_uris}}

    resp = await es.search(
        index="posts",
        query=query,
        size=len(at_uris),
        _source=["embeddings.all_MiniLM_L12_v2"],
    )

    data = unwrap_es_response(resp)
    vectors: list[list[float]] = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source") or {}
        emb = src.get("embeddings")
        if isinstance(emb, dict):
            vec = emb.get("all_MiniLM_L12_v2")
            if vec:
                vectors.append(vec)
    return vectors