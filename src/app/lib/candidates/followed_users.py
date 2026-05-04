"""Candidate generator for posts from followed users.

Returns the last N posts from users that the requesting user follows"""

import httpx
import logging

from ...models import CandidatePost
from .base import CandidateGenerator, CandidateResult
from .utils import candidate_posts_from_es_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Maximum number of followed users to use in the query
MAX_FOLLOWED_USERS = 10_000

# Maximum page size accepted by app.bsky.graph.getFollows (should not be changed)
FOLLOWS_PAGE_LIMIT = 100


# ---------------------------------------------------------------------------
# Followed users API query
# ---------------------------------------------------------------------------

class FollowedUsersLookupError(Exception):
    """Raised when followed-user lookup fails."""


async def get_followed_user_dids(user_did: str, limit: int) -> list[str]:
    base_url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"
    followed_dids: list[str] = []
    cursor: str | None = None

    if limit <= 0:
        return followed_dids

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while len(followed_dids) < limit:
                page_limit = min(FOLLOWS_PAGE_LIMIT, limit - len(followed_dids))
                params = {"actor": user_did, "limit": page_limit}
                if cursor:
                    params["cursor"] = cursor

                resp = await client.get(base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                follows = data.get("follows", [])
                if not isinstance(follows, list):
                    raise FollowedUsersLookupError(
                        f"Unexpected follows response for {user_did}"
                    )

                followed_dids.extend(
                    follow["did"]
                    for follow in follows
                    if isinstance(follow, dict) and isinstance(follow.get("did"), str)
                )

                cursor = data.get("cursor")
                if not isinstance(cursor, str) or not cursor:
                    break
    except (httpx.HTTPError, ValueError) as exc:
        raise FollowedUsersLookupError(
            f"Failed to fetch followed users for {user_did}"
        ) from exc

    return followed_dids[:limit]


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

async def followed_users_search(
    es,
    user_did: str,
    num_candidates: int,
    generator_name: str | None = None,
    video_only: bool = False,
    exclude_uris: list[str] | None = None,
) -> list[CandidatePost]:
    """Fetch posts from users followed by user_did from the ``posts`` index."""

    filters: list[dict] = []
    if video_only:
        filters.append({"term": {"contains_video": True}})

    must_not: list[dict] = []
    if exclude_uris:
        must_not.append({"terms": {"at_uri": exclude_uris}})

    followed_dids: list[str] = await get_followed_user_dids(
        user_did,
        limit=MAX_FOLLOWED_USERS
    )
    if not followed_dids:
        return []

    query = {
        "bool": {
            "filter": [
                *filters,
                {"terms": {"author_did": followed_dids}},
            ],
            **("must_not" and {"must_not": must_not} if must_not else {}),
        }
    }

    resp = await es.search(
        index="posts",
        query=query,
        size=num_candidates,
        sort=[{"created_at": "desc"}],
    )
    return candidate_posts_from_es_response(resp, generator_name=generator_name)


class FollowedUsersCandidateGenerator(CandidateGenerator):
    """Returns the last N posts from users that the requesting user follows."""

    @property
    def name(self) -> str:
        return "followed_users"

    async def generate(
        self,
        es,
        user_did: str,
        num_candidates: int = 100,
        video_only: bool = False,
        exclude_uris: list[str] | None = None,
    ) -> CandidateResult:
        candidates = await followed_users_search(
            es,
            user_did,
            num_candidates,
            generator_name=self.name,
            video_only=video_only,
            exclude_uris=exclude_uris,
        )
        return CandidateResult(generator_name=self.name, candidates=candidates)
