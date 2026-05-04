"""Candidate generator for posts from followed users.

Returns the last N posts from users that the requesting user follows"""

import httpx

from ...models import CandidatePost
from .base import CandidateGenerator, CandidateResult
from .utils import candidate_posts_from_es_response

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Maximum number of followed users to use in the query
MAX_FOLLOWED_USERS = 1000


# ---------------------------------------------------------------------------
# Followed users API query
# ---------------------------------------------------------------------------

class FollowedUsersLookupError(Exception):
    """Raised when followed-user lookup fails."""


async def get_followed_user_dids(user_did: str, limit: int) -> list[str]:
    base_url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                base_url,
                params={"actor": user_did, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise FollowedUsersLookupError(
            f"Failed to fetch followed users for {user_did}"
        ) from exc

    follows = data.get("follows", [])
    if not isinstance(follows, list):
        raise FollowedUsersLookupError(
            f"Unexpected follows response for {user_did}"
        )

    return [
        follow["did"]
        for follow in follows
        if isinstance(follow, dict) and isinstance(follow.get("did"), str)
    ]


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
