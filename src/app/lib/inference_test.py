"""Tests for shared inference helpers."""

import asyncio

import pytest

from . import inference as inference_module


@pytest.fixture(autouse=True)
def clear_post_tower_uuid_cache():
    inference_module._post_tower_uuid_cache.clear()
    inference_module._post_tower_uuid_locks.clear()
    yield
    inference_module._post_tower_uuid_cache.clear()
    inference_module._post_tower_uuid_locks.clear()


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_reuses_successful_lookup(monkeypatch):
    calls = 0

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> str:
        nonlocal calls
        calls += 1
        return f"{base_url}:{api_key}:uuid"

    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    first = await inference_module.get_cached_post_tower_uuid("https://inference", "key")
    second = await inference_module.get_cached_post_tower_uuid("https://inference", "key")

    assert first == "https://inference:key:uuid"
    assert second == first
    assert calls == 1


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_collapses_concurrent_lookups(monkeypatch):
    calls = 0

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return "post-tower-uuid"

    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    results = await asyncio.gather(
        inference_module.get_cached_post_tower_uuid("https://inference", "key"),
        inference_module.get_cached_post_tower_uuid("https://inference", "key"),
    )

    assert results == ["post-tower-uuid", "post-tower-uuid"]
    assert calls == 1


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_does_not_cache_missing_uuid(monkeypatch):
    calls = 0

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    first = await inference_module.get_cached_post_tower_uuid("https://inference", "key")
    second = await inference_module.get_cached_post_tower_uuid("https://inference", "key")

    assert first is None
    assert second is None
    assert calls == 2


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_refreshes_stale_uuid(monkeypatch):
    now = 1000.0
    key = ("https://inference", "key")
    inference_module._post_tower_uuid_cache[key] = ("stale-uuid", now - 1)

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> str:
        return "new-uuid"

    monkeypatch.setattr(inference_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    result = await inference_module.get_cached_post_tower_uuid(
        "https://inference", "key"
    )

    assert result == "new-uuid"
    assert inference_module._post_tower_uuid_cache[key] == (
        "new-uuid",
        now + inference_module._POST_TOWER_UUID_TTL_SEC,
    )


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_returns_stale_uuid_on_refresh_error(
    monkeypatch,
):
    now = 1000.0
    key = ("https://inference", "key")
    inference_module._post_tower_uuid_cache[key] = ("stale-uuid", now - 1)

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> str:
        raise RuntimeError("ready down")

    monkeypatch.setattr(inference_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    result = await inference_module.get_cached_post_tower_uuid(
        "https://inference", "key"
    )

    assert result == "stale-uuid"
    assert inference_module._post_tower_uuid_cache[key] == ("stale-uuid", now - 1)


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_returns_stale_uuid_on_missing_refresh_uuid(
    monkeypatch,
):
    now = 1000.0
    key = ("https://inference", "key")
    inference_module._post_tower_uuid_cache[key] = ("stale-uuid", now - 1)

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> None:
        return None

    monkeypatch.setattr(inference_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    result = await inference_module.get_cached_post_tower_uuid(
        "https://inference", "key"
    )

    assert result == "stale-uuid"
    assert inference_module._post_tower_uuid_cache[key] == ("stale-uuid", now - 1)


@pytest.mark.asyncio
async def test_cached_post_tower_uuid_raises_refresh_error_after_stale_grace(
    monkeypatch,
):
    now = 1000.0
    key = ("https://inference", "key")
    expires_at = now - inference_module._POST_TOWER_UUID_STALE_GRACE_SEC - 1
    inference_module._post_tower_uuid_cache[key] = ("too-old-uuid", expires_at)

    async def fake_get_post_tower_uuid(base_url: str, api_key: str) -> str:
        raise RuntimeError("ready down")

    monkeypatch.setattr(inference_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(
        inference_module, "get_post_tower_uuid", fake_get_post_tower_uuid
    )

    with pytest.raises(RuntimeError, match="ready down"):
        await inference_module.get_cached_post_tower_uuid("https://inference", "key")
