"""Telemetry helpers for timing API and Elasticsearch operations."""

import logging
import time
from contextlib import asynccontextmanager


@asynccontextmanager
async def timed(logger: logging.Logger, label: str, **extra: object):
    """Async context manager that logs elapsed time for a labelled operation.

    Usage::

        async with timed(logger, "knn_search", index="posts_recent"):
            resp = await es.search(...)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "%s elapsed_ms=%.1f",
            label,
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms, **extra},
        )
