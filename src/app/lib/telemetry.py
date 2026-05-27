"""Telemetry helpers for timing API and Elasticsearch operations."""

import logging
import os
import time
from contextlib import asynccontextmanager

from .request_context import get_request_id


@asynccontextmanager
async def timed(logger: logging.Logger, label: str, **extra: object):
    """Async context manager that logs elapsed time for a labelled operation.

    The wrapped block always runs. The log line is only emitted when
    ``GE_TIMING_LOGS=1`` is set in the environment, so production logs
    stay quiet until someone is actively debugging performance. When a
    request ID is set on the current ContextVar it is included as
    ``rid=<id>`` so a single ``grep`` ties together all spans from one
    request.

    Usage::

        async with timed(logger, "knn_search", index="posts_recent"):
            resp = await es.search(...)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        if os.environ.get("GE_TIMING_LOGS") == "1":
            elapsed_ms = (time.monotonic() - start) * 1000
            rid = get_request_id()
            if rid is not None:
                logger.info(
                    "%s rid=%s elapsed_ms=%.1f",
                    label,
                    rid,
                    elapsed_ms,
                    extra={"elapsed_ms": elapsed_ms, "rid": rid, **extra},
                )
            else:
                logger.info(
                    "%s elapsed_ms=%.1f",
                    label,
                    elapsed_ms,
                    extra={"elapsed_ms": elapsed_ms, **extra},
                )
