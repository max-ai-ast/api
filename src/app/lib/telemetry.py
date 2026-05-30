"""Telemetry helpers for timing API and Elasticsearch operations."""

import logging
import os
import time
from contextlib import asynccontextmanager

from .request_context import get_request_id


@asynccontextmanager
async def timed(logger: logging.Logger, label: str, metric_name: str | None = None, **extra: object):
    """Async context manager that times a labelled operation.

    Timing logs are only emitted when ``GE_TIMING_LOGS=1`` is set.

    When *metric_name* is provided and a MetricCollector is configured,
    the elapsed milliseconds are recorded as a metric unconditionally —
    independent of ``GE_TIMING_LOGS``.  Any keyword extras become metric
    attributes (labels).

    Usage::

        async with timed(logger, "knn_search", metric_name="es.knn.duration_ms", index="posts_recent"):
            resp = await es.search(...)
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000

        if os.environ.get("GE_TIMING_LOGS") == "1":
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

        if metric_name is not None:
            from .metrics import get_metric_collector
            collector = get_metric_collector()
            if collector is not None:
                str_attrs = {k: str(v) for k, v in extra.items()}
                collector.record(metric_name, elapsed_ms, **str_attrs)
