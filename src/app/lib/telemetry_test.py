"""Tests for telemetry.timed()."""

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from .metrics import MetricCollector, set_metric_collector
from .telemetry import timed


def _make_collector() -> tuple[MetricCollector, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    collector = MetricCollector._from_reader(reader, service_name="test", env="test")
    return collector, reader


@pytest.mark.asyncio
async def test_timed_records_metric_when_metric_name_given():
    collector, reader = _make_collector()
    set_metric_collector(collector)
    try:
        import logging
        async with timed(logging.getLogger("test"), "op", metric_name="op.duration_ms", tag="val"):
            pass
        data = reader.get_metrics_data()
        names: set[str] = set()
        if data is not None:
            names = {
                metric.name
                for rm in data.resource_metrics
                for sm in rm.scope_metrics
                for metric in sm.metrics
            }
        assert "op.duration_ms" in names
    finally:
        set_metric_collector(None)


@pytest.mark.asyncio
async def test_timed_no_metric_without_metric_name():
    collector, reader = _make_collector()
    set_metric_collector(collector)
    try:
        import logging
        async with timed(logging.getLogger("test"), "op"):
            pass
        data = reader.get_metrics_data()
        names: set[str] = set()
        if data is not None:
            names = {
                metric.name
                for rm in data.resource_metrics
                for sm in rm.scope_metrics
                for metric in sm.metrics
            }
        assert not names
    finally:
        set_metric_collector(None)


@pytest.mark.asyncio
async def test_timed_no_metric_without_collector():
    set_metric_collector(None)
    import logging
    async with timed(logging.getLogger("test"), "op", metric_name="op.duration_ms"):
        pass
    # No exception means we gracefully handle missing collector


@pytest.mark.asyncio
async def test_timed_attaches_extra_as_attributes():
    collector, reader = _make_collector()
    set_metric_collector(collector)
    try:
        import logging
        async with timed(logging.getLogger("test"), "op", metric_name="op.duration_ms", feed_name="nature"):
            pass
        data = reader.get_metrics_data()
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "op.duration_ms":
                        dp = metric.data.data_points[0]
                        assert dp.attributes.get("feed_name") == "nature"
    finally:
        set_metric_collector(None)
