"""Tests SubagentSink Protocol + LogSink par defaut.

@spec specs/orchestration.spec.md v2.1.1 §3.7 + R16 scope on_end
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest

from wincorp_odin.orchestration.result import SubagentResult, SubagentStatus
from wincorp_odin.orchestration.sinks import LogSink, SubagentSink


def _make_result(
    *,
    status: SubagentStatus = SubagentStatus.RUNNING,
    started: bool = True,
    completed: bool = False,
    error: str | None = None,
) -> SubagentResult:
    """Helper construit un SubagentResult minimal."""
    return SubagentResult(
        task_id="t-1",
        trace_id="trace-1",
        status=status,
        submitted_at=datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC),
        started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC) if started else None,
        completed_at=(
            datetime(2026, 4, 23, 14, 0, 2, tzinfo=UTC) if completed else None
        ),
        result=None,
        error=error,
        ai_messages=(),
    )


def test_protocol_duck_type_ok() -> None:
    """Une classe qui implemente on_start/on_end satisfait SubagentSink."""

    class _Noop:
        def on_start(self, result: SubagentResult) -> None:
            pass

        def on_end(self, result: SubagentResult) -> None:
            pass

    sink: SubagentSink = _Noop()
    sink.on_start(_make_result())
    sink.on_end(_make_result(status=SubagentStatus.COMPLETED, completed=True))


def test_logsink_on_start_emits_json(caplog: pytest.LogCaptureFixture) -> None:
    """LogSink.on_start emet une ligne JSON INFO."""
    caplog.set_level(logging.INFO, logger="wincorp_odin.orchestration")
    sink = LogSink()
    sink.on_start(_make_result())
    assert any("subagent_start" in r.message for r in caplog.records)
    record = next(r for r in caplog.records if "subagent_start" in r.message)
    payload = json.loads(record.message)
    assert payload["event"] == "subagent_start"
    assert payload["task_id"] == "t-1"
    assert payload["status"] == "running"


def test_logsink_on_end_emits_json_with_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LogSink.on_end emet JSON INFO avec duration_ms."""
    caplog.set_level(logging.INFO, logger="wincorp_odin.orchestration")
    sink = LogSink()
    res = _make_result(status=SubagentStatus.COMPLETED, started=True, completed=True)
    sink.on_end(res)
    record = next(r for r in caplog.records if "subagent_end" in r.message)
    payload = json.loads(record.message)
    assert payload["event"] == "subagent_end"
    assert payload["status"] == "completed"
    assert payload["duration_ms"] == pytest.approx(1000.0)


def test_logsink_on_end_error_tronque_500(caplog: pytest.LogCaptureFixture) -> None:
    """LogSink.on_end tronque error a 500 chars (R17)."""
    caplog.set_level(logging.INFO, logger="wincorp_odin.orchestration")
    sink = LogSink()
    long_error = "x" * 800
    res = _make_result(
        status=SubagentStatus.FAILED,
        started=True,
        completed=True,
        error=long_error,
    )
    sink.on_end(res)
    record = next(r for r in caplog.records if "subagent_end" in r.message)
    payload = json.loads(record.message)
    assert len(payload["error"]) == 500


def test_logsink_on_end_error_none_preserved(caplog: pytest.LogCaptureFixture) -> None:
    """error=None preserve dans le payload."""
    caplog.set_level(logging.INFO, logger="wincorp_odin.orchestration")
    sink = LogSink()
    res = _make_result(status=SubagentStatus.COMPLETED, started=True, completed=True)
    sink.on_end(res)
    record = next(r for r in caplog.records if "subagent_end" in r.message)
    payload = json.loads(record.message)
    assert payload["error"] is None


def test_logsink_custom_logger_name(caplog: pytest.LogCaptureFixture) -> None:
    """LogSink avec logger_name custom."""
    caplog.set_level(logging.INFO, logger="custom.logger")
    sink = LogSink(logger_name="custom.logger")
    sink.on_start(_make_result())
    assert any(r.name == "custom.logger" for r in caplog.records)


def test_logsink_on_start_started_at_none_preserved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_start avec started_at=None (cas limite) -> JSON None preserve."""
    caplog.set_level(logging.INFO, logger="wincorp_odin.orchestration")
    sink = LogSink()
    # Scenario atypique mais passe : PENDING snapshot pas realise en pratique.
    sink.on_start(_make_result(status=SubagentStatus.PENDING, started=False))
    record = next(r for r in caplog.records if "subagent_start" in r.message)
    payload = json.loads(record.message)
    assert payload["started_at"] is None
