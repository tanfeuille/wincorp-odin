"""Tests precedence statut final : R23 / R24 via injection directe _await_with_precedence.

@spec specs/orchestration.spec.md v2.1.1 §R9b

Tests isoles de `_await_with_precedence` avec Future pre-resolu et
cancel_event/_timeout_triggered pre-set, pour tester sans race reelle.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from datetime import UTC, datetime

from wincorp_odin.orchestration._entry import _TaskEntry
from wincorp_odin.orchestration.exceptions import SubagentCancelledException
from wincorp_odin.orchestration.executor import _await_with_precedence
from wincorp_odin.orchestration.result import SubagentStatus


def _make_entry() -> _TaskEntry:
    """Helper _TaskEntry minimal."""
    return _TaskEntry(
        task_id="t-1",
        trace_id="trace-1",
        submitted_at=datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC),
        cancel_event=threading.Event(),
    )


# --- R9b.6 : task retourne normalement -> COMPLETED ------------------------


def test_precedence_completed() -> None:
    """R9b.6 : exec_future.result() retourne valeur -> COMPLETED."""
    f: Future[int] = Future()
    f.set_result(42)
    entry = _make_entry()
    status, result, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.COMPLETED
    assert result == 42
    assert error is None


# --- R9b.2 : FuturesTimeoutError -> TIMED_OUT, _timeout_triggered, cancel_event set


def test_precedence_timeout_triggers_flag_and_event() -> None:
    """R9b.2 : timeout -> TIMED_OUT + _timeout_triggered=True + cancel_event.set()."""
    f: Future[int] = Future()  # jamais set_result -> timeout
    entry = _make_entry()
    status, result, error = _await_with_precedence(f, entry, timeout_sec=0.05)
    assert status == SubagentStatus.TIMED_OUT
    assert result is None
    assert error is not None
    assert "Timeout" in error
    assert entry._timeout_triggered is True
    assert entry.cancel_event.is_set()


# --- R9b.3a : SubagentCancelledException sans timeout -> CANCELLED ----------


def test_precedence_cancelled_exception_no_timeout() -> None:
    """R9b.3a : task raise Cancelled sans timeout_triggered -> CANCELLED."""
    f: Future[int] = Future()
    f.set_exception(SubagentCancelledException("cancelled par user"))
    entry = _make_entry()
    status, result, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.CANCELLED
    assert result is None
    assert error == "cancelled par user"


def test_precedence_cancelled_exception_default_message() -> None:
    """Cancelled sans message custom -> message par defaut."""
    f: Future[int] = Future()
    # SubagentCancelledException() avec message par defaut.
    exc = SubagentCancelledException()
    f.set_exception(exc)
    entry = _make_entry()
    _, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert error is not None


# --- R9b.3b : SubagentCancelledException + _timeout_triggered -> TIMED_OUT --


def test_precedence_r24_timeout_wins_over_cancelled() -> None:
    """R24 : si _timeout_triggered puis task raise Cancelled -> TIMED_OUT precedence."""
    f: Future[int] = Future()
    f.set_exception(SubagentCancelledException())
    entry = _make_entry()
    entry._timeout_triggered = True
    status, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.TIMED_OUT
    assert error is not None
    assert "Timeout" in error


# --- R9b.4a : autre Exception + _timeout_triggered -> TIMED_OUT -------------


def test_precedence_other_exception_timeout_triggered_to_timed_out() -> None:
    """Autre exc + _timeout_triggered -> TIMED_OUT."""
    f: Future[int] = Future()
    f.set_exception(RuntimeError("boom"))
    entry = _make_entry()
    entry._timeout_triggered = True
    status, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.TIMED_OUT
    assert error is not None
    assert "Timeout" in error


# --- R9b.4b : autre Exception + cancel_event.is_set() -> CANCELLED ---------


def test_precedence_other_exception_cancel_event_set_to_cancelled() -> None:
    """R9b.4b : autre exc + cancel_event deja set -> CANCELLED avec repr exc."""
    f: Future[int] = Future()
    f.set_exception(ValueError("manual"))
    entry = _make_entry()
    entry.cancel_event.set()
    status, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.CANCELLED
    assert error is not None
    assert "ValueError" in error


# --- R9b.4c : autre Exception sans cancel/timeout -> FAILED ----------------


def test_precedence_other_exception_no_cancel_to_failed() -> None:
    """R9b.4c : exception nominale -> FAILED."""
    f: Future[int] = Future()
    f.set_exception(RuntimeError("crash metier"))
    entry = _make_entry()
    status, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.FAILED
    assert error is not None
    assert "RuntimeError" in error
    assert "crash metier" in error


def test_precedence_error_truncated_500() -> None:
    """R17 : error tronque 500 chars."""
    f: Future[int] = Future()
    long_msg = "x" * 1000
    f.set_exception(RuntimeError(long_msg))
    entry = _make_entry()
    _, _, error = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert error is not None
    assert len(error) == 500


# --- EC55 asyncio.CancelledError traite comme Exception generique ----------


def test_ec55_asyncio_cancellederror_treated_as_failed() -> None:
    """EC55 : asyncio.CancelledError capture comme Exception -> FAILED."""
    import asyncio

    f: Future[int] = Future()
    f.set_exception(asyncio.CancelledError())
    entry = _make_entry()
    status, _, _ = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.FAILED


# --- EC56 : Cancelled sans event pre-set -> CANCELLED ----------------------


def test_ec56_cancelled_programmatic_direct() -> None:
    """EC56 : task raise Cancelled sans event set -> CANCELLED (precedence)."""
    f: Future[int] = Future()
    f.set_exception(SubagentCancelledException())
    entry = _make_entry()
    # cancel_event PAS set, _timeout_triggered False.
    status, _, _ = _await_with_precedence(f, entry, timeout_sec=1.0)
    assert status == SubagentStatus.CANCELLED
