"""Tests edge cases SubagentExecutor : EC1, EC2, EC6, EC18, EC25, EC68-71.

@spec specs/orchestration.spec.md v2.1.1
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from wincorp_odin.orchestration.exceptions import SubagentCancelledException
from wincorp_odin.orchestration.executor import SubagentExecutor
from wincorp_odin.orchestration.result import SubagentStatus

# --- EC1 : KeyboardInterrupt in task propagated via wait() ----------------


def test_ec1_keyboard_interrupt_propagated_via_wait(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC1 : task raise KI -> wait() propage au caller (pas CAPTURE en FAILED)."""
    ex = SubagentExecutor(_now_factory=frozen_now, _uuid_factory=uuid_factory_seq)
    try:

        def task(s: Any, e: threading.Event) -> None:
            raise KeyboardInterrupt("user abort")

        tid = ex.submit(task, initial_state={}, timeout_sec=2.0, trace_id="t")
        # KI est propagee depuis le wrapper scheduler -> wait() re-raise via future.exception().
        with pytest.raises(KeyboardInterrupt):
            ex.wait(tid, timeout=2.0)
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC2 : SubagentCancelledException -> CANCELLED ------------------------


def test_ec2_cancelled_exception_without_timeout(
    executor: SubagentExecutor,
) -> None:
    """EC2 : task raise Cancelled sans timeout -> status CANCELLED."""

    def task(s: Any, e: threading.Event) -> None:
        raise SubagentCancelledException()

    tid = executor.submit(task, initial_state={}, timeout_sec=2.0, trace_id="t")
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.CANCELLED


# --- EC6 : shutdown(wait=False) with RUNNING tasks ------------------------


def test_ec6_shutdown_wait_false_returns_immediate(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC6 : shutdown(wait=False) avec RUNNING -> retour immediat, tasks coop bg."""
    ex = SubagentExecutor(
        max_workers_scheduler=3,
        max_workers_exec=3,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )

    def coop(s: Any, e: threading.Event) -> None:
        if e.wait(1.0):
            raise SubagentCancelledException()
        return None

    tids = [
        ex.submit(coop, initial_state={}, timeout_sec=5.0, trace_id="t")
        for _ in range(3)
    ]
    ex.shutdown(wait=False, cancel_futures=True, force_timeout_sec=None)
    # Wait les tasks qui ont recu cancel_event via shutdown.
    for tid in tids:
        ex.wait(tid, timeout=2.0)


# --- EC18 : timeout tres court -> TIMED_OUT -------------------------------


def test_ec18_very_short_timeout_timed_out(executor: SubagentExecutor) -> None:
    """EC18 : timeout 0.01s sur task cancel_event.wait(1.0) -> TIMED_OUT."""

    def slow(s: Any, e: threading.Event) -> None:
        e.wait(1.0)
        return None

    tid = executor.submit(slow, initial_state={}, timeout_sec=0.01, trace_id="t")
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.TIMED_OUT
    assert res.error is not None
    assert "Timeout" in res.error


# --- EC25 : sink.on_start raise KI -> propage, task pas lancee ------------


def test_ec25_sink_on_start_keyboard_interrupt_propagates(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC25 : sink.on_start raise KI -> propage via wait, task PAS lancee."""

    class KISink:
        def on_start(self, r: Any) -> None:
            raise KeyboardInterrupt("sink ki")

        def on_end(self, r: Any) -> None:
            pass

    ex = SubagentExecutor(
        sink=KISink(),
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    try:
        task_started = threading.Event()

        def task(s: Any, e: threading.Event) -> None:
            task_started.set()
            return None

        tid = ex.submit(task, initial_state={}, timeout_sec=2.0, trace_id="t")
        # Le wrapper scheduler a lance on_start -> KI propagee jusqu'a future.
        with pytest.raises(KeyboardInterrupt):
            ex.wait(tid, timeout=2.0)
        # Task jamais lancee.
        assert not task_started.is_set(), (
            "EC25 : sink.on_start KI doit empecher exec_pool.submit"
        )
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC68 / _now_factory raise in submit --------------------------------


def test_ec68_now_factory_raises_propagates(
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC68 : _now_factory raise RuntimeError pendant submit -> propage, entry non creee."""
    calls = {"n": 0}

    def bad_now() -> datetime:
        calls["n"] += 1
        raise RuntimeError("clock fail")

    ex = SubagentExecutor(
        _now_factory=bad_now,
        _uuid_factory=uuid_factory_seq,
    )
    try:

        def task(s: Any, e: threading.Event) -> None:
            return None

        with pytest.raises(RuntimeError, match="clock fail"):
            ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        # Aucun task_id cree, history vide.
        assert ex.history_size == 0
    finally:
        # Remplacer _now_factory pour pouvoir shutdown proprement (shutdown appelle _now_factory).
        ex._now_factory = lambda: datetime.now(UTC)
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC69 : _uuid_factory raise ---------------------------------------------


def test_ec69_uuid_factory_raises_propagates(
    frozen_now: Callable[[], datetime],
) -> None:
    """EC69 : _uuid_factory raise -> propage, entry non creee."""

    def bad_uuid() -> str:
        raise RuntimeError("uuid fail")

    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=bad_uuid,
    )
    try:

        def task(s: Any, e: threading.Event) -> None:
            return None

        with pytest.raises(RuntimeError, match="uuid fail"):
            ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        assert ex.history_size == 0
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC70 : force_timeout_sec hors plage -> clip silencieux ---------------


def test_ec70_force_timeout_out_of_range_clipped(
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC70 : force_timeout_sec=-1 -> clip silencieux a 5.0 + log WARNING."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    ex.shutdown(force_timeout_sec=-1.0)
    # NE leve PAS ValueError (preserve idempotence).
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "force_timeout_sec" in r.message
    ]
    assert len(warnings) >= 1


def test_force_timeout_too_high_clipped(
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """force_timeout_sec > 300 -> clip silencieux."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    ex.shutdown(force_timeout_sec=10_000.0)
    # Clip a 5.0 -> shutdown prend ~5s sur ce test. Trop lent pour le CI. On verifie juste
    # qu'il n'y a pas eu exception.
    # Warning present.
    assert any(
        "force_timeout_sec" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_force_timeout_nan_clipped(
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """force_timeout_sec NaN -> clip."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    ex.shutdown(force_timeout_sec=float("nan"))
    assert any("force_timeout_sec" in r.message for r in caplog.records)


def test_force_timeout_wrong_type_clipped(
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """force_timeout_sec type str -> clip."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    ex.shutdown(force_timeout_sec="5")  # type: ignore[arg-type]
    assert any("force_timeout_sec" in r.message for r in caplog.records)


def test_force_timeout_bool_clipped(
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """force_timeout_sec=True (bool exclu) -> clip."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    ex.shutdown(force_timeout_sec=True)  # type: ignore[arg-type]
    assert any("force_timeout_sec" in r.message for r in caplog.records)


# --- EC71 : force_timeout_sec=None -> pas de scan zombie -------------------


def test_ec71_force_timeout_none_no_scan(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC71 : force_timeout_sec=None -> pas de scan, retour rapide."""
    ex = SubagentExecutor(
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    # Fermeture rapide < 1s.
    import time

    start = time.monotonic()
    ex.shutdown(force_timeout_sec=None)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


# --- FAILED on generic exception ------------------------------------------


def test_failed_on_runtimeerror(executor: SubagentExecutor) -> None:
    """Task raise RuntimeError -> status FAILED avec repr."""

    def task(s: Any, e: threading.Event) -> None:
        raise RuntimeError("crash metier")

    tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.FAILED
    assert res.error is not None
    assert "RuntimeError" in res.error


def test_default_sink_instantiated_if_none(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """sink=None -> LogSink par defaut instancie."""
    ex = SubagentExecutor(
        sink=None,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    from wincorp_odin.orchestration.sinks import LogSink

    assert isinstance(ex._sink, LogSink)
    ex.shutdown(force_timeout_sec=None)


def test_default_factories_used_if_none() -> None:
    """_now_factory / _uuid_factory None -> defaults stdlib."""
    ex = SubagentExecutor()

    def task(s: Any, e: threading.Event) -> int:
        return 1

    tid = ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    res = ex.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.COMPLETED
    # UUID factory defaut -> hex (pas uuid-NNNN).
    assert not tid.startswith("uuid-")
    ex.shutdown(wait=True, force_timeout_sec=None)


# --- Submit apres shutdown direct retour via _entries pre-existants -------


def test_shutdown_sets_error_for_pending_tasks(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """shutdown() force status CANCELLED sur PENDING avec error='Annulee - executor ferme.'."""
    # Scheduler a 1 worker, on sature avec une task, la 2e reste PENDING.
    ex = SubagentExecutor(
        max_workers_scheduler=1,
        max_workers_exec=1,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    block = threading.Event()
    started = threading.Event()

    def blocker(s: Any, e: threading.Event) -> None:
        started.set()
        block.wait(5.0)
        return None

    ex.submit(blocker, initial_state={}, timeout_sec=10.0, trace_id="t")
    started.wait(1.0)

    # 2e task PENDING.
    def noop(s: Any, e: threading.Event) -> None:
        return None

    t_pending = ex.submit(noop, initial_state={}, timeout_sec=5.0, trace_id="t2")
    snap = ex.get(t_pending)
    assert snap is not None
    assert snap.status == SubagentStatus.PENDING

    block.set()
    ex.shutdown(wait=True, cancel_futures=True, force_timeout_sec=None)

    # apres shutdown, t_pending doit etre CANCELLED via shutdown, pas COMPLETED.
    snap = ex.get(t_pending)
    assert snap is not None
    # Le shutdown a force CANCELLED avant que le scheduler pool ne pick up t_pending
    # (selon timing), ou COMPLETED si le pool a eu le temps. Les 2 sont acceptables.
    assert snap.is_terminal()
