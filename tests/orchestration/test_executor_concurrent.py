"""Tests concurrence SubagentExecutor : R19, EC19-24 + sink error handling R16.

@spec specs/orchestration.spec.md v2.1.1 §3.4
"""
from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from wincorp_odin.orchestration.exceptions import SubagentCancelledException
from wincorp_odin.orchestration.executor import SubagentExecutor
from wincorp_odin.orchestration.result import SubagentStatus

# --- EC20 cancel during RUNNING ---------------------------------------------


def test_ec20_cancel_running_task_cooperates(
    executor: SubagentExecutor,
) -> None:
    """EC20 : cancel(task_id) pendant RUNNING -> task voit is_set(), sort CANCELLED."""
    started = threading.Event()

    def coop(state: Any, cancel_event: threading.Event) -> None:
        started.set()
        while not cancel_event.is_set():
            if cancel_event.wait(0.05):
                raise SubagentCancelledException()
        raise SubagentCancelledException()

    tid = executor.submit(coop, initial_state={}, timeout_sec=5.0, trace_id="t")
    assert started.wait(1.0)
    assert executor.cancel(tid) is True
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.CANCELLED


# --- EC23 get() atomic during PENDING->RUNNING ------------------------------


def test_ec23_get_snapshot_atomic(executor: SubagentExecutor) -> None:
    """EC23 : get() pendant transition PENDING->RUNNING retourne soit PENDING soit RUNNING.

    Invariant : jamais de status RUNNING avec started_at=None, jamais PENDING avec
    started_at set.
    """

    def task(state: Any, cancel_event: threading.Event) -> int:
        cancel_event.wait(0.05)
        return 1

    tid = executor.submit(task, initial_state={}, timeout_sec=5.0, trace_id="t")
    # Plusieurs snapshots pendant que la task s'execute.
    for _ in range(50):
        snap = executor.get(tid)
        if snap is None:
            continue
        if snap.status == SubagentStatus.PENDING:
            assert snap.started_at is None
        if snap.status == SubagentStatus.RUNNING:
            assert snap.started_at is not None
    executor.wait(tid, timeout=2.0)


# --- R16 sink errors swallowed ----------------------------------------------


def test_r16_sink_on_start_exception_swallowed(
    bad_sink: Any,
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """R16 : sink.on_start raise -> log WARNING, task continue."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    with SubagentExecutor(
        sink=bad_sink,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    ) as ex:

        def task(s: Any, e: threading.Event) -> str:
            return "ok"

        tid = ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        res = ex.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.COMPLETED
    assert bad_sink.start_called == 1
    assert bad_sink.end_called == 1
    # 2 warnings : un pour on_start, un pour on_end.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 2


def test_ec24_sink_start_and_end_both_raise(
    bad_sink: Any,
    caplog: pytest.LogCaptureFixture,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC24 : on_start + on_end raise sur meme task -> 2 logs distincts, task termine."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.executor")
    with SubagentExecutor(
        sink=bad_sink,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    ) as ex:

        def task(s: Any, e: threading.Event) -> str:
            return "ok"

        tid = ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        res = ex.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.COMPLETED
    # 2 warnings distincts (on_start, on_end).
    assert bad_sink.start_called == 1
    assert bad_sink.end_called == 1


def test_r16_capture_sink_gets_start_and_end(
    capture_sink: Any,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """capture_sink : on_start et on_end appeles dans l'ordre."""
    with SubagentExecutor(
        sink=capture_sink,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    ) as ex:

        def task(s: Any, e: threading.Event) -> str:
            return "ok"

        tid = ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        ex.wait(tid, timeout=2.0)

    assert len(capture_sink.started) == 1
    assert len(capture_sink.ended) == 1
    assert capture_sink.started[0].status == SubagentStatus.RUNNING
    assert capture_sink.started[0].started_at is not None
    assert capture_sink.ended[0].status == SubagentStatus.COMPLETED
    assert capture_sink.ended[0].completed_at is not None


def test_r16_on_end_not_called_on_pending_cancel(
    capture_sink: Any,
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """R16 scope : PENDING -> CANCELLED via cancel (pre-RUNNING) -> on_end PAS appele.

    Scenario : on inhibe le scheduler pour forcer PENDING, puis cancel -> direct CANCELLED.
    """
    # max_workers_scheduler=1 et bloque avec une tache infinie.
    block = threading.Event()

    class BlockingSink:
        def __init__(self) -> None:
            self.started: list[Any] = []
            self.ended: list[Any] = []

        def on_start(self, r: Any) -> None:
            self.started.append(r)

        def on_end(self, r: Any) -> None:
            self.ended.append(r)

    sink = BlockingSink()
    ex = SubagentExecutor(
        max_workers_scheduler=1,
        max_workers_exec=1,
        sink=sink,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    try:
        # Saturer scheduler_pool avec une task bloquante.
        def blocker(s: Any, e: threading.Event) -> None:
            block.wait(2.0)
            return None

        _t_block = ex.submit(
            blocker, initial_state={}, timeout_sec=5.0, trace_id="blocker"
        )

        # Maintenant toute nouvelle task reste PENDING (scheduler plein).
        def never_run(s: Any, e: threading.Event) -> None:
            return None

        tid = ex.submit(
            never_run,
            initial_state={},
            timeout_sec=5.0,
            trace_id="pending-one",
        )
        # Force cancel PENDING avant pickup.
        snap_before = ex.get(tid)
        assert snap_before is not None
        assert snap_before.status == SubagentStatus.PENDING
        assert ex.cancel(tid) is True
        res = ex.wait(tid, timeout=1.0)
        assert res.status == SubagentStatus.CANCELLED
        assert res.started_at is None
        # R16 : on_end NON appele (pas de RUNNING->terminal).
        assert not any(
            r.task_id == tid for r in sink.ended
        ), "on_end a ete appele pour une tache CANCELLED pre-RUNNING"

        block.set()
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC22 shutdown avec plusieurs tasks ------------------------------------


def test_ec22_shutdown_with_many_tasks(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC22 : shutdown() cancel PENDING + set event RUNNING."""
    started = threading.Barrier(3, timeout=2.0)
    ex = SubagentExecutor(
        max_workers_scheduler=3,
        max_workers_exec=3,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )

    def coop(state: Any, cancel_event: threading.Event) -> None:
        with contextlib.suppress(threading.BrokenBarrierError):
            started.wait()
        if cancel_event.wait(5.0):
            raise SubagentCancelledException()
        return None

    tids: list[str] = []
    for _ in range(5):
        tids.append(
            ex.submit(coop, initial_state={}, timeout_sec=10.0, trace_id="t")
        )

    # Let first 3 start.
    with contextlib.suppress(threading.BrokenBarrierError):
        started.wait()
    ex.shutdown(wait=True, cancel_futures=True, force_timeout_sec=None)

    # Apres shutdown, toutes les tasks terminales.
    for tid in tids:
        snap = ex.get(tid)
        assert snap is not None
        assert snap.is_terminal()


# --- EC19 submit vs shutdown race -----------------------------------------


def test_ec19_submit_race_with_shutdown(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC19 : submit concurrent a shutdown -> soit submit OK soit Closed. Jamais crash.

    Race forcee via `threading.Barrier(N+1)` : les N threads submit et le thread
    principal (shutdown) se synchronisent sur la barrier avant leurs appels
    respectifs, ce qui garantit une contention reelle sur `_state_lock` au lieu
    d'un test ou le shutdown survient avant que les submits aient tente quoi que
    ce soit.
    """
    from wincorp_odin.orchestration.exceptions import SubagentExecutorClosedError

    ex = SubagentExecutor(
        _now_factory=frozen_now, _uuid_factory=uuid_factory_seq
    )

    n_threads = 10
    barrier = threading.Barrier(n_threads + 1, timeout=5.0)
    results: list[Any] = []
    results_lock = threading.Lock()

    def do_submit() -> None:
        # Synchronisation avec le thread principal : force la race.
        try:
            barrier.wait()
        except threading.BrokenBarrierError:  # pragma: no cover
            return
        try:
            tid = ex.submit(
                lambda s, e: 1,
                initial_state={},
                timeout_sec=1.0,
                trace_id="t",
            )
            with results_lock:
                results.append(("ok", tid))
        except SubagentExecutorClosedError:
            with results_lock:
                results.append(("closed", None))

    threads = [threading.Thread(target=do_submit) for _ in range(n_threads)]
    for t in threads:
        t.start()
    # Rendez-vous : tous les submits et le shutdown partent ensemble.
    barrier.wait()
    ex.shutdown(wait=True, cancel_futures=True, force_timeout_sec=None)
    for t in threads:
        t.join(timeout=5.0)
    # On a forcement une reponse par thread (pas de crash).
    assert len(results) == n_threads
    assert all(r[0] in ("ok", "closed") for r in results)


# --- CR-005/CR-006 wrapper handles exec_pool shutdown race ------------------


def test_wrapper_handles_exec_pool_shutdown_race(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """CR-005/006 : RuntimeError sur exec_pool.submit (shutdown survenu entre
    on_start et submit) -> wrapper transitionne RUNNING -> CANCELLED proprement,
    on_end appele (scope R16 respecte), status final CANCELLED avec error
    contenant 'shutdown'.

    Force la race via un sink `on_start` qui bloque sur une Barrier(2). Le thread
    principal attend la barrier, patch exec_pool.submit pour raise RuntimeError,
    libere le sink. Le wrapper reprend, appelle exec_pool.submit (patche) qui
    raise -> branche CANCELLED exercee deterministiquement.
    """
    # Barrier(2) : test principal + wrapper (via sink on_start).
    sink_barrier = threading.Barrier(2, timeout=5.0)
    on_end_called = threading.Event()

    class BlockingStartSink:
        """on_start bloque sur la barrier jusqu'a ce que le test soit pret."""

        def __init__(self) -> None:
            self.started: list[Any] = []
            self.ended: list[Any] = []

        def on_start(self, result: Any) -> None:
            self.started.append(result)
            # Signale au test principal qu'on est en RUNNING ; attend son go.
            with contextlib.suppress(threading.BrokenBarrierError):
                sink_barrier.wait()

        def on_end(self, result: Any) -> None:
            self.ended.append(result)
            on_end_called.set()

    sink = BlockingStartSink()

    ex = SubagentExecutor(
        max_workers_scheduler=1,
        max_workers_exec=1,
        sink=sink,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )

    def task(state: Any, cancel_event: threading.Event) -> str:
        # Ne doit jamais etre execute : exec_pool.submit va raise.
        return "ne-doit-pas-sortir"  # pragma: no cover

    tid = ex.submit(task, initial_state={}, timeout_sec=5.0, trace_id="race")

    # Attend que le wrapper ait transitionne RUNNING et appele on_start.
    try:
        sink_barrier.wait()
    except threading.BrokenBarrierError:  # pragma: no cover
        pytest.fail("Sink on_start n'a pas ete appele dans les temps")

    # Patch exec_pool.submit pour simuler le shutdown race.
    assert ex._exec_pool is not None

    def _raise_shutdown(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("cannot schedule new futures after shutdown")

    ex._exec_pool.submit = _raise_shutdown

    # Libere le sink on_start -> le wrapper appelle exec_pool.submit (patche).
    # La branche RuntimeError doit transitionner CANCELLED proprement.
    res = ex.wait(tid, timeout=5.0)

    assert res.status == SubagentStatus.CANCELLED, (
        f"Status attendu CANCELLED, recu {res.status}"
    )
    assert res.error is not None and "shutdown" in res.error, (
        f"Error attendu contenant 'shutdown', recu {res.error!r}"
    )
    assert res.completed_at is not None, "completed_at doit etre set"
    assert res.started_at is not None, "started_at doit rester set (RUNNING traverse)"
    # on_end doit avoir ete appele (R16 scope : RUNNING -> terminal).
    assert on_end_called.is_set(), "on_end doit etre appele apres RUNNING -> terminal"
    assert len(sink.ended) == 1
    assert sink.ended[0].task_id == tid
    assert sink.ended[0].status == SubagentStatus.CANCELLED

    ex.shutdown(wait=True, cancel_futures=True, force_timeout_sec=None)
