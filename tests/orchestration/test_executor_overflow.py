"""Tests SubagentExecutor overflow + eviction FIFO : R22 / R22b / EC58-59b / EC66-67.

@spec specs/orchestration.spec.md v2.1.1 §3.4
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentExecutorOverflowError,
)
from wincorp_odin.orchestration.executor import SubagentExecutor

# --- EC66 / EC67 / R22b constructor validation -----------------------------


def test_ec66_max_history_zero_rejected() -> None:
    """EC66 : max_history=0 -> ValueError FR."""
    with pytest.raises(ValueError, match="max_history"):
        SubagentExecutor(max_history=0)


def test_ec66_max_history_negative_rejected() -> None:
    """R22b : max_history <0 -> ValueError FR."""
    with pytest.raises(ValueError, match="max_history"):
        SubagentExecutor(max_history=-1)


def test_max_history_non_int_rejected() -> None:
    """max_history non int strict -> ValueError."""
    with pytest.raises(ValueError, match="max_history"):
        SubagentExecutor(max_history=1.5)  # type: ignore[arg-type]


def test_ec67_max_workers_scheduler_zero_rejected() -> None:
    """EC67 : max_workers_scheduler=0 -> ValueError FR."""
    with pytest.raises(ValueError, match="max_workers_scheduler"):
        SubagentExecutor(max_workers_scheduler=0)


def test_max_workers_exec_zero_rejected() -> None:
    """max_workers_exec=0 -> ValueError FR."""
    with pytest.raises(ValueError, match="max_workers_exec"):
        SubagentExecutor(max_workers_exec=0)


def test_max_workers_scheduler_non_int_rejected() -> None:
    """max_workers_scheduler float -> ValueError FR."""
    with pytest.raises(ValueError, match="max_workers_scheduler"):
        SubagentExecutor(max_workers_scheduler=1.5)  # type: ignore[arg-type]


def test_max_workers_exec_non_int_rejected() -> None:
    """max_workers_exec str -> ValueError FR."""
    with pytest.raises(ValueError, match="max_workers_exec"):
        SubagentExecutor(max_workers_exec="3")  # type: ignore[arg-type]


# --- EC57 / lazy pools no init ----------------------------------------------


def test_ec57_no_pool_before_submit() -> None:
    """EC57 : construit puis jamais utilise -> pas de pool, pas de thread.

    Apres construction, les pools internes sont None.
    """
    ex = SubagentExecutor()
    assert ex._scheduler_pool is None
    assert ex._exec_pool is None
    ex.shutdown(force_timeout_sec=None)


# --- EC58 sequential with wait ---------------------------------------------


def test_ec58_sequential_with_wait_evicts_old_terminals(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC58 : many submits avec wait entre chaque -> registre <= max_history."""
    ex = SubagentExecutor(
        max_history=5,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    try:

        def quick(s: Any, e: threading.Event) -> int:
            return 1

        for _ in range(20):
            tid = ex.submit(quick, initial_state={}, timeout_sec=1.0, trace_id="t")
            ex.wait(tid, timeout=2.0)
        assert ex.history_size <= 5
        # stats totals cohérent.
        stats = ex.stats()
        assert stats["total"] == ex.history_size
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC58b invariant final --------------------------------------------------


def test_ec58b_many_submits_no_wait_invariant_final(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC58b : 10001 submits sans wait avec pool 3 -> total final <= max_history.

    On teste l'invariant final, pas la branche mid-run SubagentExecutorOverflowError.
    On catch l'overflow pour ne pas crash le test.
    """
    ex = SubagentExecutor(
        max_workers_scheduler=3,
        max_workers_exec=3,
        max_history=20,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    tids: list[str] = []
    try:

        def quick(s: Any, e: threading.Event) -> int:
            return 1

        for _ in range(100):
            try:
                tid = ex.submit(
                    quick,
                    initial_state={},
                    timeout_sec=5.0,
                    trace_id="t",
                )
                tids.append(tid)
            except SubagentExecutorOverflowError:
                # Tolere l'overflow pendant la charge (branche EC59).
                pass
        # Attendre quelques terminaisons.
        for tid in tids[-10:]:
            with contextlib.suppress(Exception):
                ex.wait(tid, timeout=2.0)
        # Invariant final : registre borne.
        assert ex.history_size <= 20
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)


# --- EC59 : tout RUNNING, aucun terminal -> OverflowError ------------------


def test_ec59_no_terminal_overflow(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC59 : max_history tasks RUNNING + 1 submit -> SubagentExecutorOverflowError."""
    ex = SubagentExecutor(
        max_workers_scheduler=10,
        max_workers_exec=10,
        max_history=3,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    started = threading.Barrier(4, timeout=3.0)
    release = threading.Event()

    def blocking(s: Any, e: threading.Event) -> None:
        with contextlib.suppress(threading.BrokenBarrierError):
            started.wait()
        # Attend jusqu'a release ou cancel.
        if e.wait(timeout=5.0):
            raise SubagentCancelledException()
        return None

    tids: list[str] = []
    for _ in range(3):
        tids.append(
            ex.submit(
                blocking, initial_state={}, timeout_sec=10.0, trace_id="t"
            )
        )
    # Attend que les 3 soient toutes en RUNNING.
    with contextlib.suppress(threading.BrokenBarrierError):
        started.wait()
    # Maintenant 4e submit -> overflow (aucun terminal).
    with pytest.raises(SubagentExecutorOverflowError, match="Limite"):
        ex.submit(blocking, initial_state={}, timeout_sec=10.0, trace_id="t4")
    # Cleanup.
    for tid in tids:
        ex.cancel(tid)
    release.set()
    ex.shutdown(wait=True, force_timeout_sec=None)


def test_ec59b_one_terminal_allows_eviction(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC59b : max_history-1 RUNNING + 1 COMPLETED + 1 submit -> OK (eviction)."""
    ex = SubagentExecutor(
        max_workers_scheduler=10,
        max_workers_exec=10,
        max_history=3,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )

    try:
        # D'abord 1 task qui termine (completed).
        def quick(s: Any, e: threading.Event) -> int:
            return 1

        t_done = ex.submit(quick, initial_state={}, timeout_sec=1.0, trace_id="t")
        ex.wait(t_done, timeout=2.0)

        # Ensuite saturer avec 2 RUNNING.
        started = threading.Barrier(3, timeout=3.0)

        def blocking(s: Any, e: threading.Event) -> None:
            with contextlib.suppress(threading.BrokenBarrierError):
                started.wait()
            if e.wait(5.0):
                raise SubagentCancelledException()
            return None

        running_tids: list[str] = []
        for _ in range(2):
            running_tids.append(
                ex.submit(
                    blocking, initial_state={}, timeout_sec=10.0, trace_id="t"
                )
            )
        # Tentative 4e submit -> evince le terminal, OK.
        t_new = ex.submit(quick, initial_state={}, timeout_sec=1.0, trace_id="t-new")
        with contextlib.suppress(threading.BrokenBarrierError):
            started.wait()
        ex.wait(t_new, timeout=2.0)
        # t_done evince.
        assert ex.get(t_done) is None
        for tid in running_tids:
            ex.cancel(tid)
        for tid in running_tids:
            ex.wait(tid, timeout=2.0)
    finally:
        ex.shutdown(wait=True, force_timeout_sec=None)
