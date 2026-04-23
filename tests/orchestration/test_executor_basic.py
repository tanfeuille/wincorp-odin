"""Tests basic SubagentExecutor : R8-R13 happy path + EC30-35.

@spec specs/orchestration.spec.md v2.1.1 §3.4
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentExecutorClosedError,
    SubagentTaskIdConflictError,
    SubagentTaskNotFoundError,
)
from wincorp_odin.orchestration.executor import SubagentExecutor
from wincorp_odin.orchestration.result import SubagentStatus

# --- R8 submit validation ordre ---------------------------------------------


def test_submit_task_not_callable_type_error(
    executor: SubagentExecutor,
) -> None:
    """R8.1 : task non callable -> TypeError FR."""
    with pytest.raises(TypeError, match="callable"):
        executor.submit("not-callable", initial_state={}, timeout_sec=1.0, trace_id="t")  # type: ignore[arg-type]


def test_submit_initial_state_not_mapping_type_error(
    executor: SubagentExecutor,
) -> None:
    """R8.2 : initial_state non Mapping -> TypeError FR."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(TypeError, match="Mapping"):
        executor.submit(task, initial_state=[], timeout_sec=1.0, trace_id="t")  # type: ignore[arg-type]


def test_ec35c_timeout_sec_str_type_error(executor: SubagentExecutor) -> None:
    """EC35c : timeout_sec='5' -> TypeError FR."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(TypeError, match="int ou float"):
        executor.submit(task, initial_state={}, timeout_sec="5", trace_id="t")  # type: ignore[arg-type]


def test_submit_timeout_sec_bool_rejected(executor: SubagentExecutor) -> None:
    """timeout_sec bool -> TypeError (bool exclu même s'il est int)."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(TypeError, match="int ou float"):
        executor.submit(task, initial_state={}, timeout_sec=True, trace_id="t")  # type: ignore[arg-type]


def test_ec35_timeout_sec_nan_value_error(executor: SubagentExecutor) -> None:
    """EC35 : timeout_sec NaN -> ValueError FR."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="NaN"):
        executor.submit(
            task, initial_state={}, timeout_sec=float("nan"), trace_id="t"
        )


def test_ec33_timeout_sec_zero_value_error(executor: SubagentExecutor) -> None:
    """EC33 : timeout_sec=0 -> ValueError FR."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="strictement positif"):
        executor.submit(task, initial_state={}, timeout_sec=0.0, trace_id="t")


def test_ec34_timeout_sec_negative_zero_rejected(executor: SubagentExecutor) -> None:
    """EC34 : -0.0 rejete (<= 0)."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="strictement positif"):
        executor.submit(task, initial_state={}, timeout_sec=-0.0, trace_id="t")


def test_ec35b_timeout_sec_negative_inf_rejected(executor: SubagentExecutor) -> None:
    """EC35b : -inf -> ValueError FR."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="strictement positif"):
        executor.submit(
            task, initial_state={}, timeout_sec=float("-inf"), trace_id="t"
        )


def test_ec32_timeout_sec_inf_accepted(executor: SubagentExecutor) -> None:
    """EC32 : +inf -> accepte (no timeout)."""

    def task(s: Any, e: threading.Event) -> int:
        return 42

    tid = executor.submit(
        task, initial_state={}, timeout_sec=float("inf"), trace_id="t"
    )
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.COMPLETED
    assert res.result == 42


def test_submit_trace_id_empty_value_error(executor: SubagentExecutor) -> None:
    """R8.6 : trace_id vide -> ValueError."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="trace_id"):
        executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="")


def test_submit_trace_id_non_str_value_error(executor: SubagentExecutor) -> None:
    """R8.6 : trace_id non str -> ValueError."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="trace_id"):
        executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id=123)  # type: ignore[arg-type]


def test_submit_task_id_empty_value_error(executor: SubagentExecutor) -> None:
    """R8.7 : task_id='' -> ValueError."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="task_id"):
        executor.submit(
            task, initial_state={}, timeout_sec=1.0, trace_id="t", task_id=""
        )


def test_submit_task_id_non_str_value_error(executor: SubagentExecutor) -> None:
    """R8.7 : task_id=123 (non-str) -> ValueError."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    with pytest.raises(ValueError, match="task_id"):
        executor.submit(
            task, initial_state={}, timeout_sec=1.0, trace_id="t", task_id=123  # type: ignore[arg-type]
        )


# --- Happy path ------------------------------------------------------------


def test_submit_returns_task_id_from_uuid_factory(
    executor: SubagentExecutor,
) -> None:
    """submit sans task_id -> uuid_factory appele, retour task_id."""

    def task(s: Any, e: threading.Event) -> str:
        return "ok"

    tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    assert tid.startswith("uuid-")
    res = executor.wait(tid, timeout=2.0)
    assert res.task_id == tid
    assert res.status == SubagentStatus.COMPLETED
    assert res.result == "ok"


def test_submit_task_id_provided_used(executor: SubagentExecutor) -> None:
    """task_id fourni -> utilise tel quel."""

    def task(s: Any, e: threading.Event) -> str:
        return "ok"

    tid = executor.submit(
        task, initial_state={}, timeout_sec=1.0, trace_id="t", task_id="my-id"
    )
    assert tid == "my-id"


def test_ec30_task_returns_none_status_completed(
    executor: SubagentExecutor,
) -> None:
    """EC30 : task retourne None -> status COMPLETED, result None."""

    def task(s: Any, e: threading.Event) -> None:
        return None

    tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.COMPLETED
    assert res.result is None


def test_ec31_mappingproxy_initial_state_accepte(executor: SubagentExecutor) -> None:
    """EC31 : MappingProxyType accepte en initial_state."""
    from types import MappingProxyType

    captured: dict[str, Any] = {}

    def task(s: Any, e: threading.Event) -> str:
        captured["state"] = dict(s)
        return "ok"

    state = MappingProxyType({"k": "v"})
    tid = executor.submit(task, initial_state=state, timeout_sec=1.0, trace_id="t")
    executor.wait(tid, timeout=2.0)
    assert captured["state"] == {"k": "v"}


# --- R11 get() --------------------------------------------------------------


def test_get_unknown_returns_none(executor: SubagentExecutor) -> None:
    """R11 : get(inconnu) -> None."""
    assert executor.get("inconnu") is None


def test_get_returns_snapshot(executor: SubagentExecutor) -> None:
    """R11 : get(tid) retourne un SubagentResult snapshot."""

    def task(s: Any, e: threading.Event) -> int:
        return 1

    tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    executor.wait(tid, timeout=2.0)
    snap = executor.get(tid)
    assert snap is not None
    assert snap.task_id == tid
    assert snap.status == SubagentStatus.COMPLETED


# --- R11b wait() ------------------------------------------------------------


def test_ec60_wait_unknown_raises(executor: SubagentExecutor) -> None:
    """EC60 : wait(inconnu) -> SubagentTaskNotFoundError."""
    with pytest.raises(SubagentTaskNotFoundError, match="inconnu"):
        executor.wait("inconnu")


def test_ec61_wait_timeout_raises_builtin(executor: SubagentExecutor) -> None:
    """EC61 : wait(timeout court) sur task longue -> TimeoutError builtin."""

    def task(s: Any, e: threading.Event) -> None:
        e.wait(1.0)  # long
        return None

    tid = executor.submit(task, initial_state={}, timeout_sec=5.0, trace_id="t")
    with pytest.raises(TimeoutError, match="Timeout"):
        executor.wait(tid, timeout=0.05)
    # Cleanup.
    executor.cancel(tid)
    executor.wait(tid, timeout=2.0)


# --- R10 cancel() -----------------------------------------------------------


def test_ec4_cancel_unknown_returns_false(executor: SubagentExecutor) -> None:
    """EC4 : cancel(inconnu) -> False, pas d'exception."""
    assert executor.cancel("inconnu") is False


def test_ec3_cancel_terminal_returns_true(executor: SubagentExecutor) -> None:
    """EC3 : cancel(terminal) -> True, no-op."""

    def task(s: Any, e: threading.Event) -> int:
        return 42

    tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
    executor.wait(tid, timeout=2.0)
    assert executor.cancel(tid) is True
    # Status inchange (COMPLETED).
    snap = executor.get(tid)
    assert snap is not None
    assert snap.status == SubagentStatus.COMPLETED


def test_ec21_double_cancel_idempotent(executor: SubagentExecutor) -> None:
    """EC21 : 2 cancel successifs -> True True."""

    def task(s: Any, e: threading.Event) -> None:
        if e.wait(0.5):
            raise SubagentCancelledException()
        return None

    tid = executor.submit(task, initial_state={}, timeout_sec=5.0, trace_id="t")
    a = executor.cancel(tid)
    b = executor.cancel(tid)
    assert a is True
    assert b is True
    res = executor.wait(tid, timeout=2.0)
    assert res.status == SubagentStatus.CANCELLED


# --- R13 context manager ----------------------------------------------------


def test_r13_context_manager_shutdown_on_exit(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """R13 : __exit__ appelle shutdown."""
    with SubagentExecutor(
        _now_factory=frozen_now, _uuid_factory=uuid_factory_seq
    ) as ex:

        def task(s: Any, e: threading.Event) -> str:
            return "ok"

        tid = ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        ex.wait(tid, timeout=2.0)
    # Apres le with : executor ferme -> submit leve.
    with pytest.raises(SubagentExecutorClosedError):
        ex.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t2")


def test_ec63_exit_on_body_exception_propagates(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC63 : __exit__ appele meme si body raise. Exception body propagee."""
    with pytest.raises(RuntimeError, match="body boom"), SubagentExecutor(
        _now_factory=frozen_now, _uuid_factory=uuid_factory_seq
    ) as ex:
        raise RuntimeError("body boom")
    # Executor ferme apres exit.
    with pytest.raises(SubagentExecutorClosedError):
        ex.submit(lambda s, e: None, initial_state={}, timeout_sec=1.0, trace_id="t")


# --- R12 shutdown idempotent -----------------------------------------------


def test_ec5_shutdown_idempotent(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC5 : shutdown appele 2x -> 2e appel no-op."""
    ex = SubagentExecutor(_now_factory=frozen_now, _uuid_factory=uuid_factory_seq)
    ex.shutdown(force_timeout_sec=None)
    ex.shutdown(force_timeout_sec=None)  # no-op
    with pytest.raises(SubagentExecutorClosedError):
        ex.submit(lambda s, e: None, initial_state={}, timeout_sec=1.0, trace_id="t")


def test_ec7_submit_after_shutdown_raises(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> None:
    """EC7 : submit apres shutdown -> SubagentExecutorClosedError."""
    ex = SubagentExecutor(_now_factory=frozen_now, _uuid_factory=uuid_factory_seq)
    ex.shutdown(force_timeout_sec=None)
    with pytest.raises(SubagentExecutorClosedError, match="ferme"):
        ex.submit(lambda s, e: None, initial_state={}, timeout_sec=1.0, trace_id="t")


# --- R8.9/10 conflict & ecrasement -----------------------------------------


def test_ec17_submit_task_id_active_conflict(executor: SubagentExecutor) -> None:
    """EC17 : task_id actif (PENDING/RUNNING) -> SubagentTaskIdConflictError."""

    def slow(s: Any, e: threading.Event) -> None:
        e.wait(2.0)
        return None

    _ = executor.submit(
        slow,
        initial_state={},
        timeout_sec=5.0,
        trace_id="t1",
        task_id="fixed",
    )
    with pytest.raises(SubagentTaskIdConflictError, match="fixed"):
        executor.submit(
            slow,
            initial_state={},
            timeout_sec=5.0,
            trace_id="t2",
            task_id="fixed",
        )


def test_ec16_submit_task_id_terminal_overwrite(executor: SubagentExecutor) -> None:
    """EC16 : task_id existant terminal -> ecrasement silencieux (log DEBUG)."""

    def quick(s: Any, e: threading.Event) -> str:
        return "fast"

    t1 = executor.submit(
        quick,
        initial_state={},
        timeout_sec=1.0,
        trace_id="t1",
        task_id="same",
    )
    executor.wait(t1, timeout=2.0)
    # Terminal -> ecrasement OK.
    t2 = executor.submit(
        quick,
        initial_state={},
        timeout_sec=1.0,
        trace_id="t2",
        task_id="same",
    )
    assert t1 == t2 == "same"
    executor.wait(t2, timeout=2.0)


def test_ec17b_submit_cancel_submit_same_id_succeed(
    executor: SubagentExecutor,
) -> None:
    """EC17b : submit -> cancel -> submit meme id succeed (apres cancel, id est terminal)."""

    def slow(s: Any, e: threading.Event) -> None:
        if e.wait(1.0):
            raise SubagentCancelledException()
        return None

    _ = executor.submit(
        slow,
        initial_state={},
        timeout_sec=5.0,
        trace_id="t1",
        task_id="reuse",
    )
    assert executor.cancel("reuse") is True
    # cancel passe en terminal immediat (PENDING ou RUNNING). wait pour safety.
    executor.wait("reuse", timeout=2.0)
    # 2e submit meme id autorise (ecrasement silencieux).
    t2 = executor.submit(
        slow,
        initial_state={},
        timeout_sec=5.0,
        trace_id="t2",
        task_id="reuse",
    )
    assert t2 == "reuse"
    executor.cancel("reuse")
    executor.wait("reuse", timeout=2.0)


# --- stats / clear_history --------------------------------------------------


def test_stats_totals(executor: SubagentExecutor) -> None:
    """stats retourne comptage par status."""

    def quick(s: Any, e: threading.Event) -> int:
        return 1

    for _ in range(3):
        tid = executor.submit(quick, initial_state={}, timeout_sec=1.0, trace_id="t")
        executor.wait(tid, timeout=2.0)
    s = executor.stats()
    assert s["total"] == 3
    assert s["completed"] == 3
    assert s["pending"] == 0


def test_ec62_clear_history_removes_terminal_only(
    executor: SubagentExecutor,
) -> None:
    """EC62 : clear_history supprime seulement les terminaux."""
    import threading as thmod

    started = thmod.Event()
    release = thmod.Event()

    def done_task(s: Any, e: thmod.Event) -> int:
        return 1

    def running_task(s: Any, e: thmod.Event) -> None:
        started.set()
        release.wait(2.0)
        return None

    t1 = executor.submit(done_task, initial_state={}, timeout_sec=1.0, trace_id="t")
    t2 = executor.submit(done_task, initial_state={}, timeout_sec=1.0, trace_id="t")
    executor.wait(t1, timeout=2.0)
    executor.wait(t2, timeout=2.0)
    t3 = executor.submit(running_task, initial_state={}, timeout_sec=5.0, trace_id="t")
    started.wait(2.0)
    removed = executor.clear_history()
    assert removed == 2
    release.set()
    executor.wait(t3, timeout=2.0)


def test_history_size_property(executor: SubagentExecutor) -> None:
    """history_size = nombre total d'entrees."""

    def task(s: Any, e: threading.Event) -> int:
        return 1

    for _ in range(5):
        tid = executor.submit(task, initial_state={}, timeout_sec=1.0, trace_id="t")
        executor.wait(tid, timeout=2.0)
    assert executor.history_size == 5
