"""Structure interne mutable `_TaskEntry` protegee par _lock.

@spec specs/orchestration.spec.md v2.1.1 §7.6

_TaskEntry = un enregistrement mutable sous entry._lock qui suit le cycle de vie
d'une task :
    - PENDING (submitted_at, cancel_event non set, started_at/completed_at None)
    - RUNNING (status, started_at set au pickup)
    - terminal (completed_at set, _done_event.set)

`snapshot()` construit un SubagentResult frozen coherent depuis les champs sous
entry._lock.

Invariants §7.6 :
    1. submitted_at jamais mute apres creation.
    2. status transitions : PENDING -> RUNNING -> {COMPLETED/FAILED/CANCELLED/
       TIMED_OUT}. PENDING -> CANCELLED direct autorise (cancel pre-RUNNING).
    3. started_at set une seule fois (pickup RUNNING, ou None si
       PENDING->CANCELLED).
    4. completed_at set une seule fois (transition terminale).
    5. _timeout_triggered set avant cancel_event.set() dans
       _await_with_precedence.
    6. _done_event.set() apres transition terminale (wait() non-polling).
    7. Acquisition lock : toujours _state_lock -> entry._lock, jamais l'inverse.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from wincorp_odin.orchestration.result import SubagentResult, SubagentStatus


@dataclass
class _TaskEntry:
    """Mutable, protege par `_lock`. `snapshot()` retourne SubagentResult frozen."""

    task_id: str
    trace_id: str
    submitted_at: datetime
    cancel_event: threading.Event
    future: Future[Any] | None = None

    # Champs mutables sous _lock.
    status: SubagentStatus = SubagentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    ai_messages: tuple[dict[str, Any], ...] = ()
    _timeout_triggered: bool = False

    # Sync primitives.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> SubagentResult:
        """Construit un SubagentResult frozen coherent sous `_lock`.

        Acquiert self._lock le temps de lire les champs mutables, construit le
        SubagentResult hors lock (le constructeur `__post_init__` peut valider
        et dedup - operations sur un tuple copy, pas sur le registre).

        Returns:
            SubagentResult frozen snapshot.
        """
        with self._lock:
            task_id = self.task_id
            trace_id = self.trace_id
            status = self.status
            submitted_at = self.submitted_at
            started_at = self.started_at
            completed_at = self.completed_at
            result = self.result
            error = self.error
            ai_messages = self.ai_messages
        return SubagentResult(
            task_id=task_id,
            trace_id=trace_id,
            status=status,
            submitted_at=submitted_at,
            started_at=started_at,
            completed_at=completed_at,
            result=result,
            error=error,
            ai_messages=ai_messages,
        )
