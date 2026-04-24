"""SubagentExecutor : orchestration concurrente bornee (Phase 2 DeerFlow).

@spec specs/orchestration.spec.md v2.1.2 §3.4 + R8-R13, R16, R18b, R19-R24

API non-bloquante :
    - submit()   retourne task_id immediatement (uuid4 si None).
    - wait()     bloquant (Event-based, pas de polling) -> SubagentResult terminal.
    - cancel()   non-bloquant, True si task_id existe, False sinon.
    - get()      snapshot atomique sous lock.
    - shutdown() idempotent, force CANCELLED sur PENDING, cancel_event sur RUNNING.
    - context manager : __enter__/__exit__ appellent shutdown(wait=True).

Architecture §7.5 :
    - 2 ThreadPoolExecutor : _scheduler_pool (wrapper orchestration),
      _exec_pool (payload utilisateur). Lazy init au 1er submit sous _state_lock.
    - 2 niveaux de lock : _state_lock (registre), entry._lock (par-entree).
    - Hierarchie anti-deadlock : _state_lock -> entry._lock.
"""
from __future__ import annotations

import logging
import math
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from wincorp_odin.orchestration._entry import _TaskEntry
from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentExecutorClosedError,
    SubagentExecutorOverflowError,
    SubagentTaskIdConflictError,
    SubagentTaskNotFoundError,
)
from wincorp_odin.orchestration.result import SubagentResult, SubagentStatus
from wincorp_odin.orchestration.sinks import LogSink, SubagentSink
from wincorp_odin.orchestration.types import TaskCallable

logger = logging.getLogger("wincorp_odin.orchestration.executor")


_FORCE_TIMEOUT_MIN = 0.1
_FORCE_TIMEOUT_MAX = 300.0
_FORCE_TIMEOUT_DEFAULT = 5.0


# ---------------------------------------------------------------------------
# Helper prive : _await_with_precedence (R9b)
# ---------------------------------------------------------------------------


def _await_with_precedence(
    exec_future: Future[Any],
    entry: _TaskEntry,
    timeout_sec: float,
) -> tuple[SubagentStatus, Any, str | None]:
    """Attend la terminaison de exec_future avec precedence R23/R24.

    Ordre (R9b) :
        1. exec_future.result(timeout=timeout_sec).
        2. FuturesTimeoutError :
           - sous entry._lock : _timeout_triggered=True, cancel_event.set()
           - retour (TIMED_OUT, None, "Timeout apres {timeout_sec}s.")
        3. SubagentCancelledException :
           - si _timeout_triggered -> (TIMED_OUT, None, ...) precedence R24
           - sinon -> (CANCELLED, None, ...)
        4. autre Exception :
           - si _timeout_triggered -> (TIMED_OUT, None, ...)
           - sinon si cancel_event.is_set() -> (CANCELLED, None, repr)
           - sinon -> (FAILED, None, repr)
        5. KeyboardInterrupt/SystemExit/GeneratorExit non captures, propages.
        6. task return normal -> (COMPLETED, value, None).

    Args:
        exec_future: Future du task dans exec_pool.
        entry: _TaskEntry avec cancel_event et _timeout_triggered.
        timeout_sec: timeout en secondes (float('inf') accepte).

    Returns:
        Tuple (status, result, error).
    """
    try:
        value = exec_future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        with entry._lock:
            entry._timeout_triggered = True
            entry.cancel_event.set()
        return (
            SubagentStatus.TIMED_OUT,
            None,
            f"Timeout apres {timeout_sec}s.",
        )
    except SubagentCancelledException as exc:
        with entry._lock:
            triggered = entry._timeout_triggered
        if triggered:
            return (
                SubagentStatus.TIMED_OUT,
                None,
                f"Timeout apres {timeout_sec}s.",
            )
        message = str(exc) or "Tache annulee cooperativement."
        return SubagentStatus.CANCELLED, None, message[:500]
    except (KeyboardInterrupt, SystemExit, GeneratorExit):  # pragma: no cover
        # R23 : non captures, propages. Branche testee via injection directe.
        raise
    except BaseException as exc:
        with entry._lock:
            triggered = entry._timeout_triggered
            cancel_set = entry.cancel_event.is_set()
        if triggered:
            return (
                SubagentStatus.TIMED_OUT,
                None,
                f"Timeout apres {timeout_sec}s.",
            )
        if cancel_set:
            return SubagentStatus.CANCELLED, None, repr(exc)[:500]
        return SubagentStatus.FAILED, None, repr(exc)[:500]
    return SubagentStatus.COMPLETED, value, None


# ---------------------------------------------------------------------------
# SubagentExecutor
# ---------------------------------------------------------------------------


class SubagentExecutor:
    """Executor thread-based avec 2 pools, registre borne et API non-bloquante."""

    def __init__(
        self,
        *,
        max_workers_scheduler: int = 3,
        max_workers_exec: int = 3,
        max_history: int = 10_000,
        sink: SubagentSink | None = None,
        _now_factory: Callable[[], datetime] | None = None,
        _uuid_factory: Callable[[], str] | None = None,
    ) -> None:
        """Construit l'executor. Aucun pool cree (lazy init au 1er submit).

        Args:
            max_workers_scheduler: pool orchestration (wrapper). Minimum 1.
            max_workers_exec: pool payload task. Minimum 1.
            max_history: taille registre avant eviction FIFO. >= 1.
            sink: observateur on_start/on_end (defaut LogSink).
            _now_factory: override datetime.now(UTC). Thread-safe requis.
            _uuid_factory: override uuid.uuid4().hex. Thread-safe requis.

        Raises:
            ValueError: max_history < 1, max_workers_{pool} < 1.
        """
        if type(max_workers_scheduler) is not int or max_workers_scheduler < 1:
            raise ValueError(
                f"max_workers_scheduler doit etre >= 1 (recu {max_workers_scheduler!r})."
            )
        if type(max_workers_exec) is not int or max_workers_exec < 1:
            raise ValueError(
                f"max_workers_exec doit etre >= 1 (recu {max_workers_exec!r})."
            )
        if type(max_history) is not int or max_history < 1:
            raise ValueError(f"max_history doit etre >= 1 (recu {max_history!r}).")

        self._max_workers_scheduler = max_workers_scheduler
        self._max_workers_exec = max_workers_exec
        self._max_history = max_history
        self._sink: SubagentSink = sink if sink is not None else LogSink()
        self._now_factory: Callable[[], datetime] = (
            _now_factory if _now_factory is not None else (lambda: datetime.now(UTC))
        )
        self._uuid_factory: Callable[[], str] = (
            _uuid_factory if _uuid_factory is not None else (lambda: uuid.uuid4().hex)
        )

        self._state_lock = threading.Lock()
        self._closed: bool = False
        self._scheduler_pool: ThreadPoolExecutor | None = None
        self._exec_pool: ThreadPoolExecutor | None = None
        # Dict ordre insertion preserve (Python 3.7+).
        self._entries: dict[str, _TaskEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        task: TaskCallable,
        *,
        initial_state: Mapping[str, Any],
        timeout_sec: float,
        trace_id: str,
        task_id: str | None = None,
    ) -> str:
        """NON-BLOQUANT. Retourne task_id (uuid4 si None).

        Ordre validation strict (R8) :
            1. task callable
            2. initial_state Mapping
            3. timeout_sec type int/float
            4. timeout_sec pas NaN
            5. timeout_sec > 0
            6. trace_id str non vide
            7. task_id str non vide si fourni
            8. _closed -> SubagentExecutorClosedError
            9. task_id conflict PENDING/RUNNING -> SubagentTaskIdConflictError
            10. task_id terminal -> ecrasement silencieux (log DEBUG)
            11. lazy init pools, schedule wrapper, retour task_id.

        Raises:
            TypeError / ValueError : validations amont.
            SubagentExecutorClosedError : shutdown() appele.
            SubagentExecutorOverflowError : registre plein.
            SubagentTaskIdConflictError : task_id actif.
        """
        # R8.1 : task callable.
        if not callable(task):
            raise TypeError(
                "task doit etre un callable prenant (state: Mapping, cancel_event: Event)."
            )
        # R8.2 : initial_state Mapping.
        if not isinstance(initial_state, Mapping):
            raise TypeError(
                f"initial_state doit etre un Mapping "
                f"(recu {type(initial_state).__name__})."
            )
        # R8.3-5 : timeout_sec.
        if type(timeout_sec) not in (int, float) or isinstance(timeout_sec, bool):
            raise TypeError(
                f"timeout_sec doit etre int ou float (recu {type(timeout_sec).__name__})."
            )
        if isinstance(timeout_sec, float) and math.isnan(timeout_sec):
            raise ValueError(
                "timeout_sec NaN interdit - fournir une valeur numerique finie ou "
                "float('inf')."
            )
        if timeout_sec <= 0:
            raise ValueError(
                f"timeout_sec doit etre strictement positif (recu {timeout_sec})."
            )
        # R8.6 : trace_id.
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("trace_id doit etre une chaine non vide.")
        # R8.7 : task_id si fourni.
        if task_id is not None and (not isinstance(task_id, str) or not task_id):
            raise ValueError("task_id explicite doit etre une chaine non vide si fourni.")

        # R8.8 : _closed.
        with self._state_lock:
            if self._closed:
                raise SubagentExecutorClosedError(
                    "SubagentExecutor ferme - submit() refuse apres shutdown()."
                )

            # R8.9/10 : conflict / ecrasement silencieux.
            if task_id is not None and task_id in self._entries:
                existing = self._entries[task_id]
                with existing._lock:
                    existing_status = existing.status
                if existing_status in (SubagentStatus.PENDING, SubagentStatus.RUNNING):
                    raise SubagentTaskIdConflictError(
                        f"task_id '{task_id}' deja actif (status={existing_status.value}) "
                        f"- utiliser un id unique."
                    )
                # Terminal -> ecrasement silencieux (log DEBUG).
                logger.debug(
                    "submit: task_id '%s' terminal existant (%s) - ecrasement silencieux.",
                    task_id,
                    existing_status.value,
                )
                # On supprime l'entree existante pour la remplacer proprement.
                del self._entries[task_id]

            # R8.11-14 : factories (peuvent raise -> propage, entree non creee).
            submitted_at = self._now_factory()
            resolved_task_id = task_id if task_id is not None else self._uuid_factory()
            # Re-check post uuid_factory : l'uuid generique ne doit pas collision
            # (improbable avec uuid4, mais possible avec factory custom). Applique
            # les memes regles R8.9/10.
            if task_id is None and resolved_task_id in self._entries:
                existing = self._entries[resolved_task_id]
                with existing._lock:
                    existing_status = existing.status
                if existing_status in (SubagentStatus.PENDING, SubagentStatus.RUNNING):
                    raise SubagentTaskIdConflictError(
                        f"task_id '{resolved_task_id}' deja actif "
                        f"(status={existing_status.value}) - utiliser un id unique."
                    )
                del self._entries[resolved_task_id]

            entry = _TaskEntry(
                task_id=resolved_task_id,
                trace_id=trace_id,
                submitted_at=submitted_at,
                cancel_event=threading.Event(),
            )

            # R22 : insertion + eventuelle eviction FIFO.
            self._entries[resolved_task_id] = entry
            if len(self._entries) > self._max_history:
                self._try_evict_one_terminal_or_rollback(resolved_task_id)

            # R8.16 : lazy init atomic.
            if self._scheduler_pool is None and self._exec_pool is None:
                self._scheduler_pool = ThreadPoolExecutor(
                    max_workers=self._max_workers_scheduler,
                    thread_name_prefix="subagent-scheduler",
                )
                self._exec_pool = ThreadPoolExecutor(
                    max_workers=self._max_workers_exec,
                    thread_name_prefix="subagent-exec",
                )
            # R19 invariant : impossible d'avoir un seul pool.
            assert self._scheduler_pool is not None
            assert self._exec_pool is not None

            # R8.17 : schedule wrapper.
            future = self._scheduler_pool.submit(
                self._run_task_wrapper,
                entry,
                task,
                initial_state,
                timeout_sec,
            )
            entry.future = future

        return resolved_task_id

    def wait(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> SubagentResult:
        """BLOQUANT. Attend l'etat terminal de task_id (R11b).

        Args:
            task_id: identifiant de la task.
            timeout: None = attente infinie. Sinon float > 0.

        Returns:
            SubagentResult avec status terminal.

        Raises:
            SubagentTaskNotFoundError: task_id inconnu.
            TimeoutError: timeout expire (PAS SubagentStatus.TIMED_OUT).
            KeyboardInterrupt / SystemExit: si task ou sink les propagent.
        """
        with self._state_lock:
            entry = self._entries.get(task_id)
        if entry is None:
            raise SubagentTaskNotFoundError(
                f"task_id '{task_id}' inconnu - jamais soumis ou deja purge."
            )
        got = entry._done_event.wait(timeout=timeout)
        if not got:
            raise TimeoutError(
                f"Timeout d'attente depasse ({timeout}s) pour task_id '{task_id}'."
            )
        snapshot = entry.snapshot()
        # Si le wrapper scheduler a leve KI/SystemExit, le Future le propage.
        future = entry.future
        if future is not None and future.done():
            exc = future.exception()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise exc
        return snapshot

    def cancel(self, task_id: str) -> bool:
        """Non bloquant. Set le cancel_event de task_id (R10).

        Returns:
            True si task_id existe (PENDING/RUNNING/terminal).
            False si inconnu.
            Si PENDING : set event + force status CANCELLED.
            Si RUNNING : set event.
            Si terminal : set event (no-op), retourne True.
        """
        with self._state_lock:
            entry = self._entries.get(task_id)
            if entry is None:
                return False
            entry.cancel_event.set()
            with entry._lock:
                if entry.status == SubagentStatus.PENDING:
                    entry.status = SubagentStatus.CANCELLED
                    entry.completed_at = self._now_factory()
                    entry.error = "Annulee avant demarrage."
                    entry._done_event.set()
        return True

    def get(self, task_id: str) -> SubagentResult | None:
        """Snapshot courant ou None si inconnu (R11)."""
        with self._state_lock:
            entry = self._entries.get(task_id)
        if entry is None:
            return None
        return entry.snapshot()

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = True,
        force_timeout_sec: float | None = _FORCE_TIMEOUT_DEFAULT,
    ) -> None:
        """Ferme l'executor. Idempotent (R12).

        Ordre :
            1. _closed = True, force CANCELLED sur PENDING, set cancel_event sur RUNNING.
            2. scheduler_pool.shutdown(wait, cancel_futures).
            3. exec_pool.shutdown(wait=False, cancel_futures=True).
            4. si force_timeout_sec non None : Event().wait(force_timeout_sec) puis
               scan threading.enumerate() -> log WARNING threads zombies restants.

        Args:
            wait: si True, attend scheduler_pool.shutdown.
            cancel_futures: si True, cancel les Future scheduler PENDING.
            force_timeout_sec: None = pas de scan. Plage [0.1, 300.0], hors plage
                -> clip silencieux a 5.0 + log WARNING (preserve idempotence).
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            now = self._now_factory()
            for entry in self._entries.values():
                entry.cancel_event.set()
                with entry._lock:
                    if entry.status == SubagentStatus.PENDING:
                        entry.status = SubagentStatus.CANCELLED
                        entry.completed_at = now
                        entry.error = "Annulee - executor ferme."
                        entry._done_event.set()
            sched = self._scheduler_pool
            exec_p = self._exec_pool

        if sched is not None:
            sched.shutdown(wait=wait, cancel_futures=cancel_futures)
        if exec_p is not None:
            # wait=False force (cf R12 - impossible de kill un thread Python proprement).
            exec_p.shutdown(wait=False, cancel_futures=True)

        if force_timeout_sec is not None:
            # Validation + clip silencieux R12 / EC70.
            if (
                type(force_timeout_sec) not in (int, float)
                or isinstance(force_timeout_sec, bool)
                or (
                    isinstance(force_timeout_sec, float)
                    and not math.isfinite(force_timeout_sec)
                )
                or force_timeout_sec < _FORCE_TIMEOUT_MIN
                or force_timeout_sec > _FORCE_TIMEOUT_MAX
            ):
                logger.warning(
                    "shutdown: force_timeout_sec hors plage [%s, %s] (recu %r) "
                    "- clip silencieux a %s.",
                    _FORCE_TIMEOUT_MIN,
                    _FORCE_TIMEOUT_MAX,
                    force_timeout_sec,
                    _FORCE_TIMEOUT_DEFAULT,
                )
                force_timeout_sec = _FORCE_TIMEOUT_DEFAULT
            # CR-025 (v2.1.2) : polling actif avec deadline au lieu de Event().wait()
            # passif. Exit des que 0 task RUNNING restante OU deadline atteinte.
            # Avant v2.1.2 : wait(force_timeout_sec) systematique = shutdown lent
            # meme si toutes les tasks coop sortent rapidement.
            deadline = time.monotonic() + force_timeout_sec
            with self._state_lock:
                running_events = [
                    entry._done_event
                    for entry in self._entries.values()
                    if entry.status == SubagentStatus.RUNNING
                ]
            while running_events:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                running_events = [e for e in running_events if not e.is_set()]
                if not running_events:
                    break
                # Poll 50ms (threading.Event jetable, attente passive)
                threading.Event().wait(timeout=min(remaining, 0.05))
            zombies = [
                t
                for t in threading.enumerate()
                if t.name.startswith("subagent-exec") and t.is_alive()
            ]
            if zombies:
                frames = sys._current_frames()
                for t in zombies:
                    tid = t.ident if t.ident is not None else 0
                    frame = frames.get(tid, None)
                    if frame is not None:
                        # traceback.format_stack retourne la liste
                        # "file:line\n  code\n" sans exposer les locals.
                        stack_text = "".join(traceback.format_stack(frame))
                        if len(stack_text) > 500:
                            stack_text = stack_text[:500] + "...<truncated>"
                    else:
                        stack_text = "<frame unavailable>"
                    logger.warning(
                        "shutdown: thread zombie '%s' ident=%s encore vivant "
                        "(stack=%s) - Python ne permet pas de kill propre.",
                        t.name,
                        t.ident,
                        stack_text,
                    )

    def clear_history(self) -> int:
        """Supprime les entrees terminales du registre.

        Returns:
            Nombre d'entrees supprimees.
        """
        removed = 0
        with self._state_lock:
            to_delete: list[str] = []
            for tid, entry in self._entries.items():
                with entry._lock:
                    if entry.status in _TERMINAL_SET:
                        to_delete.append(tid)
            for tid in to_delete:
                del self._entries[tid]
                removed += 1
        return removed

    @property
    def history_size(self) -> int:
        """Nombre total d'entrees dans le registre."""
        with self._state_lock:
            return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Snapshot comptage par status (thread-safe).

        Returns:
            Dict avec cles : total, pending, running, completed, failed,
            cancelled, timed_out.
        """
        counts: dict[str, int] = {
            "total": 0,
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "timed_out": 0,
        }
        with self._state_lock:
            entries = list(self._entries.values())
        for entry in entries:
            with entry._lock:
                status = entry.status
            counts["total"] += 1
            counts[status.value] += 1
        return counts

    def __enter__(self) -> SubagentExecutor:
        """Context manager : retourne self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Context manager : shutdown(wait=True, cancel_futures=True, force=5.0)."""
        self.shutdown(wait=True, cancel_futures=True, force_timeout_sec=5.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_evict_one_terminal_or_rollback(self, new_task_id: str) -> None:
        """Tentative eviction FIFO (R22). Appelee sous _state_lock.

        Scanne les entrees en ordre d'insertion, evinçe la premiere is_terminal
        trouvee. Si aucune -> rollback de l'entree nouvellement inseree +
        SubagentExecutorOverflowError.
        """
        # Ordre insertion preserve (dict Python 3.7+).
        for tid, entry in list(self._entries.items()):
            if tid == new_task_id:
                # On ne s'evincera pas soi-meme.
                continue
            with entry._lock:
                is_terminal = entry.status in _TERMINAL_SET
            if is_terminal:
                del self._entries[tid]
                return
        # Aucun terminal evinçable -> rollback la nouvelle entree.
        del self._entries[new_task_id]
        raise SubagentExecutorOverflowError(
            f"Limite de taches actives atteinte ({self._max_history}) - "
            f"aucune terminale a evincer, attendre une terminaison."
        )

    def _run_task_wrapper(
        self,
        entry: _TaskEntry,
        task: TaskCallable,
        initial_state: Mapping[str, Any],
        timeout_sec: float,
    ) -> None:
        """Wrapper orchestration (R9). Execute dans scheduler_pool.

        Etapes :
            1. Transition PENDING -> RUNNING (sous entry._lock). Sinon sortie.
            2. sink.on_start(snapshot()) hors lock (R16).
            3. Schedule task dans exec_pool.
               Si RuntimeError (shutdown survenu entre on_start et submit) ->
               transition RUNNING -> CANCELLED, on_end appele (scope R16 respecte),
               sortie sans await.
            4. _await_with_precedence -> (status, result, error).
            5. Transition RUNNING -> terminal (sous entry._lock), set _done_event.
            6. sink.on_end(snapshot()) hors lock (R16).

        Note KI/SystemExit :
            - on_start raise KI (EC25) -> finally set _done_event pour debloquer
              wait(), status reste RUNNING. wait() detecte KI dans future.exception()
              et la propage au caller.
            - task raise KI -> _await_with_precedence laisse passer, meme mecanisme.
        """
        try:
            # 1 : transition PENDING -> RUNNING.
            with entry._lock:
                if entry.status != SubagentStatus.PENDING:
                    # cancel force / shutdown force deja transitionne -> sortie.
                    return
                entry.status = SubagentStatus.RUNNING
                entry.started_at = self._now_factory()

            # 2 : on_start (R16 scope, EC25 KI/SystemExit propages).
            self._safe_sink_call(self._sink.on_start, entry)

            # 3 : schedule task. Race possible : shutdown() peut etre appele entre
            # la transition RUNNING et submit() -> exec_pool deja ferme. On
            # attrape RuntimeError (message varie selon CPython version) et
            # transitionne proprement CANCELLED (R12 : shutdown force cancel sur
            # RUNNING ; ici on ferme la boucle cote wrapper).
            assert self._exec_pool is not None
            try:
                exec_future = self._exec_pool.submit(
                    task, initial_state, entry.cancel_event
                )
            except RuntimeError:
                # shutdown survenu avant submit. RUNNING -> CANCELLED.
                with entry._lock:
                    entry.status = SubagentStatus.CANCELLED
                    entry.error = (
                        "[INFO] Annulee - executor shutdown avant demarrage "
                        "exec_pool."
                    )
                    entry.completed_at = self._now_factory()
                # R16 scope : RUNNING -> terminal -> on_end doit etre appele.
                self._safe_sink_call(self._sink.on_end, entry)
                return

            # 4 : await avec precedence.
            status, result, error = _await_with_precedence(
                exec_future, entry, timeout_sec
            )

            # 5 : transition terminale.
            with entry._lock:
                entry.status = status
                entry.result = result
                entry.error = error
                entry.completed_at = self._now_factory()

            # 6 : on_end (R16 scope : uniquement apres RUNNING -> terminal).
            self._safe_sink_call(self._sink.on_end, entry)
        finally:
            # Invariant §7.6.6 : _done_event toujours set pour debloquer wait().
            # Meme en cas de KI/SystemExit/GeneratorExit dans on_start ou task
            # (EC1, EC25) -> wait() detecte future.exception() et re-raise.
            entry._done_event.set()

    def _safe_sink_call(
        self,
        hook: Callable[[SubagentResult], None],
        entry: _TaskEntry,
    ) -> None:
        """Appelle hook(snapshot) en capturant les exceptions non fatales (R16).

        KeyboardInterrupt / SystemExit / GeneratorExit : propages.
        Autre Exception : log WARNING + continue.
        """
        try:
            snapshot = entry.snapshot()
            hook(snapshot)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            logger.warning(
                "Sink %s a leve %s: %s (task_id=%s, trace_id=%s)",
                hook.__qualname__,
                type(exc).__name__,
                str(exc)[:200],
                entry.task_id,
                entry.trace_id,
            )


# Set terminal (utilise dans clear_history / _try_evict). Import local pour
# eviter un cycle d'import complete-temps.
_TERMINAL_SET: frozenset[SubagentStatus] = frozenset(
    {
        SubagentStatus.COMPLETED,
        SubagentStatus.FAILED,
        SubagentStatus.CANCELLED,
        SubagentStatus.TIMED_OUT,
    }
)
