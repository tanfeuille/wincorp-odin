"""Sinks observabilite : Protocol SubagentSink + LogSink par defaut.

@spec specs/orchestration.spec.md v2.1.1 §3.7 + R16

Contrat : on_start apres PENDING->RUNNING, on_end apres RUNNING->terminal.
Errors non-bloquants sauf KeyboardInterrupt/SystemExit/GeneratorExit.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from wincorp_odin.orchestration.result import SubagentResult


class SubagentSink(Protocol):
    """Protocol observabilite pour SubagentExecutor.

    on_start : appele apres PENDING->RUNNING, avant execution task.
        result.status == RUNNING, result.started_at non-None.
        Exceptions -> log WARNING, continue.
        KeyboardInterrupt/SystemExit/GeneratorExit -> propages (EC25).

    on_end : appele apres RUNNING->terminal uniquement (R16).
        result.is_terminal(), result.completed_at non-None.
        Exceptions -> log WARNING, continue.
        KeyboardInterrupt/SystemExit/GeneratorExit -> propages.
    """

    def on_start(self, result: SubagentResult) -> None:
        """Hook PENDING -> RUNNING."""
        ...  # pragma: no cover — Protocol body (runtime = subclass override)

    def on_end(self, result: SubagentResult) -> None:
        """Hook RUNNING -> terminal."""
        ...  # pragma: no cover — Protocol body (runtime = subclass override)


class LogSink:
    """Sink par defaut : log structure JSON via stdlib logging.

    Format : une ligne JSON par transition (event, task_id, trace_id, status,
    duration_ms, submitted_at, started_at, completed_at, error).
    """

    def __init__(self, *, logger_name: str = "wincorp_odin.orchestration") -> None:
        """Initialise le sink.

        Args:
            logger_name: nom du logger stdlib cible.
        """
        self._logger = logging.getLogger(logger_name)

    def on_start(self, result: SubagentResult) -> None:
        """Log INFO avec task_id, trace_id, status, submitted_at, started_at."""
        self._logger.info(
            json.dumps(
                {
                    "event": "subagent_start",
                    "task_id": result.task_id,
                    "trace_id": result.trace_id,
                    "status": result.status.value,
                    "submitted_at": result.submitted_at.isoformat(),
                    "started_at": (
                        result.started_at.isoformat()
                        if result.started_at is not None
                        else None
                    ),
                }
            )
        )

    def on_end(self, result: SubagentResult) -> None:
        """Log INFO avec task_id, trace_id, status, duration_ms, error tronque.

        R17 : pas de contenu complet result / ai_messages en INFO. Error tronque 500.
        """
        payload: dict[str, Any] = {
            "event": "subagent_end",
            "task_id": result.task_id,
            "trace_id": result.trace_id,
            "status": result.status.value,
            "submitted_at": result.submitted_at.isoformat(),
            "started_at": (
                result.started_at.isoformat() if result.started_at is not None else None
            ),
            "completed_at": (
                result.completed_at.isoformat()
                if result.completed_at is not None
                else None
            ),
            "duration_ms": result.duration_ms,
            "error": result.error[:500] if result.error is not None else None,
        }
        self._logger.info(json.dumps(payload))
