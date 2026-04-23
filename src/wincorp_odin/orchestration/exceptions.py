"""Exceptions dediees du module orchestration (Phase 2 DeerFlow).

@spec specs/orchestration.spec.md v2.1.1

Hierarchie exceptions orchestration, messages FR — runtime utilisateur.
"""
from __future__ import annotations


class SubagentError(Exception):
    """Racine des exceptions du module orchestration."""


class SubagentExecutorClosedError(SubagentError, RuntimeError):
    """Levee par submit() / cancel() / wait() apres shutdown()."""


class SubagentExecutorOverflowError(SubagentError, RuntimeError):
    """submit() lorsqu'aucune entree terminale n'est disponible pour eviction FIFO."""


class SubagentTaskNotFoundError(SubagentError, KeyError):
    """wait() sur task_id inconnu. cancel() renvoie False sans lever."""


class SubagentTaskIdConflictError(SubagentError, ValueError):
    """submit(task_id=X) lorsque X existe deja en PENDING ou RUNNING."""


class SubagentCancelledException(SubagentError):  # noqa: N818 — nom fige par spec §3.1
    """Levee par une task qui observe cancel_event.is_set() == True."""

    def __init__(
        self, message: str = "Tache annulee cooperativement via cancel_event."
    ) -> None:
        """Initialise l'exception avec un message FR par defaut."""
        super().__init__(message)
