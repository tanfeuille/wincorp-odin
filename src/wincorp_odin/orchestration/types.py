"""Types orchestration : AIMessage TypedDict, TaskCallable, InitialState (Phase 2 DeerFlow).

@spec specs/orchestration.spec.md v2.1.1 §3.2

Exportes via wincorp_odin.orchestration.__init__.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

Role = Literal["user", "assistant", "system", "tool"]


class AIMessage(TypedDict, total=False):
    """Message LLM generique (total=False : toutes les cles sont optionnelles).

    Cles :
        id: requis pour declenchement dedup (R20) — sinon dict conserve tel quel.
        role: un Role ("user", "assistant", "system", "tool").
        content: str, list[dict] blocs tool_use, dict — libre au caller.
        name: optionnel, renseigne si role == "tool".
        tool_call_id: optionnel, renseigne si role == "tool".
    """

    id: str
    role: Role
    content: Any
    name: str
    tool_call_id: str


TaskCallable = Callable[[Mapping[str, Any], threading.Event], Any]
"""Signature obligatoire d'une task soumise a SubagentExecutor.

Args:
    initial_state: read-only cote task (Mapping, pas dict).
    cancel_event: threading.Event a checker periodiquement (cooperatif).

Returns:
    Any - caller responsable de la serialisabilite si to_dict() sera appele.
    Si la task raise SubagentCancelledException -> status CANCELLED.
    Si la task raise autre Exception -> status FAILED, error = repr(exc)[:500].
    KeyboardInterrupt / SystemExit -> propagees a l'appelant, pas FAILED.
"""

InitialState = Mapping[str, Any]
"""Alias semantique - read-only cote task."""
