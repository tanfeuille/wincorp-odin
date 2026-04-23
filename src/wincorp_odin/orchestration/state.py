"""Fonction pure `build_initial_state` : heritage selectif parent->enfant.

@spec specs/orchestration.spec.md v2.1.1 §3.5 + R5/R6/R7/EC11-13/EC36-40

Whitelist stricte : sandbox_state, thread_data, session_id, trace_id.
Toute autre cle du parent EXCLUE (messages, tool_calls, subagent_results, ...).
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

# Whitelist R5 : cles heritees du parent vers le sub-agent.
_INHERITED_KEYS: frozenset[str] = frozenset(
    {"sandbox_state", "thread_data", "session_id", "trace_id"}
)


def build_initial_state(
    parent_state: Mapping[str, Any],
    *,
    extra_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Herite selectivement du parent_state vers un dict enfant vierge.

    Whitelist (R5) : sandbox_state, thread_data, session_id, trace_id sont
    deep-copiees via `copy.deepcopy`. Toute autre cle du parent est EXCLUE.

    `extra_overrides` (R7) est applique APRES heritage parent : remplace les
    cles presentes (pas de merge), ajoute les nouvelles.

    Args:
        parent_state: Mapping du parent.
        extra_overrides: Mapping d'overrides ou None.

    Returns:
        Dict vierge contenant uniquement les cles whitelist + overrides.

    Raises:
        TypeError: parent_state non Mapping, extra_overrides non Mapping/None.
        ValueError: deepcopy leve (valeur non copiable) avec cle en cause.
    """
    # EC40 : validation type parent_state.
    if not isinstance(parent_state, Mapping):
        raise TypeError(
            f"parent_state doit etre un Mapping "
            f"(recu {type(parent_state).__name__})."
        )
    # Validation type extra_overrides.
    if extra_overrides is not None and not isinstance(extra_overrides, Mapping):
        raise TypeError(
            f"extra_overrides doit etre un Mapping ou None "
            f"(recu {type(extra_overrides).__name__})."
        )

    out: dict[str, Any] = {}
    # R5 + R6 : whitelist deep-copie.
    for key in _INHERITED_KEYS:
        if key in parent_state:
            try:
                out[key] = copy.deepcopy(parent_state[key])
            except Exception as exc:
                raise ValueError(
                    f"Impossible de deep-copier la cle '{key}' de parent_state : {exc}."
                ) from exc

    # R7 : overrides priorite max.
    if extra_overrides is not None:
        for key, value in extra_overrides.items():
            out[key] = value

    return out
