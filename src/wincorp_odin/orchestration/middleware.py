"""Fonction pure `truncate_task_calls` : fan-out clamp.

@spec specs/orchestration.spec.md v2.1.1 §3.6 + R14/R15/EC8-10/EC41-49c

Limite stricte [1, 20], type int strict (bool rejete), Sequence strict
(generator/set rejetes via TypeError FR).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger("wincorp_odin.orchestration.middleware")

_MAX_CONCURRENT_MIN = 1
_MAX_CONCURRENT_MAX = 20


def truncate_task_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    max_concurrent: int = 3,
    tool_name: str = "task",
) -> list[Mapping[str, Any]]:
    """Tronque les tool_calls de type `tool_name` a `max_concurrent` occurrences.

    Ordre d'entree preserve. Calls `name != tool_name` (y compris `name` absent
    ou None) passent tous. Si count(tool_name) > max_concurrent, log WARNING
    avec le nombre droppe.

    Args:
        tool_calls: Sequence (list, tuple) de tool_calls. Generators / sets
            sont REJETES via TypeError FR.
        max_concurrent: limite stricte [1, 20]. Type `int` strict (bool rejete).
            Defaut 3.
        tool_name: nom du tool a limiter. String non vide.

    Returns:
        Liste tronquee (list, pas tuple, pour permettre mutation downstream).

    Raises:
        TypeError: max_concurrent pas int strict (R14), tool_calls pas Sequence
            (EC44, EC44b).
        ValueError: max_concurrent hors [1, 20] (EC48, EC49), tool_name vide.
    """
    # R14 : valide max_concurrent type `int` strict (exclut bool).
    if type(max_concurrent) is not int:
        raise TypeError(
            f"max_concurrent doit etre un entier strict "
            f"(recu {type(max_concurrent).__name__})."
        )
    # R14 : plage [1, 20].
    if max_concurrent < _MAX_CONCURRENT_MIN or max_concurrent > _MAX_CONCURRENT_MAX:
        raise ValueError(
            f"max_concurrent doit etre un entier entre {_MAX_CONCURRENT_MIN} et "
            f"{_MAX_CONCURRENT_MAX} (recu {max_concurrent})."
        )
    # R14 : tool_name str non vide.
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name doit etre une chaine non vide.")

    # R14 + EC44/EC44b : Sequence strict (exclut generator, set, dict).
    # `str` et `bytes` sont des Sequences mais semantiquement non valides ici.
    # On filtre explicitement (documentation guide : "list, tuple").
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        raise TypeError(
            "tool_calls doit etre une Sequence (list, tuple) "
            "- generateurs non supportes, faire list(gen) au prealable."
        )

    # R15 : preserver ordre, conserver tous les non-tool_name, keep max_concurrent tool_name.
    result: list[Mapping[str, Any]] = []
    kept = 0
    total_target = 0
    for call in tool_calls:
        name = call.get("name") if isinstance(call, Mapping) else None
        if name == tool_name:
            total_target += 1
            if kept < max_concurrent:
                result.append(call)
                kept += 1
            # else: drop.
        else:
            result.append(call)

    dropped = total_target - kept
    if dropped > 0:
        logger.warning(
            "truncate_task_calls: %d calls '%s' droppes (fan-out > %d).",
            dropped,
            tool_name,
            max_concurrent,
        )
    return result
