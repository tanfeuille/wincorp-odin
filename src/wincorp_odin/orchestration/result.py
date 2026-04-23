"""SubagentStatus enum + SubagentResult dataclass + helpers prives.

@spec specs/orchestration.spec.md v2.1.1 §3.3 + R1/R2/R3/R3b/R4/R20/R20b

- SubagentStatus : str Enum serialisable JSON (R1).
- SubagentResult : frozen=True, slots=True, eq=True, __hash__=None (R2, EC15b).
- _TERMINAL_STATUSES : frozenset des 4 statuts terminaux (R3).
- _dedup_messages_by_id : dernier gagne, position du dernier (R20 + §3.3 helper).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from wincorp_odin.orchestration._json_safe import _json_safe


class SubagentStatus(StrEnum):
    """Statuts possibles d'une task orchestree (R1).

    Str Enum (via `StrEnum` Python 3.11+ — equivalent semantique a
    `class X(str, Enum)` : serialisation JSON triviale via `.value`
    ou `json.dumps(status)` grace a l'heritage str).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


_TERMINAL_STATUSES: frozenset[SubagentStatus] = frozenset(
    {
        SubagentStatus.COMPLETED,
        SubagentStatus.FAILED,
        SubagentStatus.CANCELLED,
        SubagentStatus.TIMED_OUT,
    }
)


def _dedup_messages_by_id(
    messages: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Dedup conserve le dernier occurrence de chaque id, a la position du dernier.

    Exemple :
        (a1, b, a2, c) avec a1.id == a2.id == "a" -> (b, a2, c).
        msg1 retire, msg3 conserve a sa position originale (2 dans l'input, 1 dans l'output).

    Args:
        messages: tuple de messages deja valides (type dict, id str non vide si present).

    Returns:
        Tuple dedup-e (meme identite que l'input si aucun doublon).
    """
    # 1ere passe : identifier l'index du dernier msg pour chaque id.
    last_index: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        if isinstance(msg_id, str) and msg_id:
            last_index[msg_id] = idx

    # Pas de doublons -> retour identite stricte (comparaison tailles).
    ids_found = 0
    for msg in messages:
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        if isinstance(msg_id, str) and msg_id:
            ids_found += 1
    if ids_found == len(last_index):
        # Chaque id apparait une seule fois -> pas de doublons.
        return messages

    # 2e passe : ne conserver que les messages dont l'index == last_index[id]
    # (ou ceux sans id / avec id non-str / sans doublon).
    result: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        msg_id = msg.get("id") if isinstance(msg, dict) else None
        if isinstance(msg_id, str) and msg_id:
            if last_index[msg_id] == idx:
                result.append(msg)
        else:
            result.append(msg)
    return tuple(result)


@dataclass(frozen=True, slots=True, eq=True)
class SubagentResult:
    """Resultat immutable d'une task orchestree (R2).

    Attributs :
        task_id: identifiant task (str non vide).
        trace_id: identifiant trace (str non vide).
        status: SubagentStatus.
        submitted_at: UTC-aware, fige a l'instant submit() (R18b).
        started_at: UTC-aware si non None, fige au pickup RUNNING.
        completed_at: UTC-aware si non None, fige au passage terminal.
        result: valeur de retour task (None si status != COMPLETED).
        error: message d'erreur tronque 500 chars (populated FAILED/CANCELLED/TIMED_OUT).
        ai_messages: tuple de dicts dedup via R20.

    `__hash__ = None` force la non-hashabilite au niveau du type (EC15b).
    """

    task_id: str
    trace_id: str
    status: SubagentStatus
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: Any
    error: str | None
    ai_messages: tuple[dict[str, Any], ...]

    # EC15b : empeche l'Heisenbug pytest dedup internes (tuple de dict unhashable).
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validation ordonnee (R20b) :

        1. Validation tz-aware submitted_at.
        2. Validation tz-aware started_at si non None.
        3. Validation tz-aware completed_at si non None.
        4. Validation type ai_messages (tuple strict, pas list).
        5. Validation chaque msg est dict + id str non vide si present.
        6. Dedup via object.__setattr__ si doublons id detectes.
        """
        # 1-3 : tz-aware.
        if self.submitted_at.tzinfo is None:
            raise ValueError(
                "SubagentResult.submitted_at doit etre tz-aware (UTC)."
            )
        if self.started_at is not None and self.started_at.tzinfo is None:
            raise ValueError(
                "SubagentResult.started_at doit etre tz-aware (UTC)."
            )
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError(
                "SubagentResult.completed_at doit etre tz-aware (UTC)."
            )
        # 4 : type tuple strict (pas list).
        if not isinstance(self.ai_messages, tuple):
            raise TypeError(
                f"SubagentResult.ai_messages doit etre un tuple "
                f"(recu {type(self.ai_messages).__name__}). Le caller est responsable "
                f"de la conversion list->tuple."
            )
        # 5 : type chaque message + id str non vide si present.
        for idx, msg in enumerate(self.ai_messages):
            if not isinstance(msg, dict):
                raise TypeError(
                    f"SubagentResult.ai_messages[{idx}] doit etre un dict "
                    f"(recu {type(msg).__name__})."
                )
            if "id" in msg:
                value = msg["id"]
                if not (isinstance(value, str) and value):
                    raise TypeError(
                        f"ai_messages[{idx}]: champ 'id' doit etre str non vide "
                        f"(recu {type(value).__name__})."
                    )
        # 6 : dedup (dernier gagne, position du dernier).
        deduped = _dedup_messages_by_id(self.ai_messages)
        if deduped is not self.ai_messages:
            object.__setattr__(self, "ai_messages", deduped)

    def is_terminal(self) -> bool:
        """Retourne True si status est un terminal (R3).

        PENDING et RUNNING -> False. COMPLETED/FAILED/CANCELLED/TIMED_OUT -> True.
        """
        return self.status in _TERMINAL_STATUSES

    @property
    def duration_ms(self) -> float | None:
        """Duree d'execution en ms depuis le pickup RUNNING (R3b).

        Returns:
            None si `started_at is None` ou `completed_at is None`.
            Sinon (completed_at - started_at).total_seconds() * 1000.0.

        Note:
            Peut etre negatif en cas de clock skew (NTP drift). Pas de clamp
            automatique - responsabilite caller si monitoring veut >= 0.
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialisation JSON-safe recursive (R4).

        Normalise :
            - status enum -> value string.
            - submitted_at/started_at/completed_at -> ISO8601 (None preserve).
            - ai_messages tuple -> list, normalise via _json_safe recursif.
            - result -> _json_safe recursif.

        Returns:
            Dict JSON-serialisable.

        Raises:
            TypeError: type non serialisable (chemin JSONPath precis).
            ValueError: float non-fini (NaN/Inf) dans result ou ai_messages.
        """
        return {
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": (
                self.started_at.isoformat() if self.started_at is not None else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at is not None else None
            ),
            "result": _json_safe(self.result, _path="$.result"),
            "error": self.error,
            "ai_messages": [
                _json_safe(msg, _path=f"$.ai_messages[{idx}]")
                for idx, msg in enumerate(self.ai_messages)
            ],
        }
