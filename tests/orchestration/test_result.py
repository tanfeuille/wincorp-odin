"""Tests SubagentStatus + SubagentResult : R1, R2, R3, R3b, R4, R20, R20b, EC14-15b, EC26-28.

@spec specs/orchestration.spec.md v2.1.1 §3.3
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from wincorp_odin.orchestration.result import (
    _TERMINAL_STATUSES,
    SubagentResult,
    SubagentStatus,
    _dedup_messages_by_id,
)


def _fresh(
    *,
    status: SubagentStatus = SubagentStatus.PENDING,
    submitted_at: datetime | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    result: object = None,
    error: str | None = None,
    ai_messages: tuple[dict[str, object], ...] = (),
) -> SubagentResult:
    """Helper : construit un SubagentResult legal par defaut."""
    if submitted_at is None:
        submitted_at = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
    return SubagentResult(
        task_id="t-1",
        trace_id="trace-1",
        status=status,
        submitted_at=submitted_at,
        started_at=started_at,
        completed_at=completed_at,
        result=result,
        error=error,
        ai_messages=ai_messages,
    )


# --- R1 : str Enum -----------------------------------------------------------


def test_r1_status_str_enum_values_lowercase() -> None:
    """R1 : SubagentStatus valeurs str lowercase."""
    assert SubagentStatus.PENDING.value == "pending"
    assert SubagentStatus.RUNNING.value == "running"
    assert SubagentStatus.COMPLETED.value == "completed"
    assert SubagentStatus.FAILED.value == "failed"
    assert SubagentStatus.CANCELLED.value == "cancelled"
    assert SubagentStatus.TIMED_OUT.value == "timed_out"


def test_r1_status_is_str_subclass() -> None:
    """R1 : SubagentStatus est une subclass str."""
    assert isinstance(SubagentStatus.RUNNING, str)


def test_r1_status_serialisable_json() -> None:
    """R1 : json.dumps(status.value) fonctionne."""
    assert json.dumps(SubagentStatus.RUNNING.value) == '"running"'


# --- R2 : frozen / slots / non-hashable --------------------------------------


def test_r2_result_is_frozen() -> None:
    """R2 : SubagentResult immutable (frozen)."""
    res = _fresh()
    with pytest.raises((AttributeError, Exception)):
        res.task_id = "other"  # type: ignore[misc]


def test_r2_result_eq_positive() -> None:
    """R2 : eq=True, meme donnees -> meme instance equal."""
    a = _fresh(result="x")
    b = _fresh(result="x")
    assert a == b


def test_r2_result_eq_negative() -> None:
    """R2 : eq=True, donnees differentes -> neq."""
    a = _fresh(result="x")
    b = _fresh(result="y")
    assert a != b


def test_ec15b_hash_raises_type_error() -> None:
    """EC15b : `__hash__ = None` -> TypeError au niveau du type."""
    res = _fresh()
    with pytest.raises(TypeError):
        hash(res)


# --- R3 / is_terminal -------------------------------------------------------


def test_r3_is_terminal_pending_false() -> None:
    """R3 : PENDING non-terminal."""
    assert not _fresh(status=SubagentStatus.PENDING).is_terminal()


def test_r3_is_terminal_running_false() -> None:
    """R3 : RUNNING non-terminal."""
    res = _fresh(
        status=SubagentStatus.RUNNING,
        started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC),
    )
    assert not res.is_terminal()


def test_r3_is_terminal_completed_true() -> None:
    """R3 : COMPLETED terminal."""
    res = _fresh(
        status=SubagentStatus.COMPLETED,
        started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC),
        completed_at=datetime(2026, 4, 23, 14, 0, 2, tzinfo=UTC),
    )
    assert res.is_terminal()


def test_r3_is_terminal_all_terminal_statuses() -> None:
    """R3 : COMPLETED/FAILED/CANCELLED/TIMED_OUT tous terminaux."""
    for s in (
        SubagentStatus.COMPLETED,
        SubagentStatus.FAILED,
        SubagentStatus.CANCELLED,
        SubagentStatus.TIMED_OUT,
    ):
        res = _fresh(
            status=s,
            started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC),
            completed_at=datetime(2026, 4, 23, 14, 0, 2, tzinfo=UTC),
        )
        assert res.is_terminal(), s


def test_terminal_statuses_frozenset_content() -> None:
    """_TERMINAL_STATUSES contient exactement les 4 terminaux."""
    assert frozenset(
        {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }
    ) == _TERMINAL_STATUSES


# --- R3b / duration_ms ------------------------------------------------------


def test_ec14b_duration_ms_started_at_none() -> None:
    """EC14b : duration_ms None si started_at=None."""
    res = _fresh()
    assert res.duration_ms is None


def test_ec14b_duration_ms_completed_at_none() -> None:
    """EC14b : duration_ms None si completed_at=None."""
    res = _fresh(
        status=SubagentStatus.RUNNING,
        started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC),
    )
    assert res.duration_ms is None


def test_ec14c_duration_ms_nominal() -> None:
    """EC14c : (completed_at - started_at).total_seconds() * 1000."""
    res = _fresh(
        status=SubagentStatus.COMPLETED,
        started_at=datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 23, 14, 0, 0, 500_000, tzinfo=UTC),
    )
    assert res.duration_ms == pytest.approx(500.0)


def test_r3b_duration_ms_negative_clock_skew() -> None:
    """R3b : clock skew -> duration_ms peut etre negatif (pas de clamp auto)."""
    res = _fresh(
        status=SubagentStatus.COMPLETED,
        started_at=datetime(2026, 4, 23, 14, 0, 0, 600_000, tzinfo=UTC),
        completed_at=datetime(2026, 4, 23, 14, 0, 0, 100_000, tzinfo=UTC),
    )
    assert res.duration_ms is not None
    assert res.duration_ms < 0


# --- EC14 / RUNNING snapshot -----------------------------------------------


def test_ec14_running_snapshot_shape() -> None:
    """EC14 : RUNNING avec started_at non-None et completed_at None, is_terminal False."""
    res = _fresh(
        status=SubagentStatus.RUNNING,
        started_at=datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC),
    )
    assert res.status == SubagentStatus.RUNNING
    assert res.started_at is not None
    assert res.completed_at is None
    assert not res.is_terminal()


# --- R20 / R20b dedup + validation -------------------------------------------


def test_r20_dedup_last_wins_position_of_last() -> None:
    """R20 : (a1, b, a2, c) avec a1.id==a2.id -> (b, a2, c). a2 a sa position de source."""
    msg_a1: dict[str, object] = {"id": "a", "content": "first"}
    msg_b: dict[str, object] = {"id": "b", "content": "b-content"}
    msg_a2: dict[str, object] = {"id": "a", "content": "last"}
    msg_c: dict[str, object] = {"id": "c", "content": "c-content"}
    res = _fresh(ai_messages=(msg_a1, msg_b, msg_a2, msg_c))
    # Exactement 3 messages, a1 retire, a2 en position 1 (apres b).
    assert len(res.ai_messages) == 3
    assert res.ai_messages[0]["id"] == "b"
    assert res.ai_messages[1]["id"] == "a"
    assert res.ai_messages[1]["content"] == "last"
    assert res.ai_messages[2]["id"] == "c"


def test_r20_dedup_helper_no_duplicates_returns_identity() -> None:
    """Pas de doublons -> helper retourne l'input sans copie."""
    msgs = (
        {"id": "a", "x": 1},
        {"id": "b", "x": 2},
    )
    assert _dedup_messages_by_id(msgs) is msgs


def test_r20_dedup_helper_messages_without_id_preserved() -> None:
    """Messages sans id -> conserves sans dedup."""
    msgs = (
        {"role": "user", "content": "anon-1"},
        {"id": "a", "content": "with-id"},
        {"role": "user", "content": "anon-2"},
    )
    out = _dedup_messages_by_id(msgs)
    assert out is msgs  # pas de doublon


def test_r20_dedup_helper_with_only_non_str_id() -> None:
    """Helper ignore les id non-str (doit avoir deja passe validation R2)."""
    # Helper direct, bypass de __post_init__ validation. Un message avec id=123
    # n'est pas traite comme ayant un id (str check).
    # Ici on passe en direct sans passer par SubagentResult (simulate internal).
    msgs: tuple[dict[str, object], ...] = (
        {"id": 123, "content": "a"},  # id pas str -> pas dedup
        {"id": "a", "content": "b"},
        {"id": "a", "content": "c"},  # doublon str, dernier gagne
    )
    out = _dedup_messages_by_id(msgs)
    # (id=123, ignored) + (a-c au lieu de a-b).
    assert len(out) == 2
    assert out[0]["id"] == 123
    assert out[1]["content"] == "c"


def test_r20b_validation_ai_messages_not_tuple() -> None:
    """R20b : ai_messages list -> TypeError FR strict."""
    with pytest.raises(TypeError, match="tuple"):
        _fresh(ai_messages=[{"id": "a"}])  # type: ignore[arg-type]


def test_r20b_validation_message_not_dict() -> None:
    """R20b : message scalar -> TypeError FR."""
    with pytest.raises(TypeError, match="ai_messages"):
        _fresh(ai_messages=("not-a-dict",))  # type: ignore[arg-type]


def test_ec52_validation_id_non_str() -> None:
    """EC52 : id=123 (pas str) -> TypeError FR."""
    with pytest.raises(TypeError, match="id"):
        _fresh(ai_messages=({"id": 123},))


def test_ec53_validation_id_empty_str() -> None:
    """EC53 : id='' (vide) -> TypeError FR."""
    with pytest.raises(TypeError, match="id"):
        _fresh(ai_messages=({"id": ""},))


def test_ec51_message_without_id_preserved() -> None:
    """EC51 : message sans id conserve sans dedup."""
    res = _fresh(
        ai_messages=(
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        )
    )
    assert len(res.ai_messages) == 2


# --- R18 / tz-aware enforce / EC26-29 --------------------------------------


def test_ec26_naive_submitted_at_rejects() -> None:
    """EC26 : submitted_at naive -> ValueError FR."""
    with pytest.raises(ValueError, match="tz-aware"):
        _fresh(submitted_at=datetime(2026, 4, 23, 14, 0, 0))


def test_ec26_naive_started_at_rejects() -> None:
    """started_at naive si non-None -> ValueError FR."""
    with pytest.raises(ValueError, match="tz-aware"):
        _fresh(
            status=SubagentStatus.RUNNING,
            started_at=datetime(2026, 4, 23, 14, 0, 0),
        )


def test_ec26_naive_completed_at_rejects() -> None:
    """completed_at naive si non-None -> ValueError FR."""
    with pytest.raises(ValueError, match="tz-aware"):
        _fresh(
            status=SubagentStatus.COMPLETED,
            started_at=datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 4, 23, 14, 0, 1),
        )


def test_ec27_non_utc_tz_aware_accepte() -> None:
    """EC27 : timezone(timedelta(hours=2)) tz-aware accepte."""
    tz_plus2 = timezone(timedelta(hours=2))
    res = _fresh(
        submitted_at=datetime(2026, 4, 23, 14, 0, 0, tzinfo=tz_plus2),
    )
    assert res.submitted_at.tzinfo is not None


# --- R4 / to_dict ----------------------------------------------------------


def test_r4_to_dict_shape_and_iso8601() -> None:
    """R4 : to_dict serialise status.value, timestamps ISO8601, ai_messages list."""
    sub = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
    start = datetime(2026, 4, 23, 14, 0, 1, tzinfo=UTC)
    end = datetime(2026, 4, 23, 14, 0, 2, tzinfo=UTC)
    res = _fresh(
        status=SubagentStatus.COMPLETED,
        submitted_at=sub,
        started_at=start,
        completed_at=end,
        result={"x": 1},
        error=None,
        ai_messages=({"id": "m1", "role": "assistant", "content": "ok"},),
    )
    out = res.to_dict()
    assert out["status"] == "completed"
    assert out["submitted_at"] == "2026-04-23T14:00:00+00:00"
    assert out["started_at"] == "2026-04-23T14:00:01+00:00"
    assert out["completed_at"] == "2026-04-23T14:00:02+00:00"
    assert out["result"] == {"x": 1}
    assert out["error"] is None
    assert isinstance(out["ai_messages"], list)
    assert out["ai_messages"][0]["id"] == "m1"


def test_r4_to_dict_none_timestamps_preserved() -> None:
    """R4 : started_at/completed_at None -> JSON None."""
    res = _fresh()
    out = res.to_dict()
    assert out["started_at"] is None
    assert out["completed_at"] is None


def test_ec15_to_dict_empty_ai_messages_is_empty_list() -> None:
    """EC15 : ai_messages=() -> to_dict -> 'ai_messages': []."""
    res = _fresh()
    assert res.to_dict()["ai_messages"] == []


def test_ec64_to_dict_non_serialisable_result() -> None:
    """EC64 : result contient threading.Lock -> TypeError FR."""
    import threading

    res = _fresh(result=threading.Lock())
    with pytest.raises(TypeError, match=r"\$\.result"):
        res.to_dict()


def test_r4_to_dict_is_json_dumpable() -> None:
    """R4 : to_dict() donne un dict compatible json.dumps."""
    sub = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
    res = _fresh(
        submitted_at=sub,
        result=[1, "two", None, {"n": 3.14}],
    )
    payload = json.dumps(res.to_dict())
    assert "pending" in payload
