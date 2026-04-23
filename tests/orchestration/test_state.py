"""Tests build_initial_state : R5/R6/R7, EC11-13, EC36-40.

@spec specs/orchestration.spec.md v2.1.1 §3.5
"""
from __future__ import annotations

import copy
from types import MappingProxyType
from uuid import UUID

import pytest

from wincorp_odin.orchestration.state import build_initial_state


def test_ec11_empty_parent_returns_empty_dict() -> None:
    """EC11 : parent vide -> dict vide."""
    assert build_initial_state({}) == {}


def test_r5_whitelist_only() -> None:
    """R5 : seules les cles whitelist sont heritees. Messages exclus."""
    parent = {
        "sandbox_state": {"cwd": "/tmp"},
        "messages": [{"role": "user"}],
        "tool_calls": [{}],
        "subagent_results": [],
    }
    out = build_initial_state(parent)
    assert out == {"sandbox_state": {"cwd": "/tmp"}}


def test_ec12_sandbox_state_only_no_other_keys() -> None:
    """EC12 : seule sandbox_state presente -> dict avec seule cette cle."""
    parent = {"sandbox_state": {"x": 1}, "messages": []}
    assert build_initial_state(parent) == {"sandbox_state": {"x": 1}}


def test_r5_all_four_whitelist_keys() -> None:
    """R5 : les 4 cles whitelist sont heritees."""
    parent = {
        "sandbox_state": {"a": 1},
        "thread_data": {"b": 2},
        "session_id": "s-1",
        "trace_id": "t-1",
    }
    out = build_initial_state(parent)
    assert out == parent


def test_r6_deepcopy_applied() -> None:
    """R6 : deepcopy inconditionnel -> mutation out n'affecte pas parent."""
    parent = {"sandbox_state": {"nested": [1, 2, 3]}}
    out = build_initial_state(parent)
    out["sandbox_state"]["nested"].append(4)
    assert parent["sandbox_state"]["nested"] == [1, 2, 3]


def test_ec36_deepcopy_raises_wraps_valueerror() -> None:
    """EC36 : objet avec __deepcopy__ qui raise -> ValueError FR avec cle."""

    class _Boom:
        def __deepcopy__(self, memo: dict[int, object]) -> object:
            raise RuntimeError("cannot copy")

    parent = {"sandbox_state": _Boom()}
    with pytest.raises(ValueError, match=r"sandbox_state"):
        build_initial_state(parent)


def test_ec37_circular_reference_in_thread_data_ok() -> None:
    """EC37 : reference circulaire dans thread_data -> deepcopy gere."""
    circular: dict[str, object] = {"self": None}
    circular["self"] = circular
    parent = {"thread_data": circular}
    out = build_initial_state(parent)
    assert "thread_data" in out
    # La copie est distincte.
    assert out["thread_data"] is not circular


def test_ec38_path_uuid_enum_deep_copied() -> None:
    """EC38 : Path, UUID, Enum -> copiables via deepcopy."""
    from enum import Enum as StdEnum
    from pathlib import Path

    class _Role(StdEnum):
        ADMIN = "admin"

    parent = {
        "sandbox_state": {
            "path": Path("/tmp"),
            "uid": UUID("12345678-1234-5678-1234-567812345678"),
            "role": _Role.ADMIN,
        }
    }
    out = build_initial_state(parent)
    assert out["sandbox_state"]["path"] == Path("/tmp")
    assert isinstance(out["sandbox_state"]["uid"], UUID)
    assert out["sandbox_state"]["role"] == _Role.ADMIN


# --- R7 / extra_overrides --------------------------------------------------


def test_ec13_extra_overrides_replace_not_merge() -> None:
    """EC13 : extra_overrides REPLACE (pas merge) la cle entiere."""
    parent = {"sandbox_state": {"x": 1}}
    out = build_initial_state(parent, extra_overrides={"sandbox_state": {"y": 2}})
    assert out == {"sandbox_state": {"y": 2}}


def test_ec39_extra_overrides_allows_none_value() -> None:
    """EC39 : override {'sandbox_state': None} -> remplace par None."""
    parent = {"sandbox_state": {"x": 1}}
    out = build_initial_state(parent, extra_overrides={"sandbox_state": None})
    assert out == {"sandbox_state": None}


def test_r7_extra_overrides_adds_new_key() -> None:
    """R7 : override injecte une cle absente du parent."""
    parent: dict[str, object] = {}
    out = build_initial_state(parent, extra_overrides={"custom": 99})
    assert out == {"custom": 99}


def test_r7_extra_overrides_none_noop() -> None:
    """extra_overrides=None est autorise et no-op."""
    parent = {"sandbox_state": {"x": 1}}
    out = build_initial_state(parent, extra_overrides=None)
    assert out == {"sandbox_state": {"x": 1}}


# --- EC40 / TypeError parent_state -----------------------------------------


def test_ec40_parent_state_non_mapping_raises() -> None:
    """EC40 : parent_state list -> TypeError FR."""
    with pytest.raises(TypeError, match="Mapping"):
        build_initial_state([("a", 1)])  # type: ignore[arg-type]


def test_extra_overrides_non_mapping_raises() -> None:
    """extra_overrides non Mapping -> TypeError FR."""
    with pytest.raises(TypeError, match="Mapping"):
        build_initial_state({}, extra_overrides=[("a", 1)])  # type: ignore[arg-type]


def test_parent_state_mappingproxy_accepte() -> None:
    """MappingProxyType accepte (Mapping)."""
    parent = MappingProxyType({"sandbox_state": {"x": 1}})
    out = build_initial_state(parent)
    assert out == {"sandbox_state": {"x": 1}}


# --- Preserve parent identity (copy isolation) -----------------------------


def test_parent_state_not_mutated() -> None:
    """build_initial_state ne mute pas le parent."""
    parent = {"sandbox_state": {"x": 1}}
    snapshot = copy.deepcopy(parent)
    _ = build_initial_state(parent, extra_overrides={"sandbox_state": {"y": 2}})
    assert parent == snapshot
