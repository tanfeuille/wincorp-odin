"""Tests types orchestration : AIMessage TypedDict, TaskCallable, InitialState.

@spec specs/orchestration.spec.md v2.1.1 §3.2
"""
from __future__ import annotations

from wincorp_odin.orchestration.types import AIMessage, InitialState, TaskCallable


def test_aimessage_instanciation_vide() -> None:
    """AIMessage total=False : dict vide accepte."""
    msg: AIMessage = {}
    assert msg == {}


def test_aimessage_instanciation_complete() -> None:
    """AIMessage avec toutes les cles."""
    msg: AIMessage = {
        "id": "m-1",
        "role": "user",
        "content": "hello",
        "name": "tool-name",
        "tool_call_id": "tc-1",
    }
    assert msg["id"] == "m-1"
    assert msg["role"] == "user"


def test_taskcallable_is_exported() -> None:
    """TaskCallable est importable (alias Callable)."""
    assert TaskCallable is not None


def test_initialstate_is_exported() -> None:
    """InitialState est importable (alias Mapping)."""
    assert InitialState is not None
