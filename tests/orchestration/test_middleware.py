"""Tests truncate_task_calls : R14/R15, EC8-10, EC41-49c.

@spec specs/orchestration.spec.md v2.1.1 §3.6
"""
from __future__ import annotations

import logging

import pytest

from wincorp_odin.orchestration.middleware import truncate_task_calls

# --- EC8 / empty list ------------------------------------------------------


def test_ec8_empty_input_returns_empty() -> None:
    """EC8 : truncate_task_calls([]) -> []."""
    assert truncate_task_calls([]) == []


# --- EC9 / ordre preserve --------------------------------------------------


def test_ec9_mixed_order_preserved_truncation_applied() -> None:
    """EC9 : 2 task + read_file + 3 task, max=2 -> 2 task + read_file."""
    calls = [
        {"name": "task", "args": {"t": 1}},
        {"name": "task", "args": {"t": 2}},
        {"name": "read_file", "args": {"p": "a"}},
        {"name": "task", "args": {"t": 3}},
        {"name": "task", "args": {"t": 4}},
        {"name": "task", "args": {"t": 5}},
    ]
    out = truncate_task_calls(calls, max_concurrent=2)
    assert out == [
        {"name": "task", "args": {"t": 1}},
        {"name": "task", "args": {"t": 2}},
        {"name": "read_file", "args": {"p": "a"}},
    ]


def test_r15_log_warning_on_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """R15 : log WARNING avec nombre droppe."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.middleware")
    calls = [{"name": "task"}] * 5
    out = truncate_task_calls(calls, max_concurrent=2)
    assert len(out) == 2
    assert any("3 calls 'task' droppes" in r.message for r in caplog.records)


# --- EC10 / exactement max -------------------------------------------------


def test_ec10_exactly_max_no_truncation_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """EC10 : count == max -> pas de truncation ni log WARNING."""
    caplog.set_level(logging.WARNING, logger="wincorp_odin.orchestration.middleware")
    calls = [{"name": "task"}] * 3
    out = truncate_task_calls(calls, max_concurrent=3)
    assert len(out) == 3
    assert not any("droppes" in r.message for r in caplog.records)


# --- EC41 / sans name -------------------------------------------------------


def test_ec41_dict_sans_name_preserve() -> None:
    """EC41 : dict sans name -> conserve."""
    calls = [{"args": {"x": 1}}]
    assert truncate_task_calls(calls) == calls


def test_ec42_name_none_preserve() -> None:
    """EC42 : {'name': None} -> conserve."""
    calls = [{"name": None, "args": {}}]
    assert truncate_task_calls(calls) == calls


# --- EC43 / tool_name strict equality --------------------------------------


def test_ec43_tool_name_equality_not_regex() -> None:
    """EC43 : tool_name comparaison ==, pas de regex."""
    calls = [
        {"name": "task.slow"},
        {"name": "task"},
    ]
    out = truncate_task_calls(calls, max_concurrent=1, tool_name="task")
    # 'task.slow' n'est pas 'task' -> conserve. 'task' -> conserve (1 max).
    assert len(out) == 2


# --- EC44 / EC44b : Sequence strict ----------------------------------------


def test_ec44_generator_rejected() -> None:
    """EC44 : generator -> TypeError FR."""

    def gen() -> object:
        yield {"name": "task"}

    with pytest.raises(TypeError, match="Sequence"):
        truncate_task_calls(gen())  # type: ignore[arg-type]


def test_ec44b_set_rejected() -> None:
    """EC44b : set -> TypeError FR."""
    with pytest.raises(TypeError, match="Sequence"):
        truncate_task_calls({("x", 1)})  # type: ignore[arg-type]


def test_str_rejected_as_sequence() -> None:
    """str est une Sequence Python mais rejetee semantiquement."""
    with pytest.raises(TypeError, match="Sequence"):
        truncate_task_calls("task")  # type: ignore[arg-type]


def test_bytes_rejected_as_sequence() -> None:
    """bytes rejete."""
    with pytest.raises(TypeError, match="Sequence"):
        truncate_task_calls(b"task")  # type: ignore[arg-type]


def test_tuple_accepted() -> None:
    """tuple est une Sequence valide."""
    calls = ({"name": "task"},)
    out = truncate_task_calls(calls, max_concurrent=3)
    assert len(out) == 1


# --- R14 / max_concurrent type / plage -------------------------------------


def test_ec45_max_concurrent_bool_rejected() -> None:
    """EC45 : bool True -> TypeError FR (bool exclu strict int)."""
    with pytest.raises(TypeError, match="entier"):
        truncate_task_calls([], max_concurrent=True)  # type: ignore[arg-type]


def test_ec46_max_concurrent_float_rejected() -> None:
    """EC46 : 3.5 -> TypeError FR."""
    with pytest.raises(TypeError, match="entier"):
        truncate_task_calls([], max_concurrent=3.5)  # type: ignore[arg-type]


def test_ec47_max_concurrent_str_rejected() -> None:
    """EC47 : '3' str -> TypeError FR."""
    with pytest.raises(TypeError, match="entier"):
        truncate_task_calls([], max_concurrent="3")  # type: ignore[arg-type]


def test_ec48_max_concurrent_zero_rejected() -> None:
    """EC48 : 0 -> ValueError FR."""
    with pytest.raises(ValueError, match=r"entre 1 et 20"):
        truncate_task_calls([], max_concurrent=0)


def test_ec49_max_concurrent_21_rejected() -> None:
    """EC49 : 21 -> ValueError FR."""
    with pytest.raises(ValueError, match=r"entre 1 et 20"):
        truncate_task_calls([], max_concurrent=21)


def test_ec49b_max_concurrent_1_accepte() -> None:
    """EC49b : 1 borne basse accepte."""
    out = truncate_task_calls([{"name": "task"}] * 3, max_concurrent=1)
    assert len(out) == 1


def test_ec49c_max_concurrent_20_accepte() -> None:
    """EC49c : 20 borne haute accepte."""
    calls = [{"name": "task"}] * 30
    out = truncate_task_calls(calls, max_concurrent=20)
    assert len(out) == 20


def test_negative_max_concurrent_rejected() -> None:
    """Negatif -> ValueError FR."""
    with pytest.raises(ValueError, match=r"entre 1 et 20"):
        truncate_task_calls([], max_concurrent=-1)


# --- tool_name empty -------------------------------------------------------


def test_tool_name_empty_rejected() -> None:
    """tool_name vide -> ValueError FR."""
    with pytest.raises(ValueError, match="tool_name"):
        truncate_task_calls([], tool_name="")


def test_tool_name_non_str_rejected() -> None:
    """tool_name non str -> ValueError FR."""
    with pytest.raises(ValueError, match="tool_name"):
        truncate_task_calls([], tool_name=None)  # type: ignore[arg-type]


# --- default max_concurrent -----------------------------------------------


def test_default_max_concurrent_3() -> None:
    """Defaut max_concurrent=3."""
    calls = [{"name": "task"}] * 5
    out = truncate_task_calls(calls)
    # 3 task kept. Dropped 2.
    assert len(out) == 3


def test_custom_tool_name() -> None:
    """tool_name='compute' limite les compute, pas les task."""
    calls = [
        {"name": "task"},
        {"name": "compute"},
        {"name": "compute"},
        {"name": "compute"},
    ]
    out = truncate_task_calls(calls, max_concurrent=2, tool_name="compute")
    # Les 2 premiers compute + task conserves, 1 compute drop.
    assert len(out) == 3
    names = [c.get("name") for c in out]
    assert names == ["task", "compute", "compute"]
