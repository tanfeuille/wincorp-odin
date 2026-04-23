"""Tests hierarchie exceptions orchestration.

@spec specs/orchestration.spec.md v2.1.1 §3.1
"""
from __future__ import annotations

import pytest

from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentError,
    SubagentExecutorClosedError,
    SubagentExecutorOverflowError,
    SubagentTaskIdConflictError,
    SubagentTaskNotFoundError,
)


def test_subagent_error_root() -> None:
    """SubagentError est la racine (herite Exception)."""
    assert issubclass(SubagentError, Exception)
    with pytest.raises(SubagentError):
        raise SubagentError("boom")


def test_subagent_executor_closed_error_hierarchy() -> None:
    """SubagentExecutorClosedError herite SubagentError ET RuntimeError."""
    assert issubclass(SubagentExecutorClosedError, SubagentError)
    assert issubclass(SubagentExecutorClosedError, RuntimeError)


def test_subagent_executor_overflow_error_hierarchy() -> None:
    """SubagentExecutorOverflowError herite SubagentError ET RuntimeError."""
    assert issubclass(SubagentExecutorOverflowError, SubagentError)
    assert issubclass(SubagentExecutorOverflowError, RuntimeError)


def test_subagent_task_not_found_error_hierarchy() -> None:
    """SubagentTaskNotFoundError herite SubagentError ET KeyError."""
    assert issubclass(SubagentTaskNotFoundError, SubagentError)
    assert issubclass(SubagentTaskNotFoundError, KeyError)


def test_subagent_task_id_conflict_error_hierarchy() -> None:
    """SubagentTaskIdConflictError herite SubagentError ET ValueError."""
    assert issubclass(SubagentTaskIdConflictError, SubagentError)
    assert issubclass(SubagentTaskIdConflictError, ValueError)


def test_subagent_cancelled_exception_default_message_fr() -> None:
    """SubagentCancelledException message par defaut en FR."""
    exc = SubagentCancelledException()
    assert "annulee" in str(exc).lower() or "annul" in str(exc).lower()
    assert issubclass(SubagentCancelledException, SubagentError)


def test_subagent_cancelled_exception_custom_message() -> None:
    """Message personnalise preserve."""
    exc = SubagentCancelledException("motif specifique")
    assert str(exc) == "motif specifique"
