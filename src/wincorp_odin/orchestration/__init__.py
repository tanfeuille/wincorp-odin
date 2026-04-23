"""Module orchestration (Phase 2 DeerFlow).

@spec specs/orchestration.spec.md v2.1.1

Exports publics :
    - SubagentStatus : enum str (pending/running/completed/failed/cancelled/timed_out).
    - SubagentResult : dataclass frozen immutable avec timestamps + ai_messages.
    - SubagentExecutor : executor thread-based avec 2 pools + registre borne.
    - build_initial_state : heritage whitelist parent->enfant.
    - truncate_task_calls : fan-out clamp [1, 20].
    - SubagentSink : Protocol observabilite.
    - LogSink : sink par defaut stdout JSON.
    - AIMessage, TaskCallable, InitialState : types.
    - Exceptions : SubagentError + variantes.
"""
from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentError,
    SubagentExecutorClosedError,
    SubagentExecutorOverflowError,
    SubagentTaskIdConflictError,
    SubagentTaskNotFoundError,
)
from wincorp_odin.orchestration.middleware import truncate_task_calls
from wincorp_odin.orchestration.result import SubagentResult, SubagentStatus
from wincorp_odin.orchestration.sinks import LogSink, SubagentSink
from wincorp_odin.orchestration.state import build_initial_state
from wincorp_odin.orchestration.types import AIMessage, InitialState, TaskCallable

__all__ = [
    "AIMessage",
    "InitialState",
    "LogSink",
    "SubagentCancelledException",
    "SubagentError",
    "SubagentExecutorClosedError",
    "SubagentExecutorOverflowError",
    "SubagentResult",
    "SubagentSink",
    "SubagentStatus",
    "SubagentTaskIdConflictError",
    "SubagentTaskNotFoundError",
    "TaskCallable",
    "build_initial_state",
    "truncate_task_calls",
]
