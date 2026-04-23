"""Module orchestration (Phase 2 DeerFlow + Phase 3 valkyries).

@spec specs/orchestration.spec.md v2.1.1
@spec specs/valkyries.spec.md v1.2

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
    - ValkyrieConfig : dataclass frozen hashable roles produit.
    - ValkyrieToolGuard : middleware LangChain enforcement blocked_tools.
    - load_valkyrie, list_valkyries, validate_all_valkyries : API loader.
    - create_valkyrie_chat : factory principale consumer.
    - ValkyrieConfigError, ValkyrieNotFoundError, ValkyrieModelRefError,
      ValkyrieRangeError : exceptions valkyries.
"""
from wincorp_odin.orchestration.exceptions import (
    SubagentCancelledException,
    SubagentError,
    SubagentExecutorClosedError,
    SubagentExecutorOverflowError,
    SubagentTaskIdConflictError,
    SubagentTaskNotFoundError,
)
from wincorp_odin.orchestration.executor import SubagentExecutor
from wincorp_odin.orchestration.middleware import truncate_task_calls
from wincorp_odin.orchestration.result import SubagentResult, SubagentStatus
from wincorp_odin.orchestration.sinks import LogSink, SubagentSink
from wincorp_odin.orchestration.state import build_initial_state
from wincorp_odin.orchestration.types import AIMessage, InitialState, TaskCallable
from wincorp_odin.orchestration.valkyries import (
    ValkyrieConfig,
    ValkyrieConfigError,
    ValkyrieModelRefError,
    ValkyrieNotFoundError,
    ValkyrieRangeError,
    ValkyrieToolGuard,
    create_valkyrie_chat,
    list_valkyries,
    load_valkyrie,
    validate_all_valkyries,
)

__all__ = [
    "AIMessage",
    "InitialState",
    "LogSink",
    "SubagentCancelledException",
    "SubagentError",
    "SubagentExecutor",
    "SubagentExecutorClosedError",
    "SubagentExecutorOverflowError",
    "SubagentResult",
    "SubagentSink",
    "SubagentStatus",
    "SubagentTaskIdConflictError",
    "SubagentTaskNotFoundError",
    "TaskCallable",
    "ValkyrieConfig",
    "ValkyrieConfigError",
    "ValkyrieModelRefError",
    "ValkyrieNotFoundError",
    "ValkyrieRangeError",
    "ValkyrieToolGuard",
    "build_initial_state",
    "create_valkyrie_chat",
    "list_valkyries",
    "load_valkyrie",
    "truncate_task_calls",
    "validate_all_valkyries",
]
