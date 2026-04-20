"""wincorp_odin.llm — factory LLM providers.

@spec specs/llm-factory.spec.md v1.3.2

Phase 1 DeerFlow inspiration. Exports publics uniquement :
_registry et _whitelist restent prives (R8). `_reload_for_tests` n'est PAS
exporte (PB-015).

v1.3 : ajout middlewares Phase 1.4/1.5/1.6 (circuit_breaker, retry, tokens)
avec leurs exceptions et helpers de config publics.

v1.3.2 (Phase 1.9a) : ajout `create_client` (SDK Anthropic brut) pour
consommateurs `client.messages.create(...)` — voir §27.
"""
from __future__ import annotations

from wincorp_odin.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from wincorp_odin.llm.client import create_client
from wincorp_odin.llm.config import ModelConfig, load_models_config
from wincorp_odin.llm.exceptions import (
    CapabilityMismatchError,
    CircuitOpenError,
    ExtraKwargsForbiddenError,
    ModelAuthenticationError,
    ModelConfigError,
    ModelConfigSchemaError,
    ModelNotFoundError,
    OdinLlmError,
    ProviderNotInstalledError,
    RetryExhaustedError,
    SecretMissingError,
    TokenTrackingError,
)
from wincorp_odin.llm.factory import create_model, validate_all_models
from wincorp_odin.llm.helpers import is_capability_mismatch, is_model_not_found
from wincorp_odin.llm.retry import RetryConfig
from wincorp_odin.llm.tokens import (
    PricingConfig,
    TokenUsageEvent,
    clear_context,
    get_sink,
    set_context,
)

__all__ = [
    # Core Phase 1.1-1.2
    "CapabilityMismatchError",
    "ExtraKwargsForbiddenError",
    "ModelAuthenticationError",
    "ModelConfig",
    "ModelConfigError",
    "ModelConfigSchemaError",
    "ModelNotFoundError",
    "OdinLlmError",
    "ProviderNotInstalledError",
    "SecretMissingError",
    "create_model",
    "is_capability_mismatch",
    "is_model_not_found",
    "load_models_config",
    "validate_all_models",
    # v1.3 — middlewares (§22-24)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "PricingConfig",
    "RetryConfig",
    "RetryExhaustedError",
    "TokenTrackingError",
    "TokenUsageEvent",
    "clear_context",
    "get_sink",
    "set_context",
    # v1.3.2 — SDK client factory (§27, Phase 1.9a)
    "create_client",
]
