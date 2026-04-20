"""wincorp_odin.llm — factory LLM providers.

@spec specs/llm-factory.spec.md v1.2

Phase 1 DeerFlow inspiration. Exports publics uniquement :
_registry et _whitelist restent prives (R8). `_reload_for_tests` n'est PAS
exporte (PB-015).
"""
from __future__ import annotations

from wincorp_odin.llm.config import ModelConfig, load_models_config
from wincorp_odin.llm.exceptions import (
    CapabilityMismatchError,
    ExtraKwargsForbiddenError,
    ModelAuthenticationError,
    ModelConfigError,
    ModelConfigSchemaError,
    ModelNotFoundError,
    OdinLlmError,
    ProviderNotInstalledError,
    SecretMissingError,
)
from wincorp_odin.llm.factory import create_model, validate_all_models
from wincorp_odin.llm.helpers import is_capability_mismatch, is_model_not_found

__all__ = [
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
]
