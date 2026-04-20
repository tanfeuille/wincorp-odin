"""Helpers d'inspection d'exceptions — sucre syntaxique PB-011.

@spec specs/llm-factory.spec.md v1.2
"""
from __future__ import annotations

from wincorp_odin.llm.exceptions import (
    CapabilityMismatchError,
    ModelNotFoundError,
)


def is_model_not_found(exc: BaseException) -> bool:
    """Sucre syntaxique remplacant `except ModelNotFoundError`."""
    return isinstance(exc, ModelNotFoundError)


def is_capability_mismatch(exc: BaseException) -> bool:
    """Sucre syntaxique remplacant `except CapabilityMismatchError`."""
    return isinstance(exc, CapabilityMismatchError)
