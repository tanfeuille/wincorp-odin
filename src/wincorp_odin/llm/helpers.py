"""Helpers d'inspection d'exceptions + lookup model_id — Phase 1.9b.

@spec specs/llm-factory.spec.md v1.3.3
"""
from __future__ import annotations

from wincorp_odin.llm.config import load_models_config
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


def get_model_id(name: str) -> str:
    """Retourne le model_id provider (ex: claude-sonnet-4-5-20250929) depuis models.yaml.

    Complement de `create_client` / `create_model` — utile pour les consommateurs
    qui passent le model en parametre a `client.messages.create(...)` (SDK Anthropic
    brut) sans instancier via la factory.

    Args:
        name: Nom logique du modele declare dans models.yaml (ex: "claude-sonnet").

    Returns:
        Le model_id provider (ex: "claude-sonnet-4-5-20250929").

    Raises:
        ModelNotFoundError: Si `name` n'est pas declare ou est desactive
            (disabled: true).

    Example:
        >>> from wincorp_odin.llm import get_model_id
        >>> model_id = get_model_id("claude-sonnet")
        >>> client.messages.create(model=model_id, messages=[...])
    """
    configs = load_models_config()
    cfg = configs.get(name)
    if cfg is None or cfg.disabled:
        available = sorted(n for n, c in configs.items() if not c.disabled)
        raise ModelNotFoundError(
            f"[ERREUR] Modele '{name}' introuvable. Typo probable ? "
            f"Modeles disponibles : {available}."
        )
    return cfg.model
