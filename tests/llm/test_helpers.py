"""Tests helpers — get_model_id + is_model_not_found + is_capability_mismatch.

@spec specs/llm-factory.spec.md v1.3.3
"""
from __future__ import annotations

from pathlib import Path

import pytest

from wincorp_odin.llm import (
    CapabilityMismatchError,
    ModelNotFoundError,
    get_model_id,
    is_capability_mismatch,
    is_model_not_found,
)


class TestIsModelNotFound:
    """Couvre le sucre syntaxique is_model_not_found (helpers.py)."""

    def test_true_pour_model_not_found_error(self) -> None:
        exc = ModelNotFoundError("inconnue")
        assert is_model_not_found(exc) is True

    def test_false_pour_autre_exception(self) -> None:
        assert is_model_not_found(ValueError("bad")) is False

    def test_false_pour_capability_mismatch(self) -> None:
        exc = CapabilityMismatchError("pas thinking")
        assert is_model_not_found(exc) is False


class TestIsCapabilityMismatch:
    """Couvre le sucre syntaxique is_capability_mismatch (helpers.py)."""

    def test_true_pour_capability_mismatch_error(self) -> None:
        exc = CapabilityMismatchError("pas thinking")
        assert is_capability_mismatch(exc) is True

    def test_false_pour_autre_exception(self) -> None:
        assert is_capability_mismatch(TypeError("bad")) is False

    def test_false_pour_model_not_found(self) -> None:
        exc = ModelNotFoundError("inconnue")
        assert is_capability_mismatch(exc) is False


class TestGetModelId:
    """Couvre get_model_id — lookup model_id provider depuis models.yaml."""

    def test_retourne_model_id_provider(
        self,
        mock_anthropic_api_key: str,
        patched_yaml_path_full: Path,
    ) -> None:
        """Happy path : sonnet → model_id declare dans models_full.yaml."""
        model_id = get_model_id("sonnet")
        # fixture models_full.yaml contient sonnet → claude-sonnet-4-5-20250929
        assert model_id == "claude-sonnet-4-5-20250929"

    def test_leve_model_not_found_si_nom_absent(
        self,
        mock_anthropic_api_key: str,
        patched_yaml_path_full: Path,
    ) -> None:
        """Nom inexistant → ModelNotFoundError avec liste modeles dispo."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            get_model_id("claude-inexistant")
        msg = str(exc_info.value)
        assert "claude-inexistant" in msg
        assert "Modeles disponibles" in msg

    def test_message_erreur_inclut_modeles_dispo(
        self,
        mock_anthropic_api_key: str,
        patched_yaml_path_full: Path,
    ) -> None:
        """Liste des noms disponibles presente dans le message."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            get_model_id("typo-model")
        msg = str(exc_info.value)
        # Au moins un modele de la fixture full doit apparaitre
        assert "sonnet" in msg or "haiku" in msg or "opus" in msg

    def test_coherent_avec_load_models_config(
        self,
        mock_anthropic_api_key: str,
        patched_yaml_path_full: Path,
    ) -> None:
        """get_model_id(name) == load_models_config()[name].model."""
        from wincorp_odin.llm import load_models_config

        configs = load_models_config()
        for name, cfg in configs.items():
            if cfg.disabled:
                continue
            assert get_model_id(name) == cfg.model
