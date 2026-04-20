"""Test integration : lecture du vrai wincorp-urd/referentiels/models.yaml.

@spec specs/llm-factory.spec.md v1.2

PB-017 : marque @pytest.mark.integration + skipif si URD reel absent.
Execute UNIQUEMENT avec ANTHROPIC_API_KEY defini ET wincorp-urd present.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Chemin suppose du vrai URD en dev
_URD_CANDIDATES = [
    Path("C:/Users/Tanfeuille/Documents/wincorp-dev/wincorp-urd/referentiels/models.yaml"),
    Path(__file__).resolve().parents[4] / "wincorp-urd" / "referentiels" / "models.yaml",
]


def _find_real_urd() -> Path | None:
    """Retourne le chemin du vrai models.yaml ou None si absent."""
    for candidate in _URD_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY absent — integration skip (PB-017)",
)
@pytest.mark.skipif(
    _find_real_urd() is None,
    reason="wincorp-urd/referentiels/models.yaml absent — integration skip",
)
def test_integration_real_urd_yaml_loads_all_declared_models() -> None:
    """Smoke : le vrai YAML URD charge, tous les modeles sont accessibles."""
    from wincorp_odin.llm import load_models_config

    configs = load_models_config()
    assert len(configs) > 0, "Au moins un modele doit etre declare dans l'URD reel"
    for name, cfg in configs.items():
        assert cfg.name == name
        assert cfg.use.count(":") == 1, f"Format use: invalide pour {name}"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY absent — integration skip (PB-017)",
)
@pytest.mark.skipif(
    _find_real_urd() is None,
    reason="wincorp-urd/referentiels/models.yaml absent — integration skip",
)
def test_integration_validate_all_models_passes_on_real_urd() -> None:
    """validate_all_models() passe sur le vrai YAML URD sans exception."""
    from wincorp_odin.llm import validate_all_models

    validate_all_models()
