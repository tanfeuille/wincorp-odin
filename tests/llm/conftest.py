"""Fixtures pytest partagees pour tests wincorp_odin.llm.

@spec specs/llm-factory.spec.md v1.2
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_factory_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Purge integrale de l'etat factory entre chaque test.

    Couvre PB-012 (registry cache) et nouveautes v1.2 (_deferred_resolutions,
    _last_mtime_check, _STARTUP_TIMEOUT_S, _RUNTIME_TIMEOUT_S).
    """
    # Import tardif pour ne pas planter si le module n'existe pas encore (phase TDD rouge)
    try:
        from wincorp_odin.llm import _registry, factory
    except ImportError:
        return
    factory._cache.clear()
    factory._resolved_configs.clear()
    factory._deferred_resolutions.clear()
    factory._yaml_mtime = None
    factory._last_mtime_check = 0.0
    _registry._class_cache.clear()


@pytest.fixture
def mock_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Definit ANTHROPIC_API_KEY pour les tests (valeur factice mais formee)."""
    value = "sk-ant-test-xxxxxxxxxxxxxxxxxxxx"
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    return value


def _copy_fixture(src_name: str, tmp_path: Path) -> Path:
    """Copie un fichier fixture YAML vers un dossier wincorp-urd simule.

    Structure attendue par WINCORP_URD_PATH : <racine>/referentiels/models.yaml.
    """
    src = Path(__file__).parent / "fixtures" / src_name
    urd_root = tmp_path / "wincorp-urd"
    ref_dir = urd_root / "referentiels"
    ref_dir.mkdir(parents=True, exist_ok=True)
    dst = ref_dir / "models.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    """Fichier models.yaml minimal (1 modele claude-sonnet sans thinking)."""
    return _copy_fixture("models_minimal.yaml", tmp_path)


@pytest.fixture
def full_yaml(tmp_path: Path) -> Path:
    """Fichier models.yaml complet (3 modeles : sonnet, opus, haiku)."""
    return _copy_fixture("models_full.yaml", tmp_path)


@pytest.fixture
def malformed_yaml(tmp_path: Path) -> Path:
    """Fichier models.yaml avec champ obligatoire manquant (use:)."""
    return _copy_fixture("models_malformed.yaml", tmp_path)


@pytest.fixture
def patched_yaml_path(monkeypatch: pytest.MonkeyPatch, minimal_yaml: Path) -> Path:
    """Configure WINCORP_URD_PATH pour pointer vers le tmp_path/wincorp-urd."""
    urd_root = minimal_yaml.parent.parent
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd_root))
    return urd_root


@pytest.fixture
def patched_yaml_path_full(monkeypatch: pytest.MonkeyPatch, full_yaml: Path) -> Path:
    """Variante avec fixture full (3 modeles)."""
    urd_root = full_yaml.parent.parent
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd_root))
    return urd_root


@pytest.fixture
def mock_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ChatAnthropic AVANT instanciation.

    Retourne des instances uniques a chaque call pour tester l'identite cache.
    """
    call_count = {"n": 0}

    def _factory(**kwargs: Any) -> MagicMock:
        call_count["n"] += 1
        instance = MagicMock(name=f"ChatAnthropicInstance#{call_count['n']}")
        instance.init_kwargs = kwargs
        return instance

    mock = MagicMock(side_effect=_factory)
    mock.name = "ChatAnthropic"
    # Patch au niveau du module langchain_anthropic (import dynamique via _registry)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", mock)
    return mock
