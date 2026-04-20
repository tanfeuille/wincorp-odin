"""Tests factory `create_client` — SDK Anthropic brut.

@spec specs/llm-factory.spec.md v1.3.2 §27

Couvre :
- R29 : create_client retourne anthropic.Anthropic depuis models.yaml
- EC42 : name inconnu -> ModelNotFoundError (+ liste alphabetique disponibles)
- EC43 : provider non-Anthropic -> CapabilityMismatchError
- EC44 : modele disabled -> ModelNotFoundError (meme comportement que create_model)

Pattern identique a `test_factory.py` : WINCORP_URD_PATH patchke via fixture,
cle API factice via `mock_anthropic_api_key`, YAML fixture copie dans tmp_path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from tests.llm.conftest import _copy_fixture  # noqa: I001  (conftest helper)

# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture
def non_anthropic_yaml(tmp_path: Path) -> Path:
    """YAML avec 1 modele sonnet (OK), 1 deepseek (KO pour create_client),
    1 sonnet-disabled (KO).
    """
    return _copy_fixture("models_non_anthropic.yaml", tmp_path)


@pytest.fixture
def patched_non_anthropic_path(
    monkeypatch: pytest.MonkeyPatch, non_anthropic_yaml: Path
) -> Path:
    """Configure WINCORP_URD_PATH vers la fixture non-Anthropic."""
    urd_root = non_anthropic_yaml.parent.parent
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd_root))
    # Les secrets requis par la fixture — deepseek non-utilise cote create_client,
    # mais load_models_config doit resoudre les ${VAR} de TOUS les modeles actifs.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test-xxxxxxxxxxxxxxxxxxxx")
    return urd_root


# ---------------------------------------------------------------------------
# R29 — chemin nominal
# ---------------------------------------------------------------------------


def test_r29_create_client_returns_anthropic_instance(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
) -> None:
    """R29 : create_client('sonnet') retourne une instance anthropic.Anthropic."""
    from wincorp_odin.llm import create_client

    client = create_client("sonnet")

    assert isinstance(client, anthropic.Anthropic)


def test_r29_create_client_uses_api_key_from_yaml(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R29 : la cle API transmise au SDK vient du YAML (interpolation ${VAR})."""
    captured_kwargs: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock(spec=anthropic.Anthropic)

    monkeypatch.setattr("anthropic.Anthropic", _capture)

    from wincorp_odin.llm import create_client

    create_client("sonnet")

    assert captured_kwargs.get("api_key") == mock_anthropic_api_key


def test_r29_create_client_returns_new_instance_each_call(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
) -> None:
    """R29 : pas de cache cote create_client — chaque appel cree une instance.

    Le SDK Anthropic gere son pool HTTP en interne, donc l'absence de cache
    est volontaire (KISS — voir client.py module docstring).
    """
    from wincorp_odin.llm import create_client

    a = create_client("sonnet")
    b = create_client("sonnet")

    # Pas d'identite is — deux instances SDK distinctes.
    assert a is not b


# ---------------------------------------------------------------------------
# EC42 — name inconnu
# ---------------------------------------------------------------------------


def test_ec42_create_client_unknown_name_raises_model_not_found(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
) -> None:
    """EC42 : name absent du YAML -> ModelNotFoundError avec liste disponibles."""
    from wincorp_odin.llm import ModelNotFoundError, create_client

    with pytest.raises(ModelNotFoundError) as exc_info:
        create_client("modele-inexistant")

    msg = str(exc_info.value)
    assert "modele-inexistant" in msg
    # La liste des modeles disponibles doit apparaitre (minimal_yaml contient 'sonnet')
    assert "sonnet" in msg
    assert "models.yaml" in msg


def test_ec42_create_client_unknown_name_lists_available_sorted(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
) -> None:
    """EC42 : la liste disponibles est triee alpha (determinisme UX)."""
    from wincorp_odin.llm import ModelNotFoundError, create_client

    with pytest.raises(ModelNotFoundError) as exc_info:
        create_client("xxx")

    msg = str(exc_info.value)
    # full_yaml contient haiku, opus, sonnet — verifier l'ordre alpha.
    pos_haiku = msg.find("haiku")
    pos_opus = msg.find("opus")
    pos_sonnet = msg.find("sonnet")
    assert pos_haiku != -1
    assert pos_opus != -1
    assert pos_sonnet != -1
    assert pos_haiku < pos_opus < pos_sonnet


# ---------------------------------------------------------------------------
# EC43 — provider non-Anthropic
# ---------------------------------------------------------------------------


def test_ec43_create_client_rejects_non_anthropic_provider(
    mock_anthropic_api_key: str,
    patched_non_anthropic_path: Path,
) -> None:
    """EC43 : provider DeepSeek (ou autre non-Anthropic) -> CapabilityMismatchError."""
    from wincorp_odin.llm import CapabilityMismatchError, create_client

    with pytest.raises(CapabilityMismatchError) as exc_info:
        create_client("deepseek-v3")

    msg = str(exc_info.value)
    assert "deepseek-v3" in msg
    # Message doit rediriger vers create_model.
    assert "create_model" in msg
    assert "langchain_deepseek:ChatDeepSeek" in msg


def test_ec43_create_client_accepts_anthropic_prefix(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EC43 bis : prefixe `anthropic:` (SDK raw explicite) accepte.

    Cas d'usage : un YAML qui declare `use: "anthropic:Anthropic"` directement
    (sans passer par LangChain). create_client doit accepter les deux prefixes.
    """
    urd_root = tmp_path / "wincorp-urd"
    ref_dir = urd_root / "referentiels"
    ref_dir.mkdir(parents=True)
    (ref_dir / "models.yaml").write_text(
        """
config_version: 1
source: "Test anthropic: prefix"
maintainer: "Test"
updated: "2026-04-20"
models:
  - name: "sonnet-raw"
    display_name: "Claude Sonnet (SDK raw)"
    use: "anthropic:Anthropic"
    model: "claude-sonnet-4-5-20250929"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 8192
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd_root))

    from wincorp_odin.llm import create_client

    client = create_client("sonnet-raw")

    assert isinstance(client, anthropic.Anthropic)


# ---------------------------------------------------------------------------
# EC44 — modele disabled
# ---------------------------------------------------------------------------


def test_ec44_create_client_disabled_model_raises_model_not_found(
    mock_anthropic_api_key: str,
    patched_non_anthropic_path: Path,
) -> None:
    """EC44 : modele disabled=true -> ModelNotFoundError (meme UX que create_model)."""
    from wincorp_odin.llm import ModelNotFoundError, create_client

    with pytest.raises(ModelNotFoundError) as exc_info:
        create_client("sonnet-disabled")

    msg = str(exc_info.value)
    assert "sonnet-disabled" in msg
    # sonnet-disabled NE DOIT PAS apparaitre dans la liste disponibles
    # (on parse apres ':' et verifie qu'il n'y est pas dans cette portion).
    available_section = msg.split("disponibles")[-1] if "disponibles" in msg else msg
    assert "sonnet-disabled" not in available_section.split(".")[0]
