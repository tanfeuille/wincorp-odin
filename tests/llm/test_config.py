"""Tests config : parsing YAML, interpolation, resolution path URD, schema.

@spec specs/llm-factory.spec.md v1.2
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# EC1 — Fichier YAML absent
# ---------------------------------------------------------------------------


def test_ec1_yaml_file_not_found_raises_model_config_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC1/R17 : URD_PATH defini mais models.yaml absent -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    urd.mkdir()
    (urd / "referentiels").mkdir()
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        create_model("sonnet")


# ---------------------------------------------------------------------------
# EC2 — YAML syntaxe invalide
# ---------------------------------------------------------------------------


def test_ec2_malformed_yaml_raises_model_config_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC2 : YAML syntaxe cassee -> ModelConfigError (avec ligne/colonne si possible)."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        "models: [  invalid yaml : : :\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        create_model("sonnet")


# ---------------------------------------------------------------------------
# EC3 — YAML vide ou models manquant
# ---------------------------------------------------------------------------


def test_ec3_empty_models_raises_model_config_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC3 : liste models vide -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        "config_version: 1\nmodels: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        create_model("sonnet")


# ---------------------------------------------------------------------------
# EC4 — Champ obligatoire manquant
# ---------------------------------------------------------------------------


def test_ec4_missing_required_field_raises_schema_error(
    mock_anthropic_api_key: str,
    malformed_yaml: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC4 : champ use: manquant -> ModelConfigSchemaError."""
    from wincorp_odin.llm import ModelConfigSchemaError, create_model

    urd_root = malformed_yaml.parent.parent
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd_root))

    with pytest.raises(ModelConfigSchemaError):
        create_model("broken")


# ---------------------------------------------------------------------------
# EC5 — Type incorrect
# ---------------------------------------------------------------------------


def test_ec5_wrong_type_raises_schema_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC5 : supports_thinking en str au lieu de bool -> ModelConfigSchemaError."""
    from wincorp_odin.llm import ModelConfigSchemaError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "typo"
    display_name: "Typo"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: "yes"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigSchemaError):
        create_model("typo")


# ---------------------------------------------------------------------------
# EC6 — Doublon de name
# ---------------------------------------------------------------------------


def test_ec6_duplicate_name_raises_config_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC6 : meme name declare 2 fois -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "dup"
    display_name: "Un"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
  - name: "dup"
    display_name: "Deux"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test-2"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError) as excinfo:
        create_model("dup")
    assert "dup" in str(excinfo.value)


# ---------------------------------------------------------------------------
# EC7/EC8 — Secret manquant / chaine vide
# ---------------------------------------------------------------------------


def test_ec7_missing_env_var_raises_secret_missing_error(
    patched_yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC7 : ANTHROPIC_API_KEY absente -> SecretMissingError."""
    from wincorp_odin.llm import SecretMissingError, create_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SecretMissingError):
        create_model("sonnet")


def test_ec8_empty_env_var_raises_secret_missing_error(
    patched_yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC8 : ANTHROPIC_API_KEY=chaine vide -> SecretMissingError."""
    from wincorp_odin.llm import SecretMissingError, create_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(SecretMissingError):
        create_model("sonnet")


# ---------------------------------------------------------------------------
# EC9 — typo ${UNKNOWN_VAR}
# ---------------------------------------------------------------------------


def test_ec9_unknown_var_raises_config_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC9 : ${UNKNOWN_VAR} avec env var absente -> SecretMissingError."""
    from wincorp_odin.llm import SecretMissingError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "typo"
    display_name: "Typo"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${UNKNOWN_VAR_FOR_TEST}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))
    monkeypatch.delenv("UNKNOWN_VAR_FOR_TEST", raising=False)

    with pytest.raises(SecretMissingError):
        create_model("typo")


# ---------------------------------------------------------------------------
# EC17 — Conflit OneDrive
# ---------------------------------------------------------------------------


def test_ec17_onedrive_conflict_raises_error(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC17 : pattern OneDrive detecte -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    src = Path(__file__).parent / "fixtures" / "models_minimal.yaml"
    content = src.read_text(encoding="utf-8")
    (ref / "models.yaml").write_text(content, encoding="utf-8")
    # Creer un fichier conflit
    (ref / "models-DESKTOP-ABC123.yaml").write_text(content, encoding="utf-8")
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError) as excinfo:
        create_model("sonnet")
    msg = str(excinfo.value).lower()
    assert "conflit" in msg or "conflict" in msg or "onedrive" in msg


# ---------------------------------------------------------------------------
# EC24 — Taille > 1 MB
# ---------------------------------------------------------------------------


def test_ec24_yaml_size_exceeds_1mb_rejected(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R15/EC24 : YAML > 1 Mo -> ModelConfigError sans parser."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    # Generer un fichier > 1 Mo : padding via commentaire YAML
    padding = "# " + ("x" * 80) + "\n"
    body = "config_version: 1\nmodels:\n  - name: \"x\"\n"
    # Multiplier jusqu'a depasser 1 Mo
    big = body + padding * 15000  # ~1.2 Mo
    (ref / "models.yaml").write_text(big, encoding="utf-8")
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        create_model("x")


# ---------------------------------------------------------------------------
# R14 — safe_load rejet de tag !!python/object
# ---------------------------------------------------------------------------


def test_r14_yaml_unsafe_tag_rejected(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R14 : tag !!python/object rejete par safe_load -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - !!python/object:os.system
    command: "whoami"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        create_model("whatever")


# ---------------------------------------------------------------------------
# R13/EC23 — Whitelist extra_kwargs
# ---------------------------------------------------------------------------


def test_ec23_extra_kwargs_whitelist_rejects_base_url(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R13/EC23 : extra_kwargs base_url rejete -> ExtraKwargsForbiddenError."""
    from wincorp_odin.llm import ExtraKwargsForbiddenError, create_model

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "evil"
    display_name: "Evil"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    extra_kwargs:
      base_url: "https://evil.com"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ExtraKwargsForbiddenError) as excinfo:
        create_model("evil")
    assert "base_url" in str(excinfo.value)


def test_r13_extra_kwargs_whitelist_accepts_temperature(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R13 : temperature est dans la whitelist -> passe."""
    from wincorp_odin.llm import create_model

    create_model("haiku")
    kwargs = mock_chat_anthropic.call_args.kwargs
    # temperature est dans extra_kwargs du YAML full
    assert kwargs.get("temperature") == 0.7


# ---------------------------------------------------------------------------
# R17 — Installed mode (pas de .git) sans env var
# ---------------------------------------------------------------------------


def test_r17_installed_requires_env_var(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R17/EC27 : pas de .git detecte + pas de WINCORP_URD_PATH -> ModelConfigError FATAL."""
    from wincorp_odin.llm import ModelConfigError
    from wincorp_odin.llm import config as config_mod

    monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

    # Simuler un __file__ isole hors repo git
    fake_parents = [tmp_path / f"p{i}" for i in range(6)]
    for p in fake_parents:
        p.mkdir(parents=True, exist_ok=True)

    # Monkeypatcher la fonction de detection dev pour ne rien trouver
    monkeypatch.setattr(
        config_mod,
        "_detect_dev_urd_path",
        lambda: None,
    )

    with pytest.raises(ModelConfigError) as excinfo:
        config_mod._resolve_urd_path()
    msg = str(excinfo.value)
    assert "WINCORP_URD_PATH" in msg


# ---------------------------------------------------------------------------
# R17 — Path traversal : WINCORP_URD_PATH hors racines autorisees
# ---------------------------------------------------------------------------


def test_r17_urd_path_traversal_rejected(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R17 : WINCORP_URD_PATH hors racines autorisees -> ModelConfigError generique (sans chemin)."""
    from wincorp_odin.llm import ModelConfigError
    from wincorp_odin.llm import config as config_mod

    # Chemin hors $HOME et hors project_root detectable
    forbidden = Path("C:/") if Path("C:/").exists() else Path("/")
    # Forcer _assert_under_allowed_root a rejeter
    monkeypatch.setattr(config_mod, "_find_project_root", lambda: None)
    # HOME simule sur tmp_path -> forbidden ne sera pas sous home
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("WINCORP_URD_PATH", str(forbidden))

    with pytest.raises(ModelConfigError) as excinfo:
        config_mod._resolve_urd_path()
    msg = str(excinfo.value)
    # Message generique, ne doit PAS contenir le chemin interdit
    assert str(forbidden) not in msg


# ---------------------------------------------------------------------------
# R17 — Dev mode auto-detect .git
# ---------------------------------------------------------------------------


def test_r17_dev_mode_autodetects_wincorp_dev(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R17 : .git present en ancetre -> trouve wincorp-urd/referentiels/models.yaml sans env var."""
    from wincorp_odin.llm import config as config_mod

    monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

    # Creer une simili-arbo : tmp/wincorp-workspace/wincorp-odin/.git + tmp/wincorp-urd/referentiels/models.yaml
    dev_root = tmp_path / "wincorp-workspace"
    odin = dev_root / "wincorp-odin"
    odin_src = odin / "src" / "wincorp_odin" / "llm"
    odin_src.mkdir(parents=True)
    (odin / ".git").mkdir()

    urd = dev_root / "wincorp-urd" / "referentiels"
    urd.mkdir(parents=True)
    src = Path(__file__).parent / "fixtures" / "models_minimal.yaml"
    (urd / "models.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Faux __file__ dans la sous-arbo odin
    fake_file = odin_src / "config.py"
    fake_file.write_text("", encoding="utf-8")

    def fake_detect() -> Path | None:
        # Simule le scan des 5 parents : trouve .git puis renvoie wincorp-urd/referentiels/models.yaml
        # Le path retourne par _detect_dev_urd_path doit etre la racine URD (pas le yaml)
        return dev_root / "wincorp-urd"

    monkeypatch.setattr(config_mod, "_detect_dev_urd_path", fake_detect)

    resolved = config_mod._resolve_urd_path()
    assert resolved.name == "models.yaml"
    assert resolved.exists()


# ---------------------------------------------------------------------------
# R17 — PR-004 : dev mode implicite aussi passe par _assert_under_allowed_root
# ---------------------------------------------------------------------------


def test_r17_dev_mode_rejects_path_outside_allowed_root(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PR-004 : mode dev implicite appelle _assert_under_allowed_root.

    Meme symetrie que le mode explicite WINCORP_URD_PATH : si _detect_dev_urd_path
    retourne un chemin hors racines autorisees (scenario pathologique : faux .git
    ailleurs), le resolve doit remonter ModelConfigError generique.
    """
    from wincorp_odin.llm import ModelConfigError
    from wincorp_odin.llm import config as config_mod

    monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

    # Fabriquer une arbo hors racines autorisees
    forbidden_urd = tmp_path / "forbidden_root" / "wincorp-urd"
    (forbidden_urd / "referentiels").mkdir(parents=True)
    (forbidden_urd / "referentiels" / "models.yaml").write_text(
        "config_version: 1\n", encoding="utf-8"
    )

    # Forcer la detection dev a retourner ce chemin interdit
    monkeypatch.setattr(
        config_mod, "_detect_dev_urd_path", lambda: forbidden_urd
    )
    # Forcer _find_project_root a None pour eviter que tmp_path soit couvert
    monkeypatch.setattr(config_mod, "_find_project_root", lambda: None)
    # HOME simule hors forbidden_urd
    hidden_home = tmp_path / "real_home"
    hidden_home.mkdir()
    monkeypatch.setenv("HOME", str(hidden_home))
    monkeypatch.setenv("USERPROFILE", str(hidden_home))

    with pytest.raises(ModelConfigError) as excinfo:
        config_mod._resolve_urd_path()
    msg = str(excinfo.value)
    # Message generique — ne revele pas le chemin tente
    assert "hors des racines autorisees" in msg
    assert str(forbidden_urd) not in msg
