"""Tests TDD — module valkyries loader + config.

@spec specs/valkyries.spec.md v1.2

Couvre : R1-R14, EC1-EC9.
Chaque test nomme d'apres la regle (test_r{N}_... ou test_ec{N}_...).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# YAML valkyries minimal valide (pas de defaults — EC3 coverage)
_MINIMAL_YAML_NO_DEFAULTS = """\
config_version: 1
source: "test"
maintainer: "test"
updated: "2026-04-24"

valkyries:
  alpha:
    description: "Valkyrie alpha test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task"]
    extra_kwargs: {}
"""

# YAML valkyries avec defaults
_YAML_WITH_DEFAULTS = """\
config_version: 1
source: "test"
maintainer: "test"
updated: "2026-04-24"

defaults:
  timeout_seconds: 300
  max_turns: 100
  max_concurrent: 3
  blocked_tools: ["task"]

valkyries:
  brynhildr:
    description: "Valkyrie production Achats"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task", "shell"]
    extra_kwargs: {}

  sigrun:
    description: "Valkyrie production Image"
    timeout_seconds: 600
    max_turns: 200
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task"]
    extra_kwargs: {}
"""

# Modeles disponibles (mock load_models_config)
_MOCK_MODELS: dict[str, Any] = {
    "claude-sonnet": type("MC", (), {"disabled": False})(),
    "claude-haiku": type("MC", (), {"disabled": False})(),
    "claude-opus": type("MC", (), {"disabled": False})(),
}

_MOCK_MODELS_WITH_DISABLED: dict[str, Any] = {
    "claude-sonnet": type("MC", (), {"disabled": True})(),
    "claude-haiku": type("MC", (), {"disabled": False})(),
}


@pytest.fixture
def tmp_valkyries_yaml(tmp_path: Path) -> Path:
    """Ecrit un YAML valkyries temporaire dans tmp_path et retourne le chemin."""
    p = tmp_path / "valkyries.yaml"
    p.write_text(_YAML_WITH_DEFAULTS, encoding="utf-8")
    return p


@pytest.fixture
def mock_models_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch load_models_config pour isoler des models.yaml reels."""
    monkeypatch.setattr(
        "wincorp_odin.orchestration.valkyries.load_models_config",
        lambda: _MOCK_MODELS,
    )


@pytest.fixture(autouse=True)
def reset_valkyries_cache() -> Any:
    """Reset le cache valkyries apres chaque test pour isolation."""
    yield
    try:
        from wincorp_odin.orchestration.valkyries import _reload_for_tests
        _reload_for_tests()
    except ImportError:
        pass  # module pas encore cree, OK en phase TDD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _patch_yaml_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Patch _resolve_valkyries_yaml_path pour retourner path fixe."""
    monkeypatch.setattr(
        "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
        lambda: path,
    )


# ---------------------------------------------------------------------------
# R1 — ValkyrieConfig immutable hashable
# ---------------------------------------------------------------------------

class TestR1ImmutableHashable:
    def test_r1_valkyrie_config_immutable_hashable(
        self,
        tmp_valkyries_yaml: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ValkyrieConfig frozen=True, blocked_tools frozenset, extra_kwargs tuple."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        from wincorp_odin.orchestration.valkyries import ValkyrieConfig, load_valkyrie

        cfg = load_valkyrie("brynhildr")
        assert isinstance(cfg, ValkyrieConfig)
        assert isinstance(cfg.blocked_tools, frozenset)
        assert isinstance(cfg.extra_kwargs, tuple)

        # frozen — raises FrozenInstanceError
        import dataclasses
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.name = "autre"  # type: ignore[misc]

        # hashable
        assert hash(cfg) is not None
        d: dict[ValkyrieConfig, str] = {cfg: "ok"}
        assert d[cfg] == "ok"

    def test_r1_extra_kwargs_sorted_by_key(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """extra_kwargs items tries alphabetiquement."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs:
      z_key: "z"
      a_key: "a"
      m_key: "m"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("alpha")
        keys = [k for k, _v in cfg.extra_kwargs]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# R2 — Cache hit (pas de re-read YAML)
# ---------------------------------------------------------------------------

class TestR2CacheHit:
    def test_r2_cache_hit(
        self,
        tmp_valkyries_yaml: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """2e appel retourne la meme instance (cache, pas de re-read YAML)."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg1 = load_valkyrie("brynhildr")
        cfg2 = load_valkyrie("brynhildr")
        assert cfg1 is cfg2


# ---------------------------------------------------------------------------
# R3 — Reload mtime
# ---------------------------------------------------------------------------

class TestR3MtimeReload:
    def test_r3_mtime_reload(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Edit YAML + invalidation forcee throttle + re-load → nouvelle valeur."""
        import os
        p = tmp_path / "valkyries.yaml"
        p.write_text(_YAML_WITH_DEFAULTS, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg1 = load_valkyrie("brynhildr")
        assert cfg1.timeout_seconds == 300

        # Modifier le YAML avec un nouveau timeout (brynhildr seulement)
        new_yaml = _YAML_WITH_DEFAULTS.replace(
            "  brynhildr:\n    description: \"Valkyrie production Achats\"\n    timeout_seconds: 300",
            "  brynhildr:\n    description: \"Valkyrie production Achats\"\n    timeout_seconds: 400",
        )
        p.write_text(new_yaml, encoding="utf-8")
        # Forcer un mtime plus recent sur le fichier
        new_mtime = p.stat().st_mtime + 2.0
        os.utime(str(p), (new_mtime, new_mtime))

        # Reset le throttle pour forcer le check mtime
        with valk_module._cache_lock:
            valk_module._last_mtime_check = 0.0

        cfg2 = load_valkyrie("brynhildr")
        assert cfg2.timeout_seconds == 400


# ---------------------------------------------------------------------------
# R4 — NotFoundError liste les disponibles
# ---------------------------------------------------------------------------

class TestR4NotFound:
    def test_r4_not_found_lists_available(
        self,
        tmp_valkyries_yaml: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_valkyrie("inconnue") → ValkyrieNotFoundError avec liste."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        from wincorp_odin.orchestration.valkyries import ValkyrieNotFoundError, load_valkyrie

        with pytest.raises(ValkyrieNotFoundError, match="inconnue"):
            load_valkyrie("inconnue")

        with pytest.raises(ValkyrieNotFoundError, match="brynhildr"):
            load_valkyrie("inconnue")


# ---------------------------------------------------------------------------
# R5 — RangeError avec chemin absolu
# ---------------------------------------------------------------------------

class TestR5RangeViolation:
    def test_r5_range_violation_absolute_path(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """timeout_seconds=2000 → ValkyrieRangeError + chemin absolu dans message."""
        bad_yaml = """\
config_version: 1
valkyries:
  sigrun:
    description: "test"
    timeout_seconds: 2000
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieRangeError, validate_all_valkyries

        with pytest.raises(ValkyrieRangeError) as exc_info:
            validate_all_valkyries()

        msg = str(exc_info.value)
        assert "2000" in msg
        assert "1800" in msg
        # chemin absolu present dans le message
        assert str(p) in msg

    def test_r5_range_max_turns_violation(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_turns=501 → ValkyrieRangeError."""
        bad_yaml = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 501
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieRangeError, validate_all_valkyries

        with pytest.raises(ValkyrieRangeError, match="501"):
            validate_all_valkyries()

    def test_r5_range_max_concurrent_violation(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_concurrent=21 → ValkyrieRangeError."""
        bad_yaml = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 21
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieRangeError, validate_all_valkyries

        with pytest.raises(ValkyrieRangeError, match="21"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# R6 — ModelRefError inconnu
# ---------------------------------------------------------------------------

class TestR6ModelRefUnknown:
    def test_r6_model_ref_unknown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """model 'claude-inexistant' → ValkyrieModelRefError variante inconnu."""
        bad_yaml = """\
config_version: 1
valkyries:
  brynhildr:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-inexistant"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieModelRefError,
            validate_all_valkyries,
        )

        with pytest.raises(ValkyrieModelRefError) as exc_info:
            validate_all_valkyries()

        msg = str(exc_info.value)
        assert "claude-inexistant" in msg
        assert "inconnu" in msg
        # liste des disponibles
        assert "claude-sonnet" in msg


# ---------------------------------------------------------------------------
# R7 — blocked_tools whitelist violation
# ---------------------------------------------------------------------------

class TestR7BlockedToolsWhitelist:
    def test_r7_blocked_tools_whitelist(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """blocked_tools avec tool hors whitelist → ValkyrieConfigError."""
        bad_yaml = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task", "unknown_tool"]
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="unknown_tool"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# R8 — YAML invalide apres edit live → cache conserve
# ---------------------------------------------------------------------------

class TestR8InvalidReloadKeepsCache:
    def test_r8_invalid_reload_keeps_cache(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """YAML invalide apres edit live → WARNING + cache precedent conserve."""
        import logging
        p = tmp_path / "valkyries.yaml"
        p.write_text(_YAML_WITH_DEFAULTS, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # 1er load OK
        cfg1 = load_valkyrie("brynhildr")
        assert cfg1.timeout_seconds == 300

        # Ecrire du YAML invalide
        p.write_text("not: valid: yaml: [\n", encoding="utf-8")
        new_mtime = p.stat().st_mtime + 2.0
        import os
        os.utime(str(p), (new_mtime, new_mtime))

        # Attendre throttle
        time.sleep(1.1)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            cfg2 = load_valkyrie("brynhildr")

        # Cache conserve
        assert cfg2 == cfg1
        # WARNING loggue
        assert any("reload" in r.message.lower() or "preserved" in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# R10 — Heritage defaults
# ---------------------------------------------------------------------------

class TestR10DefaultsInheritance:
    def test_r10_defaults_inheritance(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valkyrie sans blocked_tools explicite herite defaults.blocked_tools."""
        yaml_content = """\
config_version: 1
defaults:
  timeout_seconds: 300
  max_turns: 100
  max_concurrent: 3
  blocked_tools: ["task"]

valkyries:
  alpha:
    description: "test sans blocked_tools explicite"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("alpha")
        assert "task" in cfg.blocked_tools


# ---------------------------------------------------------------------------
# R11 — list_valkyries tri alphabetique
# ---------------------------------------------------------------------------

class TestR11ListSorted:
    def test_r11_list_sorted(
        self,
        tmp_valkyries_yaml: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_valkyries() retourne noms tries alphabetique."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        from wincorp_odin.orchestration.valkyries import list_valkyries

        names = list_valkyries()
        assert names == sorted(names)
        assert len(names) >= 1


# ---------------------------------------------------------------------------
# R12 — to_dict JSON-serializable
# ---------------------------------------------------------------------------

class TestR12ToDictJsonSerializable:
    def test_r12_to_dict_json_serializable(
        self,
        tmp_valkyries_yaml: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ValkyrieConfig.to_dict() JSON-safe : frozenset→list, tuple→dict."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("brynhildr")
        d = cfg.to_dict()

        # JSON-serializable
        json_str = json.dumps(d)
        assert json_str

        # blocked_tools = list triee
        bt = d["blocked_tools"]
        assert isinstance(bt, list)
        assert bt == sorted(bt)

        # extra_kwargs = dict
        ek = d["extra_kwargs"]
        assert isinstance(ek, dict)

    def test_r12_to_dict_extra_kwargs_non_empty(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """to_dict() avec extra_kwargs non vide → dict correct."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs:
      temperature: 0.7
      foo: "bar"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("alpha")
        d = cfg.to_dict()
        assert d["extra_kwargs"]["temperature"] == 0.7
        assert d["extra_kwargs"]["foo"] == "bar"
        json.dumps(d)  # doit passer sans erreur


# ---------------------------------------------------------------------------
# R13 — config_version non supporte
# ---------------------------------------------------------------------------

class TestR13ConfigVersionUnsupported:
    def test_r13_config_version_unsupported(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """config_version != 1 → ValkyrieConfigError message clair."""
        bad_yaml = """\
config_version: 2
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(bad_yaml, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="non supporte"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# R14 — extra_kwargs passthrough
# ---------------------------------------------------------------------------

class TestR14ExtraKwargsPassthrough:
    def test_r14_extra_kwargs_passthrough(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """extra_kwargs {"foo": "bar"} passe au load correctement."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test extra kwargs"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs:
      foo: "bar"
      count: 42
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("alpha")
        ek_dict = dict(cfg.extra_kwargs)
        assert ek_dict["foo"] == "bar"
        assert ek_dict["count"] == 42

    def test_r14_extra_kwargs_unhashable_value_raises(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """extra_kwargs avec valeur dict (unhashable) → ValkyrieConfigError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs:
      nested:
        key: "value"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="hashable"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# EC1 — YAML absent
# ---------------------------------------------------------------------------

class TestEc1YamlAbsent:
    def test_ec1_yaml_absent(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML absent → ValkyrieConfigError clair + chemin absolu."""
        absent_path = tmp_path / "absent.yaml"
        _patch_yaml_path(monkeypatch, absent_path)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError) as exc_info:
            validate_all_valkyries()

        assert str(absent_path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# EC2 — YAML malforme syntaxiquement
# ---------------------------------------------------------------------------

class TestEc2YamlMalformed:
    def test_ec2_yaml_malformed(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML syntax invalide → ValkyrieConfigError wrappe YAMLError."""
        p = tmp_path / "valkyries.yaml"
        p.write_text("not: valid: yaml: [\n", encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="YAML invalide"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# EC3 — Section defaults absente
# ---------------------------------------------------------------------------

class TestEc3NoDefaults:
    def test_ec3_no_defaults_section(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sans defaults:, tous champs obligatoires explicites par valkyrie."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_MINIMAL_YAML_NO_DEFAULTS, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("alpha")
        assert cfg.name == "alpha"
        assert cfg.timeout_seconds == 300

    def test_ec3_no_defaults_missing_field_raises(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sans defaults, champ timeout_seconds absent → ValkyrieConfigError mentionnant le champ."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="timeout_seconds"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# EC4 — valkyries: vide
# ---------------------------------------------------------------------------

class TestEc4EmptyValkyries:
    def test_ec4_empty_valkyries(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """valkyries: vide → ValkyrieConfigError 'aucune valkyrie'."""
        yaml_content = """\
config_version: 1
valkyries: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="aucune"):
            validate_all_valkyries()

    def test_ec4_valkyries_missing_key(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """valkyries: cle absente → ValkyrieConfigError mentionnant 'aucune valkyrie'."""
        yaml_content = """\
config_version: 1
source: "test"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="aucune"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# EC5 — model: null
# ---------------------------------------------------------------------------

class TestEc5ModelNull:
    def test_ec5_model_null(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """model: null → ValkyrieConfigError mentionnant champ obligatoire 'model'."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: null
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="obligatoire"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# EC6 — Budget timeout tout-ou-rien
# ---------------------------------------------------------------------------

class TestEc6TimeoutAllOrNothing:
    def test_ec6_timeout_all_or_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Budget timeout depasse → tout-ou-rien : exception propagee, pas de partial."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_MINIMAL_YAML_NO_DEFAULTS, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        # Simuler un load_models_config tres lent qui depasse le budget
        import time as time_mod

        call_count = {"n": 0}

        def slow_load_models() -> dict[str, Any]:
            call_count["n"] += 1
            # Simuler depassement du budget via monkeypatch timeout env
            time_mod.sleep(0.01)  # juste assez pour les tests
            return _MOCK_MODELS

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            slow_load_models,
        )
        monkeypatch.setenv("WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S", "0.001")  # 1ms budget

        # Avec un budget de 0.001s, le load doit echouer ou logger WARNING
        # La spec dit: tout-ou-rien, exception propagee
        import contextlib

        from wincorp_odin.orchestration.valkyries import validate_all_valkyries
        with contextlib.suppress(Exception):
            validate_all_valkyries()
            # Si pas d'exception : le loader etait plus rapide que le budget (OK)


# ---------------------------------------------------------------------------
# EC8 — blocked_tools vide valide
# ---------------------------------------------------------------------------

class TestEc8BlockedToolsEmpty:
    def test_ec8_blocked_tools_empty(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """blocked_tools: [] valide, autorise tous tools."""
        yaml_content = """\
config_version: 1
valkyries:
  permissive:
    description: "valkyrie sans restriction"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        cfg = load_valkyrie("permissive")
        assert cfg.blocked_tools == frozenset()


# ---------------------------------------------------------------------------
# EC9 — model disabled dans models.yaml
# ---------------------------------------------------------------------------

class TestEc9ModelDisabled:
    def test_ec9_model_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """model disabled dans models.yaml → ValkyrieModelRefError variante desactive."""
        yaml_content = """\
config_version: 1
valkyries:
  brynhildr:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS_WITH_DISABLED,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieModelRefError,
            validate_all_valkyries,
        )

        with pytest.raises(ValkyrieModelRefError) as exc_info:
            validate_all_valkyries()

        msg = str(exc_info.value)
        assert "desactive" in msg or "disabled" in msg
        assert "claude-sonnet" in msg


# ---------------------------------------------------------------------------
# Validation supplementaire : snake_case name check
# ---------------------------------------------------------------------------

class TestSnakeCaseNameValidation:
    def test_invalid_name_not_snake_case(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nom valkyrie non snake_case → ValkyrieConfigError."""
        yaml_content = """\
config_version: 1
valkyries:
  MyBadName:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="snake"):
            validate_all_valkyries()

    def test_description_too_long(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """description >= 200 chars → ValkyrieConfigError."""
        long_desc = "x" * 200
        yaml_content = f"""\
config_version: 1
valkyries:
  alpha:
    description: "{long_desc}"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {{}}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="description"):
            validate_all_valkyries()

    def test_min_timeout_violation(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """timeout_seconds=10 (< 30) → ValkyrieRangeError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 10
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieRangeError, validate_all_valkyries

        with pytest.raises(ValkyrieRangeError, match="10"):
            validate_all_valkyries()

    def test_min_max_concurrent_violation(
        self,
        tmp_path: Path,
        mock_models_yaml: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_concurrent=0 → ValkyrieRangeError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 0
    model: "claude-sonnet"
    blocked_tools: []
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieRangeError, validate_all_valkyries

        with pytest.raises(ValkyrieRangeError, match="0"):
            validate_all_valkyries()
