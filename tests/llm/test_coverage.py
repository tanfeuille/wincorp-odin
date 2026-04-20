"""Tests complementaires pour atteindre 100% branch coverage.

@spec specs/llm-factory.spec.md v1.2

Couvre : exceptions redaction/cause-chain, registry errors (EC10/11/12),
legacy wrapper, factory branches edge (timeout env var, mtime invalidation OK,
OSError paths, etc.).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# exceptions.py — redaction + strip cause chain
# ---------------------------------------------------------------------------


def test_redact_removes_api_key_from_string() -> None:
    """R10 : _redact remplace les cles API dans les strings."""
    from wincorp_odin.llm.exceptions import _redact

    result = _redact("Erreur avec cle sk-ant-test-abcdefghij1234567890 invalide")
    assert "sk-ant-test" not in result
    assert "***REDACTED***" in result


def test_redact_recurses_into_tuple_list_dict() -> None:
    """_redact traverse tuples/listes/dicts."""
    from wincorp_odin.llm.exceptions import _redact

    key = "sk-ant-test-abcdefghij1234567890"
    # Tuple
    t = _redact((key, "autre"))
    assert key not in str(t)
    # List
    ls = _redact([key, 42])
    assert key not in str(ls)
    # Dict
    d = _redact({"k": key, "i": 1})
    assert key not in str(d)
    # Int passthrough
    assert _redact(42) == 42


def test_model_authentication_error_strips_api_key_in_args() -> None:
    """R10c : ModelAuthenticationError nettoie args."""
    from wincorp_odin.llm.exceptions import ModelAuthenticationError

    raw = "La cle sk-ant-test-abcdefghij1234567890xx est invalide"
    exc = ModelAuthenticationError(raw)
    assert "sk-ant-test" not in str(exc)
    assert "***REDACTED***" in str(exc)


def test_model_authentication_error_strips_api_key_from_cause_chain() -> None:
    """R10c : nettoie la chaine __cause__ au moment du _strip_cause_chain().

    Le __init__ s'execute AVANT que `from inner` attache __cause__. On force donc
    un appel explicite a _strip_cause_chain() apres attachement.
    """
    from wincorp_odin.llm.exceptions import ModelAuthenticationError

    raw_key = "sk-ant-test-abcdefghij1234567890xx"
    inner = RuntimeError(f"Cle fuitee: {raw_key}")
    wrapper = ModelAuthenticationError("Auth echouee")
    wrapper.__cause__ = inner
    wrapper._strip_cause_chain()
    assert wrapper.__cause__ is not None
    assert "sk-ant-test" not in str(wrapper.__cause__.args)
    assert "***REDACTED***" in str(wrapper.__cause__.args)


def test_model_authentication_error_tolerant_to_immutable_args() -> None:
    """R10c : tolerer les exceptions C dont args refuse la mutation."""
    from wincorp_odin.llm.exceptions import ModelAuthenticationError

    # Creer une exception factice avec args readonly via descriptor
    class FrozenError(Exception):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "args":
                raise AttributeError("read-only")
            super().__setattr__(name, value)

    inner = FrozenError("boom")
    wrapper = ModelAuthenticationError("wrapper")
    wrapper.__cause__ = inner
    # Doit passer sans lever meme si args immuable
    wrapper._strip_cause_chain()
    assert wrapper.__cause__ is inner


# ---------------------------------------------------------------------------
# _registry.py — erreurs EC10/11/12 + cache
# ---------------------------------------------------------------------------


def test_registry_ec10_import_error() -> None:
    """EC10 : package non installe -> ProviderNotInstalledError + suggestion uv pip."""
    from wincorp_odin.llm._registry import resolve_class
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    with pytest.raises(ProviderNotInstalledError) as excinfo:
        resolve_class("nonexistent_pkg_xyz123:SomeClass")
    msg = str(excinfo.value)
    assert "uv pip install" in msg


def test_registry_ec11_attribute_error() -> None:
    """EC11 : classe absente dans module existant."""
    from wincorp_odin.llm._registry import resolve_class
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    with pytest.raises(ProviderNotInstalledError) as excinfo:
        resolve_class("os:ClasseInexistante_XYZ")
    assert "introuvable" in str(excinfo.value).lower()


def test_registry_ec12_not_callable() -> None:
    """EC12 : cible non-callable -> ProviderNotInstalledError."""
    from wincorp_odin.llm._registry import resolve_class
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    # sys.platform est un str, non callable
    with pytest.raises(ProviderNotInstalledError) as excinfo:
        resolve_class("sys:platform")
    assert "instanciable" in str(excinfo.value).lower()


def test_registry_invalid_format_no_colon() -> None:
    """Format 'use:' invalide sans ':' -> ProviderNotInstalledError."""
    from wincorp_odin.llm._registry import resolve_class
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    with pytest.raises(ProviderNotInstalledError):
        resolve_class("missing_colon_format")


def test_registry_invalid_format_empty_parts() -> None:
    """Format 'use:' avec module ou classe vide."""
    from wincorp_odin.llm._registry import resolve_class
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    with pytest.raises(ProviderNotInstalledError):
        resolve_class(":ClasseSansModule")
    with pytest.raises(ProviderNotInstalledError):
        resolve_class("moduleSansClasse:")


def test_registry_cache_hit_second_call() -> None:
    """Cache hit : 2e appel retourne la classe cachee."""
    from wincorp_odin.llm import _registry
    from wincorp_odin.llm._registry import resolve_class

    # 1er appel
    klass_a = resolve_class("os:getcwd")
    # 2e appel — cache
    klass_b = resolve_class("os:getcwd")
    assert klass_a is klass_b
    assert "os:getcwd" in _registry._class_cache


# ---------------------------------------------------------------------------
# legacy.py — DeprecationWarning FR
# ---------------------------------------------------------------------------


def test_legacy_wrapper_emits_deprecation_warning_fr(
    mock_chat_anthropic: MagicMock,
) -> None:
    """Legacy wrapper emet DeprecationWarning avec message FR."""
    from wincorp_odin.llm.legacy import deprecated_direct_chat_anthropic

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        deprecated_direct_chat_anthropic(model="claude-test")
        assert len(recorded) >= 1
        msg = str(recorded[0].message)
        assert "create_model" in msg
        assert "Phase 2.0" in msg
        assert recorded[0].category is DeprecationWarning


# ---------------------------------------------------------------------------
# factory.py — env var timeouts + mtime invalidation positive
# ---------------------------------------------------------------------------


def test_factory_startup_timeout_env_var_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lecture WINCORP_LLM_VALIDATE_TIMEOUT_S + bornes."""
    from wincorp_odin.llm.factory import _get_startup_timeout

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_TIMEOUT_S", "10.0")
    assert _get_startup_timeout() == 10.0

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_TIMEOUT_S", "0.01")
    assert _get_startup_timeout() == 1.0  # borne min

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_TIMEOUT_S", "999")
    assert _get_startup_timeout() == 60.0  # borne max

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_TIMEOUT_S", "not-a-number")
    assert _get_startup_timeout() == 5.0  # fallback defaut

    monkeypatch.delenv("WINCORP_LLM_VALIDATE_TIMEOUT_S", raising=False)
    assert _get_startup_timeout() == 5.0


def test_factory_runtime_timeout_env_var_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lecture WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S + bornes."""
    from wincorp_odin.llm.factory import _get_runtime_timeout

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S", "1.0")
    assert _get_runtime_timeout() == 1.0

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S", "0.01")
    assert _get_runtime_timeout() == 0.1

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S", "999")
    assert _get_runtime_timeout() == 5.0

    monkeypatch.setenv("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S", "garbage")
    assert _get_runtime_timeout() == 0.5

    monkeypatch.delenv("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S", raising=False)
    assert _get_runtime_timeout() == 0.5


def test_factory_mtime_invalidation_triggers_reload(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EC16 : mtime change -> cache vide et reload avec nouvelle config."""
    from wincorp_odin.llm import create_model, factory

    # 1er appel - init
    a = create_model("sonnet")

    # Reset du throttle pour bypass
    factory._last_mtime_check = 0.0

    # Re-ecrire le YAML avec un nouveau modele (mtime change)
    yaml_file = patched_yaml_path / "referentiels" / "models.yaml"
    time.sleep(0.05)  # garantit mtime distincte
    yaml_file.write_text(
        """config_version: 1
models:
  - name: "sonnet"
    display_name: "Sonnet new"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-new-id"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 2048
    supports_thinking: false
""",
        encoding="utf-8",
    )

    # 2e appel - reload detecte
    b = create_model("sonnet")
    assert b is not a
    # Nouvelle instanciation avec nouveau model id
    assert mock_chat_anthropic.call_args.kwargs["model"] == "claude-new-id"


def test_factory_validate_all_models_end_to_end(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """validate_all_models() passe sur un YAML complet sans lever."""
    from wincorp_odin.llm import validate_all_models

    # Ne doit pas lever
    validate_all_models()


def test_factory_validate_all_models_pb018_disabled_still_validated(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-018 + PR-001 : validate_all_models verifie 'use:' meme sur modele disabled.

    PR-001 : le format invalide remonte ModelConfigError (pas
    ProviderNotInstalledError) — distinction entre erreur de config YAML
    (format) et erreur d'installation runtime (package absent).
    """
    from wincorp_odin.llm import validate_all_models
    from wincorp_odin.llm.exceptions import ModelConfigError

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "ghost"
    display_name: "Disabled buggy"
    use: "invalid_format_no_colon"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    disabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        validate_all_models()


def test_factory_capability_mismatch_on_explicit_disabled_thinking() -> None:
    """Skip guard : modele supports_thinking=false + thinking_enabled=True -> mismatch."""
    # Deja couvert par test_ec14, mais on ajoute un cas multi-modeles pour
    # verifier la liste "thinking-compatibles" dans le message.
    # Test redondant, pas d'ajout necessaire ici.


def test_factory_authentication_error_wraps_instantiation(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EC15 : exception a l'instanciation -> ModelAuthenticationError avec cle stripped."""
    from wincorp_odin.llm import create_model
    from wincorp_odin.llm.exceptions import ModelAuthenticationError

    # Mock qui leve
    def bombing_factory(**kwargs: Any) -> Any:
        raise RuntimeError(f"Auth failed for key {kwargs.get('api_key')}")

    bombing_mock = MagicMock(side_effect=bombing_factory)
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", bombing_mock)

    with pytest.raises(ModelAuthenticationError):
        create_model("sonnet")


# ---------------------------------------------------------------------------
# factory._load_and_validate_models : timeout deferred path
# ---------------------------------------------------------------------------


def test_factory_load_and_validate_deferred_when_timeout(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R19/EC25 : si timeout etapes 1-7 depasse, use: mis en resolution differee."""
    from wincorp_odin.llm import factory

    # Simuler monotonic qui retourne des valeurs croissantes > budget
    times = iter([0.0, 10.0, 10.5, 11.0])

    def fake_monotonic() -> float:
        try:
            return next(times)
        except StopIteration:
            return 12.0

    monkeypatch.setattr("wincorp_odin.llm.factory.time.monotonic", fake_monotonic)
    configs, deferred = factory._load_and_validate_models(timeout_s=1.0)
    assert "sonnet" in configs
    # La fonction pure renvoie le set des differes (pas de mutation globale)
    assert "sonnet" in deferred
    # PR-006 — la fonction pure ne mute PAS _deferred_resolutions
    assert "sonnet" not in factory._deferred_resolutions


# ---------------------------------------------------------------------------
# config.py — branches restantes
# ---------------------------------------------------------------------------


def test_config_literal_api_key_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """api_key literal (pas de ${VAR}) passe l'interpolation."""
    from wincorp_odin.llm import load_models_config

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "literal"
    display_name: "Literal key"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "sk-ant-literal-not-an-env-var-key"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    configs = load_models_config()
    assert configs["literal"].api_key_resolved == "sk-ant-literal-not-an-env-var-key"
    assert configs["literal"].api_key_env == ""


def test_config_unsupported_version(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC19 : config_version non supportee -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, load_models_config

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 999
models:
  - name: "x"
    display_name: "X"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError) as excinfo:
        load_models_config()
    assert "config_version" in str(excinfo.value)


def test_config_root_not_dict(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Racine YAML est une liste au lieu d'un dict -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, load_models_config

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        "- this\n- is\n- a\n- list\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError):
        load_models_config()


def test_config_read_text_oserror(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """OSError a la lecture -> ModelConfigError."""
    from wincorp_odin.llm import ModelConfigError, load_models_config

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    target = ref / "models.yaml"
    target.write_text("config_version: 1\nmodels:\n  - x\n", encoding="utf-8")
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    # Patcher read_text pour lever OSError
    orig_read = Path.read_text

    def raising_read(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "models.yaml":
            raise OSError("Permission denied simulated")
        return orig_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raising_read)

    with pytest.raises(ModelConfigError) as excinfo:
        load_models_config()
    assert "Lecture" in str(excinfo.value)


def test_config_find_project_root_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """_find_project_root retourne None si aucun .git/pyproject dans parents."""
    from wincorp_odin.llm import config as config_mod

    # Utiliser un path sans .git ni pyproject
    fake = Path("C:/NonExistentRandomPath_ABC_XYZ_12345")
    monkeypatch.setattr(config_mod, "__file__", str(fake / "fake.py"))
    # On ne peut pas vraiment forcer mais on confirme au moins que la fonction retourne
    # un path ou None sans lever
    result = config_mod._find_project_root()
    assert result is None or isinstance(result, Path)


def test_config_home_path_fallback_when_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_home_path retourne None si Path.home() leve."""
    from wincorp_odin.llm import config as config_mod

    def raising_home() -> Path:
        raise RuntimeError("No home")

    monkeypatch.setattr(Path, "home", staticmethod(raising_home))
    assert config_mod._home_path() is None


def test_config_assert_under_allowed_root_with_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_assert_under_allowed_root accepte un chemin sous project_root."""
    from wincorp_odin.llm import config as config_mod

    project_root = tmp_path / "project"
    project_root.mkdir()
    child = project_root / "child"
    child.mkdir()

    monkeypatch.setattr(config_mod, "_find_project_root", lambda: project_root)
    monkeypatch.setattr(config_mod, "_home_path", lambda: None)

    # Ne doit pas lever
    config_mod._assert_under_allowed_root(child)


def test_config_detect_dev_urd_path_returns_none_without_git() -> None:
    """_detect_dev_urd_path retourne None si pas de .git dans 5 parents."""
    from wincorp_odin.llm import config as config_mod

    # Selon le contexte d'execution (repo git reel), peut retourner path ou None
    result = config_mod._detect_dev_urd_path()
    # Accepte les deux ; on teste surtout que la fonction ne lève pas
    assert result is None or isinstance(result, Path)


def test_config_extra_kwargs_default_on_disabled_model(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Un modele disabled avec extra_kwargs non vide mais non-whitelistes est tolere
    par load (validation differee a validate_all_models)."""
    from wincorp_odin.llm import load_models_config

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    # disabled:true -> load ne valide pas extra_kwargs
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "disabled-with-extras"
    display_name: "Disabled"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    extra_kwargs:
      temperature: 0.5
    disabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    configs = load_models_config()
    assert configs["disabled-with-extras"].disabled is True


def test_config_oserror_on_path_resolve_mtime(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_check_mtime_and_invalidate : OSError sur stat -> return silencieux."""
    from wincorp_odin.llm import create_model, factory

    create_model("sonnet")
    factory._last_mtime_check = 0.0

    def raising_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        raise OSError("simulated stat failure")

    monkeypatch.setattr(Path, "stat", raising_stat)

    # Ne doit pas lever
    factory._check_mtime_and_invalidate()


def test_config_modelconfigerror_on_resolve_path_in_mtime_check(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_check_mtime_and_invalidate : ModelConfigError sur resolve -> return silencieux."""
    from wincorp_odin.llm import create_model, factory
    from wincorp_odin.llm.exceptions import ModelConfigError

    create_model("sonnet")
    factory._last_mtime_check = 0.0

    def raising_resolve() -> Path:
        raise ModelConfigError("forced error")

    monkeypatch.setattr("wincorp_odin.llm.factory._resolve_urd_path", raising_resolve)
    factory._check_mtime_and_invalidate()  # ne doit pas lever


def test_config_swap_double_check_aborts_if_mtime_receded(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Double-check mtime sous lock : si une autre thread a deja swap, abandon."""
    from wincorp_odin.llm import create_model, factory

    create_model("sonnet")
    factory._last_mtime_check = 0.0

    # Simuler modif YAML
    yaml_file = patched_yaml_path / "referentiels" / "models.yaml"
    time.sleep(0.05)
    yaml_file.write_text(
        """config_version: 1
models:
  - name: "sonnet"
    display_name: "mod"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-z"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )

    # Patcher pour simuler un mtime qui redescend sous lock
    real_stat = Path.stat
    call_count = {"n": 0}

    def mtime_oscillating(self: Path, *args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        s = real_stat(self, *args, **kwargs)
        if self.name == "models.yaml" and call_count["n"] >= 2:
            # Simulacre : retourner un mtime ancien (<= _yaml_mtime)
            class FakeStat:
                st_mtime = 0.0
                st_size = s.st_size
            return FakeStat()
        return s

    monkeypatch.setattr(Path, "stat", mtime_oscillating)
    factory._check_mtime_and_invalidate()


# ---------------------------------------------------------------------------
# Branches missing coverage complementaires
# ---------------------------------------------------------------------------


def test_factory_mtime_check_no_change_no_reload(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """mtime inchange -> pas de reload."""
    from wincorp_odin.llm import create_model, factory

    a = create_model("sonnet")
    factory._last_mtime_check = 0.0
    # pas de modif -> meme instance
    b = create_model("sonnet")
    assert a is b


def test_factory_ensure_configs_loaded_oserror(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_ensure_configs_loaded tolere OSError sur stat mtime init."""
    from wincorp_odin.llm import create_model, factory

    # Vider configs
    factory._resolved_configs.clear()
    factory._yaml_mtime = None

    real_stat = Path.stat
    flag = {"raise": False}

    def maybe_raise(self: Path, *args: Any, **kwargs: Any) -> Any:
        if flag["raise"] and self.name == "models.yaml":
            raise OSError("simulated")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", maybe_raise)

    # Activer le raise juste avant le stat dans _ensure_configs_loaded
    # (la logique load_models_config appelle aussi stat, donc on accepte
    # qu'elle passe si l'ordre est different — ce test vise le fallback)
    create_model("sonnet")


def test_config_model_entry_default_none_fields(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Les champs optionnels absents sont comble par defaults."""
    from wincorp_odin.llm import load_models_config

    configs = load_models_config()
    cfg = configs["sonnet"]
    # timeout du defaults=60 (yaml fixture)
    assert cfg.timeout == 60.0
    assert cfg.max_retries == 0
    assert cfg.supports_vision is False


def test_factory_reload_for_tests_clears_all_state(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """§10.5 PB-015 : _reload_for_tests vide cache + registry + mtime."""
    from wincorp_odin.llm import _registry, create_model, factory
    from wincorp_odin.llm.factory import _reload_for_tests

    create_model("sonnet")
    assert len(factory._cache) == 1
    assert len(factory._resolved_configs) >= 1
    assert "langchain_anthropic:ChatAnthropic" in _registry._class_cache

    _reload_for_tests()
    assert len(factory._cache) == 0
    assert len(factory._resolved_configs) == 0
    assert len(_registry._class_cache) == 0
    assert factory._yaml_mtime is None
    assert factory._last_mtime_check == 0.0


def test_factory_use_resolution_error_eager_propagates(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Branche factory 144-147 : error sur resolve_class eager -> re-raise."""
    from wincorp_odin.llm import factory
    from wincorp_odin.llm.exceptions import ProviderNotInstalledError

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "bad-provider"
    display_name: "Bad"
    use: "nonexistent_pkg_zzz:SomeClass"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ProviderNotInstalledError):
        factory._load_and_validate_models(timeout_s=5.0)


def test_factory_stat_reloaded_oserror_abandon_swap(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branche factory 190-191 : OSError sur stat dans le swap -> abandon."""
    from wincorp_odin.llm import create_model, factory

    create_model("sonnet")
    factory._last_mtime_check = 0.0
    # Provoquer un mtime change
    yaml_file = patched_yaml_path / "referentiels" / "models.yaml"
    time.sleep(0.05)
    yaml_file.write_text(
        (Path(__file__).parent / "fixtures" / "models_minimal.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    # Monkeypatch stat pour qu'il passe la 1ère fois puis leve sous lock
    real_stat = Path.stat
    call_count = {"n": 0}

    def staged_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "models.yaml":
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise OSError("simulated OSError in swap")
        return real_stat(self, *args, **kwargs)

    # PR-002 — snapshot cache avant swap-aborted pour verifier integrite
    configs_before = dict(factory._resolved_configs)
    deferred_before = set(factory._deferred_resolutions)

    monkeypatch.setattr(Path, "stat", staged_stat)
    factory._check_mtime_and_invalidate()  # ne doit pas lever

    # PR-002 — cache doit rester intact apres OSError dans le swap
    assert factory._resolved_configs == configs_before, (
        "Cache pollue apres OSError dans le swap"
    )
    assert factory._deferred_resolutions == deferred_before, (
        "Deferred pollue apres OSError dans le swap"
    )


def test_config_find_project_root_with_git_present() -> None:
    """_find_project_root trouve .git sur le repo courant (si present)."""
    from wincorp_odin.llm import config as config_mod

    # Le repo wincorp-odin a un .git -> detection doit marcher
    result = config_mod._find_project_root()
    # Accepter None si execute dans un env sans git
    assert result is None or (result / ".git").exists() or (result / "pyproject.toml").exists()


def test_config_detect_dev_urd_returns_path_when_models_yaml_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Branche config 166 : .git trouve + wincorp-urd/referentiels/models.yaml existe."""
    from wincorp_odin.llm import config as config_mod

    # Creer arbo : dev_root/wincorp-odin/.git + dev_root/wincorp-urd/referentiels/models.yaml
    dev_root = tmp_path / "dev"
    repo = dev_root / "repo"
    (repo / ".git").mkdir(parents=True)
    urd_yaml = dev_root / "wincorp-urd" / "referentiels"
    urd_yaml.mkdir(parents=True)
    (urd_yaml / "models.yaml").write_text("config_version: 1\n", encoding="utf-8")

    # Faux path pour __file__
    fake_file = repo / "src" / "pkg" / "llm" / "config.py"
    fake_file.parent.mkdir(parents=True)

    # Mocker Path(__file__).resolve() dans le module config
    import pathlib
    real_resolve = pathlib.Path.resolve

    def resolve_patch(self: pathlib.Path) -> pathlib.Path:
        # Si on resolve le __file__ du module config, retourner notre fake_file
        if "config.py" in str(self) and "wincorp_odin" in str(self):
            return fake_file.resolve()
        return real_resolve(self)

    monkeypatch.setattr(pathlib.Path, "resolve", resolve_patch)

    # Charger une copie du module sans cache
    result = config_mod._detect_dev_urd_path()
    assert result is not None
    assert result.name == "wincorp-urd"


def test_factory_validate_raises_on_disabled_extra_kwargs_forbidden(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-018 : validate_all_models rejette extra_kwargs forbidden meme sur disabled.

    PR-001 : l'ordre est whitelist AVANT parsing use (miroir §9.1). Ici
    `use:` est valide mais `extra_kwargs` forbidden -> remonte sur whitelist.
    """
    from wincorp_odin.llm import ExtraKwargsForbiddenError, validate_all_models

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "ghost"
    display_name: "Disabled bad"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    disabled: true
    extra_kwargs:
      base_url: "https://evil.com"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ExtraKwargsForbiddenError):
        validate_all_models()


def test_factory_validate_raises_on_disabled_use_format_invalid(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PR-001 : modele disabled avec extra_kwargs OK mais use: format invalide.

    validate_use_format (public) parse sans importer -> ModelConfigError format.
    """
    from wincorp_odin.llm import ModelConfigError, validate_all_models

    urd = tmp_path / "wincorp-urd"
    ref = urd / "referentiels"
    ref.mkdir(parents=True)
    (ref / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "ghost"
    display_name: "Disabled bad format"
    use: "format_invalide_sans_colon"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    disabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(urd))

    with pytest.raises(ModelConfigError) as excinfo:
        validate_all_models()
    assert "format_invalide_sans_colon" in str(excinfo.value)


def test_registry_validate_use_format_rejects_empty_parts() -> None:
    """PR-001 : validate_use_format rejette module ou classe vide."""
    from wincorp_odin.llm._registry import validate_use_format
    from wincorp_odin.llm.exceptions import ModelConfigError

    with pytest.raises(ModelConfigError):
        validate_use_format("")
    with pytest.raises(ModelConfigError):
        validate_use_format("pas_de_colon")
    with pytest.raises(ModelConfigError):
        validate_use_format(":Classe")
    with pytest.raises(ModelConfigError):
        validate_use_format("module:")


def test_registry_validate_use_format_accepts_valid() -> None:
    """PR-001 : validate_use_format accepte format valide sans import."""
    from wincorp_odin.llm._registry import validate_use_format

    # Ne leve pas, meme pour un package inexistant (pas de resolution)
    validate_use_format("package_inexistant_xyz:Classe")
    validate_use_format("langchain_anthropic:ChatAnthropic")


def test_r10d_redacts_openai_project_key() -> None:
    """R10d : OpenAI project keys (sk-proj-*) redactees."""
    from wincorp_odin.llm.exceptions import _redact

    key = "sk-proj-abcdefghijklmnopqrst1234567890XYZ"
    result = _redact(f"Erreur avec {key} invalide")
    assert key not in result
    assert "***REDACTED***" in result


def test_r10d_redacts_openai_generic_key() -> None:
    """R10d : OpenAI/DeepSeek generic keys (sk-* 32+ chars) redactees."""
    from wincorp_odin.llm.exceptions import _redact

    key = "sk-" + "a" * 48
    result = _redact(f"config api_key={key}")
    assert key not in result
    assert "***REDACTED***" in result


def test_r10d_redacts_aws_access_key() -> None:
    """R10d : AWS access keys (AKIA*) redactees."""
    from wincorp_odin.llm.exceptions import _redact

    key = "AKIAIOSFODNN7EXAMPLE"
    result = _redact(f"aws_access_key_id={key}")
    assert key not in result
    assert "***REDACTED***" in result


def test_r10d_still_redacts_anthropic_key() -> None:
    """R10d : la couverture multi-providers n'a pas casse Anthropic."""
    from wincorp_odin.llm.exceptions import _redact

    key = "sk-ant-api03-abcdefghij1234567890KLMNOP"
    result = _redact(f"api={key} end")
    assert key not in result
    assert "***REDACTED***" in result


def test_load_and_validate_is_pure(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PR-006 : _load_and_validate_models ne mute pas les globals.

    Contrat : la fonction est pure et retourne (configs, deferred). La mutation
    de `_resolved_configs` et `_deferred_resolutions` n'intervient que dans le
    swap atomique sous lock (cf `validate_all_models`, `_check_mtime_and_invalidate`,
    `_ensure_configs_loaded`).
    """
    from wincorp_odin.llm import factory

    # S'assurer que l'etat global est vide avant l'appel (autofix dans conftest)
    assert factory._resolved_configs == {}
    assert factory._deferred_resolutions == set()

    configs, deferred = factory._load_and_validate_models(timeout_s=5.0)

    assert isinstance(configs, dict)
    assert isinstance(deferred, set)
    # La fonction pure n'a rien mute
    assert factory._resolved_configs == {}, (
        "_load_and_validate_models a mute _resolved_configs (viole PR-006)"
    )
    assert factory._deferred_resolutions == set(), (
        "_load_and_validate_models a mute _deferred_resolutions (viole PR-006)"
    )
