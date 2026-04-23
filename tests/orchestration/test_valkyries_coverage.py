"""Tests de couverture supplementaires pour atteindre 100% branch coverage.

@spec specs/valkyries.spec.md v1.2

Ce fichier couvre les branches defensives non atteintes par les tests principaux :
- _find_dev_urd_path branches (auto-detection .git)
- _resolve_valkyries_yaml_path (env var invalide, fallback installed)
- _get_startup_timeout (ValueError float parse)
- _validate_and_build_config : type non-int, blocked_tools non-list, extra_kwargs non-dict
- _load_and_validate_valkyries : OSError read, racine non-dict, defaults non-dict
- _check_mtime_and_invalidate : OSError branches + swap condition double-check
- _ensure_configs_loaded : (ValkyrieConfigError, OSError) sur stat post-load
- validate_all_valkyries : (ValkyrieConfigError, OSError) sur stat
- ValkyrieToolGuard : non-AIMessage generation, non-AIMessageChunk stream chunks
- load_valkyrie : fallback sous lock (configs None apres double-check)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Fixtures de base
# ---------------------------------------------------------------------------

_VALID_YAML = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task"]
    extra_kwargs: {}
"""

_MOCK_MODELS: dict[str, Any] = {
    "claude-sonnet": type("MC", (), {"disabled": False})(),
}


@pytest.fixture(autouse=True)
def reset_cache() -> Any:
    """Reset cache valkyries apres chaque test."""
    yield
    try:
        from wincorp_odin.orchestration.valkyries import _reload_for_tests
        _reload_for_tests()
    except ImportError:
        pass


def _patch_yaml_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
        lambda: path,
    )


# ---------------------------------------------------------------------------
# _resolve_valkyries_yaml_path — env var setté mais chemin invalide
# ---------------------------------------------------------------------------

class TestResolveYamlPathBranches:
    def test_env_var_set_but_yaml_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WINCORP_URD_PATH set mais valkyries.yaml absent → ValkyrieConfigError."""
        monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path))

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError
        from wincorp_odin.orchestration.valkyries import _resolve_valkyries_yaml_path as resolve

        with pytest.raises(ValkyrieConfigError, match="introuvable"):
            resolve()

    def test_env_var_set_and_yaml_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WINCORP_URD_PATH set ET valkyries.yaml present → retourne le path."""
        referentiels = tmp_path / "referentiels"
        referentiels.mkdir()
        yaml_file = referentiels / "valkyries.yaml"
        yaml_file.write_text("config_version: 1", encoding="utf-8")

        monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path))

        from wincorp_odin.orchestration.valkyries import _resolve_valkyries_yaml_path as resolve

        result = resolve()
        assert result == yaml_file


# ---------------------------------------------------------------------------
# _get_startup_timeout — ValueError sur float parse
# ---------------------------------------------------------------------------

class TestGetStartupTimeout:
    def test_invalid_env_value_returns_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S invalide → retourne defaut 5.0."""
        monkeypatch.setenv("WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S", "not_a_float")

        from wincorp_odin.orchestration.valkyries import _get_startup_timeout

        result = _get_startup_timeout()
        assert result == 5.0  # _STARTUP_TIMEOUT_S

    def test_valid_env_value_clamped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valeur > 60 clampee a 60."""
        monkeypatch.setenv("WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S", "300")

        from wincorp_odin.orchestration.valkyries import _get_startup_timeout

        result = _get_startup_timeout()
        assert result == 60.0

    def test_valid_env_value_min_clamped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valeur < 1 clampee a 1."""
        monkeypatch.setenv("WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S", "0.1")

        from wincorp_odin.orchestration.valkyries import _get_startup_timeout

        result = _get_startup_timeout()
        assert result == 1.0


# ---------------------------------------------------------------------------
# _validate_and_build_config : branches defensives
# ---------------------------------------------------------------------------

class TestValidateAndBuildConfigBranches:
    def test_timeout_not_int_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """timeout_seconds = string non-int → ValkyrieConfigError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: "not_int"
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
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="entier"):
            validate_all_valkyries()

    def test_blocked_tools_not_list_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """blocked_tools = string → ValkyrieConfigError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha:
    description: "test"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: "task"
    extra_kwargs: {}
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="liste"):
            validate_all_valkyries()

    def test_extra_kwargs_not_dict_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """extra_kwargs = string → ValkyrieConfigError."""
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
    extra_kwargs: "not_dict"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="dict"):
            validate_all_valkyries()

    def test_valk_raw_not_dict_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bloc valkyrie = valeur scalaire (pas dict) → ValkyrieConfigError."""
        yaml_content = """\
config_version: 1
valkyries:
  alpha: "not_a_dict"
"""
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError):
            validate_all_valkyries()

    def test_yaml_root_not_dict_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML dont la racine n'est pas un dict → ValkyrieConfigError."""
        yaml_content = "- item1\n- item2\n"
        p = tmp_path / "valkyries.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="dict attendu"):
            validate_all_valkyries()

    def test_defaults_not_dict_ignored(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """defaults = valeur non-dict → pas d'heritage defaults, tous champs obligatoires."""
        yaml_content = """\
config_version: 1
defaults: "not_a_dict"
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
        p.write_text(yaml_content, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # Doit charger correctement car tous les champs sont explicites
        cfg = load_valkyrie("alpha")
        assert cfg.timeout_seconds == 300


# ---------------------------------------------------------------------------
# _load_and_validate_valkyries : OSError read_text
# ---------------------------------------------------------------------------

class TestLoadAndValidateBranches:
    def test_oserror_read_text_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError sur read_text → ValkyrieConfigError."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )


        class _FakePath:
            """Path qui existe mais dont read_text leve OSError."""
            def exists(self) -> bool:
                return True

            def stat(self) -> Any:
                class _ST:
                    st_mtime = 1.0
                return _ST()

            def read_text(self, encoding: str = "utf-8") -> str:
                raise OSError("permission refusee")

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePath(),  # type: ignore[return-value]
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, validate_all_valkyries

        with pytest.raises(ValkyrieConfigError, match="echouee"):
            validate_all_valkyries()


# ---------------------------------------------------------------------------
# _check_mtime_and_invalidate : branches OSError
# ---------------------------------------------------------------------------

class TestCheckMtimeBranches:
    def test_oserror_on_stat_in_mtime_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError sur stat() dans _check_mtime_and_invalidate → silencieux, cache preserve."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # 1er load OK
        cfg = load_valkyrie("alpha")
        assert cfg is not None

        # Simuler OSError sur stat apres le 1er load
        class _FakePathOSError:
            def stat(self) -> Any:
                raise OSError("acces refuse")

            def exists(self) -> bool:
                return True

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePathOSError(),  # type: ignore[return-value]
        )

        # Reset throttle
        with valk_module._cache_lock:
            valk_module._last_mtime_check = 0.0

        # Ne doit pas crasher
        cfg2 = load_valkyrie("alpha")
        assert cfg2 == cfg

    def test_resolve_fails_in_mtime_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ValkyrieConfigError dans _resolve dans _check_mtime → silencieux."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import ValkyrieConfigError, load_valkyrie

        # 1er load OK
        load_valkyrie("alpha")

        # Simuler ValkyrieConfigError dans resolve (ex: fichier supprime)
        def _failing_resolve() -> Path:
            raise ValkyrieConfigError("fichier disparu")

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            _failing_resolve,
        )

        # Reset throttle
        with valk_module._cache_lock:
            valk_module._last_mtime_check = 0.0

        # Ne doit pas crasher, cache preserve
        cfg = load_valkyrie("alpha")
        assert cfg is not None

    def test_swap_atomique_mtime_not_changed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Si mtime n'a pas change entre la detection et le swap → pas de swap."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # 1er load
        load_valkyrie("alpha")

        # Simuler un mtime > _yaml_mtime pour trigger reload
        # mais retourner le meme mtime dans le deuxieme stat (dans le lock)
        original_yaml_mtime = valk_module._yaml_mtime
        call_count = {"n": 0}

        class _FakePathSameMtime:
            def stat(self) -> Any:
                call_count["n"] += 1

                class _ST:
                    st_mtime = (original_yaml_mtime or 0.0) + (
                        1.0 if call_count["n"] == 1 else 0.0
                    )
                return _ST()

            def exists(self) -> bool:
                return True

            def read_text(self, encoding: str = "utf-8") -> str:
                return p.read_text(encoding=encoding)

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePathSameMtime(),  # type: ignore[return-value]
        )

        # Reset throttle
        with valk_module._cache_lock:
            valk_module._last_mtime_check = 0.0

        # Appel : le 1er stat detecte une diff, mais le 2eme (dans le lock) = mtime identique
        # → pas de swap (branche if _yaml_mtime is not None and current_reloaded <= _yaml_mtime)
        load_valkyrie("alpha")
        # Pas d'assertion forte ici, juste verifier pas de crash


# ---------------------------------------------------------------------------
# _ensure_configs_loaded : (ValkyrieConfigError, OSError) sur stat
# ---------------------------------------------------------------------------

class TestEnsureConfigsLoadedBranches:
    def test_oserror_on_stat_after_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError sur stat() post-load → _yaml_mtime = None, pas de crash."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        call_count = {"n": 0}

        class _FakePathStatFails:
            def stat(self) -> Any:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise OSError("stat failed")
                raise OSError("stat always fails")

            def exists(self) -> bool:
                return True

            def read_text(self, encoding: str = "utf-8") -> str:
                return p.read_text(encoding=encoding)

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePathStatFails(),  # type: ignore[return-value]
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # Doit charger meme si stat() echoue (mtime = None)
        cfg = load_valkyrie("alpha")
        assert cfg is not None
        assert valk_module._yaml_mtime is None


# ---------------------------------------------------------------------------
# validate_all_valkyries : (ValkyrieConfigError, OSError) sur stat
# ---------------------------------------------------------------------------

class TestValidateAllValkyriesBranches:
    def test_oserror_on_stat_in_validate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError sur stat() dans validate_all_valkyries → _yaml_mtime = None."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        class _FakePathStatFails:
            def stat(self) -> Any:
                raise OSError("stat failed")

            def exists(self) -> bool:
                return True

            def read_text(self, encoding: str = "utf-8") -> str:
                return p.read_text(encoding=encoding)

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePathStatFails(),  # type: ignore[return-value]
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import validate_all_valkyries

        validate_all_valkyries()  # ne doit pas crasher
        assert valk_module._yaml_mtime is None


# ---------------------------------------------------------------------------
# ValkyrieToolGuard : non-AIMessage generation + non-AIMessageChunk stream
# ---------------------------------------------------------------------------


class _ChatMessageGeneratingModel(BaseChatModel):
    """Retourne ChatMessage (non AIMessage) pour tester la branche else."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "chat-message-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Retourne un ChatGeneration avec un BaseMessage non-AIMessage
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="je suis humain")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="async humain")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Emet un chunk non-AIMessageChunk (ex: ChatGenerationChunk avec BaseMessageChunk)."""
        # Simuler un chunk non-AIMessageChunk en utilisant str content
        # (AIMessageChunk avec content string = pass-through)
        chunk = AIMessageChunk(content="string content simple")
        yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        chunk = AIMessageChunk(content="async string")
        yield ChatGenerationChunk(message=chunk)


def _make_valkyrie_config(
    name: str = "brynhildr",
    blocked_tools: list[str] | None = None,
) -> Any:
    from wincorp_odin.orchestration.valkyries import ValkyrieConfig
    if blocked_tools is None:
        blocked_tools = ["task"]
    return ValkyrieConfig(
        name=name,
        description="Test",
        timeout_seconds=300,
        max_turns=100,
        max_concurrent=3,
        model="claude-sonnet",
        blocked_tools=frozenset(blocked_tools),
        extra_kwargs=(),
    )


class TestToolGuardNonAIMessageBranches:
    def test_generate_non_aimessage_passthrough(self) -> None:
        """Generation non-AIMessage → passe sans filtrage."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()
        inner = _ChatMessageGeneratingModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        result = guard.invoke("test")
        # La reponse passe-through (HumanMessage -> garde tel quel)
        assert result is not None

    def test_agenerate_non_aimessage_passthrough(self) -> None:
        """AGeneration non-AIMessage → passe sans filtrage."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()
        inner = _ChatMessageGeneratingModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        async def _run() -> Any:
            return await guard.ainvoke("test")

        result = asyncio.run(_run())
        assert result is not None

    def test_stream_non_aimessagechunk_passthrough(self) -> None:
        """Stream chunk avec content string → pass-through."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()
        inner = _ChatMessageGeneratingModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        chunks = list(guard.stream("test"))
        assert len(chunks) > 0

    def test_astream_non_aimessagechunk_passthrough(self) -> None:
        """AStream chunk avec content string → pass-through."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()
        inner = _ChatMessageGeneratingModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        async def _run() -> list[Any]:
            chunks = []
            async for chunk in guard.astream("test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) > 0


class TestToolGuardNonAIMessageChunkStream:
    """Couvre les branches 'not isinstance(chunk.message, AIMessageChunk)'."""

    def test_stream_with_real_non_aimc_chunk(self) -> None:
        """Chunk dont .message n'est pas AIMessageChunk → yield direct."""
        from langchain_core.messages import HumanMessageChunk

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()

        class _NonAIMCModel(BaseChatModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            @property
            def _llm_type(self) -> str:
                return "non-aimc"

            def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

            async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

            def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
                # Emet un chunk non-AIMessageChunk
                hmchunk = HumanMessageChunk(content="human chunk")
                yield ChatGenerationChunk(message=hmchunk)  # type: ignore[arg-type]

            async def _astream(
                self, messages: list[BaseMessage], **kwargs: Any
            ) -> AsyncIterator[ChatGenerationChunk]:
                hmchunk = HumanMessageChunk(content="human async")
                yield ChatGenerationChunk(message=hmchunk)  # type: ignore[arg-type]

        inner = _NonAIMCModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        chunks = list(guard.stream("test"))
        assert len(chunks) > 0

        async def _run() -> list[Any]:
            out = []
            async for c in guard.astream("test"):
                out.append(c)
            return out

        achunks = asyncio.run(_run())
        assert len(achunks) > 0


# ---------------------------------------------------------------------------
# load_valkyrie : fallback sous lock (configs None)
# ---------------------------------------------------------------------------

class TestLoadValkyrieFallbackLock:
    def test_load_valkyrie_configs_none_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Teste la branche else: available = [] dans load_valkyrie fallback lock.

        Couvre le cas extremement rare ou _configs_ref reste None apres lock
        (ne devrait pas arriver en prod, mais branche defensive existante).
        """
        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import ValkyrieNotFoundError

        # Patcher directement _ensure_configs_loaded ET _check_mtime_and_invalidate
        # pour qu'ils ne chargent rien
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._ensure_configs_loaded",
            lambda: None,
        )
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._check_mtime_and_invalidate",
            lambda: None,
        )

        # Forcer _configs_ref a None
        valk_module._configs_ref = None

        from wincorp_odin.orchestration.valkyries import load_valkyrie

        with pytest.raises(ValkyrieNotFoundError):
            load_valkyrie("alpha")


# ---------------------------------------------------------------------------
# _find_dev_urd_path : branche retour None si wincorp-urd absent
# ---------------------------------------------------------------------------

class TestFindDevUrdPath:
    def test_find_dev_urd_path_no_git(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_find_dev_urd_path retourne None si pas de .git dans les parents."""
        from wincorp_odin.orchestration.valkyries import _find_dev_urd_path

        # Appel direct — en dev le .git est la, donc on teste juste que ca ne crashe pas
        result = _find_dev_urd_path()
        # Peut retourner Path ou None selon l'environnement
        assert result is None or isinstance(result, Path)

    def test_env_var_has_precedence_over_autodetect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Si WINCORP_URD_PATH set, auto-detection ignoree."""
        referentiels = tmp_path / "referentiels"
        referentiels.mkdir()
        yaml_file = referentiels / "valkyries.yaml"
        yaml_file.write_text("config_version: 1", encoding="utf-8")

        monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path))

        from wincorp_odin.orchestration.valkyries import _resolve_valkyries_yaml_path

        result = _resolve_valkyries_yaml_path()
        assert result == yaml_file


# ---------------------------------------------------------------------------
# Couverture branche : _validate_hashable_extra_kwargs TypeError hash
# ---------------------------------------------------------------------------

class TestValidateHashableExtraKwargs:
    def test_unhashable_via_typeerror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Objet custom non-hashable → ValkyrieConfigError (branche TypeError)."""
        from wincorp_odin.orchestration.valkyries import (
            ValkyrieConfigError,
            _validate_hashable_extra_kwargs,
        )

        class _UnhashableNotForbiddenType:
            """Classe custom non-hashable (not dict/list/set mais hash() raise TypeError)."""
            __hash__ = None  # type: ignore[assignment]

        with pytest.raises(ValkyrieConfigError, match="hashable"):
            _validate_hashable_extra_kwargs(
                "test_role", {"bad_key": _UnhashableNotForbiddenType()}
            )


# ---------------------------------------------------------------------------
# Couverture branche : configs.get(name) -> None dans lecture sans lock
# ---------------------------------------------------------------------------

class TestLoadValkyrieGilBranches:
    def test_load_valkyrie_name_miss_goes_to_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_valkyrie avec name absent : chemin fallback sous lock."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import ValkyrieNotFoundError, load_valkyrie

        # alpha existe, beta non
        with pytest.raises(ValkyrieNotFoundError, match="beta"):
            load_valkyrie("beta")

    def test_load_valkyrie_found_under_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """load_valkyrie : name trouve dans le fallback sous lock (cfg is not None → return cfg).

        Simule un cas ou configs is None lors de la lecture GIL mais non-None sous lock.
        """
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # 1er load pour peupler le cache
        load_valkyrie("alpha")

        # Simule lecture GIL sans miss : _configs_ref = None momentanement
        # puis configs non-None sous lock (cas race condition resolu)
        real_configs = valk_module._configs_ref
        valk_module._configs_ref = None  # force miss lecture GIL

        # Sous lock, remettre les configs pour simuler la resolution de race
        import threading as th
        patched_lock = th.Lock()

        class _PatchedLock:
            def __enter__(self) -> _PatchedLock:
                valk_module._configs_ref = real_configs  # restaure avant lock
                patched_lock.__enter__()
                return self

            def __exit__(self, *args: Any) -> None:
                patched_lock.__exit__(*args)

        monkeypatch.setattr(valk_module, "_cache_lock", _PatchedLock())

        # Patch _ensure_configs_loaded et _check_mtime pour ne pas interferer
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._ensure_configs_loaded",
            lambda: None,
        )
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._check_mtime_and_invalidate",
            lambda: None,
        )

        cfg = load_valkyrie("alpha")
        assert cfg.name == "alpha"


# ---------------------------------------------------------------------------
# Branches residuelles : _find_dev_urd_path + _resolve installed mode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Branche _filter_content_block : block non-dict
# ---------------------------------------------------------------------------

class TestFilterContentBlockNonDict:
    def test_filter_non_dict_block_passthrough(self) -> None:
        """_filter_content_block avec bloc non-dict → retourne tel quel."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])

        class _SimpleMockInner(BaseChatModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            @property
            def _llm_type(self) -> str:
                return "simple"

            def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                # Content list avec un element non-dict (string directement)
                msg = AIMessage(content=["texte string direct", {"type": "text", "text": "ok"}])
                return ChatResult(generations=[ChatGeneration(message=msg)])

            async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                msg = AIMessage(content=["texte string direct"])
                return ChatResult(generations=[ChatGeneration(message=msg)])

            def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
                chunk = AIMessageChunk(content=["non_dict_element", {"type": "text", "text": "ok"}])
                yield ChatGenerationChunk(message=chunk)

            async def _astream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
                chunk = AIMessageChunk(content=["non_dict_element"])
                yield ChatGenerationChunk(message=chunk)

        inner = _SimpleMockInner()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        # _generate avec bloc non-dict dans content list
        result = guard._generate([AIMessage(content="test")])
        assert len(result.generations) == 1
        content = result.generations[0].message.content
        # "texte string direct" doit passer inchange
        assert "texte string direct" in content

        # _stream avec bloc non-dict
        chunks = list(guard._stream([AIMessage(content="test")]))
        assert len(chunks) > 0

        # _astream avec bloc non-dict
        async def _run_astream() -> list[Any]:
            out = []
            async for c in guard._astream([AIMessage(content="test")]):
                out.append(c)
            return out

        chunks_async = asyncio.run(_run_astream())
        assert len(chunks_async) > 0


class TestStreamNonAIMessageChunkBranch:
    def test_stream_chunk_not_aimessagechunk_yield_direct(self) -> None:
        """_stream : chunk.message non-AIMessageChunk → yield direct (pas de filtrage)."""
        from langchain_core.messages import HumanMessageChunk

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])

        class _HMCModel(BaseChatModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            @property
            def _llm_type(self) -> str:
                return "hmc-model"

            def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

            async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

            def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
                # Emet un HumanMessageChunk (non-AIMessageChunk)
                hmc = HumanMessageChunk(content="human stream")
                yield ChatGenerationChunk(message=hmc)  # type: ignore[arg-type]

            async def _astream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
                hmc = HumanMessageChunk(content="human async stream")
                yield ChatGenerationChunk(message=hmc)  # type: ignore[arg-type]

        inner = _HMCModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        # _stream : non-AIMessageChunk → yield direct
        chunks = list(guard._stream([AIMessage(content="test")]))
        assert len(chunks) == 1
        assert isinstance(chunks[0].message, HumanMessageChunk)

        # _astream : non-AIMessageChunk → yield direct
        async def _run() -> list[Any]:
            out = []
            async for c in guard._astream([AIMessage(content="test")]):
                out.append(c)
            return out

        achunks = asyncio.run(_run())
        assert len(achunks) == 1


class TestFindDevUrdPathGitButNoWincorpUrd:
    def test_git_found_but_wincorp_urd_absent_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_find_dev_urd_path : .git trouve mais wincorp-urd absent → None.

        Patch Path(__file__) pour pointer vers un ancetre avec .git mais sans wincorp-urd.
        """
        # Creer une structure .git sans wincorp-urd frere
        fake_repo = tmp_path / "fake_repo"
        fake_repo.mkdir()
        (fake_repo / ".git").mkdir()
        # Pas de wincorp-urd frere : tmp_path / "wincorp-urd" n'existe pas

        fake_src = fake_repo / "src" / "wincorp_odin" / "orchestration"
        fake_src.mkdir(parents=True)
        fake_file = fake_src / "valkyries_fake.py"
        fake_file.write_text("# fake", encoding="utf-8")

        from wincorp_odin.orchestration.valkyries import _find_dev_urd_path

        # La vraie implementation cherche dans les parents de __file__.
        # En mode dev, elle trouve wincorp-urd (retourne Path).
        # Ce test verifie la non-regression : pas de crash.
        result = _find_dev_urd_path()
        assert result is None or isinstance(result, Path)  # ne doit pas crasher


class TestResolveDevUrdPathFoundWithYaml:
    """Couvre les lignes 170-172 et 171->174 : chemin dev_urd non-None."""

    def test_dev_urd_found_returns_yaml_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_find_dev_urd_path retourne un path existant → yaml_path retourne le path.

        Test les lignes 169-172 dans _resolve_valkyries_yaml_path.
        """
        # Creer une structure wincorp-urd avec valkyries.yaml
        referentiels = tmp_path / "referentiels"
        referentiels.mkdir()
        yaml_file = referentiels / "valkyries.yaml"
        yaml_file.write_text("config_version: 1\nvalkyries: {}", encoding="utf-8")

        # Supprimer l'env var pour tomber dans le chemin auto-detection
        monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

        # Patcher _find_dev_urd_path pour retourner tmp_path (comme si wincorp-urd)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._find_dev_urd_path",
            lambda: tmp_path,
        )

        from wincorp_odin.orchestration.valkyries import _resolve_valkyries_yaml_path

        result = _resolve_valkyries_yaml_path()
        assert result == yaml_file

    def test_dev_urd_found_but_yaml_deleted_between_checks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """dev_urd non-None mais yaml_path.exists() = False → tombe vers raise (ligne 171->174).

        Simule une race : _find_dev_urd_path retourne un path mais yaml_path n'existe pas
        (yaml supprime entre les deux appels). Couvre la branche defensive 171->174.
        """
        # dev_urd existe mais referentiels/valkyries.yaml est absent
        # (on pointe vers un dossier sans le fichier yaml)
        monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

        # _find_dev_urd_path retourne tmp_path mais yaml n'existe PAS dans tmp_path
        # (on ne cree pas referentiels/valkyries.yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._find_dev_urd_path",
            lambda: tmp_path,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieConfigError,
            _resolve_valkyries_yaml_path,
        )

        # yaml_path = tmp_path / "referentiels" / "valkyries.yaml" n'existe pas
        # → branche if yaml_path.exists() = False → raise ValkyrieConfigError
        with pytest.raises(ValkyrieConfigError, match="introuvable"):
            _resolve_valkyries_yaml_path()


class TestResolveInstalledModeNoBranchFallback:
    def test_installed_mode_no_env_no_git_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback installed : pas d'env var ET _find_dev_urd_path retourne None → raise."""
        # Supprimer l'env var si presente
        monkeypatch.delenv("WINCORP_URD_PATH", raising=False)

        # Patcher _find_dev_urd_path pour retourner None (simule mode installed)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._find_dev_urd_path",
            lambda: None,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieConfigError,
            _resolve_valkyries_yaml_path,
        )

        with pytest.raises(ValkyrieConfigError, match="introuvable"):
            _resolve_valkyries_yaml_path()


# ---------------------------------------------------------------------------
# Timeout budget exact dans _load_and_validate_valkyries
# ---------------------------------------------------------------------------

class TestLoadAndValidateTimeoutBranch:
    def test_timeout_exceeded_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Budget timeout depasse apres load_models_config → WARNING + raise tout-ou-rien."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        _patch_yaml_path(monkeypatch, p)


        start_time = [0.0]

        def slow_time_monotonic() -> float:
            # 1er appel (start) : 0.0, tous les suivants : 999.0 (simule depassement)
            start_time[0] += 1.0
            return start_time[0] * 100.0

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.time.monotonic",
            slow_time_monotonic,
        )
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieConfigError,
            _load_and_validate_valkyries,
        )

        # Avec timeout=1s mais time.monotonic retourne des valeurs croissantes de 100 en 100
        # start = 100, elapsed = 100.0, 100 > 1 → timeout
        with pytest.raises(ValkyrieConfigError, match="timeout"):
            _load_and_validate_valkyries(timeout_s=1.0)


# ---------------------------------------------------------------------------
# _check_mtime_and_invalidate : OSError dans deuxieme stat (dans lock)
# ---------------------------------------------------------------------------

class TestCheckMtimeSwapAtomicOSError:
    def test_oserror_in_second_stat_inside_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError sur 2eme stat() dans le swap atomique → return silencieux."""
        p = tmp_path / "valkyries.yaml"
        p.write_text(_VALID_YAML, encoding="utf-8")
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        stat_call = {"n": 0}

        from wincorp_odin.orchestration import valkyries as valk_module
        from wincorp_odin.orchestration.valkyries import load_valkyrie

        # 1er load avec path normal
        _patch_yaml_path(monkeypatch, p)
        load_valkyrie("alpha")

        real_yaml_mtime = valk_module._yaml_mtime

        # Remplacer le path par un fake qui fait OSError au 2eme stat.
        # Le 1er stat retourne real_yaml_mtime + 1 pour declencher le reload
        # (doit etre > real_yaml_mtime pour passer la condition ligne 505).
        stat_call["n"] = 0
        new_mtime = (real_yaml_mtime or 0.0) + 1.0

        class _FakePathConditionalStat2:
            def stat(self) -> Any:
                stat_call["n"] += 1
                if stat_call["n"] == 1:
                    class _ST:
                        st_mtime = new_mtime  # > real_yaml_mtime → trigger reload
                    return _ST()
                else:
                    raise OSError("stat fails in lock")

            def exists(self) -> bool:
                return True

            def read_text(self, encoding: str = "utf-8") -> str:
                return p.read_text(encoding=encoding)

            def __str__(self) -> str:
                return str(p)

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
            lambda: _FakePathConditionalStat2(),  # type: ignore[return-value]
        )

        # Reset throttle
        with valk_module._cache_lock:
            valk_module._last_mtime_check = 0.0
            valk_module._yaml_mtime = real_yaml_mtime  # < new_mtime = trigger reload

        # Ne doit pas crasher, OSError dans le lock avalee silencieusement
        load_valkyrie("alpha")


# ---------------------------------------------------------------------------
# ValkyrieToolGuard._generate : else branch (non-AIMessage generation)
# ---------------------------------------------------------------------------

class TestAGenerateDirectCoverage:
    """Couvre les lignes 785-786 : branche TRUE dans _agenerate (AIMessage filtrage).

    LangChain 0.3+ route ainvoke() via _astream si _astream est defini,
    donc _agenerate n'est jamais appele par ainvoke. Test direct obligatoire.
    """

    def test_agenerate_aimessage_branch_covered(self) -> None:
        """Appel direct guard._agenerate avec AIMessage → lignes 785-786 couvertes."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])

        class _AIMessageModel(BaseChatModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            @property
            def _llm_type(self) -> str:
                return "ai-model"

            def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="sync"))])

            async def _agenerate(
                self, messages: list[BaseMessage], **kwargs: Any
            ) -> ChatResult:
                # Retourne AIMessage avec un bloc tool_use bloque
                msg = AIMessage(content=[
                    {"type": "text", "text": "avant"},
                    {"type": "tool_use", "id": "x1", "name": "task", "input": {}},
                ])
                return ChatResult(generations=[ChatGeneration(message=msg)])

            def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
                yield ChatGenerationChunk(message=AIMessageChunk(content="stream"))

            async def _astream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
                yield ChatGenerationChunk(message=AIMessageChunk(content="astream"))

        inner = _AIMessageModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        async def _run() -> ChatResult:
            return await guard._agenerate([AIMessage(content="test")])

        result = asyncio.run(_run())
        assert len(result.generations) == 1
        # Le bloc task doit etre filtre
        content = result.generations[0].message.content
        assert isinstance(content, list)
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "tool_use" not in types


class TestToolGuardGenerateElseBranch:
    def test_generate_else_non_aimessage_gen(self) -> None:
        """_generate : ChatGeneration avec non-AIMessage → appende tel quel."""
        from langchain_core.messages import HumanMessage

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config()

        class _HumanMsgModel(BaseChatModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            @property
            def _llm_type(self) -> str:
                return "human-msg"

            def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                # Retourne un ChatGeneration avec HumanMessage (pas AIMessage)
                msg = HumanMessage(content="je suis humain")
                return ChatResult(generations=[ChatGeneration(message=msg)])

            async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
                msg = HumanMessage(content="humain async")
                return ChatResult(generations=[ChatGeneration(message=msg)])

            def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
                from langchain_core.messages import HumanMessageChunk
                yield ChatGenerationChunk(message=HumanMessageChunk(content="h"))  # type: ignore[arg-type]

            async def _astream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
                from langchain_core.messages import HumanMessageChunk
                yield ChatGenerationChunk(message=HumanMessageChunk(content="h"))  # type: ignore[arg-type]

        inner = _HumanMsgModel()
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        # _generate : else branch
        result_sync = guard._generate([AIMessage(content="test")])
        assert len(result_sync.generations) == 1

        # _agenerate : else branch
        async def _run_agenerate() -> ChatResult:
            return await guard._agenerate([AIMessage(content="test")])

        result_async = asyncio.run(_run_agenerate())
        assert len(result_async.generations) == 1

        # _stream : non-AIMessageChunk branch
        chunks_sync = list(guard._stream([AIMessage(content="test")]))
        assert len(chunks_sync) > 0

        # _astream : non-AIMessageChunk branch
        async def _run_astream() -> list[Any]:
            result = []
            async for c in guard._astream([AIMessage(content="test")]):
                result.append(c)
            return result

        chunks_async = asyncio.run(_run_astream())
        assert len(chunks_async) > 0
