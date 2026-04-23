"""Tests TDD — create_valkyrie_chat factory (R18).

@spec specs/valkyries.spec.md v1.2

Tests de composition loader + LLM + ValkyrieToolGuard.
Zero appel reseau — mocks.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Modele de test reutilisable (evite MagicMock qui echoue avec Pydantic/LangChain)
# ---------------------------------------------------------------------------


class _SimpleMockModel(BaseChatModel):
    """BaseChatModel minimal pour tests factory sans appel reseau."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "simple-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content="mock response")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content="mock response async")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        chunk = AIMessageChunk(content="mock")
        yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        chunk = AIMessageChunk(content="mock")
        yield ChatGenerationChunk(message=chunk)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_YAML_WITH_DEFAULTS = """\
config_version: 1
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
"""


@pytest.fixture(autouse=True)
def reset_valkyries_cache() -> Any:
    """Reset cache valkyries apres chaque test."""
    yield
    try:
        from wincorp_odin.orchestration.valkyries import _reload_for_tests
        _reload_for_tests()
    except ImportError:
        pass


@pytest.fixture
def tmp_valkyries_yaml(tmp_path: Path) -> Path:
    """Ecrit YAML temporaire."""
    p = tmp_path / "valkyries.yaml"
    p.write_text(_YAML_WITH_DEFAULTS, encoding="utf-8")
    return p


def _patch_yaml_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        "wincorp_odin.orchestration.valkyries._resolve_valkyries_yaml_path",
        lambda: path,
    )


_MOCK_MODELS: dict[str, Any] = {
    "claude-sonnet": type("MC", (), {"disabled": False})(),
    "claude-haiku": type("MC", (), {"disabled": False})(),
}


# ---------------------------------------------------------------------------
# R18 — create_valkyrie_chat compose loader + LLM + guard
# ---------------------------------------------------------------------------

class TestR18FactoryComposesGuard:
    def test_r18_factory_composes_guard(
        self,
        tmp_valkyries_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """create_valkyrie_chat retourne ValkyrieToolGuard wrappant le modele."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        # Mock create_model pour eviter instanciation reelle
        mock_inner_model = _SimpleMockModel()
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.create_model",
            lambda name, **kwargs: mock_inner_model,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieToolGuard,
            create_valkyrie_chat,
        )

        result = create_valkyrie_chat("brynhildr")

        # Retourne un ValkyrieToolGuard
        assert isinstance(result, ValkyrieToolGuard)

        # Wrappant le modele interne
        assert result.wrapped is mock_inner_model

        # Config brynhildr chargee
        assert result.config.name == "brynhildr"
        assert "task" in result.config.blocked_tools
        assert "shell" in result.config.blocked_tools

    def test_r18_factory_not_found_raises(
        self,
        tmp_valkyries_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """create_valkyrie_chat sur role inexistant → ValkyrieNotFoundError."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        from wincorp_odin.orchestration.valkyries import (
            ValkyrieNotFoundError,
            create_valkyrie_chat,
        )

        with pytest.raises(ValkyrieNotFoundError):
            create_valkyrie_chat("role_inexistant")

    def test_r18_factory_returns_basechatmodel(
        self,
        tmp_valkyries_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Le retour de create_valkyrie_chat est un BaseChatModel."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.create_model",
            lambda name, **kwargs: _SimpleMockModel(),
        )

        from wincorp_odin.orchestration.valkyries import create_valkyrie_chat

        result = create_valkyrie_chat("brynhildr")
        assert isinstance(result, BaseChatModel)

    def test_r18_factory_no_cache_different_instances(
        self,
        tmp_valkyries_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """2 appels crees = 2 instances differentes (pas de cache v1.0)."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )

        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.create_model",
            lambda name, **kwargs: _SimpleMockModel(),
        )

        from wincorp_odin.orchestration.valkyries import create_valkyrie_chat

        guard1 = create_valkyrie_chat("brynhildr")
        guard2 = create_valkyrie_chat("brynhildr")

        # Pas de cache v1.0 : objets differents
        assert guard1 is not guard2

    def test_r18_valkyrie_llm_type(
        self,
        tmp_valkyries_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ValkyrieToolGuard._llm_type == 'wincorp-valkyrie-guard'."""
        _patch_yaml_path(monkeypatch, tmp_valkyries_yaml)
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.load_models_config",
            lambda: _MOCK_MODELS,
        )
        monkeypatch.setattr(
            "wincorp_odin.orchestration.valkyries.create_model",
            lambda name, **kwargs: _SimpleMockModel(),
        )

        from wincorp_odin.orchestration.valkyries import create_valkyrie_chat

        result = create_valkyrie_chat("brynhildr")
        assert result._llm_type == "wincorp-valkyrie-guard"
