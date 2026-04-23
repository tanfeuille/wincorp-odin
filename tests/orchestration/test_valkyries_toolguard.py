"""Tests TDD — ValkyrieToolGuard middleware (R15-R17, EC10-EC12).

@spec specs/valkyries.spec.md v1.2

Tests enforcement reel des blocked_tools via middleware LangChain.
Zero appel reseau — mocks BaseChatModel.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Fixture mock_chat_model
# ---------------------------------------------------------------------------

class MockChatModel(BaseChatModel):
    """BaseChatModel de test retournant AIMessage parametrable."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response_content: list[Any] | str = ""

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content=self.response_content)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content=self.response_content)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Emet content comme chunks — si list, emet bloc par bloc."""
        content = self.response_content
        if isinstance(content, str):
            chunk = AIMessageChunk(content=content)
            yield ChatGenerationChunk(message=chunk)
        else:
            for block in content:
                chunk = AIMessageChunk(content=[block])
                yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Version async de _stream."""
        content = self.response_content
        if isinstance(content, str):
            chunk = AIMessageChunk(content=content)
            yield ChatGenerationChunk(message=chunk)
        else:
            for block in content:
                chunk = AIMessageChunk(content=[block])
                yield ChatGenerationChunk(message=chunk)


@pytest.fixture
def mock_chat_model() -> MockChatModel:
    """Retourne un MockChatModel configurable."""
    return MockChatModel(response_content="")


@pytest.fixture(autouse=True)
def reset_valkyries_cache() -> Any:
    """Reset cache valkyries apres chaque test."""
    yield
    try:
        from wincorp_odin.orchestration.valkyries import _reload_for_tests
        _reload_for_tests()
    except ImportError:
        pass


def _make_valkyrie_config(
    name: str = "brynhildr",
    blocked_tools: list[str] | None = None,
) -> Any:
    """Construit un ValkyrieConfig minimal pour les tests ToolGuard."""
    from wincorp_odin.orchestration.valkyries import ValkyrieConfig
    if blocked_tools is None:
        blocked_tools = ["task", "shell"]
    return ValkyrieConfig(
        name=name,
        description="Test valkyrie",
        timeout_seconds=300,
        max_turns=100,
        max_concurrent=3,
        model="claude-sonnet",
        blocked_tools=frozenset(blocked_tools),
        extra_kwargs=(),
    )


# ---------------------------------------------------------------------------
# R15 — Filtre tool_use bloque
# ---------------------------------------------------------------------------

class TestR15ToolGuardFiltersBlocked:
    def test_r15_toolguard_filters_blocked(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ToolGuard filtre tool_use dont name in blocked_tools + WARNING."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task", "shell"])
        inner = MockChatModel(response_content=[
            {"type": "text", "text": "Je vais appeler task"},
            {"type": "tool_use", "id": "abc", "name": "task", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            result = guard.invoke("test")

        assert isinstance(result, AIMessage)
        content = result.content
        assert isinstance(content, list)

        # bloc tool_use "task" remplace par text synthetique
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "tool_use" not in types

        # texte synthetique present
        texts = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        assert "task" in texts
        assert "rejete" in texts or "rejeté" in texts

        # WARNING loggue
        assert any("tool_blocked" in r.message or "task" in r.message
                   for r in caplog.records)

    def test_r15_toolguard_warning_structured(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """WARNING contient role= et tool= pour log structure."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(name="sigrun", blocked_tools=["shell"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "x1", "name": "shell", "input": {"cmd": "ls"}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            guard.invoke("test")

        log_messages = " ".join(r.message for r in caplog.records)
        assert "sigrun" in log_messages
        assert "shell" in log_messages


# ---------------------------------------------------------------------------
# R16 — Pass-through tool_use autorise
# ---------------------------------------------------------------------------

class TestR16ToolGuardPassthrough:
    def test_r16_toolguard_passthrough_allowed(self) -> None:
        """tool_use dont name NOT in blocked_tools → pass-through integrale."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # "read" est dans la whitelist mais pas dans blocked_tools de cette config
        inner = MockChatModel(response_content=[
            {"type": "text", "text": "je lis"},
            {"type": "tool_use", "id": "r1", "name": "read", "input": {"path": "/tmp/f"}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)
        result = guard.invoke("test")

        assert isinstance(result, AIMessage)
        content = result.content
        assert isinstance(content, list)

        # Le bloc tool_use "read" est preserve
        tool_use_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "read"

    def test_r16_all_allowed_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Si aucun bloc filtre → aucun WARNING valkyrie_tool_blocked."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "r2", "name": "read", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            guard.invoke("test")

        # Aucun WARNING tool_blocked
        assert not any("tool_blocked" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# R17 — Streaming filtre les blocs bloques
# ---------------------------------------------------------------------------

class TestR17ToolGuardStreamFilters:
    def test_r17_toolguard_stream_filters(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_stream : tool_use bloque dans chunk → remplace par text + WARNING."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "t1", "name": "task", "input": {}},
            {"type": "text", "text": "fin"},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            chunks = list(guard.stream("test"))

        assert len(chunks) > 0
        # Reconstituer le contenu
        all_content: list[Any] = []
        for chunk in chunks:
            c = chunk.content
            if isinstance(c, list):
                all_content.extend(c)
            elif isinstance(c, str) and c:
                all_content.append({"type": "text", "text": c})

        # Aucun tool_use "task" dans le stream
        tool_uses = [b for b in all_content if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert all(b.get("name") != "task" for b in tool_uses)

        # WARNING loggue
        assert any("task" in r.message or "tool_blocked" in r.message for r in caplog.records)

    def test_r17_astream_filters(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_astream : tool_use bloque remplace par text + WARNING."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["shell"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "s1", "name": "shell", "input": {"cmd": "ls"}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        async def _run() -> list[Any]:
            chunks = []
            with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
                async for chunk in guard.astream("test"):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) > 0
        # WARNING loggue
        assert any("shell" in r.message or "tool_blocked" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# EC10 — content = str (pas de tool_use possible)
# ---------------------------------------------------------------------------

class TestEc10ContentStr:
    def test_ec10_toolguard_content_str(self) -> None:
        """AIMessage.content = str → pass-through integral, pas de filtrage."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task", "shell"])
        inner = MockChatModel(response_content="Bonjour, je suis un assistant.")
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        result = guard.invoke("test")
        assert isinstance(result, AIMessage)
        assert result.content == "Bonjour, je suis un assistant."


# ---------------------------------------------------------------------------
# EC11 — bloc tool_use malforme (name absent)
# ---------------------------------------------------------------------------

class TestEc11ToolGuardMalformedBlock:
    def test_ec11_toolguard_malformed_block(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bloc tool_use sans name → remplace par text malforme + WARNING."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "bad1"},  # name absent
            {"type": "text", "text": "ok"},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            result = guard.invoke("test")

        assert isinstance(result, AIMessage)
        content = result.content
        assert isinstance(content, list)

        # Bloc malforme remplace par text
        texts = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "malforme" in texts or "malformé" in texts or "tool_use" in texts

        # WARNING loggue
        assert any("malforme" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# EC12 — blocs mixtes (text + tool_use autorise + tool_use bloque)
# ---------------------------------------------------------------------------

class TestEc12ToolGuardMixedBlocks:
    def test_ec12_toolguard_mixed_blocks(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Blocs mixtes : filtre uniquement bloque, preserve ordre des autres."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])  # "read" autorise
        inner = MockChatModel(response_content=[
            {"type": "text", "text": "bloc1"},
            {"type": "tool_use", "id": "r1", "name": "read", "input": {}},  # autorise
            {"type": "tool_use", "id": "t1", "name": "task", "input": {}},  # bloque
            {"type": "text", "text": "bloc3"},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            result = guard.invoke("test")

        assert isinstance(result, AIMessage)
        content = result.content
        assert isinstance(content, list)
        assert len(content) == 4

        # bloc 0 = text bloc1 inchange
        assert content[0] == {"type": "text", "text": "bloc1"}
        # bloc 1 = tool_use read inchange
        assert content[1]["name"] == "read"
        assert content[1]["type"] == "tool_use"
        # bloc 2 = text synthetique (task filtre)
        assert content[2]["type"] == "text"
        assert "task" in content[2]["text"]
        # bloc 3 = text bloc3 inchange
        assert content[3] == {"type": "text", "text": "bloc3"}

        # WARNING pour task
        assert any("task" in r.message or "tool_blocked" in r.message for r in caplog.records)

    def test_ec12_astream_mixed_blocks(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """astream avec blocs mixtes : meme logique de filtrage."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "text", "text": "debut"},
            {"type": "tool_use", "id": "t2", "name": "task", "input": {}},
            {"type": "text", "text": "fin"},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        async def _run() -> list[Any]:
            chunks = []
            async for chunk in guard.astream("test"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        # Verifier que task est filtre dans l'ensemble des chunks
        all_tool_use_names: list[str] = []
        for chunk in chunks:
            c = chunk.content
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        name = b.get("name", "")
                        if name:
                            all_tool_use_names.append(name)

        assert "task" not in all_tool_use_names
