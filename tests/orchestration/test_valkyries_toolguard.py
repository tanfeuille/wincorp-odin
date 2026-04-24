"""Tests TDD — ValkyrieToolGuard middleware (R15-R17, EC10-EC12).

@spec specs/valkyries.spec.md v1.4

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

# ---------------------------------------------------------------------------
# I1 — trace_id dynamique depuis run_manager.run_id (spec §5.7)
# ---------------------------------------------------------------------------

class TestTraceIdFromRunManager:
    def test_trace_id_propagated_to_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """run_manager.run_id → trace_id=<id> dans le WARNING loggue (spec §5.7)."""
        import logging

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "x1", "name": "task", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        # Mock run_manager avec run_id
        class _FakeRunManager:
            run_id = "test-trace-123"

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            guard._generate(
                messages=[AIMessage(content="test")],
                run_manager=_FakeRunManager(),
            )

        log_messages = " ".join(r.message for r in caplog.records)
        # trace_id doit etre test-trace-123, pas "unknown"
        assert "test-trace-123" in log_messages
        assert "unknown" not in log_messages

    def test_trace_id_unknown_when_no_run_manager(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """run_manager=None → trace_id=unknown dans le WARNING."""
        import logging

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "x2", "name": "task", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            guard._generate(
                messages=[AIMessage(content="test")],
                run_manager=None,
            )

        log_messages = " ".join(r.message for r in caplog.records)
        assert "trace_id=unknown" in log_messages

    def test_trace_id_stream_propagated(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """trace_id propagé dans _stream via run_manager.run_id."""
        import logging

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["shell"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "s1", "name": "shell", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        class _FakeRunManager:
            run_id = "stream-trace-456"

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            list(guard._stream(
                messages=[AIMessage(content="test")],
                run_manager=_FakeRunManager(),
            ))

        log_messages = " ".join(r.message for r in caplog.records)
        assert "stream-trace-456" in log_messages

    def test_trace_id_unknown_when_run_id_is_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """run_manager.run_id = None → trace_id=unknown (branche ligne 774)."""
        import logging

        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = MockChatModel(response_content=[
            {"type": "tool_use", "id": "x3", "name": "task", "input": {}},
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        class _FakeRunManagerNoId:
            run_id = None  # run_id explicitement None

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            guard._generate(
                messages=[AIMessage(content="test")],
                run_manager=_FakeRunManagerNoId(),
            )

        log_messages = " ".join(r.message for r in caplog.records)
        assert "trace_id=unknown" in log_messages


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


# ---------------------------------------------------------------------------
# TestR17StreamBufferAccumulation — accumulation inter-chunks (spec §5.5 v1.4)
# ---------------------------------------------------------------------------
#
# Ces tests valident la classe _StreamToolBuffer et son integration dans _stream
# et _astream. Les mocks simulent un provider qui fragmente les blocs tool_use
# sur plusieurs chunks (comportement DeepSeek / OpenAI-compat / Anthropic variable).
#
# Convention mocks :
#   - FragmentedMockChatModel : modele de test dont _stream/_astream emettent
#     les blocs contenus dans ``chunks_sequence`` — une liste de listes de dicts,
#     chaque sous-liste etant le contenu d'un chunk distinct.
# ---------------------------------------------------------------------------


class FragmentedMockChatModel(BaseChatModel):
    """BaseChatModel de test emettant des chunks fragmentes selon une sequence.

    Attributes:
        chunks_sequence: Liste de listes de blocs. Chaque sous-liste devient
            le content d'un AIMessageChunk distinct emis par _stream/_astream.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks_sequence: list[list[Any]]

    @property
    def _llm_type(self) -> str:
        return "fragmented-mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Non utilise dans ces tests — requis par BaseChatModel
        msg = AIMessage(content=[])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(content=[])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Emet chaque sous-liste comme un chunk AIMessageChunk distinct."""
        for block_list in self.chunks_sequence:
            chunk = AIMessageChunk(content=block_list)
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Version async de _stream."""
        for block_list in self.chunks_sequence:
            chunk = AIMessageChunk(content=block_list)
            yield ChatGenerationChunk(message=chunk)


def _collect_stream_blocks(guard: Any, prompt: str = "test") -> list[Any]:
    """Helper : collecte tous les blocs de contenu emis par guard.stream()."""
    all_blocks: list[Any] = []
    for chunk in guard.stream(prompt):
        c = chunk.content
        if isinstance(c, list):
            all_blocks.extend(c)
        elif isinstance(c, str) and c:
            all_blocks.append({"type": "text", "text": c})
    return all_blocks


async def _collect_astream_blocks(guard: Any, prompt: str = "test") -> list[Any]:
    """Helper async : collecte tous les blocs emis par guard.astream()."""
    all_blocks: list[Any] = []
    async for chunk in guard.astream(prompt):
        c = chunk.content
        if isinstance(c, list):
            all_blocks.extend(c)
        elif isinstance(c, str) and c:
            all_blocks.append({"type": "text", "text": c})
    return all_blocks


class TestR17StreamBufferAccumulation:
    """Tests d'accumulation inter-chunks pour _StreamToolBuffer (spec §5.5 v1.4).

    Verifie que :
    - Aucun bloc tool_use n'est emis avant sa completion (accumulation respectee).
    - Le filtre est applique sur le bloc complet (pas sur les fragments).
    - L'ordre des blocs sortants respecte l'ordre d'accumulation FIFO.
    - La logique est identique pour _stream (sync) et _astream (async).
    """

    def test_tool_use_fragmented_type_then_name_filtered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Chunk 1 = type+id sans name, chunk 2 = meme index + name → bloque.

        Assert :
        - Aucun bloc tool_use emis avant la completion (accumulation respectee).
        - Apres completion, le bloc est filtre (name='task' in blocked_tools).
        - Bloc text synthetique emis a la place.
        - WARNING loggue avec role + tool='task' + trace_id.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Chunk 1 : debut du bloc tool_use, name absent
        # Chunk 2 : completion du meme index avec name
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "abc1", "index": 0}],  # name absent
            [{"type": "tool_use", "name": "task", "input": {}, "index": 0}],  # completion
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Aucun bloc tool_use dans la sortie (filtre)
        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 0, f"Un bloc tool_use a ete emis : {tool_use_blocks}"

        # Bloc text synthetique present avec 'task' et 'rejete'
        text_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_blocks) >= 1
        texts = " ".join(b.get("text", "") for b in text_blocks)
        assert "task" in texts
        assert "rejete" in texts

        # WARNING loggue avec role + tool + trace_id
        log_messages = " ".join(r.message for r in caplog.records)
        assert "task" in log_messages
        assert "brynhildr" in log_messages

    def test_tool_use_fragmented_type_then_name_filtered_async(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Meme scenario que le test sync, valide via _astream."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "abc1", "index": 0}],
            [{"type": "tool_use", "name": "task", "input": {}, "index": 0}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = asyncio.run(_collect_astream_blocks(guard))

        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 0

        text_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "text"]
        texts = " ".join(b.get("text", "") for b in text_blocks)
        assert "task" in texts
        assert "rejete" in texts

    def test_tool_use_fragmented_input_accumulated_passthrough(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Tool autorise fragmente en plusieurs chunks → accumule et emet en pass-through.

        Schema :
        - Chunk 1 : debut tool_use name='allowed_tool', input vide
        - Chunk 2 : input_json_delta (fragment input)
        - Chunk 3 : nouveau input_json_delta

        Assert :
        - Le bloc tool_use 'allowed_tool' est emis en pass-through.
        - Aucun WARNING valkyrie_tool_blocked.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        # 'read' est dans la whitelist mais PAS dans blocked_tools de ce config
        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "r1", "name": "read", "input": {}, "index": 0}],
            [{"type": "input_json_delta", "partial_json": '{"path":', "index": 0}],
            [{"type": "input_json_delta", "partial_json": '"/tmp/f"}', "index": 0}],
            # Bloc text qui declenche le flush du tool_use pending
            [{"type": "text", "text": "ok"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Le bloc tool_use 'read' est present en pass-through
        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "read"

        # Input JSON reconstitue depuis les input_json_delta accumules
        # (v1.4.1 : regression fix post-audit C) — consommateur aval recevrait
        # un input tronque sinon.
        assert tool_use_blocks[0]["input"] == {"path": "/tmp/f"}

        # Aucun WARNING tool_blocked
        assert not any("tool_blocked" in r.message for r in caplog.records)

    def test_input_json_invalid_fallback_to_raw_string(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """JSON malforme du provider → fallback chaine brute + WARNING.

        Regression fix post-audit C : si un provider emet un input_json_delta
        qui ne se parse pas en JSON valide, ne pas crasher mais conserver la
        chaine brute et logger WARNING.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "r1", "name": "read", "input": {}, "index": 0}],
            [{"type": "input_json_delta", "partial_json": '{"malformed": ', "index": 0}],
            [{"type": "text", "text": "fin"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        # Fallback chaine brute preserve le contenu (pas de crash, pas de perte)
        assert tool_use_blocks[0]["input"] == '{"malformed": '
        # WARNING emis avec role + tool + trace_id
        assert any("valkyrie_tool_input_json_invalid" in r.message for r in caplog.records)

    def test_multiple_tool_uses_different_indices_evaluated_independently(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Deux tool_use concurrents (index=0 bloque, index=1 autorise).

        Assert :
        - Index 0 ('task') → filtre, texte synthetique.
        - Index 1 ('read') → pass-through.
        - Ordre preserve : bloc index=0 avant bloc index=1 dans la sortie.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Les deux blocs arrivent fragmentes sur des chunks distincts
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "t1", "name": "task", "input": {}, "index": 0}],
            [{"type": "tool_use", "id": "r1", "name": "read", "input": {}, "index": 1}],
            # Bloc text qui force le flush des deux pending
            [{"type": "text", "text": "fin"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Index 0 filtre : aucun tool_use 'task'
        task_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("name") == "task"]
        assert len(task_blocks) == 0

        # Index 1 pass-through : un tool_use 'read'
        read_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("name") == "read"]
        assert len(read_blocks) == 1
        assert read_blocks[0]["type"] == "tool_use"

        # Ordre : texte synthetique (remplacement task) avant read
        task_replacement_idx = next(
            i for i, b in enumerate(all_blocks)
            if isinstance(b, dict) and b.get("type") == "text" and "task" in b.get("text", "")
        )
        read_idx = next(
            i for i, b in enumerate(all_blocks)
            if isinstance(b, dict) and b.get("name") == "read"
        )
        assert task_replacement_idx < read_idx, (
            f"Bloc 'task' remplace attendu avant 'read' : task_idx={task_replacement_idx}, read_idx={read_idx}"
        )

        # WARNING pour task
        assert any("task" in r.message for r in caplog.records)

    def test_final_flush_emits_pending_blocks(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Stream termine sans bloc final declencheur → flush emis en fin de stream.

        Simule un provider qui envoie un tool_use complet mais sans bloc
        subsequence (pas de bloc text apres) : le flush final de _stream
        doit emettre le bloc accumule.

        Assert :
        - Si bloque : bloc text synthetique emis via flush final + WARNING.
        - Aucune perte de blocs.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Un seul chunk, pas de bloc subsequent : flush final obligatoire
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "t1", "name": "task", "input": {}, "index": 0}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Bloc filtre emis via flush final
        assert len(all_blocks) >= 1
        text_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_blocks) >= 1
        texts = " ".join(b.get("text", "") for b in text_blocks)
        assert "task" in texts

        # WARNING loggue
        assert any("task" in r.message for r in caplog.records)

    def test_final_flush_passthrough_allowed_tool(self) -> None:
        """Flush final sur tool autorise : pass-through sans WARNING."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "r1", "name": "read", "input": {}, "index": 0}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        all_blocks = _collect_stream_blocks(guard)

        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "read"

    def test_text_block_flushes_pending_tool_use_first(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bloc text apres tool_use fragmente → tool_use flush AVANT le texte (ordre FIFO).

        Assert :
        - Le bloc text synthetique (remplacement du tool_use bloque) apparait
          avant le bloc text normal dans la sortie.
        - L'ordre des chunks sortants respecte l'ordre d'accumulation.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            # Chunk 1 : debut tool_use bloque (index=0)
            [{"type": "tool_use", "id": "t1", "name": "task", "input": {}, "index": 0}],
            # Chunk 2 : nouveau bloc text (declenche le flush du tool_use pending)
            [{"type": "text", "text": "texte_apres"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Identifier les blocs text
        text_blocks_with_idx = [
            (i, b) for i, b in enumerate(all_blocks)
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks_with_idx) >= 2

        # Le remplacement de task doit etre avant 'texte_apres'
        replacement_positions = [
            i for i, b in text_blocks_with_idx if "task" in b.get("text", "")
        ]
        normal_text_positions = [
            i for i, b in text_blocks_with_idx if b.get("text") == "texte_apres"
        ]

        assert len(replacement_positions) >= 1, "Bloc remplacement 'task' attendu"
        assert len(normal_text_positions) >= 1, "Bloc 'texte_apres' attendu"
        assert replacement_positions[0] < normal_text_positions[0], (
            f"Remplacement attendu avant texte_apres : positions={replacement_positions}/{normal_text_positions}"
        )

    def test_text_block_flushes_pending_tool_use_first_async(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Meme scenario ordre FIFO, valide via _astream."""
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "t1", "name": "task", "input": {}, "index": 0}],
            [{"type": "text", "text": "texte_apres"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = asyncio.run(_collect_astream_blocks(guard))

        text_blocks_with_idx = [
            (i, b) for i, b in enumerate(all_blocks)
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        replacement_positions = [
            i for i, b in text_blocks_with_idx if "task" in b.get("text", "")
        ]
        normal_text_positions = [
            i for i, b in text_blocks_with_idx if b.get("text") == "texte_apres"
        ]

        assert replacement_positions[0] < normal_text_positions[0]

    def test_openai_compat_fragmented_name_filtered(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Simule provider OpenAI-compat ou le name arrive fragmente sur 2 chunks.

        Le mock emet des blocs dict avec le schema normalise (type, index, name)
        comme si le caller avait normalise depuis tool_calls list. On valide que
        le buffer accumule correctement et filtre le bloc complet.

        Note : ce test valide le chemin 'name absent puis complete sur chunk 2'.
        La normalisation depuis le format tool_calls OpenAI natif est
        responsabilite du caller (hors scope direct du buffer).
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Schema normalise simule : name fragmente sur 2 chunks
        inner = FragmentedMockChatModel(chunks_sequence=[
            # Chunk 1 : id + type, name absent (tel qu'un parser OpenAI-compat l'emettrait)
            [{"type": "tool_use", "id": "oai1", "index": 0}],
            # Chunk 2 : completion avec name
            [{"type": "tool_use", "name": "task", "index": 0}],
            # Chunk 3 : bloc text qui force le flush
            [{"type": "text", "text": "done"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Bloc 'task' doit etre filtre
        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 0

        text_replacements = [
            b for b in all_blocks
            if isinstance(b, dict) and b.get("type") == "text" and "task" in b.get("text", "")
        ]
        assert len(text_replacements) >= 1

        # WARNING loggue
        log_messages = " ".join(r.message for r in caplog.records)
        assert "task" in log_messages

    def test_input_json_delta_orphan_no_pending_ignored(self) -> None:
        """input_json_delta avec index non present dans pending → ignore silencieusement.

        Couvre la branche 'idx not in _pending' de la logique input_json_delta :
        un delta arrive pour un index qu'on n'a pas encore vu (orphelin).
        Le buffer doit ignorer le delta sans crash ni emission.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Chunk 1 : delta orphelin (index=99 jamais vu dans _pending)
        # Chunk 2 : tool autorise complet + bloc text pour flush
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "input_json_delta", "partial_json": '{"orphan":', "index": 99}],
            [{"type": "tool_use", "id": "r1", "name": "read", "input": {}, "index": 0}],
            [{"type": "text", "text": "fin"}],
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        all_blocks = _collect_stream_blocks(guard)

        # Le delta orphelin n'a produit aucun bloc fantome
        tool_use_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "read"

        # Texte normal preserve
        text_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "text"]
        assert any(b.get("text") == "fin" for b in text_blocks)

    def test_buffer_malformed_accumulated_block_no_name_after_flush(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bloc tool_use accumule sans name en fin de stream → malforme + WARNING.

        Couvre la branche 'tool_name is None' dans _evaluate_block du buffer :
        un bloc qui a commence a s'accumuler mais dont le name n'est jamais arrive
        avant la fin du stream. Le flush final doit le traiter comme malforme.
        """
        from wincorp_odin.orchestration.valkyries import ValkyrieToolGuard

        cfg = _make_valkyrie_config(blocked_tools=["task"])
        # Un seul chunk : tool_use sans name — flush final evaluera ce bloc sans name
        inner = FragmentedMockChatModel(chunks_sequence=[
            [{"type": "tool_use", "id": "bad1", "index": 0}],  # name absent
        ])
        guard = ValkyrieToolGuard(wrapped=inner, config=cfg)

        with caplog.at_level(logging.WARNING, logger="wincorp_odin.orchestration.valkyries"):
            all_blocks = _collect_stream_blocks(guard)

        # Bloc malforme emis via flush final
        assert len(all_blocks) >= 1
        text_blocks = [b for b in all_blocks if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_blocks) >= 1
        texts = " ".join(b.get("text", "") for b in text_blocks)
        assert "malforme" in texts

        # WARNING loggue (malforme)
        log_messages = " ".join(r.message for r in caplog.records)
        assert "malforme" in log_messages.lower()
