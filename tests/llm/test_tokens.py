"""Tests token usage tracking (Phase 1.6).

@spec specs/llm-factory.spec.md v1.3 §24
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from wincorp_odin.llm.exceptions import TokenTrackingError
from wincorp_odin.llm.tokens import (
    FileSink,
    LogSink,
    PricingConfig,
    SupabaseSink,
    TokenTrackingWrapper,
    TokenUsageEvent,
    _extract_usage_metadata,
    clear_context,
    get_sink,
    set_context,
)

# ---------------------------------------------------------------------------
# PricingConfig & cost calculation (R27)
# ---------------------------------------------------------------------------


def test_r27_pricing_compute_cost_sonnet() -> None:
    """R27 : cout = (in/1M)*rate_in + (out/1M)*rate_out."""
    p = PricingConfig(input_per_million_eur=2.76, output_per_million_eur=13.80)
    # 1000 input + 500 output
    cost = p.compute_cost(1000, 500)
    expected = (1000 / 1_000_000) * 2.76 + (500 / 1_000_000) * 13.80
    assert cost == round(expected, 6)


def test_r27_pricing_zero_tokens() -> None:
    """Cout = 0 pour 0 tokens."""
    p = PricingConfig(input_per_million_eur=2.76, output_per_million_eur=13.80)
    assert p.compute_cost(0, 0) == 0.0


# ---------------------------------------------------------------------------
# TokenUsageEvent
# ---------------------------------------------------------------------------


def test_token_usage_event_to_json_dict() -> None:
    """Event serialisable en dict JSON."""
    evt = TokenUsageEvent(
        timestamp=1234567890.0,
        model_name="sonnet",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_eur=0.001,
        session_id="sess-1",
        agent_name="brynhildr",
    )
    d = evt.to_json_dict()
    assert d["model_name"] == "sonnet"
    assert d["cost_eur"] == 0.001
    assert d["session_id"] == "sess-1"
    assert d["client_id"] is None
    # JSON-serialisable
    json.dumps(d)


# ---------------------------------------------------------------------------
# R26 — extract usage_metadata
# ---------------------------------------------------------------------------


def test_r26_extract_usage_metadata_present() -> None:
    """R26 : usage_metadata standard LangChain lu correctement."""
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert _extract_usage_metadata(result) == (100, 50, 150)


def test_r26_extract_usage_metadata_missing_returns_zero() -> None:
    """EC37 : usage_metadata absent -> (0, 0, 0) + WARNING."""

    class FakeResult:
        usage_metadata = None

    assert _extract_usage_metadata(FakeResult()) == (0, 0, 0)


def test_r26_extract_usage_metadata_no_attr_returns_zero() -> None:
    """Pas d'attribut usage_metadata -> (0, 0, 0)."""

    class Plain:
        pass

    assert _extract_usage_metadata(Plain()) == (0, 0, 0)


def test_r26_extract_usage_metadata_partial_total_fallback() -> None:
    """total_tokens absent -> calcule depuis input + output."""
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 80, "output_tokens": 20}
    assert _extract_usage_metadata(result) == (80, 20, 100)


def test_r26_extract_usage_metadata_invalid_format() -> None:
    """Format invalide -> (0, 0, 0)."""
    result = MagicMock()
    result.usage_metadata = "not-a-dict"
    assert _extract_usage_metadata(result) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Sinks — R28
# ---------------------------------------------------------------------------


def test_log_sink_emits_via_logger(caplog: pytest.LogCaptureFixture) -> None:
    """LogSink emit via logger.info."""
    sink = LogSink()
    evt = TokenUsageEvent(
        timestamp=1.0,
        model_name="sonnet",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        cost_eur=0.0,
    )
    with caplog.at_level("INFO", logger="wincorp_odin.llm.tokens"):
        sink.emit(evt)
    assert any("llm_usage_event" in rec.message for rec in caplog.records)


def test_file_sink_appends_jsonl(tmp_path: Path) -> None:
    """FileSink append JSONL + cree dossier parent."""
    path = tmp_path / "nested" / "events.jsonl"
    sink = FileSink(path=path)
    evt = TokenUsageEvent(
        timestamp=1.0,
        model_name="sonnet",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cost_eur=0.0001,
    )
    sink.emit(evt)
    sink.emit(evt)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["model_name"] == "sonnet"


def test_file_sink_write_error_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """EC40 : erreur ecriture -> WARNING, pas d'exception."""
    path = tmp_path / "events.jsonl"
    sink = FileSink(path=path)

    # Simule echec open
    def bad_open(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", bad_open)

    evt = TokenUsageEvent(
        timestamp=1.0,
        model_name="sonnet",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_eur=0.0,
    )
    with caplog.at_level("WARNING", logger="wincorp_odin.llm.tokens"):
        sink.emit(evt)  # Ne doit pas lever
    assert any("FileSink echec ecriture" in rec.message for rec in caplog.records)


def test_supabase_sink_raises_not_implemented() -> None:
    """EC41 : SupabaseSink stub -> NotImplementedError."""
    sink = SupabaseSink()
    evt = TokenUsageEvent(
        timestamp=1.0,
        model_name="sonnet",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_eur=0.0,
    )
    with pytest.raises(NotImplementedError):
        sink.emit(evt)


# ---------------------------------------------------------------------------
# get_sink (§24.4, R28)
# ---------------------------------------------------------------------------


def test_r28_get_sink_log_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default = log."""
    monkeypatch.delenv("WINCORP_LLM_TOKEN_SINK", raising=False)
    sink = get_sink()
    assert isinstance(sink, LogSink)


def test_r28_get_sink_explicit_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Name explicite 'file' avec override path."""
    monkeypatch.setenv("WINCORP_LLM_TOKEN_SINK_FILE", str(tmp_path / "events.jsonl"))
    sink = get_sink("file")
    assert isinstance(sink, FileSink)


def test_r28_get_sink_env_supabase_returns_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var supabase -> SupabaseSink stub."""
    monkeypatch.setenv("WINCORP_LLM_TOKEN_SINK", "supabase")
    sink = get_sink()
    assert isinstance(sink, SupabaseSink)


def test_r28_get_sink_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC39 : valeur invalide -> TokenTrackingError."""
    monkeypatch.setenv("WINCORP_LLM_TOKEN_SINK", "inconnu")
    with pytest.raises(TokenTrackingError) as excinfo:
        get_sink()
    assert "WINCORP_LLM_TOKEN_SINK" in str(excinfo.value)


# ---------------------------------------------------------------------------
# TokenTrackingWrapper integration
# ---------------------------------------------------------------------------


def test_wrapper_emits_event_after_invoke() -> None:
    """Wrapper emit un event apres chaque invoke reussi."""
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}
    mock_model.invoke.return_value = result

    pricing = PricingConfig(input_per_million_eur=2.76, output_per_million_eur=13.80)
    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=pricing,
        sink=CaptureSink(),
    ).wrap()

    returned = wrapper.invoke("prompt")
    assert returned is result
    assert len(events) == 1
    evt = events[0]
    assert evt.model_name == "sonnet"
    assert evt.input_tokens == 200
    assert evt.output_tokens == 80
    assert evt.total_tokens == 280
    assert evt.cost_eur > 0


def test_wrapper_pricing_missing_cost_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """EC38 : pricing None -> cost_eur=0.0 + WARNING."""
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    mock_model.invoke.return_value = result

    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="custom",
        pricing=None,
        sink=CaptureSink(),
    ).wrap()

    with caplog.at_level("WARNING", logger="wincorp_odin.llm.tokens"):
        wrapper.invoke("prompt")

    assert len(events) == 1
    assert events[0].cost_eur == 0.0
    assert any("Pricing manquant" in rec.message for rec in caplog.records)


def test_wrapper_sink_error_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """R28 : sink.emit() exception -> WARNING, pas d'exception caller."""

    class BadSink:
        def emit(self, event: TokenUsageEvent) -> None:
            raise RuntimeError("sink disaster")

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    mock_model.invoke.return_value = result

    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=PricingConfig(input_per_million_eur=1.0, output_per_million_eur=1.0),
        sink=BadSink(),
    ).wrap()

    with caplog.at_level("WARNING", logger="wincorp_odin.llm.tokens"):
        result_returned = wrapper.invoke("prompt")

    assert result_returned is result
    assert any("Sink emit echec" in rec.message for rec in caplog.records)


def test_wrapper_call_fails_no_event_emitted() -> None:
    """Si invoke leve -> pas d'event emis, exception propage."""
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    mock_model.invoke.side_effect = RuntimeError("fail")

    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=PricingConfig(input_per_million_eur=1.0, output_per_million_eur=1.0),
        sink=CaptureSink(),
    ).wrap()

    with pytest.raises(RuntimeError):
        wrapper.invoke("prompt")
    assert events == []


def test_wrapper_context_threadlocal() -> None:
    """Le context (session_id, agent_name, client_id) est attache aux events."""
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    mock_model.invoke.return_value = result

    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=None,
        sink=CaptureSink(),
    ).wrap()

    set_context(session_id="sess-42", agent_name="brynhildr", client_id="SPINEX")
    try:
        wrapper.invoke("prompt")
    finally:
        clear_context()

    assert events[0].session_id == "sess-42"
    assert events[0].agent_name == "brynhildr"
    assert events[0].client_id == "SPINEX"


def test_wrapper_no_context_set_defaults_none() -> None:
    """Sans set_context, tous les champs context = None."""
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    mock_model.invoke.return_value = result

    clear_context()
    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=None,
        sink=CaptureSink(),
    ).wrap()

    wrapper.invoke("prompt")
    assert events[0].session_id is None
    assert events[0].agent_name is None


def test_wrapper_async_emits_event() -> None:
    """ainvoke -> emit event."""
    import asyncio

    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    mock_model = MagicMock()
    result = MagicMock()
    result.usage_metadata = {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75}

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> Any:
        return result

    mock_model.ainvoke = fake_ainvoke

    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="haiku",
        pricing=PricingConfig(input_per_million_eur=0.92, output_per_million_eur=4.60),
        sink=CaptureSink(),
    ).wrap()

    result_returned = asyncio.run(wrapper.ainvoke("prompt"))
    assert result_returned is result
    assert len(events) == 1
    assert events[0].model_name == "haiku"


def test_wrapper_delegates_other_attributes() -> None:
    """Wrapper delegue bind_tools, stream, etc."""
    mock_model = MagicMock()
    mock_model.bind_tools = MagicMock(return_value="bound")
    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=None,
        sink=LogSink(),
    ).wrap()
    assert wrapper.bind_tools("x") == "bound"


def test_default_context_when_not_set() -> None:
    """Thread sans set_context() -> _current_context retourne context vide."""
    from wincorp_odin.llm.tokens import _context_local, _current_context

    # Nettoyer eventuel context parent
    if hasattr(_context_local, "ctx"):
        delattr(_context_local, "ctx")

    ctx = _current_context()
    assert ctx.session_id is None
    assert ctx.agent_name is None
    assert ctx.client_id is None


def test_get_sink_file_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """FileSink sans override env -> path default wincorp-odin/.token_usage/."""
    monkeypatch.delenv("WINCORP_LLM_TOKEN_SINK_FILE", raising=False)
    sink = get_sink("file")
    assert isinstance(sink, FileSink)
    # Le path doit contenir .token_usage (defaut spec §24.4)
    assert ".token_usage" in str(sink._path)


# ---------------------------------------------------------------------------
# PR-013 — __setattr__ delegation sur _TokenTrackingWrapped
# ---------------------------------------------------------------------------


def test_tokens_setattr_delegates_to_model() -> None:
    """PR-013 : wrapped.callbacks = [...] -> setattr delegue vers model."""
    mock_model = MagicMock()
    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=None,
        sink=LogSink(),
    ).wrap()
    wrapper.callbacks = ["cb1"]
    assert mock_model.callbacks == ["cb1"]


def test_tokens_setattr_slot_stays_internal() -> None:
    """PR-013 : assignation __slots__ reste en interne."""
    mock_model = MagicMock()
    wrapper = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=None,
        sink=LogSink(),
    ).wrap()
    new_model = MagicMock()
    wrapper._model = new_model
    assert wrapper._model is new_model


# ---------------------------------------------------------------------------
# PR-015 — FileSink mkdir failure swallowed
# ---------------------------------------------------------------------------


def test_file_sink_mkdir_failure_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """PR-015 : mkdir PermissionError -> warning log, pas de crash."""
    target = tmp_path / "forbidden" / "events.jsonl"

    original_mkdir = Path.mkdir

    def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if "forbidden" in str(self):
            raise PermissionError(f"denied: {self}")
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    with caplog.at_level("WARNING", logger="wincorp_odin.llm.tokens"):
        sink = FileSink(path=target)  # Ne doit pas crash

    assert isinstance(sink, FileSink)
    assert any(
        "impossible de creer le repertoire" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# PR-019 — ContextVar isolation
# ---------------------------------------------------------------------------


def test_set_context_uses_contextvar() -> None:
    """PR-019 : set_context() fixe la valeur dans le ContextVar."""
    from wincorp_odin.llm.tokens import _ctx_var

    clear_context()
    assert _ctx_var.get() is None
    set_context(session_id="s1", agent_name="a1", client_id="c1")
    ctx = _ctx_var.get()
    assert ctx is not None
    assert ctx.session_id == "s1"
    assert ctx.agent_name == "a1"
    assert ctx.client_id == "c1"
    clear_context()
    assert _ctx_var.get() is None


def test_current_context_fallback_threadlocal() -> None:
    """PR-019 retrocompat : si ContextVar vide mais thread-local pose, retourne thread-local."""
    from wincorp_odin.llm.tokens import (
        TokenTrackingContext,
        _context_local,
        _ctx_var,
        _current_context,
    )

    # Reset ContextVar
    _ctx_var.set(None)
    # Pose directement sur le thread-local (bypasser set_context pour le test)
    _context_local.ctx = TokenTrackingContext(session_id="tl-session")
    try:
        ctx = _current_context()
        assert ctx.session_id == "tl-session"
    finally:
        if hasattr(_context_local, "ctx"):
            delattr(_context_local, "ctx")
