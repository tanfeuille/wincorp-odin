"""Tests Phase 1.7 — auto-patch stream_usage=True pour OpenAI-compat.

@spec specs/llm-factory.spec.md v1.3.3 §24 + factory._build_kwargs
"""
from __future__ import annotations

from typing import Any

from wincorp_odin.llm.config import ModelConfig
from wincorp_odin.llm.factory import _build_kwargs


def _make_cfg(
    use: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> ModelConfig:
    """Helper : ModelConfig minimal pour tests _build_kwargs."""
    return ModelConfig(
        name="test-model",
        display_name="Test Model",
        use=use,
        model="test/model-id",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_resolved="sk-fake",
        max_tokens=1000,
        timeout=30.0,
        max_retries=0,
        supports_thinking=False,
        supports_vision=False,
        supports_reasoning_effort=False,
        when_thinking_enabled=None,
        when_thinking_disabled=None,
        extra_kwargs=extra_kwargs or {},
        disabled=False,
        circuit_breaker_config=None,
        retry_config=None,
        pricing_config=None,
    )


def test_stream_usage_auto_set_for_openai() -> None:
    """EC46 : langchain_openai -> stream_usage=True auto-injecte."""
    cfg = _make_cfg(use="langchain_openai:ChatOpenAI")
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    assert kwargs.get("stream_usage") is True


def test_stream_usage_auto_set_for_deepseek() -> None:
    """EC46 : langchain_deepseek -> stream_usage=True auto-injecte."""
    cfg = _make_cfg(use="langchain_deepseek:ChatDeepSeek")
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    assert kwargs.get("stream_usage") is True


def test_stream_usage_auto_set_for_community() -> None:
    """EC46 : langchain_community (Ollama/vLLM) -> stream_usage=True."""
    cfg = _make_cfg(use="langchain_community:ChatOllama")
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    assert kwargs.get("stream_usage") is True


def test_stream_usage_not_set_for_anthropic() -> None:
    """Anthropic : tokens inclus par defaut, pas de stream_usage."""
    cfg = _make_cfg(use="langchain_anthropic:ChatAnthropic")
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    assert "stream_usage" not in kwargs


def test_stream_usage_user_override_respected_false() -> None:
    """EC47 : user fixe stream_usage=False -> valeur preservee, pas d'override."""
    cfg = _make_cfg(
        use="langchain_openai:ChatOpenAI",
        extra_kwargs={"stream_usage": False},
    )
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    # extra_kwargs merge avant auto-patch, et auto-patch skip si la cle existe.
    assert kwargs.get("stream_usage") is False


def test_stream_usage_user_override_respected_true() -> None:
    """EC47 : user fixe stream_usage=True explicite -> valeur preservee."""
    cfg = _make_cfg(
        use="langchain_deepseek:ChatDeepSeek",
        extra_kwargs={"stream_usage": True},
    )
    kwargs = _build_kwargs(cfg, thinking_enabled=False)
    assert kwargs.get("stream_usage") is True
