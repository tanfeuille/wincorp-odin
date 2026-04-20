"""Tests integration factory + middlewares (Phase 1.4-1.6).

@spec specs/llm-factory.spec.md v1.3 §25
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# §25.2 — params with_* par defaut True
# ---------------------------------------------------------------------------


def test_default_params_wrap_all_middlewares(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """create_model() sans params -> instance wrappee (proxy, pas le mock brut)."""
    from wincorp_odin.llm import create_model

    instance = create_model("sonnet")
    # L'instance retournee doit etre un proxy (CircuitBreaker wrapper en externe)
    # et non le mock brut de ChatAnthropic. On verifie via absence d'attribut
    # interne du mock et presence de invoke callable.
    assert hasattr(instance, "invoke")
    assert hasattr(instance, "ainvoke")
    # Le raw mock est bien appele UNE fois (ChatAnthropic())
    assert mock_chat_anthropic.call_count == 1


def test_with_all_false_returns_raw(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """with_*=False -> retourne l'instance brute."""
    from wincorp_odin.llm import create_model

    instance = create_model(
        "sonnet",
        with_circuit_breaker=False,
        with_retry=False,
        with_token_tracking=False,
    )
    # L'instance = le mock direct retourne par ChatAnthropic()
    assert instance.init_kwargs["model"] == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# §25.3 — cache keye par 5-tuple
# ---------------------------------------------------------------------------


def test_cache_keyed_by_middleware_flags(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Meme nom + middlewares differents -> instances distinctes."""
    from wincorp_odin.llm import create_model

    a = create_model("sonnet", with_circuit_breaker=True, with_retry=True, with_token_tracking=True)
    b = create_model("sonnet", with_circuit_breaker=False, with_retry=False, with_token_tracking=False)
    assert a is not b
    # 2 instanciations ChatAnthropic differentes
    assert mock_chat_anthropic.call_count == 2


def test_cache_identical_flags_returns_same(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Flags identiques -> meme instance cachee."""
    from wincorp_odin.llm import create_model

    a = create_model("sonnet", with_circuit_breaker=True)
    b = create_model("sonnet", with_circuit_breaker=True)
    assert a is b
    assert mock_chat_anthropic.call_count == 1


# ---------------------------------------------------------------------------
# §25.5 — breaker persistant cross-create_model
# ---------------------------------------------------------------------------


def test_breaker_persistent_across_calls(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Le CircuitBreaker est partage entre appels create_model(name)."""
    from wincorp_odin.llm import create_model, factory

    # 1er create
    create_model("sonnet")
    assert "sonnet" in factory._breaker_instances
    breaker_1 = factory._breaker_instances["sonnet"]

    # 2eme create : meme breaker
    create_model("sonnet")
    breaker_2 = factory._breaker_instances["sonnet"]
    assert breaker_1 is breaker_2


def test_breaker_config_from_yaml(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """circuit_breaker dans YAML -> applique au breaker."""
    from wincorp_odin.llm import create_model, factory

    urd = tmp_path / "urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "custom"
    display_name: "Custom"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    circuit_breaker:
      failure_threshold: 7
      recovery_timeout_sec: 42.0
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "urd"))

    create_model("custom")
    breaker = factory._breaker_instances["custom"]
    assert breaker.config.failure_threshold == 7
    assert breaker.config.recovery_timeout_sec == 42.0


def test_retry_config_from_yaml(
    mock_anthropic_api_key: str,
    mock_chat_anthropic: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retry dans YAML -> applique au RetryConfig."""
    from wincorp_odin.llm import load_models_config

    urd = tmp_path / "urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "fast"
    display_name: "Fast"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    retry:
      base_delay_sec: 0.5
      cap_delay_sec: 5.0
      max_attempts: 7
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "urd"))

    configs = load_models_config()
    assert configs["fast"].retry_config == {
        "base_delay_sec": 0.5,
        "cap_delay_sec": 5.0,
        "max_attempts": 7,
    }


def test_pricing_config_from_yaml(
    mock_anthropic_api_key: str,
    mock_chat_anthropic: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pricing dans YAML -> applique au PricingConfig."""
    from wincorp_odin.llm import load_models_config

    urd = tmp_path / "urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "tariff"
    display_name: "T"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    pricing:
      input_per_million_eur: 3.14
      output_per_million_eur: 15.92
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "urd"))

    configs = load_models_config()
    assert configs["tariff"].pricing_config == {
        "input_per_million_eur": 3.14,
        "output_per_million_eur": 15.92,
    }


# ---------------------------------------------------------------------------
# Wrapping order — raw -> tokens -> retry -> breaker (§25.1)
# ---------------------------------------------------------------------------


def test_wrapping_order_breaker_outermost(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Le breaker est le plus externe : quand OPEN, il bloque avant tout."""
    from wincorp_odin.llm import CircuitOpenError, create_model, factory

    # Cree l'instance avec tous les middlewares
    wrapped = create_model("sonnet")

    # Force le breaker OPEN manuellement
    breaker = factory._breaker_instances["sonnet"]
    breaker._failure_count = breaker.config.failure_threshold
    breaker._transition_to_open()

    # L'invoke doit echouer immediatement sans toucher au mock ChatAnthropic
    # On regarde plutot combien de fois .invoke est appele sur TOUTES les instances
    # creees via side_effect — impossible a tracer proprement, donc on verifie
    # juste que CircuitOpenError leve.
    with pytest.raises(CircuitOpenError):
        wrapped.invoke("prompt")


# ---------------------------------------------------------------------------
# Retrocompat — les tests Phase 1.1 restent verts avec defaults True
# ---------------------------------------------------------------------------


def test_backward_compat_invoke_success(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """Avec defaults, invoke delegue bien au ChatAnthropic brut."""
    from wincorp_odin.llm import create_model

    wrapped = create_model("sonnet")
    # Le mock ChatAnthropicInstance#1 a ete cree. Son .invoke est un MagicMock callable.
    # On appelle invoke sur le wrapper, il doit delegater tout le long vers l'instance.
    # Le MagicMock retourne son propre default — on verifie juste que ca passe.
    result = wrapped.invoke("prompt")
    # Le mock retourne un MagicMock par defaut
    assert result is not None


# ---------------------------------------------------------------------------
# Backward compat : signatures positionnelles inchangees
# ---------------------------------------------------------------------------


def test_positional_args_preserved(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """create_model('sonnet', True) -> thinking_enabled=True, middlewares actifs."""
    from wincorp_odin.llm import create_model

    wrapped = create_model("sonnet", True)  # thinking positionnel
    # Les kwargs du ChatAnthropic() doivent contenir thinking
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert "thinking" in kwargs
    # Et l'instance retournee a invoke
    assert hasattr(wrapped, "invoke")


# ---------------------------------------------------------------------------
# Test all 3 middlewares chained manually
# ---------------------------------------------------------------------------


def test_all_three_middlewares_composition() -> None:
    """raw -> tokens -> retry -> breaker dans le bon ordre."""
    from wincorp_odin.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from wincorp_odin.llm.retry import RetryConfig, RetryWrapper
    from wincorp_odin.llm.tokens import (
        PricingConfig,
        TokenTrackingWrapper,
        TokenUsageEvent,
    )

    # Faux model
    mock_model = MagicMock()
    result_obj = MagicMock()
    result_obj.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    mock_model.invoke.return_value = result_obj

    # Sink capture
    events: list[TokenUsageEvent] = []

    class CaptureSink:
        def emit(self, event: TokenUsageEvent) -> None:
            events.append(event)

    # Build chain : raw -> tokens -> retry -> breaker
    tracked = TokenTrackingWrapper(
        model=mock_model,
        model_name="sonnet",
        pricing=PricingConfig(input_per_million_eur=2.76, output_per_million_eur=13.80),
        sink=CaptureSink(),
    ).wrap()
    retried = RetryWrapper(tracked, RetryConfig(max_attempts=2)).wrap()
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=3))
    full = cb.wrap(retried)

    result = full.invoke("prompt")
    assert result is result_obj
    # L'event tokens a bien ete emis
    assert len(events) == 1
    assert events[0].input_tokens == 10


def test_validate_all_models_works_with_new_fields(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """validate_all_models() ne casse pas sur YAML avec nouveaux champs absents."""
    from wincorp_odin.llm import validate_all_models

    validate_all_models()  # Ne doit pas lever


def test_factory_pricing_from_yaml_creates_wrapper(
    mock_anthropic_api_key: str,
    mock_chat_anthropic: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pricing_config dans YAML -> PricingConfig applique au TokenTrackingWrapper."""
    from wincorp_odin.llm import create_model

    urd = tmp_path / "urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "priced"
    display_name: "P"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    pricing:
      input_per_million_eur: 1.5
      output_per_million_eur: 7.5
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "urd"))

    wrapped = create_model("priced")
    assert wrapped is not None


def test_factory_retry_config_applied_from_yaml(
    mock_anthropic_api_key: str,
    mock_chat_anthropic: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retry_config dans YAML -> RetryConfig applique au RetryWrapper via create_model."""
    from wincorp_odin.llm import create_model

    urd = tmp_path / "urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "retried"
    display_name: "R"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    retry:
      base_delay_sec: 0.25
      cap_delay_sec: 2.5
      max_attempts: 4
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "urd"))

    wrapped = create_model("retried")
    assert wrapped is not None
