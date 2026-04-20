"""Tests retry exponentiel + Retry-After (Phase 1.5).

@spec specs/llm-factory.spec.md v1.3 §23
"""
from __future__ import annotations

from datetime import UTC
from typing import Any
from unittest.mock import MagicMock

import pytest

from wincorp_odin.llm.exceptions import RetryExhaustedError
from wincorp_odin.llm.retry import (
    RetryConfig,
    RetryWrapper,
    _compute_delay,
    _jitter_enabled_from_env,
    _parse_retry_after,
)

# ---------------------------------------------------------------------------
# _compute_delay — exponentiel
# ---------------------------------------------------------------------------


def test_compute_delay_exponentiel_base_cap() -> None:
    """Delai exponentiel : base * 2^(attempt-1), capped a cap_delay_sec."""
    cfg = RetryConfig(base_delay_sec=1.0, cap_delay_sec=30.0, max_attempts=5)
    assert _compute_delay(1, cfg, None) == 1.0
    assert _compute_delay(2, cfg, None) == 2.0
    assert _compute_delay(3, cfg, None) == 4.0
    assert _compute_delay(4, cfg, None) == 8.0
    assert _compute_delay(5, cfg, None) == 16.0
    # Au-dela du cap
    assert _compute_delay(10, cfg, None) == 30.0


def test_compute_delay_retry_after_priority() -> None:
    """R24 : Retry-After present -> ecrase l'exponentiel, capped au cap."""
    cfg = RetryConfig(base_delay_sec=1.0, cap_delay_sec=30.0, max_attempts=3)
    # Retry-After = 5s : ecrase 2s (2^1) pour attempt=2
    assert _compute_delay(2, cfg, retry_after=5.0) == 5.0
    # Retry-After > cap : capped
    assert _compute_delay(1, cfg, retry_after=120.0) == 30.0


def test_compute_delay_jitter_enabled() -> None:
    """Jitter +/-20% : le delai varie entre 80% et 120% du nominal."""
    cfg = RetryConfig(base_delay_sec=10.0, cap_delay_sec=100.0, max_attempts=3, jitter_enabled=True)
    # 100 tirages — tous dans [8, 12]
    for _ in range(100):
        d = _compute_delay(1, cfg, None)
        assert 8.0 <= d <= 12.0


def test_jitter_enabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var WINCORP_LLM_RETRY_JITTER = '1' -> True, sinon False."""
    monkeypatch.setenv("WINCORP_LLM_RETRY_JITTER", "1")
    assert _jitter_enabled_from_env() is True
    monkeypatch.setenv("WINCORP_LLM_RETRY_JITTER", "0")
    assert _jitter_enabled_from_env() is False
    monkeypatch.delenv("WINCORP_LLM_RETRY_JITTER", raising=False)
    assert _jitter_enabled_from_env() is False


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------


def test_parse_retry_after_seconds_response_headers() -> None:
    """Retry-After en secondes, via response.headers."""

    class FakeResp:
        headers = {"Retry-After": "5"}

    class FakeExc(Exception):
        response = FakeResp()

    result = _parse_retry_after(FakeExc(), cap=30.0)
    assert result == 5.0


def test_parse_retry_after_ms_priority() -> None:
    """EC36 : Retry-After-Ms prioritaire sur Retry-After."""

    class FakeResp:
        headers = {"Retry-After": "10", "Retry-After-Ms": "2500"}

    class FakeExc(Exception):
        response = FakeResp()

    result = _parse_retry_after(FakeExc(), cap=30.0)
    assert result == 2.5


def test_parse_retry_after_capped_to_cap() -> None:
    """EC35 : Retry-After > cap -> capped."""

    class FakeResp:
        headers = {"Retry-After": "120"}

    class FakeExc(Exception):
        response = FakeResp()

    result = _parse_retry_after(FakeExc(), cap=30.0)
    assert result == 30.0


def test_parse_retry_after_absent_returns_none() -> None:
    """Pas de headers -> None."""
    assert _parse_retry_after(Exception("plain"), cap=30.0) is None


def test_parse_retry_after_invalid_returns_none() -> None:
    """Valeur invalide -> None."""

    class FakeResp:
        headers = {"Retry-After": "not-a-number-or-date"}

    class FakeExc(Exception):
        response = FakeResp()

    assert _parse_retry_after(FakeExc(), cap=30.0) is None


def test_parse_retry_after_direct_attribute() -> None:
    """Headers directement sur l'exception (sans response)."""

    class FakeExc(Exception):
        headers = {"Retry-After": "3"}

    assert _parse_retry_after(FakeExc(), cap=30.0) == 3.0


def test_parse_retry_after_ms_invalid_falls_back_to_seconds() -> None:
    """Retry-After-Ms invalide -> tente Retry-After seconds."""

    class FakeResp:
        headers = {"Retry-After": "4", "Retry-After-Ms": "garbage"}

    class FakeExc(Exception):
        response = FakeResp()

    assert _parse_retry_after(FakeExc(), cap=30.0) == 4.0


# ---------------------------------------------------------------------------
# RetryWrapper integration
# ---------------------------------------------------------------------------


def test_r23_non_retryable_propagates_immediatly() -> None:
    """R23 : erreur terminal (401) propagee sans retry."""

    class AuthenticationError(Exception):
        pass

    mock_model = MagicMock()
    mock_model.invoke.side_effect = AuthenticationError("401")
    wrapped = RetryWrapper(mock_model, RetryConfig(max_attempts=3)).wrap()

    with pytest.raises(AuthenticationError):
        wrapped.invoke("prompt")
    # 1 seul appel
    assert mock_model.invoke.call_count == 1


def test_r25_max_attempts_raises_retry_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """R25/EC33 : max_attempts atteint -> RetryExhaustedError chainee."""

    class RateLimitError(Exception):
        pass

    mock_model = MagicMock()
    mock_model.invoke.side_effect = RateLimitError("429")
    cfg = RetryConfig(base_delay_sec=0.001, cap_delay_sec=0.01, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg).wrap()

    # Monkeypatch time.sleep pour tests rapides
    sleep_calls = []
    monkeypatch.setattr("wincorp_odin.llm.retry.time.sleep", lambda d: sleep_calls.append(d))

    with pytest.raises(RetryExhaustedError) as excinfo:
        wrapped.invoke("prompt")
    assert excinfo.value.attempts == 3
    assert excinfo.value.last_error_class == "RateLimitError"
    assert isinstance(excinfo.value.__cause__, RateLimitError)
    # 3 appels (1 initial + 2 retries)
    assert mock_model.invoke.call_count == 3
    # 2 sleeps (entre attempts 1-2 et 2-3)
    assert len(sleep_calls) == 2


def test_retry_succeeds_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry reussi au 2e essai -> retour resultat."""

    class RateLimitError(Exception):
        pass

    mock_model = MagicMock()
    mock_model.invoke.side_effect = [RateLimitError("429"), "ok"]
    cfg = RetryConfig(base_delay_sec=0.001, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg).wrap()

    monkeypatch.setattr("wincorp_odin.llm.retry.time.sleep", lambda d: None)

    result = wrapped.invoke("prompt")
    assert result == "ok"
    assert mock_model.invoke.call_count == 2


def test_retry_respects_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC34 : Retry-After respecte."""

    class RateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("429")

            class _Resp:
                headers = {"Retry-After": "5"}

            self.response = _Resp()

    mock_model = MagicMock()
    mock_model.invoke.side_effect = [RateLimitError(), "ok"]
    cfg = RetryConfig(base_delay_sec=1.0, cap_delay_sec=30.0, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg).wrap()

    sleep_calls = []
    monkeypatch.setattr("wincorp_odin.llm.retry.time.sleep", lambda d: sleep_calls.append(d))

    wrapped.invoke("prompt")
    # Le delai doit venir du header, pas de l'exponentiel
    assert sleep_calls == [5.0]


def test_retry_wrapper_delegates_attributes() -> None:
    """RetryWrapper delegue bind_tools et autres."""
    mock_model = MagicMock()
    mock_model.bind_tools = MagicMock(return_value="bound")
    wrapped = RetryWrapper(mock_model, RetryConfig()).wrap()
    assert wrapped.bind_tools("x") == "bound"


def test_retry_async_non_retryable_propagates() -> None:
    """Async : terminal propage sans retry."""
    import asyncio

    class AuthenticationError(Exception):
        pass

    mock_model = MagicMock()

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> Any:
        raise AuthenticationError("401")

    mock_model.ainvoke = fake_ainvoke
    wrapped = RetryWrapper(mock_model, RetryConfig(max_attempts=3)).wrap()

    with pytest.raises(AuthenticationError):
        asyncio.run(wrapped.ainvoke("prompt"))


def test_retry_async_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async : retry puis succes."""
    import asyncio

    class RateLimitError(Exception):
        pass

    call_count = {"n": 0}

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> str:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RateLimitError("429")
        return "ok"

    mock_model = MagicMock()
    mock_model.ainvoke = fake_ainvoke
    cfg = RetryConfig(base_delay_sec=0.001, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg).wrap()

    # monkeypatch asyncio.sleep pour tests rapides
    async def noop(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", noop)

    result = asyncio.run(wrapped.ainvoke("prompt"))
    assert result == "ok"
    assert call_count["n"] == 2


def test_retry_async_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async : retry exhausted."""
    import asyncio

    class RateLimitError(Exception):
        pass

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> Any:
        raise RateLimitError("429")

    mock_model = MagicMock()
    mock_model.ainvoke = fake_ainvoke
    cfg = RetryConfig(base_delay_sec=0.001, max_attempts=2)
    wrapped = RetryWrapper(mock_model, cfg).wrap()

    async def noop(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", noop)

    with pytest.raises(RetryExhaustedError) as excinfo:
        asyncio.run(wrapped.ainvoke("prompt"))
    assert excinfo.value.attempts == 2


def test_retry_config_defaults() -> None:
    """RetryConfig defaults correspondent a la spec §23.5."""
    cfg = RetryConfig()
    assert cfg.base_delay_sec == 1.0
    assert cfg.cap_delay_sec == 30.0
    assert cfg.max_attempts == 3
    assert cfg.jitter_enabled is False


def test_parse_retry_after_response_without_headers() -> None:
    """Response present mais headers=None -> fallback direct."""

    class FakeResp:
        headers = None

    class FakeExc(Exception):
        response = FakeResp()
        # Fallback direct pour tester la branche headers is None dans la boucle
        headers = {"Retry-After": "2"}

    assert _parse_retry_after(FakeExc(), cap=30.0) == 2.0


def test_parse_retry_after_rfc_date() -> None:
    """Retry-After format RFC 7231 date -> delai positif."""
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    future = datetime.now(UTC) + timedelta(seconds=10)
    date_header = format_datetime(future)

    class FakeResp:
        headers = {"Retry-After": date_header}

    class FakeExc(Exception):
        response = FakeResp()

    result = _parse_retry_after(FakeExc(), cap=30.0)
    assert result is not None
    # ~10s d'ecart (tolerance 2s)
    assert 8.0 <= result <= 12.0


# ---------------------------------------------------------------------------
# PR-018 — Validation RetryConfig a la creation (__post_init__)
# ---------------------------------------------------------------------------


def test_retry_config_rejects_invalid_max_attempts() -> None:
    """PR-018 : max_attempts < 1 -> ValueError avec message actionnable FR."""
    with pytest.raises(ValueError, match="max_attempts doit etre >= 1"):
        RetryConfig(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts doit etre >= 1"):
        RetryConfig(max_attempts=-5)


def test_retry_config_rejects_zero_base_delay() -> None:
    """PR-018 : base_delay_sec <= 0 -> ValueError."""
    with pytest.raises(ValueError, match="base_delay_sec doit etre > 0"):
        RetryConfig(base_delay_sec=0.0)
    with pytest.raises(ValueError, match="base_delay_sec doit etre > 0"):
        RetryConfig(base_delay_sec=-1.0)


def test_retry_config_rejects_cap_below_base() -> None:
    """PR-018 : cap_delay_sec < base_delay_sec -> ValueError."""
    with pytest.raises(ValueError, match="cap_delay_sec"):
        RetryConfig(base_delay_sec=10.0, cap_delay_sec=5.0)


def test_retry_config_valid_boundary_values() -> None:
    """PR-018 : les valeurs limites valides passent (max_attempts=1, cap=base)."""
    cfg = RetryConfig(max_attempts=1, base_delay_sec=1.0, cap_delay_sec=1.0)
    assert cfg.max_attempts == 1
    assert cfg.base_delay_sec == 1.0
    assert cfg.cap_delay_sec == 1.0


# ---------------------------------------------------------------------------
# PR-013 — __setattr__ delegation sur _RetryWrapped
# ---------------------------------------------------------------------------


def test_retry_setattr_delegates_to_model() -> None:
    """PR-013 : wrapped.some_attr = X -> setattr sur model."""
    mock_model = MagicMock()
    wrapped = RetryWrapper(mock_model, RetryConfig()).wrap()
    wrapped.callbacks = ["cb1", "cb2"]
    assert mock_model.callbacks == ["cb1", "cb2"]


def test_retry_setattr_slot_stays_internal() -> None:
    """PR-013 : l'assignation d'un attribut __slots__ ne fuit PAS vers model."""
    mock_model = MagicMock()
    wrapped = RetryWrapper(mock_model, RetryConfig()).wrap()
    # _model est dans __slots__, ne doit pas etre delegue
    new_model = MagicMock()
    wrapped._model = new_model
    assert wrapped._model is new_model


# ---------------------------------------------------------------------------
# PR-016 — probe HALF_OPEN ne doit pas consommer max_attempts retry
# ---------------------------------------------------------------------------


def test_half_open_probe_consumes_only_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-016 : si breaker est HALF_OPEN, retry ne fait qu'UNE tentative."""
    from wincorp_odin.llm.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
    )

    class RateLimitError(Exception):
        pass

    # Breaker en HALF_OPEN des le depart
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
    breaker._state = CircuitState.HALF_OPEN

    mock_model = MagicMock()
    mock_model.invoke.side_effect = RateLimitError("429")
    cfg = RetryConfig(base_delay_sec=0.001, cap_delay_sec=0.01, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg, breaker_ref=breaker).wrap()

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "wincorp_odin.llm.retry.time.sleep", lambda d: sleep_calls.append(d)
    )

    with pytest.raises(RetryExhaustedError):
        wrapped.invoke("prompt")

    # Une seule tentative reelle (pas de retry en HALF_OPEN)
    assert mock_model.invoke.call_count == 1
    # Aucun sleep declenche
    assert sleep_calls == []


def test_closed_breaker_retry_uses_full_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-016 : breaker CLOSED, retry fait max_attempts tentatives normalement."""
    from wincorp_odin.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    class RateLimitError(Exception):
        pass

    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=100))
    mock_model = MagicMock()
    mock_model.invoke.side_effect = RateLimitError("429")
    cfg = RetryConfig(base_delay_sec=0.001, cap_delay_sec=0.01, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg, breaker_ref=breaker).wrap()

    monkeypatch.setattr("wincorp_odin.llm.retry.time.sleep", lambda d: None)

    with pytest.raises(RetryExhaustedError):
        wrapped.invoke("prompt")
    # max_attempts = 3 tentatives
    assert mock_model.invoke.call_count == 3


def test_state_transition_during_retry_stops_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-016 : breaker bascule HALF_OPEN pendant retry -> stop."""
    from wincorp_odin.llm.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
    )

    class RateLimitError(Exception):
        pass

    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=10))
    # Breaker demarre CLOSED
    assert breaker.state == CircuitState.CLOSED

    call_count = {"n": 0}

    def failing_invoke(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        # Apres la 1ere tentative, on simule passage en HALF_OPEN
        if call_count["n"] == 1:
            breaker._state = CircuitState.HALF_OPEN
        raise RateLimitError("429")

    mock_model = MagicMock()
    mock_model.invoke.side_effect = failing_invoke
    cfg = RetryConfig(base_delay_sec=0.001, cap_delay_sec=0.01, max_attempts=5)
    wrapped = RetryWrapper(mock_model, cfg, breaker_ref=breaker).wrap()

    monkeypatch.setattr("wincorp_odin.llm.retry.time.sleep", lambda d: None)

    with pytest.raises(RetryExhaustedError):
        wrapped.invoke("prompt")
    # La bascule HALF_OPEN apres call 1 stoppe la boucle -> 1 seul call
    assert mock_model.invoke.call_count == 1


def test_retry_without_breaker_ref_unchanged() -> None:
    """PR-016 : sans breaker_ref, comportement inchange (compat)."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = "ok"
    wrapped = RetryWrapper(mock_model, RetryConfig()).wrap()
    assert wrapped.invoke("prompt") == "ok"


def test_half_open_probe_async_single_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-016 async : ainvoke stoppe aussi si breaker HALF_OPEN."""
    import asyncio

    from wincorp_odin.llm.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
    )

    class RateLimitError(Exception):
        pass

    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
    breaker._state = CircuitState.HALF_OPEN

    call_count = {"n": 0}

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        raise RateLimitError("429")

    mock_model = MagicMock()
    mock_model.ainvoke = fake_ainvoke
    cfg = RetryConfig(base_delay_sec=0.001, max_attempts=3)
    wrapped = RetryWrapper(mock_model, cfg, breaker_ref=breaker).wrap()

    async def noop(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", noop)

    with pytest.raises(RetryExhaustedError):
        asyncio.run(wrapped.ainvoke("prompt"))
    assert call_count["n"] == 1
