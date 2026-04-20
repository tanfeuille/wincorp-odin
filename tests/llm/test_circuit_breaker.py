"""Tests circuit breaker thread-safe (Phase 1.4).

@spec specs/llm-factory.spec.md v1.3 §22
"""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from wincorp_odin.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    _classify_http_error,
)
from wincorp_odin.llm.exceptions import CircuitOpenError, RetryExhaustedError

# ---------------------------------------------------------------------------
# R22 — Classification erreurs
# ---------------------------------------------------------------------------


def test_r22_transient_by_class_name() -> None:
    """R22 : classification transient par nom de classe (isolation SDK)."""

    class RateLimitError(Exception):
        pass

    assert _classify_http_error(RateLimitError("429")) == "transient"


def test_r22_terminal_by_class_name() -> None:
    """R22 : classification terminal par nom de classe."""

    class AuthenticationError(Exception):
        pass

    assert _classify_http_error(AuthenticationError("401")) == "terminal"


def test_r22_transient_by_status_code() -> None:
    """R22 : classification transient via attribut status_code."""

    class SomeApiError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.status_code = code

    assert _classify_http_error(SomeApiError(429)) == "transient"
    assert _classify_http_error(SomeApiError(503)) == "transient"


def test_r22_terminal_by_status_code() -> None:
    """R22 : classification terminal via status_code."""

    class SomeApiError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"HTTP {code}")
            self.status_code = code

    assert _classify_http_error(SomeApiError(401)) == "terminal"
    assert _classify_http_error(SomeApiError(404)) == "terminal"


def test_r22_builtin_timeout_is_transient() -> None:
    """R22 : TimeoutError builtin = transient."""
    assert _classify_http_error(TimeoutError("timeout")) == "transient"


def test_r22_builtin_connection_is_transient() -> None:
    """R22 : ConnectionError builtin = transient."""
    assert _classify_http_error(ConnectionError("refused")) == "transient"


def test_r22_status_401_is_terminal_before_fallback() -> None:
    """R22 branche : status_code=401 classe terminal (boucle if status not in transient)."""

    class SomeErr(Exception):
        def __init__(self) -> None:
            super().__init__("401")
            self.status_code = 401

    # Le nom SomeErr n'est ni dans TRANSIENT ni dans TERMINAL_CLASS_NAMES
    # -> on passe a la branche status_code, qui matche terminal
    assert _classify_http_error(SomeErr()) == "terminal"


def test_r22_status_600_goes_to_unknown() -> None:
    """status_code hors listes -> fallback isinstance check."""

    class SomeErr(Exception):
        def __init__(self) -> None:
            super().__init__("???")
            self.status_code = 600  # hors listes

    # Pas isinstance de TimeoutError/ConnectionError -> unknown
    assert _classify_http_error(SomeErr()) == "unknown"


def test_r22_status_unknown_code_with_timeout_subclass() -> None:
    """TimeoutError subclass avec status_code hors listes -> transient via isinstance fallback."""

    class WeirdTimeout(TimeoutError):
        def __init__(self) -> None:
            super().__init__("weird")
            self.status_code = 600  # hors listes

    # Nom "WeirdTimeout" hors TRANSIENT_CLASS_NAMES/TERMINAL_CLASS_NAMES,
    # status_code hors listes -> on passe au fallback isinstance TimeoutError -> transient
    assert _classify_http_error(WeirdTimeout()) == "transient"


def test_r22_unknown_exception_returns_unknown() -> None:
    """R22 : exception inconnue sans status_code -> 'unknown' (traitee comme transient par breaker)."""
    assert _classify_http_error(RuntimeError("custom")) == "unknown"


def test_r22_retry_exhausted_counts_as_transient() -> None:
    """R22 : RetryExhaustedError compte comme 1 transient (§23.7)."""
    exc = RetryExhaustedError(attempts=3, last_error_class="RateLimitError")
    assert _classify_http_error(exc) == "transient"


# ---------------------------------------------------------------------------
# R20 — Transitions CLOSED -> OPEN
# ---------------------------------------------------------------------------


def test_r20_closed_to_open_after_threshold() -> None:
    """R20 : `failure_threshold` transient successifs -> OPEN."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=3))
    assert cb.state == CircuitState.CLOSED
    for _ in range(3):
        cb.on_failure(TimeoutError("x"))
    assert cb.state == CircuitState.OPEN


def test_r20_terminal_does_not_count() -> None:
    """R22 : erreur terminal n'incrmente PAS le compteur."""

    class AuthenticationError(Exception):
        pass

    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=2))
    for _ in range(5):
        cb.on_failure(AuthenticationError("401"))
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_ec28_open_circuit_blocks_call() -> None:
    """EC28 : before_call() leve CircuitOpenError en mode OPEN."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=60.0))
    cb.on_failure(TimeoutError("x"))
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError) as excinfo:
        cb.before_call()
    assert excinfo.value.model_name == "sonnet"
    assert excinfo.value.retry_after_sec >= 0.0


# ---------------------------------------------------------------------------
# R21 — OPEN -> HALF_OPEN -> CLOSED / OPEN
# ---------------------------------------------------------------------------


def test_r21_half_open_probe_success_returns_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """R21 : probe HALF_OPEN reussie -> retour CLOSED + reset."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=0.1))
    cb.on_failure(TimeoutError("x"))
    assert cb.state == CircuitState.OPEN

    # Avancer le temps au-dela du recovery
    base = time.monotonic()
    monkeypatch.setattr("wincorp_odin.llm.circuit_breaker.time.monotonic", lambda: base + 10.0)

    cb.before_call()  # HALF_OPEN + reserve probe
    assert cb.state == CircuitState.HALF_OPEN

    cb.on_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_r21_half_open_probe_failure_returns_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC30 : probe HALF_OPEN echouee -> retour OPEN."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=0.1))
    cb.on_failure(TimeoutError("x"))

    base = time.monotonic()
    monkeypatch.setattr("wincorp_odin.llm.circuit_breaker.time.monotonic", lambda: base + 10.0)

    cb.before_call()
    assert cb.state == CircuitState.HALF_OPEN
    cb.on_failure(TimeoutError("y"))
    assert cb.state == CircuitState.OPEN


def test_ec31_half_open_concurrent_second_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC31 : HALF_OPEN + 2 requetes concurrentes -> 1 probe, l'autre CircuitOpenError."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=0.1))
    cb.on_failure(TimeoutError("x"))

    base = time.monotonic()
    monkeypatch.setattr("wincorp_odin.llm.circuit_breaker.time.monotonic", lambda: base + 10.0)

    # 1er thread reserve le probe
    cb.before_call()
    assert cb.state == CircuitState.HALF_OPEN

    # 2eme thread doit etre refuse
    with pytest.raises(CircuitOpenError):
        cb.before_call()


# ---------------------------------------------------------------------------
# R22b — Thread safety
# ---------------------------------------------------------------------------


def test_r22b_concurrent_failures_counter_consistent() -> None:
    """R22b : 100 threads incrementent failure_count de maniere coherente."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1000))

    def worker() -> None:
        cb.on_failure(TimeoutError("x"))

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.failure_count == 100
    assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Wrapper — integration invoke
# ---------------------------------------------------------------------------


def test_wrapper_invoke_delegates_and_returns() -> None:
    """Le wrapper delegue invoke + retourne le resultat brut."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=3))
    mock_model = MagicMock()
    mock_model.invoke.return_value = "result"
    wrapped = cb.wrap(mock_model)

    result = wrapped.invoke("prompt")
    assert result == "result"
    mock_model.invoke.assert_called_once_with("prompt")
    assert cb.state == CircuitState.CLOSED


def test_wrapper_invoke_raises_transient_and_counts() -> None:
    """Le wrapper compte les erreurs transient."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=2))
    mock_model = MagicMock()
    mock_model.invoke.side_effect = TimeoutError("timeout")
    wrapped = cb.wrap(mock_model)

    with pytest.raises(TimeoutError):
        wrapped.invoke("prompt")
    assert cb.failure_count == 1

    with pytest.raises(TimeoutError):
        wrapped.invoke("prompt")
    # 2 echecs = threshold, breaker doit etre OPEN
    assert cb.state == CircuitState.OPEN


def test_wrapper_invoke_open_blocks_immediately() -> None:
    """Wrapper + breaker OPEN -> CircuitOpenError sans toucher au modele."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=60.0))
    cb.on_failure(TimeoutError("x"))
    mock_model = MagicMock()
    wrapped = cb.wrap(mock_model)

    with pytest.raises(CircuitOpenError):
        wrapped.invoke("prompt")
    mock_model.invoke.assert_not_called()


def test_wrapper_delegates_other_attributes() -> None:
    """Wrapper delegue bind_tools, stream, etc. (non interceptes)."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig())
    mock_model = MagicMock()
    mock_model.bind_tools = MagicMock(return_value="bound")
    wrapped = cb.wrap(mock_model)

    assert wrapped.bind_tools("tools") == "bound"


def test_wrapper_async_invoke_transient() -> None:
    """Test ainvoke delegation + classification."""
    import asyncio

    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1))
    mock_model = MagicMock()

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> str:
        raise TimeoutError("async-timeout")

    mock_model.ainvoke = fake_ainvoke
    wrapped = cb.wrap(mock_model)

    with pytest.raises(TimeoutError):
        asyncio.run(wrapped.ainvoke("prompt"))
    assert cb.state == CircuitState.OPEN


def test_wrapper_async_invoke_success() -> None:
    """ainvoke success -> pas de failure, on_success appele."""
    import asyncio

    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=0.01))
    mock_model = MagicMock()

    async def fake_ainvoke(*args: Any, **kwargs: Any) -> str:
        return "async-ok"

    mock_model.ainvoke = fake_ainvoke

    # Mise en OPEN
    cb.on_failure(TimeoutError("x"))
    time.sleep(0.05)

    wrapped = cb.wrap(mock_model)
    result = asyncio.run(wrapped.ainvoke("prompt"))
    assert result == "async-ok"
    assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Config env vars override
# ---------------------------------------------------------------------------


def test_config_from_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars CB_* ecrasent les defauts du dataclass."""
    monkeypatch.setenv("WINCORP_LLM_CB_FAILURE_THRESHOLD", "7")
    monkeypatch.setenv("WINCORP_LLM_CB_RECOVERY_SEC", "45.5")
    cfg = CircuitBreakerConfig.from_env_or_default()
    assert cfg.failure_threshold == 7
    assert cfg.recovery_timeout_sec == 45.5


def test_config_from_env_invalid_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var invalide -> defaut (pas de crash)."""
    monkeypatch.setenv("WINCORP_LLM_CB_FAILURE_THRESHOLD", "abc")
    monkeypatch.setenv("WINCORP_LLM_CB_RECOVERY_SEC", "xyz")
    cfg = CircuitBreakerConfig.from_env_or_default()
    assert cfg.failure_threshold == CircuitBreakerConfig.failure_threshold
    assert cfg.recovery_timeout_sec == CircuitBreakerConfig.recovery_timeout_sec


def test_config_from_env_empty_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var absente -> defaut."""
    monkeypatch.delenv("WINCORP_LLM_CB_FAILURE_THRESHOLD", raising=False)
    monkeypatch.delenv("WINCORP_LLM_CB_RECOVERY_SEC", raising=False)
    cfg = CircuitBreakerConfig.from_env_or_default()
    assert cfg.failure_threshold == CircuitBreakerConfig.failure_threshold


def test_circuit_breaker_setattr_delegates_to_model() -> None:
    """PR-013 : wrapped.callbacks = [...] -> setattr delegue vers model."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig())
    mock_model = MagicMock()
    wrapped = cb.wrap(mock_model)
    wrapped.callbacks = ["cb1", "cb2"]
    assert mock_model.callbacks == ["cb1", "cb2"]


def test_circuit_breaker_setattr_slot_stays_internal() -> None:
    """PR-013 : assignation __slots__ reste en interne (pas de fuite vers model)."""
    cb = CircuitBreaker("sonnet", CircuitBreakerConfig())
    mock_model = MagicMock()
    wrapped = cb.wrap(mock_model)
    new_model = MagicMock()
    wrapped._model = new_model
    assert wrapped._model is new_model


def test_half_open_terminal_releases_probe_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """EC32 adapte : terminal en HALF_OPEN ne bascule pas OPEN mais libere le slot.

    Le breaker ne doit pas decider de rester HALF_OPEN locked sur une 401.
    Nom de classe 'AuthenticationError' dans _TERMINAL_CLASS_NAMES (§22.3).
    """

    # Nom de classe doit matcher _TERMINAL_CLASS_NAMES pour etre classe 'terminal'
    class AuthenticationError(Exception):
        pass

    cb = CircuitBreaker("sonnet", CircuitBreakerConfig(failure_threshold=1, recovery_timeout_sec=0.1))
    cb.on_failure(TimeoutError("x"))

    base = time.monotonic()
    monkeypatch.setattr("wincorp_odin.llm.circuit_breaker.time.monotonic", lambda: base + 10.0)

    cb.before_call()
    assert cb.state == CircuitState.HALF_OPEN

    # Terminal en HALF_OPEN — on ne bascule pas OPEN, le probe slot est libere
    cb.on_failure(AuthenticationError("401"))
    assert cb.state == CircuitState.HALF_OPEN
    # Un nouveau probe doit etre reservable (slot libere)
    cb.before_call()  # ne doit pas lever
