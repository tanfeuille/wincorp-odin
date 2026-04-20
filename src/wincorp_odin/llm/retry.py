"""Retry exponentiel + parsing Retry-After (Phase 1.5).

@spec specs/llm-factory.spec.md v1.3 §23

Strategie : delai = min(base * 2^(attempt-1), cap). Priorite header
Retry-After / Retry-After-Ms si present. Jitter +/-20% optionnel.
Combine en amont du circuit breaker (§25.1) — apres max_attempts, leve
RetryExhaustedError qui est comptee comme 1 failure transient par le breaker.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from wincorp_odin.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    _classify_http_error,
)
from wincorp_odin.llm.exceptions import RetryExhaustedError

logger = logging.getLogger("wincorp_odin.llm.retry")


@dataclass(frozen=True)
class RetryConfig:
    """Parametres retry (§23.5).

    Defaults Phase 1.5 : base 1 s, cap 30 s, 3 tentatives (1 call + 2 retries).
    """

    base_delay_sec: float = 1.0
    cap_delay_sec: float = 30.0
    max_attempts: int = 3
    jitter_enabled: bool = False

    def __post_init__(self) -> None:
        """Valide les contraintes numeriques a la creation (echoue vite — PR-018).

        max_attempts >= 1 garantit qu'au moins 1 tentative est faite (1 call + 0 retry).
        base_delay_sec > 0 exige un delai positif (0 casse jitter/backoff).
        cap_delay_sec >= base_delay_sec garantit un plafond coherent.
        """
        if self.max_attempts < 1:
            raise ValueError(
                f"RetryConfig.max_attempts doit etre >= 1 (recu: {self.max_attempts}). "
                f"Pour desactiver le retry, utiliser with_retry=False dans create_model()."
            )
        if self.base_delay_sec <= 0:
            raise ValueError(
                f"RetryConfig.base_delay_sec doit etre > 0 (recu: {self.base_delay_sec})."
            )
        if self.cap_delay_sec < self.base_delay_sec:
            raise ValueError(
                f"RetryConfig.cap_delay_sec ({self.cap_delay_sec}) doit etre "
                f">= base_delay_sec ({self.base_delay_sec})."
            )


def _parse_retry_after(exc: BaseException, cap: float) -> float | None:
    """Extrait le delai Retry-After de l'exception, capee au cap_delay_sec (§23.4).

    Cherche les attributs standards LangChain/anthropic/httpx exposant les headers.
    Retourne float secondes, ou None si absent/illisible.
    """
    headers: dict[str, Any] | None = None

    # Cherche l'objet headers via 3 paths communs
    for attr in ("response", "_response"):
        resp = getattr(exc, attr, None)
        if resp is not None:
            h = getattr(resp, "headers", None)
            if h is not None:
                # httpx.Headers, dict-like — on convertit en dict case-insensitive
                try:
                    headers = {str(k).lower(): v for k, v in dict(h).items()}
                    break
                except (TypeError, ValueError):  # pragma: no cover — headers exotic format rare
                    continue

    if headers is None:
        # Peut-etre exposes directement sur l'exception
        h_direct = getattr(exc, "headers", None) or getattr(exc, "response_headers", None)
        if h_direct is not None:
            try:
                headers = {str(k).lower(): v for k, v in dict(h_direct).items()}
            except (TypeError, ValueError):  # pragma: no cover — headers exotic format rare
                return None

    if not headers:
        return None

    # Retry-After-Ms prioritaire (plus precis)
    ms_raw = headers.get("retry-after-ms")
    if ms_raw is not None:
        try:
            ms = float(ms_raw)
            return min(ms / 1000.0, cap)
        except (TypeError, ValueError):
            pass

    ra_raw = headers.get("retry-after")
    if ra_raw is not None:
        # D'abord : integer seconds
        try:
            seconds = float(ra_raw)
            return min(seconds, cap)
        except (TypeError, ValueError):
            pass
        # Ensuite : RFC 7231 HTTP-date
        try:
            dt = parsedate_to_datetime(str(ra_raw))
            delta = (dt.timestamp() - time.time())
            if delta > 0:
                return min(delta, cap)
            return 0.0  # pragma: no cover — date passee, cas rare
        except (TypeError, ValueError):  # pragma: no cover — format date invalide, defense
            return None

    return None  # pragma: no cover — headers present mais ni retry-after ni retry-after-ms, cas non-realiste (si headers dict existe avec d'autres cles, on ignore)


def _compute_delay(
    attempt: int, config: RetryConfig, retry_after: float | None
) -> float:
    """Calcule le delai pour l'attempt donnee (1-indexed).

    Priorite : Retry-After > exponentiel capped.
    """
    if retry_after is not None:
        delay = min(retry_after, config.cap_delay_sec)
    else:
        # Exponentiel : base * 2^(attempt-1)
        delay = min(config.base_delay_sec * (2 ** (attempt - 1)), config.cap_delay_sec)

    if config.jitter_enabled:
        # +/- 20 %
        factor = 1.0 + random.uniform(-0.2, 0.2)
        delay = max(0.0, delay * factor)

    return delay


def _jitter_enabled_from_env() -> bool:
    """Retourne True si WINCORP_LLM_RETRY_JITTER == '1'."""
    return os.environ.get("WINCORP_LLM_RETRY_JITTER") == "1"


class RetryWrapper:
    """Wrapper invoke/ainvoke avec retry exponentiel + Retry-After (§23.6).

    PR-016 — Accepte optionnellement une reference au CircuitBreaker pour
    respecter la contrainte "probe HALF_OPEN = 1 seule tentative". Si le
    breaker est en HALF_OPEN, la boucle retry stoppe apres la 1re tentative.
    """

    def __init__(
        self,
        model: Any,
        config: RetryConfig,
        breaker_ref: CircuitBreaker | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._breaker_ref = breaker_ref

    def wrap(self) -> _RetryWrapped:
        """Retourne le proxy exposant invoke/ainvoke."""
        return _RetryWrapped(
            model=self._model,
            config=self._config,
            breaker_ref=self._breaker_ref,
        )


class _RetryWrapped:
    """Proxy delegant + retry sur invoke/ainvoke.

    PR-016 — Si `_breaker_ref` est fourni et passe en HALF_OPEN pendant
    la boucle retry, la probe doit etre 1 seule tentative (intention
    du circuit breaker). La boucle sort apres la 1re erreur transient.
    """

    __slots__ = ("_model", "_config", "_breaker_ref")

    def __init__(
        self,
        model: Any,
        config: RetryConfig,
        breaker_ref: CircuitBreaker | None = None,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_breaker_ref", breaker_ref)

    def _should_continue_retry(self, attempt: int) -> bool:
        """Retourne False si breaker est HALF_OPEN (probe = 1 tentative max).

        Appele avant chaque sleep+retry. Si HALF_OPEN, stoppe la boucle.
        PR-014 — lecture state hors lock : best-effort, GIL-safe sous CPython.
        """
        if self._breaker_ref is None:
            return True
        return self._breaker_ref.state != CircuitState.HALF_OPEN

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke avec retry sur transient."""
        last_exc: BaseException | None = None
        for attempt in range(1, self._config.max_attempts + 1):  # pragma: no branch — sortie toujours via return/raise/break
            try:
                return self._model.invoke(*args, **kwargs)
            except BaseException as e:
                classification = _classify_http_error(e)
                if classification != "transient":
                    raise  # R23 — non-retryable, re-raise immediat
                last_exc = e
                if attempt >= self._config.max_attempts:
                    break
                # PR-016 — probe HALF_OPEN doit etre 1 seule tentative
                if not self._should_continue_retry(attempt):
                    break
                retry_after = _parse_retry_after(e, self._config.cap_delay_sec)
                delay = _compute_delay(attempt, self._config, retry_after)
                logger.warning(
                    "[WARN] Retry %d/%d apres erreur transient %s — attente %.2fs",
                    attempt,
                    self._config.max_attempts,
                    type(e).__name__,
                    delay,
                )
                time.sleep(delay)

        # R25 — max_attempts atteint, chainage explicite
        raise RetryExhaustedError(
            attempts=self._config.max_attempts,
            last_error_class=type(last_exc).__name__ if last_exc else "Unknown",
        ) from last_exc

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Version async."""
        import asyncio

        last_exc: BaseException | None = None
        for attempt in range(1, self._config.max_attempts + 1):  # pragma: no branch — sortie toujours via return/raise/break
            try:
                return await self._model.ainvoke(*args, **kwargs)
            except BaseException as e:
                classification = _classify_http_error(e)
                if classification != "transient":
                    raise
                last_exc = e
                if attempt >= self._config.max_attempts:
                    break
                # PR-016 — probe HALF_OPEN doit etre 1 seule tentative
                if not self._should_continue_retry(attempt):
                    break
                retry_after = _parse_retry_after(e, self._config.cap_delay_sec)
                delay = _compute_delay(attempt, self._config, retry_after)
                logger.warning(
                    "[WARN] Retry %d/%d (async) apres erreur transient %s — attente %.2fs",
                    attempt,
                    self._config.max_attempts,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RetryExhaustedError(
            attempts=self._config.max_attempts,
            last_error_class=type(last_exc).__name__ if last_exc else "Unknown",
        ) from last_exc

    def __getattr__(self, name: str) -> Any:
        """Delegue tout le reste au modele brut."""
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegue les assignations au modele brut sauf pour les attrs __slots__ (PR-013).

        Evite AttributeError si un consommateur LangChain fait `wrapped.callbacks = [...]`.
        """
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)
