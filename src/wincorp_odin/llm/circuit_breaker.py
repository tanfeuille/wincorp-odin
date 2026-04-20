"""Circuit breaker thread-safe closed/half-open/open (Phase 1.4).

@spec specs/llm-factory.spec.md v1.3 §22

Protection contre pannes fournisseur (rate-limit sustained, API down).
Inspire DeerFlow llm_error_handling_middleware avec adaptations Yggdrasil :
- isolation dure (pas d'import SDK anthropic direct, classification par nom)
- thread-safety via threading.Lock par instance
- config YAML-driven (models.yaml champ circuit_breaker optionnel par modele)
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from wincorp_odin.llm.exceptions import CircuitOpenError


class CircuitState(Enum):
    """Etat du breaker (§22.2)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Parametres breaker (§22.4).

    Defaults Phase 1.4 : 5 echecs avant ouverture, 60 s avant probe half-open.
    Surcharges globales via env vars WINCORP_LLM_CB_FAILURE_THRESHOLD /
    WINCORP_LLM_CB_RECOVERY_SEC si non fixe par le YAML.
    """

    failure_threshold: int = 5
    recovery_timeout_sec: float = 60.0

    @classmethod
    def from_env_or_default(cls) -> CircuitBreakerConfig:
        """Lit les surcharges env vars (bornees) sinon defauts du dataclass."""
        ft_raw = os.environ.get("WINCORP_LLM_CB_FAILURE_THRESHOLD")
        rt_raw = os.environ.get("WINCORP_LLM_CB_RECOVERY_SEC")
        ft = cls.failure_threshold
        rt = cls.recovery_timeout_sec
        if ft_raw:
            with contextlib.suppress(ValueError):
                ft = max(1, int(ft_raw))
        if rt_raw:
            with contextlib.suppress(ValueError):
                rt = max(0.1, float(rt_raw))
        return cls(failure_threshold=ft, recovery_timeout_sec=rt)


# Noms de classes indicateurs (detection par nom — isolation SDK, §22.3)
_TRANSIENT_CLASS_NAMES = frozenset({
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",  # 5xx chez Anthropic SDK
    "ServiceUnavailableError",
    "InternalServerError",
    "ConnectionError",
    "TimeoutError",
    "RetryExhaustedError",  # on considere une chaine retry-exhausted comme 1 transient
})

_TERMINAL_CLASS_NAMES = frozenset({
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "UnprocessableEntityError",
    "ModelAuthenticationError",
})

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_TERMINAL_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


def _classify_http_error(exc: BaseException) -> str:
    """Retourne 'transient' | 'terminal' | 'unknown' (§22.3).

    Classification par nom de classe (isolation SDK) puis par status_code si present.
    """
    cls_name = type(exc).__name__
    if cls_name in _TRANSIENT_CLASS_NAMES:
        return "transient"
    if cls_name in _TERMINAL_CLASS_NAMES:
        return "terminal"

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in _TRANSIENT_STATUS_CODES:
            return "transient"
        if status in _TERMINAL_STATUS_CODES:
            return "terminal"

    # Fallback : builtin TimeoutError / ConnectionError
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"

    return "unknown"


class CircuitBreaker:
    """Breaker par modele — thread-safe (§22).

    Usage :
        cb = CircuitBreaker(name="sonnet", config=CircuitBreakerConfig())
        wrapped = cb.wrap(model_instance)  # proxy .invoke/.ainvoke proteges
    """

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self._lock = threading.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight: bool = False

    @property
    def state(self) -> CircuitState:
        """Etat courant (snapshot).

        PR-014 — Lecture sans lock, best-effort pour observation/monitoring.
        Sous CPython GIL, l'acces a une reference objet reste atomique.
        Sous free-threading (Python 3.13t+), cette lecture peut retourner
        une valeur transitoire entre deux transitions. Les *transitions*
        (on_success/on_failure/before_call) restent protegees par self._lock.
        """
        return self._state

    @property
    def failure_count(self) -> int:
        """Nombre d'echecs transient consecutifs.

        PR-014 — Lecture sans lock, memes contraintes que `state` :
        best-effort pour observation, les *ecritures* sont sous self._lock.
        """
        return self._failure_count

    def _transition_to_open(self) -> None:
        """Passage OPEN + timestamp. Appele sous lock."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_probe_in_flight = False

    def _transition_to_closed(self) -> None:
        """Retour CLOSED + reset compteur. Appele sous lock."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._half_open_probe_in_flight = False

    def _should_attempt_reset(self) -> bool:
        """Vrai si le timeout de recovery est ecoule depuis l'ouverture."""
        if self._opened_at is None:  # pragma: no cover — invariant : OPEN sans timestamp impossible
            return False
        return (time.monotonic() - self._opened_at) >= self.config.recovery_timeout_sec

    def _acquire_probe_slot(self) -> bool:
        """Tente de reserver le slot probe HALF_OPEN. Appele sous lock.

        Retourne True si le thread a l'exclusivite du probe, False sinon.
        """
        if self._half_open_probe_in_flight:
            return False
        self._half_open_probe_in_flight = True
        return True

    def before_call(self) -> None:
        """Verifie si l'appel est autorise.

        Leve CircuitOpenError si le breaker refuse la requete.
        Sinon, met eventuellement l'etat a HALF_OPEN + reserve le probe.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Transition OPEN -> HALF_OPEN, reserve le probe pour ce thread
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_probe_in_flight = True
                    return
                # Encore dans la periode de recovery
                retry_after = max(
                    0.0,
                    self.config.recovery_timeout_sec
                    - (time.monotonic() - (self._opened_at or 0.0)),
                )
                raise CircuitOpenError(
                    model_name=self.name, retry_after_sec=retry_after
                )

            # HALF_OPEN — un probe deja en vol, les autres refuses
            if not self._acquire_probe_slot():
                raise CircuitOpenError(
                    model_name=self.name,
                    retry_after_sec=0.0,
                    message=(
                        f"[ERREUR] Circuit breaker en mode demi-ouvert pour modele "
                        f"'{self.name}' — probe deja en vol, requete concurrente refusee. "
                        f"Reessayer apres resultat de la probe."
                    ),
                )

    def on_success(self) -> None:
        """Un appel a reussi — reset si HALF_OPEN, no-op si CLOSED."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
            # CLOSED : pas de decrement du failure_count (design R20).

    def on_failure(self, exc: BaseException) -> None:
        """Un appel a leve — classifie et transitionne si transient."""
        classification = _classify_http_error(exc)

        if classification == "terminal":
            # R22 — terminal ne compte pas, juste relacher le probe eventuel
            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    # On ne bascule pas OPEN sur terminal (le breaker n'est pas concerne)
                    self._half_open_probe_in_flight = False
            return

        # transient OU unknown -> compte comme failure
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Probe echoue -> retour OPEN
                self._transition_to_open()
                return

            # CLOSED -> increment + eventuel passage OPEN
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def wrap(self, model: Any) -> _CircuitBreakerWrapped:
        """Retourne un proxy qui intercepte invoke/ainvoke."""
        return _CircuitBreakerWrapped(model=model, breaker=self)


class _CircuitBreakerWrapped:
    """Proxy thread-safe delegant les attributs au modele brut, sauf invoke/ainvoke.

    Usage interne — instancier via CircuitBreaker.wrap(model).
    """

    __slots__ = ("_model", "_breaker")

    def __init__(self, model: Any, breaker: CircuitBreaker) -> None:
        # __setattr__ via object pour eviter boucle __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_breaker", breaker)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept invoke — verifie breaker avant, update apres."""
        self._breaker.before_call()
        try:
            result = self._model.invoke(*args, **kwargs)
        except BaseException as e:
            self._breaker.on_failure(e)
            raise
        self._breaker.on_success()
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept ainvoke — idem async."""
        self._breaker.before_call()
        try:
            result = await self._model.ainvoke(*args, **kwargs)
        except BaseException as e:
            self._breaker.on_failure(e)
            raise
        self._breaker.on_success()
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegue tout le reste au modele brut (bind_tools, stream, etc.)."""
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegue les assignations au modele brut sauf pour les attrs __slots__ (PR-013).

        Evite AttributeError si un consommateur LangChain fait `wrapped.callbacks = [...]`.
        Les attributs de __slots__ restent gerables en interne via object.__setattr__.
        """
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)
