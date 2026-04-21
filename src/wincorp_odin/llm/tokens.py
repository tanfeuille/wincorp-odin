"""Token usage middleware : tracking input/output + cout EUR (Phase 1.6).

@spec specs/llm-factory.spec.md v1.3.3 §24

Intercepte invoke/ainvoke, lit result.usage_metadata (contrat LangChain
standard), emit vers un sink pluggable (log/file/supabase).
Pricing depuis wincorp-urd/referentiels/models.yaml champ pricing: par modele.

Phase 1.6b (v1.3.3) — SupabaseSink reel : insert dans table llm_usage via
PostgREST (httpx direct, pas de dep supabase-py). Batching queue flush
5s ou 10 events. Erreurs swallowed (R28 — observabilite ne casse pas le caller).
"""
from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from wincorp_odin.llm.exceptions import TokenTrackingError

logger = logging.getLogger("wincorp_odin.llm.tokens")


# ---------------------------------------------------------------------------
# Dataclass event (§24.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricingConfig:
    """Tarifs provider (§24.3). Source : models.yaml pricing: par modele.

    Valeurs en EUR pour 1 million de tokens.
    """

    input_per_million_eur: float
    output_per_million_eur: float

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Cout EUR = input*rate_in + output*rate_out."""
        cost_in = (input_tokens / 1_000_000) * self.input_per_million_eur
        cost_out = (output_tokens / 1_000_000) * self.output_per_million_eur
        return round(cost_in + cost_out, 6)


@dataclass(frozen=True)
class TokenUsageEvent:
    """Evenement d'usage tokens emis apres chaque appel LLM reussi (§24.2).

    timestamp : unix epoch (time.time()).
    session_id / agent_name / client_id : metadata optionnelle (tracing).
    """

    timestamp: float
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_eur: float
    session_id: str | None = None
    agent_name: str | None = None
    client_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Dict JSON-compatible pour sinks."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Sinks (§24.4)
# ---------------------------------------------------------------------------


class TokenSink(Protocol):
    """Contrat d'un sink : recoit les events et les persiste."""

    def emit(self, event: TokenUsageEvent) -> None:
        ...  # pragma: no cover — Protocol signature only


class LogSink:
    """Sink par defaut : logger.info JSON compact."""

    def emit(self, event: TokenUsageEvent) -> None:
        try:
            payload = json.dumps(event.to_json_dict(), ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover — event dataclass est toujours JSON-safe
            payload = str(event.to_json_dict())
        logger.info("llm_usage_event %s", payload)


class FileSink:
    """Sink JSONL append thread-safe."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        # PR-015 — mkdir best-effort : si FS protege / disque plein / permission
        # denied, on log warning et on continue. Les emit() ulterieurs echoueront
        # aussi mais seront swallowed (R28) pour ne jamais casser le caller.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.warning(
                "[WARN] FileSink: impossible de creer le repertoire %s (%s). "
                "Les events tokens seront perdus. Verifier WINCORP_LLM_TOKEN_SINK_FILE.",
                self._path.parent,
                e,
            )

    def emit(self, event: TokenUsageEvent) -> None:
        line = json.dumps(event.to_json_dict(), ensure_ascii=False)
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            # R28 / EC40 — erreur disque ne casse pas le caller
            logger.warning(
                "[WARN] FileSink echec ecriture %s : %s. Event perdu (observabilite n'interrompt pas la prod).",
                self._path,
                e,
            )


class SupabaseSink:
    """Sink Supabase reel (Phase 1.6b) — insert batch vers table llm_usage.

    Architecture pragmatique : appel REST PostgREST direct via httpx (pas de
    dep `supabase-py`). Ecritures en batch pour eviter de saturer le provider
    LLM avec un POST reseau apres chaque invoke.

    Env vars obligatoires :
      - SUPABASE_URL
      - SUPABASE_SERVICE_ROLE_KEY

    Flush automatique :
      - Toutes les `flush_interval_sec` secondes (defaut 5s) via timer daemon.
      - Ou quand la queue atteint `batch_size` events (defaut 10).

    R28 — les erreurs reseau/HTTP sont swallowed (WARNING log), le caller
    n'est JAMAIS impacte. Un batch qui echoue est perdu, pas retente
    (l'observabilite n'interrompt pas la prod).
    """

    DEFAULT_BATCH_SIZE = 10
    DEFAULT_FLUSH_INTERVAL_SEC = 5.0
    _HTTP_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        url: str | None = None,
        service_role_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_url = url if url is not None else os.environ.get("SUPABASE_URL")
        resolved_key = (
            service_role_key
            if service_role_key is not None
            else os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )
        if not resolved_url or not resolved_key:
            raise ValueError(
                "[ERREUR] SupabaseSink requiert SUPABASE_URL et "
                "SUPABASE_SERVICE_ROLE_KEY. Variables d'environnement absentes. "
                "Fallback : WINCORP_LLM_TOKEN_SINK=log (defaut) ou =file."
            )
        self._url = resolved_url.rstrip("/")
        self._key = resolved_key
        self._endpoint = f"{self._url}/rest/v1/llm_usage"
        self._headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        self._batch_size = batch_size
        self._flush_interval_sec = flush_interval_sec
        self._queue: list[TokenUsageEvent] = []
        self._lock = threading.Lock()
        self._http = http_client if http_client is not None else httpx.Client(
            timeout=self._HTTP_TIMEOUT_SEC
        )
        self._owns_http = http_client is None
        self._stopping = False
        # Timer daemon pour flush periodique.
        self._timer: threading.Timer | None = None
        self._schedule_timer()
        # Best-effort flush a l'arret du process (event perdus sinon).
        atexit.register(self._atexit_flush)

    def _schedule_timer(self) -> None:
        """Planifie un flush dans flush_interval_sec (thread daemon)."""
        if self._stopping:
            return
        self._timer = threading.Timer(self._flush_interval_sec, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        """Callback timer — flush puis reschedule."""
        try:
            self.flush()
        finally:
            self._schedule_timer()

    def _atexit_flush(self) -> None:
        """Flush final au shutdown process."""
        self._stopping = True
        if self._timer is not None:
            self._timer.cancel()
        self.flush()
        if self._owns_http:
            with contextlib.suppress(Exception):  # best-effort close shutdown
                self._http.close()

    def _event_to_payload(self, event: TokenUsageEvent) -> dict[str, Any]:
        """Mappe TokenUsageEvent -> row llm_usage (timestamp ISO-8601 UTC)."""
        iso_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.timestamp))
        return {
            "timestamp": iso_ts,
            "model_name": event.model_name,
            "session_id": event.session_id,
            "agent_name": event.agent_name,
            "client_id": event.client_id,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "total_tokens": event.total_tokens,
            "cost_eur": event.cost_eur,
        }

    def emit(self, event: TokenUsageEvent) -> None:
        """Enqueue l'event, flush si seuil batch_size atteint."""
        with self._lock:
            self._queue.append(event)
            should_flush = len(self._queue) >= self._batch_size
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """POST la queue vers Supabase. Swallow toute erreur (R28)."""
        with self._lock:
            if not self._queue:
                return
            pending = self._queue
            self._queue = []
        payload = [self._event_to_payload(e) for e in pending]
        try:
            resp = self._http.post(
                self._endpoint, headers=self._headers, json=payload
            )
            if resp.status_code >= 400:
                logger.warning(
                    "[WARN] SupabaseSink HTTP %d : %s. Batch de %d events perdu.",
                    resp.status_code,
                    resp.text[:200],
                    len(payload),
                )
        except (httpx.HTTPError, OSError) as e:
            logger.warning(
                "[WARN] SupabaseSink erreur reseau : %s. Batch de %d events perdu "
                "(observabilite n'interrompt pas la prod).",
                e,
                len(payload),
            )


def get_sink(name: str | None = None) -> TokenSink:
    """Factory sink (§24.4, R28).

    Resolution :
      - Si `name` explicite fourni, utilise cette valeur.
      - Sinon, lit env var WINCORP_LLM_TOKEN_SINK (defaut 'log').

    Leve TokenTrackingError si valeur invalide.
    """
    effective = name if name is not None else os.environ.get("WINCORP_LLM_TOKEN_SINK", "log")
    effective = effective.lower().strip()

    if effective == "log":
        return LogSink()
    if effective == "file":
        default_path = Path(__file__).resolve().parents[3] / ".token_usage" / "events.jsonl"
        override = os.environ.get("WINCORP_LLM_TOKEN_SINK_FILE")
        path = Path(override).resolve() if override else default_path
        return FileSink(path=path)
    if effective == "supabase":
        return SupabaseSink()

    raise TokenTrackingError(
        f"[ERREUR] Valeur WINCORP_LLM_TOKEN_SINK invalide : '{effective}'. "
        f"Valeurs acceptees : 'log' (defaut), 'file', 'supabase'. "
        f"Verifier la configuration env."
    )


# ---------------------------------------------------------------------------
# Wrapper (§24.5)
# ---------------------------------------------------------------------------


@dataclass
class TokenTrackingContext:
    """Metadata optionnelle attachee aux events (tracing cross-agent).

    Chaque session/agent/client pose son context via set_context() avant le call.
    """

    session_id: str | None = None
    agent_name: str | None = None
    client_id: str | None = None


# PR-019 — ContextVar pour propagation cross-agent avec isolation asyncio native.
# ContextVar respecte les frontieres de coroutines (Phase 2) tout en gardant
# l'isolation thread sous execution synchrone. Remplace threading.local().
_ctx_var: ContextVar[TokenTrackingContext | None] = ContextVar(
    "wincorp_odin_llm_tracking_context", default=None
)

# Conserve pour retrocompat du test public `test_default_context_when_not_set`.
# Lu par _current_context() en fallback si le ContextVar n'a jamais ete set.
_context_local = threading.local()


def set_context(
    session_id: str | None = None,
    agent_name: str | None = None,
    client_id: str | None = None,
) -> None:
    """Fixe le context pour les prochains events emis (isolation asyncio via ContextVar)."""
    ctx = TokenTrackingContext(
        session_id=session_id,
        agent_name=agent_name,
        client_id=client_id,
    )
    _ctx_var.set(ctx)
    # Miroir thread-local pour retrocompat introspection tests.
    _context_local.ctx = ctx


def clear_context() -> None:
    """Reset le context courant."""
    _ctx_var.set(None)
    if hasattr(_context_local, "ctx"):
        delattr(_context_local, "ctx")


def _current_context() -> TokenTrackingContext:
    """Retourne le context courant (ou vide si non defini)."""
    ctx = _ctx_var.get()
    if ctx is not None:
        return ctx
    # Fallback thread-local (retrocompat test manipulant _context_local directement).
    ctx_tl = getattr(_context_local, "ctx", None)
    if ctx_tl is None:
        return TokenTrackingContext()
    return ctx_tl


def _extract_usage_metadata(result: Any) -> tuple[int, int, int]:
    """Extrait (input, output, total) depuis result.usage_metadata (R26).

    Retourne (0, 0, 0) + WARNING si absent.
    """
    usage = getattr(result, "usage_metadata", None)
    if usage is None:
        logger.warning(
            "[WARN] result.usage_metadata absent sur appel LLM — event tokens a zero."
        )
        return 0, 0, 0
    try:
        input_t = int(usage.get("input_tokens", 0))
        output_t = int(usage.get("output_tokens", 0))
        total_t = int(usage.get("total_tokens", input_t + output_t))
    except (AttributeError, TypeError, ValueError):
        logger.warning("[WARN] usage_metadata format inattendu — event tokens a zero.")
        return 0, 0, 0
    return input_t, output_t, total_t


class TokenTrackingWrapper:
    """Wrapper invoke/ainvoke emettant TokenUsageEvent apres chaque call reussi."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        pricing: PricingConfig | None,
        sink: TokenSink | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._pricing = pricing
        self._sink = sink if sink is not None else get_sink()

    def wrap(self) -> _TokenTrackingWrapped:
        """Retourne le proxy exposant invoke/ainvoke."""
        return _TokenTrackingWrapped(
            model=self._model,
            model_name=self._model_name,
            pricing=self._pricing,
            sink=self._sink,
        )


class _TokenTrackingWrapped:
    """Proxy — emet un event apres chaque invoke/ainvoke reussi."""

    __slots__ = ("_model", "_model_name", "_pricing", "_sink")

    def __init__(
        self,
        model: Any,
        model_name: str,
        pricing: PricingConfig | None,
        sink: TokenSink,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(self, "_pricing", pricing)
        object.__setattr__(self, "_sink", sink)

    def _emit(self, input_t: int, output_t: int, total_t: int) -> None:
        """Construit + emit l'event (swallow sink errors)."""
        cost = 0.0
        if self._pricing is not None:
            cost = self._pricing.compute_cost(input_t, output_t)
        else:
            logger.warning(
                "[WARN] Pricing manquant pour modele '%s' — cost_eur=0.0.",
                self._model_name,
            )
        ctx = _current_context()
        event = TokenUsageEvent(
            timestamp=time.time(),
            model_name=self._model_name,
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=total_t,
            cost_eur=cost,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            client_id=ctx.client_id,
        )
        try:
            self._sink.emit(event)
        except Exception as e:
            # R28 — observabilite ne casse pas la prod
            logger.warning(
                "[WARN] Sink emit echec pour modele '%s' : %s. Event perdu.",
                self._model_name,
                e,
            )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept + emit event post-success."""
        result = self._model.invoke(*args, **kwargs)
        input_t, output_t, total_t = _extract_usage_metadata(result)
        self._emit(input_t, output_t, total_t)
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Version async."""
        result = await self._model.ainvoke(*args, **kwargs)
        input_t, output_t, total_t = _extract_usage_metadata(result)
        self._emit(input_t, output_t, total_t)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegue tout le reste."""
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegue les assignations au modele brut sauf pour les attrs __slots__ (PR-013).

        Evite AttributeError si un consommateur LangChain fait `wrapped.callbacks = [...]`.
        """
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)
