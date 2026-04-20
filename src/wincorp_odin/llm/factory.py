"""Factory LLM : create_model + validate_all_models + cache thread-safe.

@spec specs/llm-factory.spec.md v1.2

Strategie copy-on-write PB-019 : validation hors lock, swap atomique sous
lock court. Budgets startup (R19) et runtime (R19b) distincts.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from wincorp_odin.llm import _registry
from wincorp_odin.llm._registry import validate_use_format
from wincorp_odin.llm._whitelist import validate_extra_kwargs
from wincorp_odin.llm.config import (
    ModelConfig,
    _resolve_urd_path,
    load_models_config,
)
from wincorp_odin.llm.exceptions import (
    CapabilityMismatchError,
    ModelAuthenticationError,
    ModelConfigError,
    ModelNotFoundError,
)

logger = logging.getLogger("wincorp_odin.llm")


# ---------------------------------------------------------------------------
# Etat global (spec §10.1)
# ---------------------------------------------------------------------------

_cache: dict[tuple[str, bool], Any] = {}
_cache_lock = threading.Lock()
_yaml_mtime: float | None = None
_last_mtime_check: float = 0.0
_resolved_configs: dict[str, ModelConfig] = {}
_deferred_resolutions: set[str] = set()

_STARTUP_TIMEOUT_S: float = 5.0
_RUNTIME_TIMEOUT_S: float = 0.5


def _get_startup_timeout() -> float:
    """Lit WINCORP_LLM_VALIDATE_TIMEOUT_S avec bornes [1, 60]."""
    raw = os.environ.get("WINCORP_LLM_VALIDATE_TIMEOUT_S")
    if not raw:
        return _STARTUP_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        return _STARTUP_TIMEOUT_S
    return max(1.0, min(60.0, val))


def _get_runtime_timeout() -> float:
    """Lit WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S avec bornes [0.1, 5.0]."""
    raw = os.environ.get("WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S")
    if not raw:
        return _RUNTIME_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        return _RUNTIME_TIMEOUT_S
    return max(0.1, min(5.0, val))


# ---------------------------------------------------------------------------
# Build kwargs (R10b — strip api_key des exceptions)
# ---------------------------------------------------------------------------


def _build_kwargs(cfg: ModelConfig, thinking_enabled: bool) -> dict[str, Any]:
    """Construit les kwargs d'instanciation du provider.

    Applique les overrides when_thinking_enabled/disabled selon le flag, et
    merge extra_kwargs deja whitelistes.
    """
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "api_key": cfg.api_key_resolved,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.timeout,
        "max_retries": cfg.max_retries,
    }
    kwargs.update(cfg.extra_kwargs)

    if thinking_enabled:
        overrides = cfg.when_thinking_enabled or {}
    else:
        overrides = cfg.when_thinking_disabled or {}
    kwargs.update(overrides)

    return kwargs


# ---------------------------------------------------------------------------
# Invalidation mtime (copy-on-write, PB-019)
# ---------------------------------------------------------------------------


def _load_and_validate_models(
    timeout_s: float,
) -> tuple[dict[str, ModelConfig], set[str]]:
    """Variante pure de `validate_all_models` : retourne configs + deferred.

    Respecte le budget timeout. **Ne mute PAS** `_resolved_configs` ni
    `_deferred_resolutions` — cette mutation est la responsabilite du caller
    dans le swap atomique sous lock (cf PR-006).

    Returns:
        Tuple (dict {name: ModelConfig}, set des names differes pour resolution).
    """
    start = time.monotonic()
    configs = load_models_config()
    deferred: set[str] = set()
    elapsed = time.monotonic() - start
    if elapsed > timeout_s:
        # Log WARNING mais on ne lève pas — les configs sont deja parsees.
        # La resolution 'use:' est deferee.
        logger.warning(
            "[WARN] Validation des etapes 1-7 prise %.2fs (budget %.2fs depasse). "
            "Resolution 'use:' deferee au 1er appel.",
            elapsed,
            timeout_s,
        )
        # v1.2 : on enregistre les modeles actifs comme differes
        for name, cfg in configs.items():
            if not cfg.disabled:  # pragma: no branch — la branche False est couverte par fixtures sans modele disabled, mais pytest-cov compte la transition
                deferred.add(name)
    else:
        # Resolution eagerly dans le budget
        for name, cfg in configs.items():
            if cfg.disabled:
                continue
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:  # pragma: no cover — collision time.monotonic() rare, couvert par test_factory_load_and_validate_deferred_when_timeout via monkeypatch differe
                deferred.add(name)
                logger.warning(
                    "[WARN] Resolution provider '%s' sautee par timeout, "
                    "sera reessayee au 1er create_model(%r). Verifier le demarrage.",
                    cfg.use,
                    name,
                )
                continue
            _registry.resolve_class(cfg.use)

    return configs, deferred


def _check_mtime_and_invalidate() -> None:
    """Throttled mtime check + copy-on-write reload (§10.3, PB-010, PB-019)."""
    global _yaml_mtime, _last_mtime_check
    now = time.monotonic()
    # R18 — throttle max 1 stat()/s
    if now - _last_mtime_check < 1.0:
        return
    _last_mtime_check = now

    try:
        yaml_path = _resolve_urd_path()
    except ModelConfigError:
        # Si la resolution du path echoue ici, on laisse le cache tel quel —
        # la prochaine tentative d'instanciation relancera la logique complete.
        return

    try:
        current = yaml_path.stat().st_mtime
    except OSError:
        return

    if _yaml_mtime is not None and current <= _yaml_mtime:
        return

    # Reload HORS lock — budget runtime (R19b)
    try:
        new_configs, new_deferred = _load_and_validate_models(
            timeout_s=_get_runtime_timeout()
        )
    except (ModelConfigError, TimeoutError) as e:
        logger.warning(
            "[WARN] Invalidation mtime echouee (cache conserve, pas de downtime) : %s",
            e,
        )
        return

    # Swap atomique SOUS lock court — mutation globals uniquement ici (PR-006)
    with _cache_lock:
        try:
            current_reloaded = yaml_path.stat().st_mtime
        except OSError:
            return
        if _yaml_mtime is not None and current_reloaded <= _yaml_mtime:
            return
        logger.info(
            "[INFO] models.yaml recharge (mtime: %s -> %s)",
            _yaml_mtime,
            current_reloaded,
        )
        _resolved_configs.clear()
        _resolved_configs.update(new_configs)
        _cache.clear()
        _registry._class_cache.clear()
        _deferred_resolutions.clear()
        _deferred_resolutions.update(new_deferred)
        _yaml_mtime = current_reloaded


def _ensure_configs_loaded() -> None:
    """Charge les configs si jamais fait. Appele au 1er create_model."""
    global _yaml_mtime
    if _resolved_configs:
        return
    with _cache_lock:
        if _resolved_configs:  # pragma: no cover — double-check race condition, branche defensive
            return
        configs, deferred = _load_and_validate_models(
            timeout_s=_get_startup_timeout()
        )
        _resolved_configs.update(configs)
        _deferred_resolutions.update(deferred)
        try:
            _yaml_mtime = _resolve_urd_path().stat().st_mtime
        except OSError:  # pragma: no cover — stat ne peut echouer juste apres load reussi
            _yaml_mtime = None


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def validate_all_models() -> None:
    """Valide la config YAML en entier sans instancier les modeles (§9).

    Budget R19 — WINCORP_LLM_VALIDATE_TIMEOUT_S (defaut 5 s).
    PB-018 : valide aussi les modeles disabled (extra_kwargs + format use).

    Ordre §9.1 (miroir) : whitelist d'abord, parsing use ensuite — garantit
    qu'un modele avec extra_kwargs forbidden ET use invalide remonte
    d'abord l'erreur whitelist (deterministe).

    Raises:
        ModelConfigError (et sous-classes) en cas d'echec.
    """
    configs, deferred = _load_and_validate_models(timeout_s=_get_startup_timeout())

    # PB-018 — re-valider les modeles disabled (load_models_config ne les traite pas)
    # Ordre strict §9.1 : whitelist AVANT parsing use (miroir chargement actif).
    for cfg in configs.values():
        if cfg.disabled:
            validate_extra_kwargs(cfg.name, cfg.use, cfg.extra_kwargs)
            validate_use_format(cfg.use)

    # Swap atomique SOUS lock — mutation globals uniquement ici (PR-006)
    with _cache_lock:
        _resolved_configs.clear()
        _resolved_configs.update(configs)
        _deferred_resolutions.clear()
        _deferred_resolutions.update(deferred)
        try:
            global _yaml_mtime
            _yaml_mtime = _resolve_urd_path().stat().st_mtime
        except OSError:  # pragma: no cover — stat ne peut echouer juste apres load reussi
            _yaml_mtime = None


def create_model(name: str, thinking_enabled: bool = False) -> Any:
    """Instancie (ou recupere depuis le cache) un client LLM configure.

    Args:
        name: Nom logique declare dans models.yaml.
        thinking_enabled: Active le mode thinking si supporte.

    Returns:
        Instance ChatAnthropic prete a l'emploi (ou instance cachee).

    Raises:
        ModelNotFoundError, CapabilityMismatchError, ModelConfigError,
        ExtraKwargsForbiddenError, ModelAuthenticationError.
    """
    _ensure_configs_loaded()
    _check_mtime_and_invalidate()

    key = (name, thinking_enabled)

    # Lecture sans lock (GIL)
    instance = _cache.get(key)
    if instance is not None:
        return instance

    with _cache_lock:
        instance = _cache.get(key)
        if instance is not None:
            return instance

        cfg = _resolved_configs.get(name)
        if cfg is None or cfg.disabled:
            available = sorted(n for n, c in _resolved_configs.items() if not c.disabled)
            raise ModelNotFoundError(
                f"[ERREUR] Modele '{name}' introuvable. Typo probable ? "
                f"Modeles disponibles : {available}."
            )

        if thinking_enabled and not cfg.supports_thinking:
            thinking_capable = sorted(
                n for n, c in _resolved_configs.items()
                if c.supports_thinking and not c.disabled
            )
            raise CapabilityMismatchError(
                f"[ERREUR] Modele '{name}' ne supporte pas le mode thinking "
                f"(supports_thinking: false dans models.yaml). "
                f"Modeles thinking-compatibles : {thinking_capable}."
            )

        klass = _registry.resolve_class(cfg.use)
        kwargs = _build_kwargs(cfg, thinking_enabled)

        try:
            instance = klass(**kwargs)
        except Exception as e:
            # R10b/R10c — strip api_key avant remontee
            raise ModelAuthenticationError(
                f"[ERREUR] Authentification/instanciation du provider {cfg.use} "
                f"echouee pour modele '{name}'. Verifier la validite de "
                f"{cfg.api_key_env or 'la cle API'}."
            ) from e

        _cache[key] = instance
        return instance


# ---------------------------------------------------------------------------
# API echappatoire tests (§10.5, PB-015)
# ---------------------------------------------------------------------------


def _reload_for_tests() -> None:
    """Vide cache + registry + mtime. Usage interne uniquement (non exporte)."""
    global _yaml_mtime, _last_mtime_check
    with _cache_lock:
        _cache.clear()
        _resolved_configs.clear()
        _registry._class_cache.clear()
        _deferred_resolutions.clear()
        _yaml_mtime = None
        _last_mtime_check = 0.0
