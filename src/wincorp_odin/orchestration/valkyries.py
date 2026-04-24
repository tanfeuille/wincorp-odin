"""Registre declaratif des roles produit d'agents (valkyries).

@spec specs/valkyries.spec.md v1.4

Charge wincorp-urd/referentiels/valkyries.yaml (source unique).
Expose ValkyrieConfig (dataclass frozen hashable), loader + middleware
LangChain ValkyrieToolGuard + factory create_valkyrie_chat.

Strategy copy-on-write PB-019 : validation hors lock, swap atomique sous
lock court. Aucune fenetre visible contrairement au pattern clear+update.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

from wincorp_odin.llm.config import load_models_config
from wincorp_odin.llm.factory import create_model

logger = logging.getLogger("wincorp_odin.orchestration.valkyries")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_STARTUP_TIMEOUT_S: float = 5.0
_THROTTLE_INTERVAL_S: float = 1.0

# Whitelist statique des tools bloqueables (v1.0)
_BLOCKED_TOOLS_WHITELIST: frozenset[str] = frozenset(
    {"task", "shell", "bash", "write", "edit", "read"}
)

# Regex snake_case valkyrie name
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Versions supportees
_SUPPORTED_VERSIONS = {1}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ValkyrieConfigError(ValueError):
    """Erreur de configuration valkyrie (YAML invalide ou schema casse)."""


class ValkyrieNotFoundError(ValkyrieConfigError):
    """Name de valkyrie absent du YAML."""


class ValkyrieModelRefError(ValkyrieConfigError):
    """Modele reference non resolvable dans models.yaml (inconnu ou disabled)."""


class ValkyrieRangeError(ValkyrieConfigError):
    """Champ numerique hors plage [min, max]."""


# ---------------------------------------------------------------------------
# Dataclass ValkyrieConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValkyrieConfig:
    """Configuration resolue d'une valkyrie (post-validation YAML).

    Attributes:
        name: Cle du role (brynhildr, sigrun, thor).
        description: Texte libre < 200 chars.
        timeout_seconds: Duree max de la tache en secondes [30, 1800].
        max_turns: Nombre max de tours LLM [1, 500].
        max_concurrent: Concurrence max [1, 20].
        model: Reference name dans models.yaml (valide au load).
        blocked_tools: Tools interdits au runtime (frozenset, immuable).
        extra_kwargs: Parametres supplementaires (tuple[tuple[str, Any], ...] trie).
    """

    name: str
    description: str
    timeout_seconds: int
    max_turns: int
    max_concurrent: int
    model: str
    blocked_tools: frozenset[str]
    extra_kwargs: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialisation JSON-safe : frozenset → list triee, tuple items → dict.

        Returns:
            Dict JSON-serializable consommable par heimdall/bifrost.
        """
        return {
            "name": self.name,
            "description": self.description,
            "timeout_seconds": self.timeout_seconds,
            "max_turns": self.max_turns,
            "max_concurrent": self.max_concurrent,
            "model": self.model,
            "blocked_tools": sorted(self.blocked_tools),
            "extra_kwargs": dict(self.extra_kwargs),
        }


# ---------------------------------------------------------------------------
# Etat global (cache thread-safe, copy-on-write PB-019)
# ---------------------------------------------------------------------------

# Swap atomique : reference simple, pas de clear+update (fenetre visible eliminee)
_configs_ref: dict[str, ValkyrieConfig] | None = None
_cache_lock = threading.Lock()
_yaml_mtime: float | None = None
_last_mtime_check: float = 0.0


# ---------------------------------------------------------------------------
# Resolution chemin YAML (miroir llm/config.py)
# ---------------------------------------------------------------------------


def _find_dev_urd_path() -> Path | None:
    """Auto-detection mode dev : remonte les parents en cherchant wincorp-urd frere."""
    start = Path(__file__).resolve()
    for ancestor in start.parents[:8]:
        if (ancestor / ".git").exists():
            candidate = ancestor.parent / "wincorp-urd"
            if (candidate / "referentiels" / "valkyries.yaml").exists():
                return candidate
            return None  # pragma: no cover — git trouve mais wincorp-urd sans yaml (mode dev rare)
    return None  # pragma: no cover — uniquement si 8 ancetres sans .git (mode installed pur)


def _resolve_valkyries_yaml_path() -> Path:
    """Resolution du chemin vers valkyries.yaml (env var ou auto-detection .git).

    Returns:
        Chemin absolu vers valkyries.yaml.

    Raises:
        ValkyrieConfigError: chemin introuvable ou mode installed sans env var.
    """
    env_value = os.environ.get("WINCORP_URD_PATH")
    if env_value:
        yaml_path = Path(env_value).resolve() / "referentiels" / "valkyries.yaml"
        if not yaml_path.exists():
            raise ValkyrieConfigError(
                f"[ERREUR] Fichier valkyries.yaml introuvable. "
                f"Chemin tente : {yaml_path}. Verifier WINCORP_URD_PATH."
            )
        return yaml_path

    dev_urd = _find_dev_urd_path()
    if dev_urd is not None:
        yaml_path = dev_urd / "referentiels" / "valkyries.yaml"
        if yaml_path.exists():
            return yaml_path

    module_path = Path(__file__).resolve()
    raise ValkyrieConfigError(
        f"[ERREUR] valkyries.yaml introuvable. "
        f"Definir WINCORP_URD_PATH ou placer wincorp-urd/ frere de wincorp-odin/. "
        f"Detection depuis : {module_path}"
    )


def _get_startup_timeout() -> float:
    """Lit WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S avec bornes [1, 60]."""
    raw = os.environ.get("WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S")
    if not raw:
        return _STARTUP_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        return _STARTUP_TIMEOUT_S
    return max(1.0, min(60.0, val))


# ---------------------------------------------------------------------------
# Validation interne
# ---------------------------------------------------------------------------


def _validate_hashable_extra_kwargs(name: str, extra_kwargs: dict[str, Any]) -> None:
    """Verifie que toutes les values de extra_kwargs sont hashables.

    Types autorises : str, int, float, bool, None, tuple, frozenset.
    Types interdits : dict, list, set.

    Raises:
        ValkyrieConfigError: si une value n'est pas hashable.
    """
    forbidden_types = (dict, list, set)
    for k, v in extra_kwargs.items():
        if isinstance(v, forbidden_types):
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : extra_kwargs values doivent etre hashable "
                f"(primitives OU tuples). Cle fautive : '{k}' (type={type(v).__name__})."
            )
        # Verification hash effective
        try:
            hash(v)
        except TypeError as exc:
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : extra_kwargs values doivent etre hashable. "
                f"Cle fautive : '{k}' (type={type(v).__name__})."
            ) from exc


def _apply_defaults(
    raw_valk: dict[str, Any],
    defaults: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Applique les defaults sur les champs optionnels manquants.

    Returns:
        Dict fusionne avec defaults appliques.
    """
    merged = dict(raw_valk)
    for field in ("timeout_seconds", "max_turns", "max_concurrent", "blocked_tools"):
        if field not in merged and field in defaults:
            merged[field] = defaults[field]
    return merged


def _validate_and_build_config(
    name: str,
    raw: dict[str, Any],
    yaml_path: Path,
    models: dict[str, Any],
) -> ValkyrieConfig:
    """Valide un bloc valkyrie et construit le ValkyrieConfig.

    Ordre de validation §6.1 :
    1. name snake_case
    2. champs obligatoires
    3. types
    4. plages numeriques
    5. blocked_tools whitelist
    6. extra_kwargs hashable
    7. model resolu dans models
    8. description longueur

    Raises:
        ValkyrieConfigError, ValkyrieRangeError, ValkyrieModelRefError
    """
    # Ordre 1 : snake_case
    if not _SNAKE_CASE_RE.match(name):
        raise ValkyrieConfigError(
            f"[ERREUR] Valkyrie '{name}' : le nom doit etre en snake_case "
            f"(^[a-z][a-z0-9_]*$). Fichier : {yaml_path}"
        )

    # Ordre 2 : champs obligatoires
    required_fields = ["description", "timeout_seconds", "max_turns", "max_concurrent", "model"]
    for field in required_fields:
        if field not in raw or raw[field] is None:
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : champ obligatoire '{field}' absent ou null. "
                f"Fichier : {yaml_path}"
            )

    # Ordre 3 : types numeriques
    for int_field in ("timeout_seconds", "max_turns", "max_concurrent"):
        val = raw[int_field]
        if not isinstance(val, int):
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : '{int_field}' doit etre un entier, "
                f"recu : {type(val).__name__}. Fichier : {yaml_path}"
            )

    # Ordre 4 : plages numeriques
    timeout = int(raw["timeout_seconds"])
    if not (30 <= timeout <= 1800):
        raise ValkyrieRangeError(
            f"[ERREUR] Valkyrie '{name}' : timeout_seconds={timeout} hors plage [30, 1800].\n"
            f"         Fichier : {yaml_path}"
        )

    max_turns = int(raw["max_turns"])
    if not (1 <= max_turns <= 500):
        raise ValkyrieRangeError(
            f"[ERREUR] Valkyrie '{name}' : max_turns={max_turns} hors plage [1, 500].\n"
            f"         Fichier : {yaml_path}"
        )

    max_concurrent = int(raw["max_concurrent"])
    if not (1 <= max_concurrent <= 20):
        raise ValkyrieRangeError(
            f"[ERREUR] Valkyrie '{name}' : max_concurrent={max_concurrent} hors plage [1, 20].\n"
            f"         Fichier : {yaml_path}"
        )

    # Ordre 5 : blocked_tools whitelist
    raw_blocked = raw.get("blocked_tools", [])
    if not isinstance(raw_blocked, list):
        raise ValkyrieConfigError(
            f"[ERREUR] Valkyrie '{name}' : blocked_tools doit etre une liste. "
            f"Fichier : {yaml_path}"
        )
    for tool in raw_blocked:
        if tool not in _BLOCKED_TOOLS_WHITELIST:
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : tool '{tool}' absent de la whitelist "
                f"autorisee {sorted(_BLOCKED_TOOLS_WHITELIST)}. "
                f"Fichier : {yaml_path}"
            )

    # Ordre 6 : extra_kwargs hashable
    raw_extra = raw.get("extra_kwargs", {})
    if not isinstance(raw_extra, dict):
        raise ValkyrieConfigError(
            f"[ERREUR] Valkyrie '{name}' : extra_kwargs doit etre un dict. "
            f"Fichier : {yaml_path}"
        )
    _validate_hashable_extra_kwargs(name, raw_extra)

    # Ordre 7 : model resolu
    model_name = str(raw["model"])
    model_cfg = models.get(model_name)
    if model_cfg is None:
        available = sorted(models.keys())
        raise ValkyrieModelRefError(
            f"[ERREUR] Valkyrie '{name}' : modele '{model_name}' inconnu.\n"
            f"         Modeles disponibles (models.yaml) : {available}.\n"
            f"         Fichier : {yaml_path}"
        )
    if getattr(model_cfg, "disabled", False):
        raise ValkyrieModelRefError(
            f"[ERREUR] Valkyrie '{name}' : modele '{model_name}' est desactive "
            f"dans models.yaml (disabled: true).\n"
            f"         Reactiver le modele OU changer la reference dans valkyries.yaml.\n"
            f"         Fichier : {yaml_path}"
        )

    # Ordre 8 : description longueur
    description = str(raw["description"])
    if not (1 <= len(description) < 200):
        raise ValkyrieConfigError(
            f"[ERREUR] Valkyrie '{name}' : description doit avoir entre 1 et 199 chars, "
            f"recu : {len(description)}. Fichier : {yaml_path}"
        )

    # Construction tuple extra_kwargs trie par key
    extra_kwargs_tuple: tuple[tuple[str, Any], ...] = tuple(
        sorted(raw_extra.items(), key=lambda kv: kv[0])
    )

    return ValkyrieConfig(
        name=name,
        description=description,
        timeout_seconds=timeout,
        max_turns=max_turns,
        max_concurrent=max_concurrent,
        model=model_name,
        blocked_tools=frozenset(raw_blocked),
        extra_kwargs=extra_kwargs_tuple,
    )


def _load_and_validate_valkyries(timeout_s: float) -> dict[str, ValkyrieConfig]:
    """Fonction pure : lit YAML, valide, retourne dict configs.

    Appele HORS lock valkyries. Conforme spec §5.1 et §5.4 (ordre locks).
    Budget timeout_s : si depasse, leve ValkyrieConfigError (tout-ou-rien §5.2).

    Args:
        timeout_s: Budget en secondes pour le chargement complet.

    Returns:
        Dict {name: ValkyrieConfig} valide.

    Raises:
        ValkyrieConfigError (et sous-classes) en cas d'echec.
    """
    start = time.monotonic()

    # Resolution path (propagation directe si ValkyrieConfigError)
    yaml_path = _resolve_valkyries_yaml_path()

    # Lecture YAML
    if not yaml_path.exists():
        raise ValkyrieConfigError(
            f"[ERREUR] valkyries.yaml introuvable : {yaml_path}"
        )

    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValkyrieConfigError(
            f"[ERREUR] Lecture de {yaml_path} echouee : {e}."
        ) from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValkyrieConfigError(
            f"[ERREUR] YAML invalide dans {yaml_path} : {e}"
        ) from e

    if not isinstance(data, dict):
        raise ValkyrieConfigError(
            f"[ERREUR] Racine YAML invalide (dict attendu) : {yaml_path}"
        )

    # Check 1 : config_version
    version = data.get("config_version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValkyrieConfigError(
            f"[ERREUR] config_version {version!r} non supporte. "
            f"Mettre a jour wincorp-odin ou downgrade valkyries.yaml. "
            f"Fichier : {yaml_path}"
        )

    # Check 2 : presence valkyries dict non vide
    raw_valkyries = data.get("valkyries")
    if not isinstance(raw_valkyries, dict) or not raw_valkyries:
        raise ValkyrieConfigError(
            f"[ERREUR] aucune valkyrie declaree dans {yaml_path} "
            f"(cle 'valkyries' absente ou vide)."
        )

    # Defaults
    defaults: dict[str, Any] = {}
    raw_defaults = data.get("defaults", {})
    if isinstance(raw_defaults, dict):
        defaults = raw_defaults

    # Charge models HORS lock valkyries (§5.4 ordre locks)
    # Budget check
    elapsed = time.monotonic() - start
    if elapsed > timeout_s:
        logger.warning(
            "valkyries_load_timeout budget_s=%.3f elapsed_s=%.3f",
            timeout_s,
            elapsed,
        )
        raise ValkyrieConfigError(
            f"[ERREUR] Budget timeout {timeout_s}s depasse au chargement valkyries "
            f"({elapsed:.3f}s ecoule). Tout-ou-rien : aucune config partiellement chargee."
        )

    models = load_models_config()

    # Validation + construction configs
    result: dict[str, ValkyrieConfig] = {}
    for name, raw_valk in raw_valkyries.items():
        if not isinstance(raw_valk, dict):
            raise ValkyrieConfigError(
                f"[ERREUR] Valkyrie '{name}' : bloc YAML invalide (dict attendu). "
                f"Fichier : {yaml_path}"
            )
        merged = _apply_defaults(raw_valk, defaults, name)
        cfg = _validate_and_build_config(name, merged, yaml_path, models)
        result[name] = cfg

    return result


# ---------------------------------------------------------------------------
# Cache mtime + swap atomique (§5.1)
# ---------------------------------------------------------------------------


def _check_mtime_and_invalidate() -> None:
    """Throttle mtime check + swap atomique copy-on-write (§5.1, PB-019).

    Throttle max 1 stat()/s. Si YAML modifie : reload hors lock, swap sous lock.
    Echec reload → WARNING + cache precedent conserve (pas de downtime).
    """
    global _yaml_mtime, _last_mtime_check, _configs_ref

    now = time.monotonic()
    if now - _last_mtime_check < _THROTTLE_INTERVAL_S:
        return
    _last_mtime_check = now

    try:
        yaml_path = _resolve_valkyries_yaml_path()
    except ValkyrieConfigError:
        return

    try:
        current = yaml_path.stat().st_mtime
    except OSError:
        return

    if _yaml_mtime is not None and current <= _yaml_mtime:
        return

    # Reload HORS lock (potentiellement long)
    try:
        new_configs = _load_and_validate_valkyries(timeout_s=_get_startup_timeout())
    except Exception as e:
        logger.warning(
            "valkyries_reload_failed error=%s cache_preserved=true", e
        )
        return

    # Swap atomique SOUS lock court (§5.1)
    with _cache_lock:
        try:
            current_reloaded = yaml_path.stat().st_mtime
        except OSError:
            return
        if _yaml_mtime is not None and current_reloaded <= _yaml_mtime:
            return

        old_mtime = _yaml_mtime
        # Swap atomique : attribution simple, pas de clear+update
        _configs_ref = new_configs
        _yaml_mtime = current_reloaded

    logger.info(
        "valkyries_reloaded count=%d mtime_old=%s mtime_new=%s",
        len(new_configs),
        old_mtime,
        current_reloaded,
    )


def _ensure_configs_loaded() -> None:
    """Charge les configs si jamais fait (double-check lock pattern).

    Thread-safe : double-check sous lock (EC7). Swap atomique §5.1.
    """
    global _configs_ref, _yaml_mtime

    # Lecture sans lock (GIL safe — attribution Python atomique)
    if _configs_ref is not None:
        return

    with _cache_lock:
        # Double-check : un autre thread a peut-etre deja charge
        if _configs_ref is not None:
            return

        start = time.monotonic()
        configs = _load_and_validate_valkyries(timeout_s=_get_startup_timeout())
        elapsed = time.monotonic() - start

        try:
            yaml_path = _resolve_valkyries_yaml_path()
            _yaml_mtime = yaml_path.stat().st_mtime
        except (ValkyrieConfigError, OSError):
            _yaml_mtime = None

        # Swap atomique
        _configs_ref = configs

    logger.info(
        "valkyries_loaded count=%d duration_ms=%.0f mtime=%s",
        len(configs),
        elapsed * 1000,
        _yaml_mtime,
    )


# ---------------------------------------------------------------------------
# API publique — loader
# ---------------------------------------------------------------------------


def load_valkyrie(name: str) -> ValkyrieConfig:
    """Charge la config d'une valkyrie (cache thread-safe avec invalidation mtime).

    Args:
        name: Nom du role (brynhildr, sigrun, thor).

    Returns:
        ValkyrieConfig frozen hashable.

    Raises:
        ValkyrieNotFoundError: si name absent du YAML.
        ValkyrieConfigError: si YAML invalide.
    """
    _ensure_configs_loaded()
    _check_mtime_and_invalidate()

    # Lecture sans lock (CPython GIL : attribution dict atomique)
    configs = _configs_ref
    if configs is not None:
        cfg = configs.get(name)
        if cfg is not None:
            return cfg

    # Fallback sous lock si miss
    with _cache_lock:
        configs = _configs_ref
        if configs is not None:
            cfg = configs.get(name)
            if cfg is not None:
                return cfg
            available = sorted(configs.keys())
        else:
            available = []

    raise ValkyrieNotFoundError(
        f"[ERREUR] Valkyrie '{name}' introuvable. "
        f"Roles disponibles : {available}."
    )


def list_valkyries() -> list[str]:
    """Retourne la liste des noms de valkyries disponibles (tries alphabetique).

    Returns:
        Liste de noms tries.
    """
    _ensure_configs_loaded()
    _check_mtime_and_invalidate()
    configs = _configs_ref or {}
    return sorted(configs.keys())


def validate_all_valkyries() -> None:
    """Valide le YAML en entier. Raise ValkyrieConfigError si invalide.

    A appeler au demarrage pour un check eager. Budget WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S.

    Raises:
        ValkyrieConfigError (et sous-classes) en cas d'echec.
    """
    global _configs_ref, _yaml_mtime

    configs = _load_and_validate_valkyries(timeout_s=_get_startup_timeout())

    with _cache_lock:
        try:
            yaml_path = _resolve_valkyries_yaml_path()
            _yaml_mtime = yaml_path.stat().st_mtime
        except (ValkyrieConfigError, OSError):
            _yaml_mtime = None
        # Swap atomique
        _configs_ref = configs


# ---------------------------------------------------------------------------
# _StreamToolBuffer — accumulation inter-chunks (spec §5.5 v1.4)
# ---------------------------------------------------------------------------


class _StreamToolBuffer:
    """Accumule les fragments tool_use par index jusqu'au bloc complet.

    Strategie :
    - Chaque bloc tool_use est identifie par son ``index`` (cle de deduplication).
    - Accumule ``name`` (peut arriver fragmente sur plusieurs chunks) +
      ``input`` (partial_json_delta cumule comme chaine).
    - Flush un bloc = emet le bloc accumule apres evaluation filtre/pass-through.
    - Triggers de flush : nouveau bloc d'index different, fin de stream explicite
      via ``flush()``.
    - Les blocs non-tool_use (text, etc.) sont retournes immediatement, apres
      flush des blocs tool_use pending d'index precedents (ordre FIFO preserve).
    - Blocs incomplets en fin de stream (name absent) : emis comme blocs malformes
      (meme comportement que ``_filter_content_block`` sur name=None) + WARNING.

    Multi-provider : le buffer fonctionne pour tout provider emettant des blocs
    fragmentes (Anthropic, OpenAI-compat via tool_calls). Pas d'hypothese sur le
    schema interne — seul ``index`` est utilise pour l'accumulation tool_use. Pour
    les formats OpenAI-compat (``tool_calls`` list), le caller doit normaliser en
    blocs dict avec ``type``/``name``/``index`` avant d'appeler ``accumulate``.
    """

    def __init__(
        self, blocked_tools: frozenset[str], role: str, trace_id: str
    ) -> None:
        # index → dict accumule en cours de construction
        self._pending: dict[int, dict[str, Any]] = {}
        # Ordre d'arrivee des index (FIFO pour flush ordonne)
        self._pending_order: list[int] = []
        self._blocked_tools = blocked_tools
        self._role = role
        self._trace_id = trace_id

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def accumulate(self, block: Any) -> list[Any]:
        """Accumule un bloc entrant.

        Retourne la liste des blocs a emettre immediatement (blocs completes
        ou blocs non-tool_use). Les blocs tool_use en cours d'accumulation ne
        sont pas emis tant qu'ils ne sont pas completes.

        Args:
            block: Bloc dict (peut etre un fragment tool_use, un delta input,
                   un bloc text, etc.) ou valeur non-dict (pass-through brut).

        Returns:
            Liste (possiblement vide) de blocs a emettre dans le chunk courant.
        """
        if not isinstance(block, dict):
            # Non-dict : flush pending PUIS retourne le bloc brut
            return self._flush_all() + [block]

        block_type = block.get("type")

        # --- Fragments input JSON (delta input d'un tool_use en cours) ---
        if block_type == "input_json_delta":
            idx = block.get("index")
            if idx is not None and idx in self._pending:
                existing = self._pending[idx]
                partial = block.get("partial_json", "")
                existing["_buf_partial_json"] = existing.get("_buf_partial_json", "") + str(partial)
            # delta seul : rien a emettre
            return []

        # --- Bloc tool_use (complet ou partiel) ---
        if block_type == "tool_use":
            idx = block.get("index")
            if idx is None:
                # Pas d'index : traiter comme bloc complet sans accumulation
                return self._flush_all() + [self._evaluate_block(block)]

            if idx not in self._pending:
                # Nouvel index : demarrer accumulation
                self._pending[idx] = dict(block)
                self._pending_order.append(idx)
                return []
            else:
                # Meme index : fusionner les champs arrivant fragmentes
                existing = self._pending[idx]
                # Completer les champs manquants avec ceux du nouveau fragment.
                # Note : _buf_partial_json est une cle interne, jamais presente dans
                # les blocs entrants (providers). La condition couvre uniquement
                # les champs natifs du bloc (id, name, input, type...).
                for key, val in block.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = val
                return []

        # --- Tout autre bloc (text, etc.) ---
        # Flush tous les tool_use pending AVANT d'emettre le bloc courant
        return self._flush_all() + [block]

    def flush(self) -> list[Any]:
        """Flush tous les blocs tool_use pending (appele en fin de stream).

        Les blocs dont le name est present sont evalues (filtre ou pass-through).
        Les blocs incomplets (name absent) sont traites comme malformes + WARNING.

        Returns:
            Liste de blocs emis (evalues).
        """
        return self._flush_all()

    # ------------------------------------------------------------------
    # Helpers prives
    # ------------------------------------------------------------------

    def _flush_all(self) -> list[Any]:
        """Flush et retourne tous les blocs pending dans l'ordre d'arrivee."""
        if not self._pending_order:
            return []
        result: list[Any] = []
        for idx in self._pending_order:
            block = self._pending[idx]
            result.append(self._evaluate_block(block))
        self._pending.clear()
        self._pending_order.clear()
        return result

    def _evaluate_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Evalue un bloc tool_use complet : filtre si bloque, sinon pass-through.

        Applique la meme semantique que ``ValkyrieToolGuard._filter_content_block``
        mais sur un bloc potentiellement reconstruit par accumulation.

        Args:
            block: Bloc tool_use accumule (peut avoir _buf_partial_json interne).

        Returns:
            Bloc original nettoy, bloc text synthetique si bloque, ou bloc
            malforme si name absent.
        """
        tool_name = block.get("name")

        # Bloc malforme (name absent meme apres accumulation complete)
        if tool_name is None:
            logger.warning(
                "valkyrie_tool_blocked role=%s tool=<malforme> trace_id=%s "
                "[tool_use malforme filtre]",
                self._role,
                self._trace_id,
            )
            return {"type": "text", "text": "[tool_use malforme filtre]"}

        # Reconstituer `input` depuis les input_json_delta accumules.
        # Sans cela, le consommateur aval (agent LLM, Phase 3.5) recoit un `input`
        # tronque sur les tools autorises (pass-through).
        clean_block = {k: v for k, v in block.items() if k != "_buf_partial_json"}
        partial = block.get("_buf_partial_json", "")
        if partial:
            try:
                clean_block["input"] = json.loads(partial)
            except json.JSONDecodeError:
                # Fallback : conserver la chaine brute (caller decidera).
                # Log WARNING pour signaler JSON malforme du provider.
                logger.warning(
                    "valkyrie_tool_input_json_invalid role=%s tool=%s trace_id=%s",
                    self._role,
                    tool_name,
                    self._trace_id,
                )
                clean_block["input"] = partial

        # Bloc bloque
        if tool_name in self._blocked_tools:
            logger.warning(
                "valkyrie_tool_blocked role=%s tool=%s trace_id=%s",
                self._role,
                tool_name,
                self._trace_id,
            )
            return {
                "type": "text",
                "text": (
                    f"[tool_use '{tool_name}' rejete : valkyrie '{self._role}' "
                    f"n'a pas ce tool. Utiliser un autre moyen.]"
                ),
            }

        return clean_block


# ---------------------------------------------------------------------------
# ValkyrieToolGuard — middleware LangChain (enforcement reel)
# ---------------------------------------------------------------------------


class ValkyrieToolGuard(BaseChatModel):
    """Wrapper BaseChatModel qui filtre les tool_use blocks interdits.

    Compose un chat model sous-jacent + ValkyrieConfig. Override _generate,
    _stream, _astream. Parcourt AIMessage.content (list Anthropic
    content blocks), filtre blocs type='tool_use' dont name appartient a
    config.blocked_tools.

    Note : _agenerate non implementé. LangChain 0.3+ route ainvoke() via
    _astream quand disponible. Le middleware couvre sync (_generate) +
    async (_astream) sans _agenerate redondant.

    Strategie : remplacement par bloc texte synthetique, jamais de raise.
    L'agent recoit feedback et peut adapter sa strategie.

    Logs : WARNING structure a chaque filtre (role, tool_name, trace_id).

    OBLIGATOIRE (audit #1bis C2) : ConfigDict(arbitrary_types_allowed=True)
    car BaseChatModel herite de Pydantic BaseModel — Pydantic refuserait sinon
    BaseChatModel et ValkyrieConfig comme types de champs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseChatModel
    config: ValkyrieConfig

    @property
    def _llm_type(self) -> str:
        """Type LLM obligatoire LangChain (BaseChatModel est abstract sur ce point)."""
        return "wincorp-valkyrie-guard"

    def _filter_content_block(self, block: Any, trace_id: str = "unknown") -> Any:
        """Filtre un bloc content Anthropic.

        Args:
            block: Bloc dict (type, ...) ou autre.
            trace_id: Identifiant de trace pour les logs (run_manager.run_id).

        Returns:
            Bloc original si autorise, bloc texte synthetique si bloque/malforme.
        """
        if not isinstance(block, dict):
            return block

        block_type = block.get("type")
        if block_type != "tool_use":
            return block

        tool_name = block.get("name")

        # Bloc malforme (name absent)
        if tool_name is None:
            logger.warning(
                "valkyrie_tool_blocked role=%s tool=<malforme> trace_id=%s "
                "[tool_use malforme filtre]",
                self.config.name,
                trace_id,
            )
            return {"type": "text", "text": "[tool_use malforme filtre]"}

        # Bloc bloque
        if tool_name in self.config.blocked_tools:
            logger.warning(
                "valkyrie_tool_blocked role=%s tool=%s trace_id=%s",
                self.config.name,
                tool_name,
                trace_id,
            )
            return {
                "type": "text",
                "text": (
                    f"[tool_use '{tool_name}' rejete : valkyrie '{self.config.name}' "
                    f"n'a pas ce tool. Utiliser un autre moyen.]"
                ),
            }

        return block

    def _filter_response(self, response: AIMessage, trace_id: str = "unknown") -> AIMessage:
        """Filtre les tool_use blocks interdits dans le contenu de la reponse.

        Args:
            response: AIMessage retourne par le modele sous-jacent.
            trace_id: Identifiant de trace pour les logs (run_manager.run_id).

        Returns:
            AIMessage avec blocks bloques remplaces par texte synthetique.
        """
        content = response.content

        # Si content est une chaine simple : aucun tool_use possible
        if isinstance(content, str):
            return response

        # Content est une liste de blocs
        filtered = [self._filter_content_block(block, trace_id=trace_id) for block in content]

        # Reconstruire AIMessage avec contenu filtre
        return AIMessage(content=filtered)

    @staticmethod
    def _extract_trace_id(run_manager: Any) -> str:
        """Extrait le trace_id depuis run_manager.run_id (LangChain CallbackManager).

        Args:
            run_manager: CallbackManagerForLLMRun ou None.

        Returns:
            str(run_id) si present, sinon "unknown".
        """
        if run_manager is None:
            return "unknown"
        run_id = getattr(run_manager, "run_id", None)
        if run_id is None:
            return "unknown"
        return str(run_id)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override : genere via wrapped model + filtre blocked tools."""
        trace_id = self._extract_trace_id(run_manager)
        result = self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        filtered_generations = []
        for gen in result.generations:
            if isinstance(gen.message, AIMessage):
                filtered_msg = self._filter_response(gen.message, trace_id=trace_id)
                filtered_generations.append(ChatGeneration(message=filtered_msg))
            else:
                filtered_generations.append(gen)
        return ChatResult(generations=filtered_generations)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Override stream : accumulation inter-chunks + filtre tool_use bloques.

        Utilise ``_StreamToolBuffer`` pour accumuler les blocs tool_use fragmentes
        (nom ou input arrives sur plusieurs chunks) avant evaluation filtre. Garantit
        que la decision de filtrage est prise sur le bloc complet, quelque soit le
        provider (Anthropic, OpenAI-compat, DeepSeek).

        L'ordre des blocs sortants respecte l'ordre d'accumulation FIFO.
        """
        trace_id = self._extract_trace_id(run_manager)
        buffer = _StreamToolBuffer(
            blocked_tools=self.config.blocked_tools,
            role=self.config.name,
            trace_id=trace_id,
        )

        for chunk in self.wrapped._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if not isinstance(chunk.message, AIMessageChunk):
                yield chunk
                continue

            content = chunk.message.content
            if isinstance(content, str):
                yield chunk
                continue

            # Liste de blocs : passer au buffer
            emitted_blocks: list[Any] = []
            for block in content:
                emitted_blocks.extend(buffer.accumulate(block))

            if emitted_blocks:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=emitted_blocks),
                    generation_info=chunk.generation_info,
                )

        # Fin du stream : flush les blocs tool_use en cours d'accumulation
        final_blocks = buffer.flush()
        if final_blocks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=final_blocks))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Override async stream : accumulation inter-chunks + filtre tool_use bloques.

        Meme logique que ``_stream`` (buffer ``_StreamToolBuffer``), version async.
        LangChain 0.3+ route ainvoke() via _astream quand disponible — ce chemin
        couvre donc sync (_generate) + async (this method) sans _agenerate redondant.
        """
        trace_id = self._extract_trace_id(run_manager)
        buffer = _StreamToolBuffer(
            blocked_tools=self.config.blocked_tools,
            role=self.config.name,
            trace_id=trace_id,
        )

        async for chunk in self.wrapped._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if not isinstance(chunk.message, AIMessageChunk):
                yield chunk
                continue

            content = chunk.message.content
            if isinstance(content, str):
                yield chunk
                continue

            # Liste de blocs : passer au buffer
            emitted_blocks: list[Any] = []
            for block in content:
                emitted_blocks.extend(buffer.accumulate(block))

            if emitted_blocks:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=emitted_blocks),
                    generation_info=chunk.generation_info,
                )

        # Fin du stream : flush les blocs tool_use en cours d'accumulation
        final_blocks = buffer.flush()
        if final_blocks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=final_blocks))


# ---------------------------------------------------------------------------
# Factory principale
# ---------------------------------------------------------------------------


def create_valkyrie_chat(role: str) -> BaseChatModel:
    """Compose loader + LLM + ValkyrieToolGuard pour un role donne.

    Pas de cache v1.0 : chaque appel reconstruit le guard (deterministe,
    cout negligeable, cf spec §3.3 rationale). Cache ajoutable v1.1+
    si hotspot observe.

    Note : les instances existantes conservent leur config (snapshot
    a la creation). Recreer via ce factory apres reload mtime si necessaire.

    Args:
        role: Nom du role valkyrie (brynhildr, sigrun, thor).

    Returns:
        ValkyrieToolGuard pret a l'emploi (BaseChatModel).

    Raises:
        ValkyrieNotFoundError: si role absent.
        ValkyrieConfigError: si YAML invalide.
    """
    config = load_valkyrie(role)
    inner_model = create_model(config.model)
    return ValkyrieToolGuard(wrapped=inner_model, config=config)


# ---------------------------------------------------------------------------
# API echappatoire tests (§3.5 spec)
# ---------------------------------------------------------------------------


def _reload_for_tests() -> None:
    """Vide cache + mtime. Usage interne tests uniquement (non exporte).

    Note : les instances ValkyrieToolGuard existantes conservent leur ref
    a l'ancienne config (frozen immuable) — comportement documente §5.6.
    """
    global _configs_ref, _yaml_mtime, _last_mtime_check
    with _cache_lock:
        _configs_ref = None
        _yaml_mtime = None
        _last_mtime_check = 0.0
