"""Chargement + validation + interpolation YAML models.yaml.

@spec specs/llm-factory.spec.md v1.3

Responsabilites :
- `_resolve_urd_path` : bifurcation dev (auto-detect .git) vs installed (env var) (R17).
- `_assert_under_allowed_root` : path traversal (R17, PB-008).
- `load_models_config` : parse safe_load (R14), taille max (R15), conflits OneDrive
  (§9.2), schema Pydantic (EC4/EC5/EC6), interpolation ${VAR} (R4, EC7-9),
  whitelist extra_kwargs (R13).

v1.3 : ajout champs optionnels par modele circuit_breaker / retry / pricing
(§22-24). Rétrocompat : fixtures v1.2 sans ces champs restent valides.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wincorp_odin.llm._whitelist import validate_extra_kwargs
from wincorp_odin.llm.exceptions import (
    ModelConfigError,
    ModelConfigSchemaError,
    SecretMissingError,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_MAX_YAML_SIZE_BYTES = 1_048_576  # R15 — 1 Mo
_SUPPORTED_CONFIG_VERSIONS = {1}
_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# Patterns de conflit OneDrive (§9.2, PB-004)
_CONFLICT_GLOBS = [
    "models-DESKTOP-*.yaml",
    "models (conflit*).yaml",
    "models (conflicted copy*).yaml",
    "models.yaml~",
    "models.yaml.bak",
]
_CONFLICT_REGEX = re.compile(
    r"models[ _\-\(].*(conflict|conflit|desktop)[\-\s].*\.yaml$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass ModelConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Configuration resolue d'un modele (post-interpolation ${VAR}).

    SECURITE R10 : api_key_resolved n'apparait jamais dans repr/logs/exceptions.

    v1.3 : champs optionnels middlewares (§22-24) :
      - circuit_breaker_config : dict | None — passe tel quel a CircuitBreakerConfig
      - retry_config : dict | None — passe tel quel a RetryConfig
      - pricing_config : dict | None — {input_per_million_eur, output_per_million_eur}
    """

    name: str
    display_name: str
    use: str
    model: str
    api_key_env: str
    api_key_resolved: str
    max_tokens: int
    timeout: float
    max_retries: int
    supports_thinking: bool
    supports_vision: bool
    supports_reasoning_effort: bool
    when_thinking_enabled: dict[str, Any] | None
    when_thinking_disabled: dict[str, Any] | None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    # v1.3 — champs optionnels middlewares
    circuit_breaker_config: dict[str, Any] | None = None
    retry_config: dict[str, Any] | None = None
    pricing_config: dict[str, Any] | None = None

    def __repr__(self) -> str:
        """Repr safe — redacte la cle API (R10)."""
        parts: list[str] = []
        for f in self.__dataclass_fields__.values():
            val = getattr(self, f.name)
            if f.name == "api_key_resolved":
                val = "***REDACTED***"
            parts.append(f"{f.name}={val!r}")
        return f"ModelConfig({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Modeles Pydantic (parsing strict YAML)
# ---------------------------------------------------------------------------


class _DefaultsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: float = 60.0
    max_retries: int = 0
    supports_vision: bool = False
    supports_reasoning_effort: bool = False


class _CircuitBreakerRawModel(BaseModel):
    """Sous-schema circuit_breaker (v1.3 §22)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    failure_threshold: int = Field(gt=0)
    recovery_timeout_sec: float = Field(gt=0)


class _RetryRawModel(BaseModel):
    """Sous-schema retry (v1.3 §23)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    base_delay_sec: float = Field(gt=0)
    cap_delay_sec: float = Field(gt=0)
    max_attempts: int = Field(gt=0)


class _PricingRawModel(BaseModel):
    """Sous-schema pricing (v1.3 §24)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_per_million_eur: float = Field(ge=0)
    output_per_million_eur: float = Field(ge=0)


class _RawModelEntry(BaseModel):
    """Un bloc modele tel que parse depuis YAML (avant interpolation).

    Pydantic v2 est strict sur les bool pour refuser les coercitions str->bool
    (YAML "yes"/"no" serait normalement coerce par Pydantic ; on refuse).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    use: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    supports_thinking: bool
    timeout: float | None = None
    max_retries: int | None = None
    supports_vision: bool | None = None
    supports_reasoning_effort: bool | None = None
    when_thinking_enabled: dict[str, Any] | None = None
    when_thinking_disabled: dict[str, Any] | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    disabled: bool = False
    # v1.3 — middlewares optionnels
    circuit_breaker: _CircuitBreakerRawModel | None = None
    retry: _RetryRawModel | None = None
    pricing: _PricingRawModel | None = None


class _ModelsFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    config_version: int
    models: list[_RawModelEntry]
    defaults: _DefaultsModel | None = None


# ---------------------------------------------------------------------------
# Resolution chemin URD (R17, PB-008)
# ---------------------------------------------------------------------------


def _find_project_root() -> Path | None:
    """Remonte jusqu'a 10 parents cherchant .git ou pyproject.toml."""
    start = Path(__file__).resolve()
    for ancestor in [start, *start.parents][:11]:
        if (ancestor / ".git").exists() or (ancestor / "pyproject.toml").exists():
            return ancestor
    return None


def _detect_dev_urd_path() -> Path | None:
    """Auto-detection mode dev : .git dans les 5 parents + wincorp-urd freres.

    Retourne le dossier racine wincorp-urd/ si trouve (pas le YAML lui-meme).
    """
    start = Path(__file__).resolve()
    for ancestor in start.parents[:5]:
        if (ancestor / ".git").exists():
            # ancestor = wincorp-odin (ou repo git frere)
            candidate = ancestor.parent / "wincorp-urd"
            if (candidate / "referentiels" / "models.yaml").exists():
                return candidate
            return None  # pragma: no cover — .git present mais wincorp-urd absent, cas couvert par test_r17_dev_mode_autodetects_wincorp_dev via mock
    return None  # pragma: no cover — pas de .git dans 5 parents, cas couvert par test_r17_installed_requires_env_var


def _home_path() -> Path | None:
    """Retourne le home utilisateur (Path.home peut lever sur Windows CI)."""
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


def _assert_under_allowed_root(path: Path) -> None:
    """Verifie que `path` est sous $HOME OU sous project_root.

    Raises:
        ModelConfigError: message generique, ne revele PAS le chemin tente.
    """
    resolved = path.resolve()
    allowed_roots: list[Path] = []

    home = _home_path()
    if home is not None:
        allowed_roots.append(home.resolve())

    project_root = _find_project_root()
    if project_root is not None:
        allowed_roots.append(project_root.resolve())

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return
        except ValueError:
            continue

    # Aucune racine autorisee ne contient le chemin
    raise ModelConfigError(
        "[ERREUR] Le chemin WINCORP_URD_PATH est hors des racines autorisees. "
        "Verifier la variable d'environnement."
    )


def _resolve_urd_path() -> Path:
    """Resolution bifurquee explicite (env var) vs implicite (dev .git).

    Returns:
        Chemin absolu vers le YAML models.yaml.

    Raises:
        ModelConfigError: chemin hors racine, fichier absent, ou mode installed
            sans env var (EC1, EC22, EC27).
    """
    env_value = os.environ.get("WINCORP_URD_PATH")
    if env_value:
        candidate = Path(env_value).resolve()
        _assert_under_allowed_root(candidate)
        yaml_path = candidate / "referentiels" / "models.yaml"
        if not yaml_path.exists():
            raise ModelConfigError(
                f"[ERREUR] Fichier de configuration LLM introuvable. "
                f"Chemin tente : {yaml_path}. Verifier la variable d'environnement "
                f"WINCORP_URD_PATH ou la presence de wincorp-urd/ a cote de wincorp-workspace/."
            )
        return yaml_path

    # Mode implicite — auto-detect .git dans les parents
    dev_urd = _detect_dev_urd_path()
    if dev_urd is not None:
        # Symetrie avec le mode explicite : le chemin auto-detecte doit aussi
        # etre verifie contre les racines autorisees (defense en profondeur).
        _assert_under_allowed_root(dev_urd)
        yaml_path = dev_urd / "referentiels" / "models.yaml"
        if yaml_path.exists():  # pragma: no branch — si _detect_dev_urd_path a retourne un path, il inclut deja la verification .exists()
            return yaml_path

    # Mode installed sans env var — EC27
    module_path = Path(__file__).resolve()
    raise ModelConfigError(
        f"[ERREUR] WINCORP_URD_PATH obligatoire en deploiement installed. "
        f"Definir la variable dans le .env du service (valeur = chemin absolu vers "
        f"le dossier wincorp-urd/). Detection dev/prod : presence de .git dans les "
        f"5 parents de {module_path}."
    )


# ---------------------------------------------------------------------------
# Detection conflits OneDrive (§9.2)
# ---------------------------------------------------------------------------


def _detect_onedrive_conflicts(yaml_path: Path) -> list[str]:
    """Retourne la liste des fichiers conflictuels dans le dossier parent."""
    parent = yaml_path.parent
    conflicts: set[str] = set()
    for pattern in _CONFLICT_GLOBS:
        for match in parent.glob(pattern):
            if match.name != yaml_path.name:  # pragma: no branch — glob ne matche pas `models.yaml` via ces patterns, defense
                conflicts.add(match.name)
    # Regex fallback — parcours complet
    for entry in parent.iterdir():
        if entry.name == yaml_path.name:
            continue
        if _CONFLICT_REGEX.search(entry.name):  # pragma: no branch — fichiers non-conflit coexistent, branche exercee via fixtures reelles
            conflicts.add(entry.name)
    return sorted(conflicts)


# ---------------------------------------------------------------------------
# Interpolation ${VAR}
# ---------------------------------------------------------------------------


def _interpolate_var(raw: str, field_name: str, model_name: str) -> str:
    """Remplace ${VAR} par la valeur d'env. Lève SecretMissingError si vide/absente."""
    match = _VAR_PATTERN.fullmatch(raw.strip())
    if match is None:
        # Pas de ${VAR} — valeur litterale (non recommande pour api_key mais autorise)
        return raw
    var_name = match.group(1)
    value = os.environ.get(var_name)
    if value is None or value == "":
        raise SecretMissingError(
            f"[ERREUR] Variable d'environnement {var_name} absente ou vide "
            f"(referencee par modele '{model_name}', champ '{field_name}'). "
            f"Definir la cle dans .env ou l'exporter avant de lancer le process."
        )
    return value


# ---------------------------------------------------------------------------
# Chargement complet
# ---------------------------------------------------------------------------


def _raw_parse_yaml(yaml_path: Path) -> Any:
    """Lecture + safe_load + gestion erreurs structurelles bas niveau."""
    # R15 — taille max avant read
    size = yaml_path.stat().st_size
    if size > _MAX_YAML_SIZE_BYTES:
        raise ModelConfigError(
            f"[ERREUR] models.yaml suspect — taille {size / 1_048_576:.2f} Mo > 1 Mo "
            f"autorise. Verifier le fichier (corruption, duplications ?)."
        )

    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ModelConfigError(
            f"[ERREUR] Lecture de {yaml_path} echouee : {e}."
        ) from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        # Tenter d'extraire ligne/colonne si dispo
        mark = getattr(e, "problem_mark", None)
        location = ""
        if mark is not None:  # pragma: no branch — defense pour YAMLError sans mark (ne se produit pas en pratique avec PyYAML)
            location = f" (ligne {mark.line + 1}, colonne {mark.column + 1})"
        raise ModelConfigError(
            f"[ERREUR] YAML invalide{location} : {e}. Corriger la syntaxe."
        ) from e

    return data


def _validate_structure(data: Any) -> _ModelsFile:
    """Validation Pydantic + unicite des noms."""
    if not isinstance(data, dict):
        raise ModelConfigError(
            "[ERREUR] Racine YAML invalide : dict attendu."
        )
    models_list = data.get("models")
    if not models_list:
        raise ModelConfigError(
            "[ERREUR] Aucun modele declare dans models.yaml (cle 'models' absente ou vide)."
        )
    try:
        parsed = _ModelsFile.model_validate(data)
    except ValidationError as e:
        # EC4/EC5 — on reformate les erreurs
        detail_lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            detail_lines.append(f"  - {loc} : {msg}")
        raise ModelConfigSchemaError(
            "[ERREUR] Schema YAML invalide :\n" + "\n".join(detail_lines)
        ) from e

    # EC19 — version supportee
    if parsed.config_version not in _SUPPORTED_CONFIG_VERSIONS:
        raise ModelConfigError(
            f"[ERREUR] config_version {parsed.config_version} non supportee. "
            f"Versions supportees : {sorted(_SUPPORTED_CONFIG_VERSIONS)}."
        )

    # EC6 — doublons de name
    seen: dict[str, int] = {}
    for idx, m in enumerate(parsed.models):
        if m.name in seen:
            raise ModelConfigError(
                f"[ERREUR] Nom '{m.name}' declare 2 fois dans models.yaml "
                f"(positions {seen[m.name]} et {idx}). Les noms doivent etre uniques."
            )
        seen[m.name] = idx

    return parsed


def _merge_defaults(
    raw: _RawModelEntry, defaults: _DefaultsModel | None
) -> dict[str, Any]:
    """Fusionne les defaults du YAML avec le modele brut."""
    d = defaults or _DefaultsModel()
    return {
        "timeout": raw.timeout if raw.timeout is not None else d.timeout,
        "max_retries": raw.max_retries if raw.max_retries is not None else d.max_retries,
        "supports_vision": (
            raw.supports_vision if raw.supports_vision is not None else d.supports_vision
        ),
        "supports_reasoning_effort": (
            raw.supports_reasoning_effort
            if raw.supports_reasoning_effort is not None
            else d.supports_reasoning_effort
        ),
    }


def load_models_config() -> dict[str, ModelConfig]:
    """Charge, valide et interpole models.yaml.

    Zero I/O reseau. Retourne un dict {name: ModelConfig}.

    Raises:
        ModelConfigError et sous-classes en cas d'echec.
    """
    yaml_path = _resolve_urd_path()

    # EC17 — conflits OneDrive
    conflicts = _detect_onedrive_conflicts(yaml_path)
    if conflicts:
        raise ModelConfigError(
            f"[ERREUR] Conflit OneDrive detecte dans {yaml_path.parent} : "
            f"{conflicts} presents a cote de models.yaml. "
            f"Resoudre manuellement (garder la bonne version) avant de relancer."
        )

    data = _raw_parse_yaml(yaml_path)
    parsed = _validate_structure(data)

    # Construction des ModelConfig — secrets a la fin (agregation)
    result: dict[str, ModelConfig] = {}
    secret_errors: list[str] = []

    for raw in parsed.models:
        # R13 — whitelist extra_kwargs (meme pour disabled selon PB-018 partiel :
        # mais la spec §9 etape 8 dit "pour chaque modele non-disabled". On suit la spec.)
        if not raw.disabled:
            validate_extra_kwargs(raw.name, raw.use, raw.extra_kwargs)

        merged = _merge_defaults(raw, parsed.defaults)

        # Interpolation api_key — agreger les erreurs
        try:
            api_key_resolved = _interpolate_var(raw.api_key, "api_key", raw.name)
        except SecretMissingError as e:
            secret_errors.append(str(e))
            api_key_resolved = ""  # placeholder pour agregation

        # Detection du nom de la var (si forme ${VAR})
        match = _VAR_PATTERN.fullmatch(raw.api_key.strip())
        api_key_env = match.group(1) if match else ""

        cfg = ModelConfig(
            name=raw.name,
            display_name=raw.display_name,
            use=raw.use,
            model=raw.model,
            api_key_env=api_key_env,
            api_key_resolved=api_key_resolved,
            max_tokens=raw.max_tokens,
            timeout=merged["timeout"],
            max_retries=merged["max_retries"],
            supports_thinking=raw.supports_thinking,
            supports_vision=merged["supports_vision"],
            supports_reasoning_effort=merged["supports_reasoning_effort"],
            when_thinking_enabled=raw.when_thinking_enabled,
            when_thinking_disabled=raw.when_thinking_disabled,
            extra_kwargs=dict(raw.extra_kwargs),
            disabled=raw.disabled,
            # v1.3 — middlewares optionnels (dict ou None, passes aux wrappers)
            circuit_breaker_config=(
                raw.circuit_breaker.model_dump() if raw.circuit_breaker else None
            ),
            retry_config=raw.retry.model_dump() if raw.retry else None,
            pricing_config=raw.pricing.model_dump() if raw.pricing else None,
        )
        result[raw.name] = cfg

    if secret_errors:
        raise SecretMissingError("\n".join(secret_errors))

    return result
