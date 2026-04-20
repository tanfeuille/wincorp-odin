# wincorp_odin.llm — Specification (Phase 1.1 DeerFlow)

> **Statut :** DRAFT (post re-review adversarial — prêt build)
> **Version :** 1.2.0
> **Niveau :** 3 (exhaustif)
> **Auteur :** Tan Phi HUYNH (consolidation 3 specs SDD Opus + 8 arbitrages review adversarial + 15 correctifs PB-001→PB-015 + 3 correctifs structurels PB-019 / points ouverts 1 et 2)
> **Date de création :** 2026-04-20
> **@plan** `memory/project_deerflow_inspiration_plan.md` Phase 1.1
> **Nom logique** : `wincorp_odin.llm` (pas d'alias `mimir.llm` — isolation dure Odin↔Mimir)
> **Package Python réel** : `wincorp_odin.llm` (repo `wincorp-odin`, Yggdrasil Tronc)

---

## 1. Objectif

Factory YAML-driven qui instancie des clients LLM (`ChatAnthropic` aujourd'hui, DeepSeek / Kimi / modèles locaux demain) **sans que le code consommateur connaisse la classe concrète**. Remplace les imports hardcodés dispersés dans Heimdall, Bifrost, Thor par un appel unique `create_model("sonnet")`. Sépare la **config** (YAML versionné dans `wincorp-urd/referentiels/models.yaml`) du **code** (Python dans `wincorp-odin/src/wincorp_odin/llm/`). Rétrocompatible **sans garantie** : les appels hardcodés existants continuent de fonctionner mais **Phase 1.1 n'apporte aucune garantie aux appels non migrés** (ni cache, ni capability check, ni secrets audit). Le plan de migration est documenté §18 et exécuté en Phase 1.9.

---

## 2. Périmètre

### IN — Ce que le module fait (Phase 1.1)

- Charger `models.yaml` une fois au boot du process (validation exhaustive, non-lazy).
- Interpoler `${VAR}` depuis l'environnement (stricte, pas de fallback vide).
- Résoudre `use: "pkg.module:Classe"` via `importlib` (zéro import hardcodé dans `factory.py`).
- Instancier le modèle à la demande via `create_model(name, thinking_enabled=False)`.
- Appliquer les overrides `when_thinking_enabled` / `when_thinking_disabled`.
- Whitelist stricte des `extra_kwargs` par provider (cf. R13).
- Cache mémoire thread-safe keyé `(name, thinking_enabled)`.
- Invalidation **lazy sur mtime avec throttle 1 Hz** (cf. R18, §10.3).
- Lever des exceptions FR explicites (hiérarchie `OdinLlmError`).
- Exposer `validate_all_models()` pour la validation startup, avec budget timeout total configurable (cf. R19).
- Masquer toute clé API en `repr()`, logs et traces d'exception (R10 / R10b / R10c).

### OUT — Ce que le module ne fait PAS (Phase 1.1)

- Circuit breaker (Phase 1.4 — `wincorp_odin.llm.circuit_breaker`).
- Retry exponentiel (Phase 1.5 — `wincorp_odin.llm.retry`).
- Middleware tokens / logging usage (Phase 1.6 — `wincorp_odin.llm.tokens`).
- Patch `stream_usage` LangChain (Phase 1.7).
- Hot reload actif / fsnotify (mtime-check throttled suffit).
- Découverte auto du fichier URD (chemin explicite via env var + fallback contraint, cf. R17).
- Support providers autres qu'Anthropic (structure prête, activation Phase 1.2+).
- Migration des appels `ChatAnthropic()` hardcodés (Phase 1.9, cf. §18).
- Round-trip réseau au boot (authentification vérifiée au 1er appel réel).
- Endpoint HTTP `admin_reload` (futur, hors scope 1.1 — cf. PB-015).

---

## 3. Interface

### 3.1 Exports publics (`src/wincorp_odin/llm/__init__.py`)

```python
from wincorp_odin.llm.factory import create_model, validate_all_models
from wincorp_odin.llm.config import ModelConfig, load_models_config
from wincorp_odin.llm.exceptions import (
    OdinLlmError,
    ModelConfigError,
    ModelConfigSchemaError,
    SecretMissingError,
    ProviderNotInstalledError,
    ModelNotFoundError,
    CapabilityMismatchError,
    ModelAuthenticationError,
    ExtraKwargsForbiddenError,
)
from wincorp_odin.llm.helpers import is_model_not_found, is_capability_mismatch

__all__ = [
    "create_model",
    "validate_all_models",
    "ModelConfig",
    "load_models_config",
    "OdinLlmError",
    "ModelConfigError",
    "ModelConfigSchemaError",
    "SecretMissingError",
    "ProviderNotInstalledError",
    "ModelNotFoundError",
    "CapabilityMismatchError",
    "ModelAuthenticationError",
    "ExtraKwargsForbiddenError",
    "is_model_not_found",
    "is_capability_mismatch",
]
```

**Note** : `reload()` **n'est PAS exporté** (renommé `_reload_for_tests`, usage interne uniquement — cf. §10.5 et PB-015).

### 3.2 Signatures

```python
def create_model(name: str, thinking_enabled: bool = False) -> ChatAnthropic:
    """Instancie (ou récupère depuis le cache) un client LLM configuré.

    Phase 1.1 : retour typé strict ``ChatAnthropic`` (seul provider actif).
    Phase 1.2 : relaxation vers ``BaseChatModel`` à la consolidation multi-provider.

    Args:
        name: Nom logique déclaré dans models.yaml (ex "sonnet", "haiku").
        thinking_enabled: Active le mode thinking si supporté par le modèle.

    Returns:
        Instance ``ChatAnthropic`` prête à l'emploi (ou instance cachée).

    Raises:
        ModelNotFoundError: nom inconnu dans la config.
        CapabilityMismatchError: thinking demandé sur modèle non-compatible.
        ModelConfigError: config YAML invalide détectée à l'invalidation.
        ExtraKwargsForbiddenError: extra_kwargs contient une clé non whitelistée.
    """


def validate_all_models() -> None:
    """Valide la config YAML en entier sans instancier les modèles.

    Ordre strict (cf. §9). Aucun round-trip réseau. Destiné au boot du process.
    Budget total par défaut 5s (env ``WINCORP_LLM_VALIDATE_TIMEOUT_S``). Au-delà,
    la résolution ``use:`` des modèles non-critiques est sautée avec WARNING FR
    et sera re-tentée au 1er ``create_model`` concerné (cf. R19).

    Raises:
        ModelConfigError (et sous-classes) en cas d'échec. Agrège les erreurs
        par modèle avant de lever — Tan doit voir TOUS les problèmes d'un coup.
    """


def _reload_for_tests() -> None:
    """Vide le cache et force un rechargement au prochain ``create_model``.

    USAGE INTERNE : fixtures pytest et shell admin local uniquement.
    **N'est PAS exporté** (pas dans ``__all__``, préfixe underscore).
    Pour un reload prod futur, utiliser une API séparée ``admin_reload(token)``
    avec authentification et rate limit — hors scope Phase 1.1.
    """
```

### 3.3 Dataclass `ModelConfig`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ModelConfig:
    """Configuration résolue d'un modèle (post-interpolation ${VAR}).

    SÉCURITÉ : ``api_key_resolved`` ne doit jamais apparaître dans logs, repr,
    sérialisation ou message d'exception. Le ``__repr__`` override ci-dessous
    la remplace par ``***REDACTED***`` (R10). ``_build_kwargs`` strip la clé
    des traces d'exception (R10b). ``ModelAuthenticationError.__init__`` nettoie
    ``args`` et la chaîne ``__cause__`` (R10c).
    """
    name: str                                     # clé logique unique ("sonnet")
    display_name: str                             # label humain ("Claude Sonnet 4.5")
    use: str                                      # "langchain_anthropic:ChatAnthropic"
    model: str                                    # model_id provider
    api_key_env: str                              # nom env var ("ANTHROPIC_API_KEY")
    api_key_resolved: str                         # SECRET — redacted par __repr__
    max_tokens: int
    timeout: float
    max_retries: int                              # 0 = pas de retry Phase 1.1
    supports_thinking: bool
    supports_vision: bool
    supports_reasoning_effort: bool
    when_thinking_enabled: dict[str, Any] | None
    when_thinking_disabled: dict[str, Any] | None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False

    def __repr__(self) -> str:
        """Repr safe — redacte la clé API (R10)."""
        fields_repr = []
        for f in self.__dataclass_fields__.values():
            val = getattr(self, f.name)
            if f.name == "api_key_resolved":
                val = "***REDACTED***"
            fields_repr.append(f"{f.name}={val!r}")
        return f"ModelConfig({', '.join(fields_repr)})"
```

### 3.4 Erreurs

| Code / Type | Condition de déclenchement | Comportement |
|-------------|----------------------------|--------------|
| `ModelConfigError` | YAML absent / invalide / schéma incomplet / doublon name / conflit OneDrive / taille max dépassée / path hors racine | Erreur FR pointant ligne/colonne, liste agrégée de problèmes, FATAL startup. N'expose JAMAIS un chemin hors racine (PB-008). |
| `ModelConfigSchemaError` | Type incorrect sur un champ (ex `supports_thinking: "yes"`) | JSONPath + type attendu vs reçu |
| `SecretMissingError` | `${ANTHROPIC_API_KEY}` absent ou chaîne vide | Liste agrégée des secrets manquants, FATAL startup |
| `ProviderNotInstalledError` | `use:` pointe vers package non installé / classe inexistante / non-callable | Suggère la commande `uv pip install <pkg>` exacte |
| `ModelNotFoundError` | `create_model("inconnu")` | Hérite de `OdinLlmError` **uniquement** (plus de `KeyError`, cf. PB-011). Helper `is_model_not_found(exc)` disponible. |
| `CapabilityMismatchError` | `thinking_enabled=True` sur modèle `supports_thinking: false` | Hérite de `OdinLlmError` **uniquement** (plus de `ValueError`, cf. PB-011). |
| `ModelAuthenticationError` | 401 Anthropic au 1er appel | Wrap l'erreur SDK en strippant la clé API des `args` et `.__cause__` (R10c). |
| `ExtraKwargsForbiddenError` | `extra_kwargs` contient une clé hors whitelist provider | Liste les kwargs rejetés et la whitelist applicable (R13). FATAL startup. |

Hiérarchie complète : cf. §8.

---

## 4. Schéma YAML unique (`wincorp-urd/referentiels/models.yaml`)

### 4.1 Structure

```yaml
config_version: 1                # obligatoire — bump si breaking change
source: "Plan DeerFlow Phase 1.1"
maintainer: "Tan Phi HUYNH"
updated: "2026-04-20"

defaults:                        # optionnel — valeurs par défaut héritées
  timeout: 60.0
  max_retries: 0                 # pas de retry Phase 1.1
  supports_vision: false
  supports_reasoning_effort: false

models:
  - name: "sonnet"               # OBLIGATOIRE — clé logique unique
    display_name: "Claude Sonnet 4.5"
    use: "langchain_anthropic:ChatAnthropic"  # OBLIGATOIRE — pkg.module:Classe
    model: "claude-sonnet-4-5-20250929"        # OBLIGATOIRE — model_id provider
    api_key: "${ANTHROPIC_API_KEY}"            # OBLIGATOIRE — interpolation ${VAR}
    max_tokens: 8192                            # OBLIGATOIRE
    supports_thinking: false                    # OBLIGATOIRE (bool strict)
    timeout: 120.0                              # OPTIONNEL (override defaults)
    max_retries: 0                              # OPTIONNEL
    supports_vision: true                       # OPTIONNEL
    supports_reasoning_effort: false            # OPTIONNEL
    when_thinking_enabled: null                 # OPTIONNEL (null si non applicable)
    when_thinking_disabled: null                # OPTIONNEL
    extra_kwargs: {}                            # OPTIONNEL — whitelist stricte (R13)
    disabled: false                             # OPTIONNEL — désactive sans supprimer

  - name: "opus-thinking"
    display_name: "Claude Opus 4.7 (thinking)"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-opus-4-7-20260115"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 16384
    supports_thinking: true
    supports_vision: true
    supports_reasoning_effort: true
    when_thinking_enabled:
      thinking:
        type: "enabled"
        budget_tokens: 8192
    when_thinking_disabled: null
```

### 4.2 Champs obligatoires vs optionnels

| Champ | Obligatoire | Défaut | Remarque |
|-------|:-----------:|--------|----------|
| `name` | OUI | — | Unicité stricte (EC6) |
| `display_name` | OUI | — | Humain-lisible |
| `use` | OUI | — | Format `pkg.module:Classe` (EC10/11/12) |
| `model` | OUI | — | model_id exposé par le provider |
| `api_key` | OUI | — | Syntaxe `${VAR}` obligatoire (EC7/8/9) |
| `max_tokens` | OUI | — | int > 0 |
| `supports_thinking` | OUI | — | bool strict |
| `timeout` | NON | `defaults.timeout` ou 60.0 | float > 0 |
| `max_retries` | NON | `defaults.max_retries` ou 0 | int ≥ 0 |
| `supports_vision` | NON | `false` | bool |
| `supports_reasoning_effort` | NON | `false` | bool |
| `when_thinking_enabled` | NON | `null` | dict ou null |
| `when_thinking_disabled` | NON | `null` | dict ou null |
| `extra_kwargs` | NON | `{}` | dict — whitelist stricte par provider (R13, EC23) |
| `disabled` | NON | `false` | bool — exclut du cache et de validate_all_models |

**Rejet explicite des champs inconnus** (Pydantic `model_config = ConfigDict(extra='forbid')`) pour éviter les typos silencieuses.

---

## 5. Architecture interne

```
wincorp-odin/src/wincorp_odin/llm/
├── __init__.py          # Ré-exports publics uniquement (cf. §3.1)
├── config.py            # Pydantic BaseModel ModelsFile + ModelConfig + loader YAML + interpolation ${VAR}
│                        # + vérification taille max (R15) + safe_load strict (R14)
├── factory.py           # create_model + validate_all_models + _reload_for_tests + cache + Lock
├── _registry.py         # resolve_class("pkg.module:Classe") via importlib — PRIVÉ, cache classe
├── _whitelist.py        # PRIVÉ — tables de whitelist extra_kwargs par provider (R13)
├── exceptions.py        # OdinLlmError + 8 sous-classes (cf. §8)
├── helpers.py           # is_model_not_found / is_capability_mismatch (sucre syntaxique PB-011)
└── legacy.py            # deprecated_direct_chat_anthropic() wrapper migration Phase 1.9 (§18)
```

**Responsabilités** :
- `config.py` : parsing YAML safe → validation Pydantic → interpolation `${VAR}` → `list[ModelConfig]`. Zéro I/O réseau.
- `_registry.py` : `resolve_class(use: str) -> type` + cache classe résolue (invalidé par `_reload_for_tests`). Gère `ImportError`, `AttributeError`, `callable()`.
- `_whitelist.py` : dict `PROVIDER_EXTRA_KWARGS_WHITELIST: dict[str, frozenset[str]]`. Pour `langchain_anthropic:ChatAnthropic` → `frozenset({"temperature", "top_p", "top_k", "stop_sequences", "streaming"})`.
- `factory.py` : orchestre — `config.load_models_config()` + `_registry.resolve_class()` + filtre whitelist + instancie + met en cache. Thread-safe.
- `exceptions.py` : hiérarchie complète, messages FR, redaction secrets.
- `helpers.py` : `is_model_not_found(exc) -> bool`, `is_capability_mismatch(exc) -> bool`.
- `legacy.py` : wrapper deprecated avec `DeprecationWarning` FR, pointe vers `create_model()`.
- `__init__.py` : façade stable, aucune logique.

**Règle d'or** : zéro `import langchain_anthropic` ou `import ChatAnthropic` en dehors de la résolution dynamique (`_registry.py`).

---

## 6. Règles métier (Rx)

- **R1** — Cache par `(name, thinking_enabled)` : deux appels identiques retournent la MÊME instance (identité `is`). Deux noms différents retournent des instances distinctes.
- **R2** — Cache invalidé par `thinking_enabled` : `create_model("opus", thinking_enabled=True)` et `create_model("opus", thinking_enabled=False)` retournent deux instances distinctes.
- **R3** — Validation startup non-lazy : `validate_all_models()` exécute tout l'ordre §9, fail-fast au boot du process.
- **R4** — Interpolation `${VAR}` stricte : regex `\$\{([A-Z_][A-Z0-9_]*)\}`, raise `SecretMissingError` si env var absente ou chaîne vide.
- **R5** — Registry dynamique sans hardcode : zéro `import ChatAnthropic` dans `factory.py`.
- **R6** — `thinking_enabled=True` sur modèle `supports_thinking: false` lève `CapabilityMismatchError` — runtime (pas startup), message FR listant les modèles thinking-compatibles.
- **R7** — Thread-safety : `threading.Lock` protège l'écriture du cache. **Double-checked locking** pour le cache instance ET pour `_yaml_mtime` (cf. PB-003 — re-lire `stat().st_mtime` APRÈS acquisition du lock avant de décider d'invalider). La lecture (dict lookup) s'appuie sur le GIL.
- **R8** — `_registry.py` est privé et **n'est pas ré-exporté** par `__init__.py`.
- **R9** — Typage retour `-> ChatAnthropic` strict en Phase 1.1. Relaxation vers `-> BaseChatModel` en Phase 1.2.
- **R10** — Interpolation `api_key_resolved` **jamais loggée, jamais sérialisée, jamais présente dans `repr()`**. `ModelConfig.__repr__` override explicite (cf. §3.3).
- **R10b** — `_build_kwargs` strip l'`api_key` des traces d'exception : tout `except` qui ré-émet en wrap reconstruit `args` sans la valeur de la clé.
- **R10c** — `ModelAuthenticationError.__init__` nettoie `args` (match regex `r"sk-ant-[A-Za-z0-9_\-]{20,}"` remplacé par `***REDACTED***`) ET la chaîne `__cause__` (récurre sur `e.__cause__.args` tant que non-None).
- **R11** — Modèle `disabled: true` exclu du cache et ignoré par `create_model` (lève `ModelNotFoundError` même si le bloc existe). Usage vs commentaire YAML : `disabled: true` conserve la validation syntaxique et permet de réactiver par patch minimal ; commenter un bloc perd cette garantie.
- **R12** — `config_version` non supportée → `ModelConfigError` avec suggestion de migration.
- **R13 — Whitelist `extra_kwargs` stricte par provider** :
  - `_whitelist.py` contient `PROVIDER_EXTRA_KWARGS_WHITELIST: dict[str, frozenset[str]]`.
  - Pour `langchain_anthropic:ChatAnthropic` : `frozenset({"temperature", "top_p", "top_k", "stop_sequences", "streaming"})`.
  - **Rejetés explicitement** : `base_url`, `default_headers`, `http_client`, `api_key`, `anthropic_api_url` (échappatoire réseau/secrets).
  - À la validation schéma, tout `extra_kwargs` contenant une clé hors whitelist → `ExtraKwargsForbiddenError`.
  - Si `use:` pointe vers un provider inconnu dans la table → whitelist vide → `extra_kwargs` doit être `{}` ou `ExtraKwargsForbiddenError`.
- **R14 — YAML safe_load obligatoire** : `config.py` utilise `yaml.safe_load()` exclusivement. Un appel à `yaml.load()` ou `yaml.unsafe_load()` est interdit — lint Ruff règle custom (ou commit-hook grep) à ajouter. Empêche l'exécution de tags `!!python/object` dans un YAML compromis (attack injection via URD contaminé).
- **R15 — Taille max YAML 1 MB** : `Path.stat().st_size` vérifié **avant** `read_text()`. Si > 1 Mo → `ModelConfigError` FR « models.yaml > 1 Mo, suspicion fichier corrompu ou bourré de duplications ». Borne arbitraire mais très supérieure aux usages légitimes (~2 Ko pour 5-10 modèles). Voir §19 « Mitigation parsing » pour la justification de l'abandon d'un timeout parsing dédié.
- **R17 — Résolution path URD bifurquée dev vs installed** :
  - **Mode explicite** (`WINCORP_URD_PATH` env var définie) : `Path(env).resolve()` → passe par `_assert_under_allowed_root(resolved)` qui vérifie l'appartenance à `Path.home()` OU à `project_root` (remontée 10 parents depuis `__file__` cherchant `.git` ou `pyproject.toml`). Sinon → `ModelConfigError` FR **générique** sans exposer le chemin tenté. Retourne ensuite `resolved / "referentiels" / "models.yaml"`.
  - **Mode implicite** (env var absente) : parcourt `Path(__file__).resolve().parents[:5]` à la recherche d'un ancêtre contenant `.git`. Si trouvé : dev local, ancêtre = `wincorp-odin/`, on remonte à `wincorp-dev/` et on tente `wincorp-urd/referentiels/models.yaml`. Si `.exists()` → OK. Sinon → break (ne pas dépasser l'ancêtre `.git`).
  - **Installed sans env var** (pas de `.git` trouvé, pas d'env var) : `ModelConfigError` **FATAL startup** FR actionnable — `"WINCORP_URD_PATH obligatoire en déploiement installed. Définir la variable dans le .env du service (valeur = chemin absolu vers le dossier wincorp-urd/). Détection dev/prod : présence de .git dans les 5 parents."`
  - **Plus de fallback silencieux vers `$HOME`** : v1.1 autorisait implicitement toute racine `$HOME` en l'absence de `.git`, ouvrant une surface d'attaque. v1.2 force l'explicite en mode installed.
  - Helper `_assert_under_allowed_root(path: Path) -> None` : raise `ModelConfigError("Path hors racine autorisée")` **générique** (pas de révélation du path tenté). Chemin journalisé en DEBUG pour investigation Tan.
- **R18 — mtime check throttled 1/s** :
  - Variable `_last_mtime_check: float` stocke `time.monotonic()` du dernier check.
  - Si `time.monotonic() - _last_mtime_check < 1.0` → skip `stat()`, retourne cache directement.
  - Sinon → `stat().st_mtime` + mise à jour `_last_mtime_check`.
  - Garantit max 1 `stat()` / seconde même sous burst `create_model`.
  - Alternative documentée (non retenue) : check mtime **uniquement au cache miss** — rejeté car rate le scénario multi-PC OneDrive où une édition est suivie de hits immédiats.
- **R19 — Budget timeout `validate_all_models()` startup** :
  - Env var `WINCORP_LLM_VALIDATE_TIMEOUT_S` (défaut 5.0, min 1.0, max 60.0, float).
  - Horloge `time.monotonic()` démarrée à l'entrée de `validate_all_models()`.
  - Étapes 1-7 de §9 (chemin, OneDrive, parse, structure, version, schéma, interpolation) : doivent toutes passer dans le budget. Si dépassement ici → `ModelConfigError` FR « Validation des étapes 1-7 > {budget}s ».
  - Étape 8 (résolution `use:`) : chaque `importlib.import_module` + `getattr` tente individuellement. Si le cumul dépasse le budget, les modèles non-résolus sont marqués `_use_resolution_deferred=True` + log WARNING FR « Résolution provider {use} sautée par timeout, vérifier au 1er appel ».
  - La résolution différée se déclenche au 1er `create_model("<ce_nom>")` — si elle échoue à ce moment-là → `ProviderNotInstalledError` comme en mode strict.
- **R19b — Budget timeout `_load_and_validate_models()` runtime (reload mtime)** :
  - Distincte de R19 : invalidation runtime ne doit **jamais** être aussi permissive que startup.
  - Env var `WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S` (défaut **0.5**, min 0.1, max 5.0, float).
  - Rationale : le reload runtime tourne sous pression utilisateur (thread `create_model` en attente). Un budget court évite qu'un YAML lent/buggué gèle la prod.
  - En cas de dépassement (`TimeoutError`) OU d'erreur config (`ModelConfigError`) pendant un reload runtime : le cache **précédent est conservé**, WARNING FR loggué, pas de downtime (cf. EC26).
  - L'invalidation est retentée au prochain `_check_mtime_and_invalidate()` après que le throttle R18 soit retombé (1 s).

---

## 7. Edge cases (ECx)

| # | Scénario | Comportement | Sévérité |
|---|----------|--------------|----------|
| EC1 | `models.yaml` absent | `ModelConfigError` FR pointant le chemin tenté **uniquement si sous racine autorisée**, sinon message générique (cf. R17) + env var `WINCORP_URD_PATH` non définie | FATAL startup |
| EC2 | YAML syntaxe invalide | `ModelConfigError` avec ligne + colonne (via `yaml.YAMLError.problem_mark`) | FATAL startup |
| EC3 | YAML vide / pas de clé `models` / `models: []` | `ModelConfigError` « aucun modèle déclaré » | FATAL startup |
| EC4 | Champ obligatoire manquant | `ModelConfigSchemaError` listant **TOUS** les champs manquants du modèle fautif | FATAL startup |
| EC5 | Type incorrect | `ModelConfigSchemaError` JSONPath + type attendu vs reçu | FATAL startup |
| EC6 | Doublon de `name` | `ModelConfigError` « name 'sonnet' déclaré 2 fois lignes X et Y » | FATAL startup |
| EC7 | `${ANTHROPIC_API_KEY}` env var absente | `SecretMissingError` listant **TOUTES** les secrets manquantes | FATAL startup |
| EC8 | `${VAR}` = chaîne vide | `SecretMissingError` (assimilé absent) | FATAL startup |
| EC9 | `${UNKNOWN_VAR}` typo | `ModelConfigError` « variable ${UNKNOWN_VAR} introuvable » | FATAL startup |
| EC10 | `use:` package non installé (`ImportError`) | `ProviderNotInstalledError` + commande `uv pip install <pkg>` suggérée | FATAL startup |
| EC11 | `use:` classe inexistante (`AttributeError`) | `ProviderNotInstalledError` « classe `XYZ` introuvable dans `pkg.module` » | FATAL startup |
| EC12 | `use:` objet non-callable | `ProviderNotInstalledError` « `pkg.module:Classe` n'est pas instanciable » | FATAL startup |
| EC13 | `create_model("inconnu")` | `ModelNotFoundError` listant les noms disponibles (triés, exclus disabled) | RUNTIME |
| EC14 | `create_model("haiku", thinking_enabled=True)` avec `supports_thinking: false` | `CapabilityMismatchError` listant les modèles thinking-compatibles | RUNTIME |
| EC15 | API key invalide / tronquée | 1er appel LLM → `ModelAuthenticationError` FR « Authentification Anthropic échouée, vérifier ANTHROPIC_API_KEY ». La clé est **strippée** de `args` ET de `__cause__` (R10c). Stack originale préservée via `raise ... from e`. | RUNTIME |
| EC16 | `models.yaml` modifié pendant le run | Check mtime throttled 1/s, log WARNING FR, invalidate cache, re-valide, re-instancie | WARNING |
| EC17 | Conflit OneDrive | Détecté par 4 patterns glob + regex fallback (cf. §9.2), `ModelConfigError` FR demandant résolution manuelle | FATAL startup |
| EC18 | Multi-threads `create_model("sonnet")` simultanés 1er appel | Double-checked locking → **une seule instanciation** | — |
| EC19 | `config_version` non supportée | `ModelConfigError` avec suggestion de migration | FATAL startup |
| EC20 | `supports_thinking: true` sans `when_thinking_enabled` | Warning FR + défaut `{thinking: {type: "enabled", budget_tokens: 4096}}` | WARNING |
| EC21 | Modèle `disabled: true` + `create_model("ce-nom")` | `ModelNotFoundError` comme si absent (R11) | RUNTIME |
| EC22 | `WINCORP_URD_PATH` pointe vers dossier inexistant | `ModelConfigError` FR distinguant env absente vs chemin invalide | FATAL startup |
| EC23 | `extra_kwargs` contient clé hors whitelist (ex `base_url: "evil.com"`) | `ExtraKwargsForbiddenError` listant clés rejetées + whitelist applicable | FATAL startup |
| EC24 | YAML > 1 Mo OU tag `!!python/object` | `ModelConfigError` FR « fichier suspect — taille/tag dangereux » — détaille quelle règle a déclenché (R14/R15) | FATAL startup |
| EC25 | `validate_all_models()` dépasse `WINCORP_LLM_VALIDATE_TIMEOUT_S` pendant résolution `use:` | Log WARNING FR par modèle différé, modèles critiques (flag futur) tombent en `ModelConfigError`. Aujourd'hui : tous modèles non résolus sont différés au 1er appel. | WARNING |
| EC26 | Invalidation runtime (`_check_mtime_and_invalidate`) dépasse `WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S` OU lève `ModelConfigError` | Cache **précédent conservé**, WARNING FR loggué, pas de downtime. Retry automatique au prochain check throttlé 1 s (R18). | WARNING |
| EC27 | Mode installed (pas de `.git` détectable) sans `WINCORP_URD_PATH` défini | `ModelConfigError` FATAL startup FR actionnable pointant la définition de la variable dans le `.env` du service. | FATAL startup |

---

## 8. Hiérarchie des exceptions (`exceptions.py`)

```
OdinLlmError (base, hérite de Exception)
├── ModelConfigError (EC1, EC2, EC3, EC6, EC9, EC17, EC19, EC22, EC24)
│   └── ModelConfigSchemaError (EC4, EC5) — JSONPath précis
├── SecretMissingError (EC7, EC8) — hérite de ModelConfigError
├── ProviderNotInstalledError (EC10, EC11, EC12)
├── ExtraKwargsForbiddenError (EC23) — hérite de ModelConfigError
├── ModelNotFoundError (EC13, EC21) — hérite de OdinLlmError UNIQUEMENT (plus de KeyError)
├── CapabilityMismatchError (EC14, EC20) — hérite de OdinLlmError UNIQUEMENT (plus de ValueError)
└── ModelAuthenticationError (EC15) — wrap anthropic.AuthenticationError, clé strippée (R10c)
```

### Changement de design vs v1.0 (PB-011)

- `ModelNotFoundError` ne dérive **plus** de `KeyError`, `CapabilityMismatchError` ne dérive **plus** de `ValueError`.
- Raison : héritage multiple de types built-in mélange sémantique métier et sémantique Python, piège les `except KeyError:` génériques qui avalent d'autres problèmes.
- Sucre syntaxique via helpers : `is_model_not_found(exc)`, `is_capability_mismatch(exc)` (module `helpers.py`).
- Base `MimirLlmError` **renommée** `OdinLlmError` (cohérence repo `wincorp-odin`).

```python
# helpers.py
def is_model_not_found(exc: BaseException) -> bool:
    """Sucre syntaxique remplace ``except ModelNotFoundError``."""
    return isinstance(exc, ModelNotFoundError)

def is_capability_mismatch(exc: BaseException) -> bool:
    return isinstance(exc, CapabilityMismatchError)
```

---

## 9. Validation startup — ordre strict

Exécuté par `validate_all_models()` sous budget timeout (R19).

### 9.1 Ordre

1. **Résolution du chemin (R17)** — bifurcation explicite selon contexte :
   - Si `WINCORP_URD_PATH` défini : `Path(env).resolve()` + `_assert_under_allowed_root(resolved)` ($HOME ou project_root détecté en remontant jusqu'à 10 parents cherchant `.git`/`pyproject.toml`). Si hors racine autorisée → `ModelConfigError` générique FR (aucun chemin révélé, log DEBUG local uniquement).
   - Sinon, détection dev local : parcourir `Path(__file__).resolve().parents[:5]` à la recherche d'un ancêtre contenant `.git`. Si trouvé → `ancêtre.parent / "wincorp-urd" / "referentiels" / "models.yaml"` si existe, sinon break sans fallback.
   - Sinon (installed sans env var) → `ModelConfigError` FATAL startup FR actionnable (EC27).
   - Erreurs possibles : EC1 (fichier non trouvé), EC22 (env var pointe vers dossier inexistant), EC27 (installed sans env var).
2. **Détection conflits OneDrive** — cf. §9.2 détail 4 patterns + regex.
3. **Vérification taille** — `stat().st_size > 1_048_576` → `ModelConfigError` (R15, EC24).
4. **Parsing YAML** — `yaml.safe_load()` strict (R14). Erreur → `ModelConfigError` avec ligne/colonne (EC2) ou « safe_load violation » (EC24).
5. **Structure racine** — présence clé `models`, liste non vide. Erreur → `ModelConfigError` (EC3).
6. **Version de schéma** — `config_version` supportée. Erreur → `ModelConfigError` (EC19).
7. **Schéma per-model** — Pydantic `ModelsFile` valide chaque bloc. **Agrégation** des erreurs (EC4, EC5, EC6).
8. **Whitelist `extra_kwargs`** — pour chaque modèle non-disabled : chaque clé de `extra_kwargs` ∈ `PROVIDER_EXTRA_KWARGS_WHITELIST[provider]` sinon `ExtraKwargsForbiddenError` (EC23, R13).
9. **Interpolation `${VAR}`** — collecte toutes les variables manquantes avant de lever (EC7, EC8, EC9).
10. **Résolution `use:`** — `importlib.import_module` + `getattr` + `callable()` pour chaque modèle non-disabled, **SANS INSTANCIER** (EC10, EC11, EC12). Sous budget R19 — modèles non résolus marqués différés (EC25).
11. **Pas de round-trip Anthropic** — clé API vérifiée au **1er appel réel** (EC15).

**Comportement lazy** : si le consommateur appelle directement `create_model` sans `validate_all_models`, la validation s'exécute au 1er appel (same logic, same order). Validation **idempotente**.

### 9.2 Détection conflits OneDrive (PB-004)

Glob **à combiner** dans le répertoire de `models.yaml` :

| Pattern | OS / Origine |
|---------|--------------|
| `models-DESKTOP-*.yaml` | Windows OneDrive version EN (ex `models-DESKTOP-ABC123.yaml`) |
| `models (conflit*).yaml` | Windows OneDrive version FR (ex `models (conflit 1).yaml`) |
| `models (conflicted copy*).yaml` | macOS/Linux OneDrive (ex `models (conflicted copy 2025-04-20).yaml`) |
| `models.yaml~`, `models.yaml.bak` | Backups éditeurs (vim, nano, éditeurs génériques) |

**Regex fallback** (catch divers patterns exotiques) :
```python
CONFLICT_REGEX = re.compile(
    r"models[ _\-\(].*(conflict|conflit|DESKTOP)[\-\s].*\.yaml$",
    re.IGNORECASE,
)
```

Si **l'un** des patterns match → `ModelConfigError` FR listant les fichiers conflictuels + instruction « résoudre manuellement (garder la bonne version), puis relancer ».

**Prérequis bloquant** : `wincorp-urd/` doit être hors OneDrive (junction NTFS ou clone local). Vérification documentée §16.

---

## 10. Thread safety + cache + invalidation mtime

### 10.1 Structures

```python
# src/wincorp_odin/llm/factory.py
import threading, time
from pathlib import Path

_cache: dict[tuple[str, bool], Any] = {}          # clé = (name, thinking_enabled)
_cache_lock = threading.Lock()                     # protège swap atomique cache + configs + mtime
_yaml_mtime: float | None = None                   # dernière mtime vue
_last_mtime_check: float = 0.0                     # R18 throttle 1/s
_resolved_configs: dict[str, ModelConfig] = {}     # post-validation (swap atomique, cf. §10.3)
_deferred_resolutions: set[str] = set()            # noms modèles à résoudre lazy (R19, EC25)
_STARTUP_TIMEOUT_S: float = 5.0                    # R19 — override via WINCORP_LLM_VALIDATE_TIMEOUT_S
_RUNTIME_TIMEOUT_S: float = 0.5                    # R19b — override via WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S
```

### 10.2 Clé de cache

**Choix tranché** : `tuple[str, bool] = (name, thinking_enabled)`.

Rationale : tuples hashables nativement, pas de parsing de string, pas de risque de collision.

### 10.3 Invalidation mtime — copy-on-write (PB-003 + PB-010 + PB-019)

**Choix tranché v1.2** : stratégie **copy-on-write** — validation **hors lock** (peut prendre plusieurs centaines de ms sous pression), swap atomique **sous lock court** (dict swap O(n) mais sans I/O).

Rationale PB-019 : la v1.1 appelait `validate_all_models()` **sous** `_cache_lock`, détenant potentiellement le lock 5 s (budget R19 startup). Sous cette fenêtre, tous les threads `create_model` concurrents stallent. v1.2 sépare la phase lente (validation, I/O disque, résolution imports) de la phase atomique (swap des références de cache). Budget runtime distinct (R19b, défaut 500 ms) pour éviter qu'un YAML lent gèle la prod.

```python
def _check_mtime_and_invalidate() -> None:
    """Throttled mtime check + copy-on-write reload."""
    global _yaml_mtime, _last_mtime_check
    now = time.monotonic()
    # R18 — throttle : max 1 stat()/s
    if now - _last_mtime_check < 1.0:
        return
    _last_mtime_check = now

    yaml_path = _resolve_yaml_path()
    current = yaml_path.stat().st_mtime
    if _yaml_mtime is not None and current <= _yaml_mtime:
        return

    # --- Reload HORS lock (peut prendre plusieurs centaines de ms) ---
    try:
        new_configs = _load_and_validate_models(
            timeout_s=_RUNTIME_TIMEOUT_S,  # R19b, défaut 0.5s
        )
    except (ModelConfigError, TimeoutError) as e:
        # EC26 — cache conservé, pas de downtime
        logger.warning(
            "Invalidation mtime échouée (cache conservé, pas de downtime) : %s", e,
        )
        return

    # --- Swap atomique SOUS lock court (double-check R7) ---
    with _cache_lock:
        current_reloaded = yaml_path.stat().st_mtime
        if current_reloaded <= (_yaml_mtime or 0):
            # Autre thread a déjà fait le swap, ou mtime est redescendu — abandon
            return
        logger.info(
            "[INFO] models.yaml rechargé (mtime: %s -> %s)",
            _yaml_mtime, current_reloaded,
        )
        _resolved_configs.clear()
        _resolved_configs.update(new_configs)
        _cache.clear()
        _registry._class_cache.clear()
        _deferred_resolutions.clear()
        _yaml_mtime = current_reloaded
```

**Invariants** :
- Le lock n'est jamais détenu pendant I/O disque ni pendant `importlib.import_module`.
- En cas d'échec du reload (timeout runtime, config invalide), `_resolved_configs` / `_cache` / `_yaml_mtime` restent **strictement inchangés**. Les threads concurrents continuent d'utiliser la config précédente.
- Le `_last_mtime_check` est mis à jour **avant** la tentative de reload : un reload échoué re-déclenchera une tentative 1 s plus tard (throttle R18). Pas de retry serré.
- `_load_and_validate_models()` est une variante interne de `validate_all_models()` qui **retourne** un `dict[str, ModelConfig]` frais sans muter l'état global (pur). `validate_all_models()` au startup appelle `_load_and_validate_models(timeout_s=_STARTUP_TIMEOUT_S)` puis mute l'état.
- Distinction des budgets : `_STARTUP_TIMEOUT_S` = `WINCORP_LLM_VALIDATE_TIMEOUT_S` (R19, défaut 5 s) / `_RUNTIME_TIMEOUT_S` = `WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S` (R19b, défaut 0.5 s).

### 10.4 Double-checked locking pour instanciation

```python
def create_model(name: str, thinking_enabled: bool = False) -> ChatAnthropic:
    _check_mtime_and_invalidate()
    key = (name, thinking_enabled)
    # Lecture sans lock (GIL)
    instance = _cache.get(key)
    if instance is not None:
        return instance
    # Écriture sous lock
    with _cache_lock:
        instance = _cache.get(key)  # double-check après acquisition
        if instance is not None:
            return instance
        cfg = _resolved_configs.get(name)
        if cfg is None or cfg.disabled:
            raise ModelNotFoundError(...)
        if thinking_enabled and not cfg.supports_thinking:
            raise CapabilityMismatchError(...)
        klass = _registry.resolve_class(cfg.use)
        kwargs = _build_kwargs(cfg, thinking_enabled)   # strip api_key des logs (R10b)
        try:
            instance = klass(**kwargs)
        except Exception as e:
            # Wrap en strippant la clé (R10b/R10c)
            raise ModelAuthenticationError(...) from e
        _cache[key] = instance
        return instance
```

### 10.5 API échappatoire `_reload_for_tests()` (PB-015)

```python
def _reload_for_tests() -> None:
    """USAGE INTERNE — fixtures + REPL admin local.
    N'est PAS exporté (pas dans __all__). Pour prod futur : admin_reload(token) séparé.
    """
    global _yaml_mtime, _last_mtime_check
    with _cache_lock:
        _cache.clear()
        _resolved_configs.clear()
        _registry._class_cache.clear()   # invalider aussi le cache classe (PB-012)
        _deferred_resolutions.clear()    # v1.2 — purge résolution différée (R19)
        _yaml_mtime = None
        _last_mtime_check = 0.0
```

---

## 11. Stratégie TDD — ordre d'écriture des tests

**Règle** : test **rouge** avant implémentation. Cible : **100% branch** sur `wincorp_odin.llm`.

### 11.1 Ordre figé (17 tests — +2 vs v1.0 pour PB-003, PB-010)

| # | Test | Couvre |
|---|------|--------|
| 1 | `test_r1_create_model_known_name_returns_chat_anthropic` | R1, R5 |
| 2 | `test_r1_create_model_passes_model_id_from_yaml` | R1 |
| 3 | `test_r1_create_model_passes_max_tokens_from_yaml` | R1 |
| 4 | `test_r1_create_model_same_key_returns_same_instance` | R1, R7 |
| 5 | `test_r1_create_model_different_names_return_different_instances` | R1 |
| 6 | `test_r2_thinking_enabled_true_applies_when_thinking_enabled_kwargs` | R2, R6 |
| 7 | `test_r2_thinking_enabled_false_no_thinking_kwarg` | R2 |
| 8 | `test_r2_cache_distingue_thinking_variants` | R2, R7 |
| 9 | `test_r4_api_key_interpolated_from_env` | R4 |
| 10 | `test_ec13_unknown_name_raises_model_not_found_error` | EC13, §8 |
| 11 | `test_ec7_missing_env_var_raises_secret_missing_error` | EC7, §8 |
| 12 | `test_ec2_malformed_yaml_raises_model_config_error` | EC2 |
| 13 | `test_ec1_yaml_file_not_found_raises_model_config_error` | EC1, R17 |
| 14 | `test_ec14_thinking_on_non_capable_raises_capability_mismatch` | EC14, R6 |
| 15 | `test_integration_real_urd_yaml_loads_all_declared_models` | Smoke URD réel |
| 16 | `test_r7_double_checked_mtime_no_double_clear` | R7, PB-003 |
| 17 | `test_r18_mtime_check_throttled` | R18, PB-010 |

Note v1.2 : les 17 tests ordonnés ci-dessus restent numérotés à l'identique. Les 3 tests structurels ajoutés (R19b copy-on-write, R17 installed, R17 path traversal) vivent dans §11.2 (complémentaires, non-bloquants pour l'ordre TDD initial).

### 11.2 Tests complémentaires

- `test_r7_concurrent_create_model_instantiates_once` (threading + `Barrier`)
- `test_ec16_yaml_mtime_change_triggers_invalidation`
- `test_ec17_onedrive_conflict_raises_error` (4 patterns + regex)
- `test_ec6_duplicate_name_raises_config_error`
- `test_ec10_missing_provider_raises_provider_not_installed`
- `test_r11_disabled_model_raises_not_found`
- `test_r13_extra_kwargs_whitelist_rejects_base_url` (EC23)
- `test_r13_extra_kwargs_whitelist_accepts_temperature`
- `test_r14_yaml_unsafe_tag_rejected` (ex `!!python/object`)
- `test_r15_yaml_size_exceeds_1mb_rejected` (EC24)
- `test_r17_urd_path_traversal_rejected` (chemin hors `$HOME` + hors `project_root`, message générique **sans** exposer le path)
- `test_r17_installed_requires_env_var` (v1.2 — pas de `.git` détectable + pas de `WINCORP_URD_PATH` → `ModelConfigError` FATAL avec message FR actionnable, EC27)
- `test_r17_dev_mode_autodetects_wincorp_dev` (v1.2 — `.git` présent dans un parent → trouve `../wincorp-urd/referentiels/models.yaml` sans env var)
- `test_r19_validate_timeout_defers_use_resolution` (EC25)
- `test_r19b_runtime_reload_copy_on_write_no_stall` (v1.2 — PB-019 : 2 threads, l'un déclenche invalidation lente via `_check_mtime_and_invalidate` sous `WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S=2.0` avec YAML ralenti artificiellement, l'autre appelle `create_model` concurrent → l'appel concurrent ne doit pas stall plus de 10 ms ; assertion via `time.monotonic()` avant/après)
- `test_r19b_runtime_reload_failure_keeps_previous_cache` (v1.2 — EC26 : reload runtime échoue (timeout ou YAML invalide) → `_resolved_configs` précédent inchangé, WARNING loggué, `create_model` concurrent continue de fonctionner)
- `test_reload_for_tests_clears_cache_and_registry` (PB-012)
- `test_registry_cache_invalidated_by_reload` (PB-012)
- `test_api_key_never_in_repr` (R10)
- `test_api_key_stripped_from_authentication_error` (R10c)
- `test_api_key_stripped_from_build_kwargs_exception` (R10b)
- `test_model_not_found_not_a_keyerror` (PB-011 — assertion `not isinstance(exc, KeyError)`)
- `test_capability_mismatch_not_a_valueerror` (PB-011)
- `test_helper_is_model_not_found_returns_true_on_match` (PB-011)
- `test_legacy_wrapper_emits_deprecation_warning_fr` (§18)

### 11.3 Structure fichiers tests

```
wincorp-odin/tests/llm/
├── __init__.py
├── conftest.py                     # fixtures + _reset_factory_state autouse
├── test_factory.py                 # R1-R11, EC13, EC14, EC16, EC18, R7, R18
├── test_config.py                  # parsing + interpolation, EC1-EC12, EC17, EC19, EC22, R14-R17
├── test_registry.py                # resolve_class, EC10-EC12, cache invalidation
├── test_whitelist.py               # R13, EC23
├── test_exceptions.py              # hiérarchie, redaction, PB-011
├── test_legacy.py                  # §18 deprecation warning
├── test_security.py                # R10/R10b/R10c, R17 path traversal
├── test_integration.py             # @pytest.mark.integration smoke URD réel
└── fixtures/
    ├── models_minimal.yaml
    ├── models_full.yaml
    ├── models_malformed_syntax.yaml
    ├── models_missing_field.yaml
    ├── models_duplicate_name.yaml
    ├── models_wrong_type.yaml
    ├── models_extra_kwargs_forbidden.yaml
    ├── models_unsafe_yaml_tag.yaml       # !!python/object
    ├── models_huge.yaml                  # > 1 Mo généré au test
    └── models_conflict_<pattern>.yaml    # 4 fichiers pour les 4 globs
```

### 11.4 Fixtures `conftest.py`

```python
@pytest.fixture(autouse=True)
def _reset_factory_state(monkeypatch):
    """Vide cache + mtime + registry cache + deferred resolutions entre chaque test (PB-012 + v1.2)."""
    from wincorp_odin.llm import factory, _registry
    factory._cache.clear()
    factory._resolved_configs.clear()
    factory._deferred_resolutions.clear()   # v1.2 — purge état différé R19
    factory._yaml_mtime = None
    factory._last_mtime_check = 0.0
    _registry._class_cache.clear()

@pytest.fixture
def mock_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-xxxxxxxxxxxxxxxxxxxx")

@pytest.fixture
def minimal_yaml(tmp_path):
    src = Path(__file__).parent / "fixtures" / "models_minimal.yaml"
    dst = tmp_path / "models.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst

@pytest.fixture
def patched_yaml_path(monkeypatch, minimal_yaml):
    monkeypatch.setenv("WINCORP_URD_PATH", str(minimal_yaml.parent.parent))

@pytest.fixture
def mock_chat_anthropic(monkeypatch):
    mock = MagicMock(name="ChatAnthropic")
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", mock)
    return mock
```

### 11.5 Mocks — règles

- **Zéro appel réseau**. `unittest.mock.patch` sur `ChatAnthropic`.
- **Assert sur `call_args.kwargs`**, pas sur l'objet retourné.
- **`monkeypatch` pour env vars et paths**, jamais `os.environ` direct.
- **Pas de mock pour les yaml fixtures** — vrais fichiers dans `tmp_path`.
- **Tests concurrents** : `threading.Barrier(N)` pour synchroniser les threads avant la course.

---

## 12. Coverage & CI

### 12.1 Commandes

```bash
# Local, module — cible 100% branch
cd wincorp-odin
uv run pytest tests/llm/ --cov=wincorp_odin.llm --cov-branch --cov-report=term-missing --cov-fail-under=100

# CI (ANTHROPIC_API_KEY absent) — doit passer 100%
ANTHROPIC_API_KEY= uv run pytest tests/llm/ -v
```

### 12.2 Règles CI

- **Jamais `skipif ANTHROPIC_API_KEY present`** — tests unitaires ET intégration sans clé.
- **`@pytest.mark.integration`** sur smoke URD réel — skipif `URD_REAL.exists()` uniquement.
- **`--cov-fail-under=100` sur `wincorp_odin.llm`** bloque tout merge.
- **Grep hook** : CI échoue si `yaml.load(` ou `yaml.unsafe_load(` trouvé dans `src/` (R14).

### 12.3 Portabilité multi-PC

- Jamais de chemins en dur. Toujours `Path()` + `tmp_path`.
- `wincorp-odin/` reste **hors OneDrive** (junction NTFS). Confirmation `memory/user_machines.md`.
- Encoding UTF-8 explicite partout.
- `.gitattributes` : `*.yaml text eol=lf`.

---

## 13. Dépendances

### 13.0 Impact architecture (PB-001 — décision B appliquée)

**Décision structurante validée 20/04/2026 soir** :

- Module **déplacé** de `wincorp-mimir/src/mimir/llm/` vers **`wincorp-odin/src/wincorp_odin/llm/`**.
- Package Python : `wincorp_odin.llm` (pas `wincorp_common.llm`, pas `mimir.llm`).
- Raison : règle `wincorp-mimir/.claude/CLAUDE.md` ligne 49 « Pas de dépendance externe autre que Pydantic » préservée. `wincorp-mimir` reste librairie métier autonome (fiscal/FEC/PCG/juridique).
- **Règles d'isolation DURES (non négociables)** :
  1. `wincorp-odin` n'importe JAMAIS `wincorp_common` (mimir).
  2. `wincorp-mimir` n'importe JAMAIS `wincorp_odin`.
  3. Namespaces strictement séparés : `wincorp_common.*` vs `wincorp_odin.*`. Jamais de namespace-package fusionné.
  4. Un consommateur métier (heimdall, bifrost, thor) qui a besoin d'un LLM pour une opération métier **importe les deux** et **compose** côté caller. Aucune fonction mimir ne reçoit un LLM implicitement — passage en argument explicite via signature.
- Documenté dans `wincorp-odin/.claude/CLAUDE.md` ET `wincorp-mimir/.claude/CLAUDE.md`. Vérification manuelle à chaque PR des deux côtés.

### 13.1 Dépendances runtime (`pyproject.toml` existant `wincorp-odin`)

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.5.0",
    "pyyaml>=6.0.1",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "anthropic>=0.40.0",
]
```

Impact disque : ~+15 MB (langchain-core + anthropic SDK). **Déjà installé** dans `wincorp-odin` (skeleton livré 20/04 soir). Aucune modification `wincorp-mimir/pyproject.toml`.

### 13.2 Python 3.12 strict (PB-002)

- `requires-python = ">=3.12"` **strict**, pas 3.10 ni 3.11.
- `[tool.ruff] target-version = "py312"`.
- `[tool.mypy] python_version = "3.12"`.
- Retirer toute mention 3.10 dans docstrings, commentaires, docs. Cohérence avec `.claude/CLAUDE.md` d'Odin ligne 42 « Python 3.12+ strict (pas de 3.10 pour éviter incohérences mimir) ».
- Exploite typage PEP 695 (`type X = ...`), pattern matching, `ExceptionGroup` pour agrégation erreurs dans `validate_all_models()`.

### 13.3 Dépendances de dev

Déjà présentes dans `wincorp-odin/[project.optional-dependencies].dev` : `pytest`, `pytest-cov`, `ruff`, `mypy`. Aucun ajout.

---

## 14. Messages d'erreur FR — types représentatifs

> Objectif : Tan non-dev doit comprendre l'erreur sans regarder la stack.

1. **EC1** — `[ERREUR] Fichier de configuration LLM introuvable. Chemin tenté : <chemin SI sous racine autorisée, sinon [masqué]>. Vérifier la variable d'environnement WINCORP_URD_PATH ou la présence de wincorp-urd/ à côté de wincorp-dev/.`
2. **EC4** — `[ERREUR] Modèle 'sonnet' (ligne 12 de models.yaml) incomplet. Champs obligatoires manquants : [use, model, api_key]. Ajouter ces champs ou retirer ce bloc.`
3. **EC5** — `[ERREUR] Modèle 'haiku' : champ 'supports_thinking' attendu bool (true/false), reçu str ("yes"). Corriger la valeur.`
4. **EC6** — `[ERREUR] Nom 'sonnet' déclaré 2 fois dans models.yaml (lignes 12 et 34). Les noms doivent être uniques.`
5. **EC7** — `[ERREUR] Variable d'environnement ANTHROPIC_API_KEY absente. Définir la clé dans .env ou l'exporter avant de lancer le process.`
6. **EC9** — `[ERREUR] Variable d'environnement ${UNKNWN_API_KEY} référencée dans models.yaml (ligne 14) mais introuvable. Typo probable ?`
7. **EC10** — `[ERREUR] Package 'langchain_openai' requis par models.yaml (modèle 'deepseek') mais non installé. Exécuter : uv pip install langchain-openai`
8. **EC13** — `[ERREUR] Modèle 'clade-sonet' introuvable. Typo probable ? Modèles disponibles : [haiku, opus, opus-thinking, sonnet].`
9. **EC14** — `[ERREUR] Modèle 'haiku' ne supporte pas le mode thinking (supports_thinking: false dans models.yaml). Modèles thinking-compatibles : [opus-thinking, sonnet-thinking].`
10. **EC15** — `[ERREUR] Authentification Anthropic échouée (401). Vérifier la validité de ANTHROPIC_API_KEY — clé tronquée, révoquée ou invalide.` (La clé originale est **masquée** dans le message ET dans la chaîne `__cause__`.)
11. **EC16** — `[INFO] models.yaml modifié sur le disque (14:23 → 14:47). Invalidation du cache LLM et rechargement en cours.`
12. **EC17** — `[ERREUR] Conflit OneDrive détecté dans wincorp-urd/referentiels/ : [models-DESKTOP-ABC123.yaml, models (conflit 1).yaml] présents à côté de models.yaml. Résoudre manuellement (garder la bonne version) avant de relancer.`
13. **EC23** — `[ERREUR] Modèle 'custom' : extra_kwargs contient les clés interdites [base_url, api_key]. Provider langchain_anthropic:ChatAnthropic — whitelist : [temperature, top_p, top_k, stop_sequences, streaming]. Retirer les clés interdites.`
14. **EC24** — `[ERREUR] models.yaml suspect — taille 2.3 Mo > 1 Mo autorisé. Vérifier le fichier (corruption, duplications ?).`
15. **EC25** — `[WARN] Validation startup > 5s — résolution du provider 'langchain_custom:ChatCustom' sautée par timeout, sera réessayée au 1er create_model("custom"). Vérifier le démarrage au premier appel.`
16. **EC26** — `[WARN] Invalidation mtime échouée (cache conservé, pas de downtime) : <erreur>.` (v1.2 — reload runtime échoué, retry automatique au prochain throttle 1 s.)
17. **EC27** — `[ERREUR] WINCORP_URD_PATH obligatoire en déploiement installed. Définir la variable dans le .env du service (valeur = chemin absolu vers le dossier wincorp-urd/). Détection dev/prod : présence de .git dans les 5 parents de <chemin module>.` (v1.2 — mode installed sans env var.)
18. **R17** — `[ERREUR] Le chemin WINCORP_URD_PATH est hors des racines autorisées. Vérifier la variable d'environnement.` (**Aucune** information sur le chemin tenté — log DEBUG local uniquement.)

---

## 15. Extensibilité — scénario DeepSeek

Phase 1.2 : ajouter DeepSeek en production avec un **coût d'ajout minimal**.

1. `uv pip install langchain-deepseek` (dépendance provider).
2. Ajouter la clé dans `_whitelist.py` :
   ```python
   PROVIDER_EXTRA_KWARGS_WHITELIST["langchain_deepseek:ChatDeepSeek"] = frozenset({
       "temperature", "top_p", "max_tokens", "stop",
   })
   ```
3. Ajouter bloc dans `wincorp-urd/referentiels/models.yaml` :
   ```yaml
   - name: "deepseek-chat"
     display_name: "DeepSeek Chat V3"
     use: "langchain_deepseek:ChatDeepSeek"
     model: "deepseek-chat"
     api_key: "${DEEPSEEK_API_KEY}"
     max_tokens: 8192
     supports_thinking: false
     supports_vision: false
   ```
4. Exporter `DEEPSEEK_API_KEY`.
5. Appeler `create_model("deepseek-chat")`.

### Précision v1.1 — « zéro ligne » rectifiée (PB-014)

**Zéro ligne dans `factory.py`/`config.py`/`_registry.py`**. Mais nécessite :
- (a) **Bump typage retour** Phase 1.2 (`-> ChatAnthropic` → `-> BaseChatModel`) — 1 ligne dans `factory.py`.
- (b) **3 tests fixtures** minimum : `test_deepseek_create_model_ok`, `test_deepseek_extra_kwargs_whitelist`, `test_deepseek_mocked_instantiation`.
- (c) **Validation compatibilité** côté consommateurs : `.astream()`, `.invoke()`, `.bind_tools()` — DeepSeek implémente le contrat `BaseChatModel` mais avec des quirks (ex pas de `.astream_events` v2). Tester chaque consommateur (heimdall/api `/chat`, bifrost pipeline) AVANT déploiement prod.
- (d) **Entry whitelist `_whitelist.py`** à ajouter (cf. point 2 ci-dessus) — 1 ligne mais **obligatoire** (sinon `extra_kwargs` rejeté d'office).

Donc : **~5 lignes Python + 3 tests + 1 doc consommateur**. C'est la validation de l'architecture, mais pas un « zéro absolu » — on reste honnête sur le coût marginal.

---

## 16. Prérequis bloquants

- [ ] **Hook `block-secrets-commit.sh` durci** — scan `.yaml` pour patterns `sk-ant-*`, `sk-proj-*`, `sk-...`, `AKIA*`. Pré-requis absolu avant de commiter `models.yaml`.
- [ ] **`wincorp-urd/` hors OneDrive** — junction NTFS ou clone local. Si `wincorp-urd` est synchronisé OneDrive, les patterns de détection de conflit (§9.2) vont potentiellement matcher sur des copies légitimes du dossier lui-même. Vérification : `Get-Item wincorp-urd\` ne doit pas avoir attribut `OneDrive`.
- [ ] **`langchain-anthropic >= 0.3.0`** — versions antérieures n'exposent pas l'argument `thinking` proprement.
- [ ] **Création `wincorp-urd/referentiels/models.yaml`** (Phase 1.2 du plan DeerFlow).
- [ ] **`.gitattributes` `*.yaml text eol=lf`** ajouté dans `wincorp-urd`.
- [ ] **Python 3.12 installé sur tanfeuille ET tanph** — `uv python pin 3.12`.
- [ ] **Tests tanfeuille ET tanph verts** avant merge Phase 1.1.
- [ ] **Doc isolation Odin↔Mimir** propagée dans `wincorp-saga/SPEC-DRIVEN-DEVELOPMENT.md` (ajout section « packages séparés »).

---

## 17. Changelog arbitrage (hérité v1.0, allégé)

### 17.1 ~~Divergence 1 — Nommage package~~ → **RÉSOLU v1.1 par décision B**

La v1.0 gardait `wincorp_common.llm` dans `wincorp-mimir`. La v1.1 tranche **option B** (nouveau repo `wincorp-odin`, package `wincorp_odin.llm`) suite au verdict review adversarial PB-001. Voir §13.0.

### 17.2-17.9 — Autres arbitrages v1.0

Les 7 autres divergences de la v1.0 (schéma YAML unifié, structure fichiers, hiérarchie exceptions, clé cache tuple, signature retour typée, invalidation mtime à chaque appel, validation statique pure) **restent valides** sauf modifications explicites par les PB-001→PB-015. Cf. spec v1.0 archivée pour détail.

**Changements dérivés v1.1** :
- **17.4** — Hiérarchie exceptions : retrait de `KeyError` et `ValueError` comme parents (PB-011). `OdinLlmError` remplace `MimirLlmError`. +`ExtraKwargsForbiddenError`.
- **17.7** — Invalidation mtime : throttle 1 Hz ajouté (PB-010, R18) + double-checked mtime (PB-003, R7).
- **17.9** — `reload()` public → `_reload_for_tests()` privé (PB-015).

---

## 18. Plan de migration Phase 1.9

### 18.1 Objectif

Remplacer **tous** les usages directs de `ChatAnthropic()` dans l'écosystème par `create_model("<nom>")`. Pas d'effet Phase 1.1, mais les usages non migrés n'héritent d'aucune garantie (pas de cache, pas de whitelist, pas de masquage secrets).

### 18.2 Wrapper deprecated (`wincorp_odin.llm.legacy`)

```python
# src/wincorp_odin/llm/legacy.py
from __future__ import annotations
import warnings
from langchain_anthropic import ChatAnthropic

def deprecated_direct_chat_anthropic(**kwargs) -> ChatAnthropic:
    """Wrapper transitoire — pointe vers create_model.

    Usage uniquement transitoire pendant la migration Phase 1.9.
    Émet DeprecationWarning FR. À supprimer en Phase 2.0.
    """
    warnings.warn(
        "Usage direct de ChatAnthropic détecté. "
        "Migrer vers wincorp_odin.llm.create_model('<nom>') "
        "en définissant le modèle dans wincorp-urd/referentiels/models.yaml. "
        "Ce wrapper sera retiré en Phase 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ChatAnthropic(**kwargs)
```

### 18.3 Repos cible migration (ordre de priorité)

| Ordre | Repo | Fichiers concernés (grep `ChatAnthropic(`) | Criticité |
|-------|------|--------------------------------------------|-----------|
| 1 | `wincorp-heimdall/api/**` | `routers/chat.py`, `services/llm_client.py` | HIGH — entrée utilisateur |
| 2 | `wincorp-thor/worker/**` | `worker/image_decider.py`, `worker/builder.py` | HIGH — pipeline prod |
| 3 | `wincorp-bifrost/app/**` | `app/api/chat/route.ts` (JS SDK — remplacer par appel à heimdall) | MEDIUM — UI |
| 4 | Repos secondaires (skadi, muninn…) | cas par cas | LOW |

### 18.4 Procédure par repo

1. `rg "ChatAnthropic\(" src/` pour lister les sites d'appel.
2. Pour chaque site : choisir le nom logique (ex `"sonnet"`, `"haiku"`, `"opus-thinking"`) et vérifier sa présence dans `models.yaml`. Sinon : l'ajouter en Phase 1.2 **avant** migration.
3. Remplacer `ChatAnthropic(model="...", api_key=os.environ["..."], ...)` par `create_model("<nom>", thinking_enabled=...)`.
4. Retirer les `os.environ["ANTHROPIC_API_KEY"]` directs (plus nécessaires, gérés par factory).
5. Rebaser les tests : mocker `wincorp_odin.llm.create_model` au lieu de `langchain_anthropic.ChatAnthropic`.
6. CI verte → commit → review PR → merge.

### 18.5 Flag `@deprecated`

Dans chaque repo consommateur : lint custom (ruff règle custom ou `pytest_deprecated_call`) qui marque tout `import langchain_anthropic` ou `from langchain_anthropic import ...` hors `wincorp_odin/llm/_registry.py`. Warning en v1.1, erreur en Phase 2.0.

### 18.6 Scope v1.1

**Phase 1.1 ne migre rien**. Elle fournit l'infrastructure (`create_model`, wrapper legacy, doc) + le plan. La migration effective est l'objet de Phase 1.9 (post-1.4 circuit breaker, 1.5 retry, 1.6 tokens pour bénéficier de ces couches).

---

## 19. Benchmarks perf (à compléter pendant build)

Section créée pour remplir les chiffres mesurés pendant l'implémentation (cf. PB-010 — « < 1μs » remplacé par vraies mesures).

### 19.1 Environnements cibles

| Env | OS | Stockage YAML | Description |
|-----|----|--------------| ------------|
| E1 | Windows 11 (tanfeuille) | NVMe local, hors OneDrive (junction) | PC principal |
| E2 | Windows 11 (tanph) | NVMe local, hors OneDrive (junction) | PC bureau |
| E3 | VPS OVH Ubuntu | SSD | Prod worker |

### 19.2 Mesures attendues

| Métrique | Cible | Méthode | Résultat E1 | Résultat E2 | Résultat E3 |
|----------|-------|---------|-------------|-------------|-------------|
| `Path.stat().st_mtime` | < 50 μs | `timeit` 10k iter | TBD | TBD | TBD |
| `_check_mtime_and_invalidate()` cache hit (sous throttle) | < 200 ns | `timeit` 100k iter | TBD | TBD | TBD |
| `_check_mtime_and_invalidate()` cache hit (throttle expiré) | < 100 μs | — | TBD | TBD | TBD |
| `create_model` cache hit | < 500 ns | — | TBD | TBD | TBD |
| `validate_all_models()` 5 modèles | < 200 ms | — | TBD | TBD | TBD |
| `validate_all_models()` 20 modèles | < 800 ms | — | TBD | TBD | TBD |

**Note** : chiffres à figer au build, puis re-mesurés à chaque Phase majeure (1.4, 1.5, 1.6) pour détecter régression perf.

### 19.3 Mitigation si dépassement

- Si `stat()` > 500 μs sur E3 (SSD VPS) — investiguer FS backing, envisager tmpfs cache ou `inotify`/`fsnotify` opt-in.
- Si `validate_all_models()` > 2s avec < 20 modèles — profiler `importlib.import_module` (peut loader LangChain lourd en cascade).

### 19.4 Mitigation parsing (v1.2 — R16 retiré)

R15 (taille max 1 Mo via `Path.stat().st_size` **avant** parsing) couvre 99 % des cas pathologiques. Un parsing `yaml.safe_load` > 2s sans taille > 1 Mo est quasi-impossible sur un YAML bien formé — le parseur C natif traite ~50-100 Mo/s sur toute machine moderne.

Le DoS volontaire par YAML **anchors/alias pathologiques** (« billion laughs », explosion exponentielle via anchors répétés) reste théoriquement possible **mais borné** par R15 : l'expansion s'arrête au plus tard quand le buffer intermédiaire dépasse la taille acceptable en RAM — un YAML source < 1 Mo ne peut pas produire une explosion de RAM significative sans lever `MemoryError` bien avant 2 s de CPU.

R14 (`yaml.safe_load` obligatoire) protège contre les **constructeurs dangereux** (`!!python/object`, `!!python/module`, etc.). Les expansions d'anchors restent possibles mais bornées par R15.

Le timeout parsing dédié (ex-R16) a été **retiré en v1.2** : la seule implémentation portable multi-OS (`signal.alarm` Unix + `threading.Timer` Windows) ne peut pas interrompre proprement un call C natif de `yaml.safe_load`. Le timeout Windows ne faisait que logguer post-parse sans arrêter quoi que ce soit — théâtre de garde-fou. R15 couvre le risque réel au bon niveau.

---

## 20. Changelog v1.0 → v1.1 — 15 corrections appliquées

| PB | Sévérité | Titre | Statut v1.1 | Section modifiée |
|----|----------|-------|-------------|-------------------|
| PB-001 | BLOCKER | Conflit dépendance mimir | **APPLIQUÉ — option B** : module migré `wincorp-odin`, package `wincorp_odin.llm`, isolation dure documentée §13.0 | Tout le doc (renommage), §13.0, §16, §18 |
| PB-002 | BLOCKER | Version Python | **APPLIQUÉ** — `>=3.12` strict, cohérence ruff/mypy, retrait 3.10 | §13.2 |
| PB-003 | BLOCKER | Race condition `_yaml_mtime` | **APPLIQUÉ** — double-checked mtime post-lock, re-`stat()`, test test_r7_double_checked_mtime | R7, §10.3, §11.1 test 16 |
| PB-004 | MAJOR | Détection OneDrive sous-spécifiée | **APPLIQUÉ** — 4 patterns glob + regex fallback + prérequis §16 | §9.2, EC17, §16 |
| PB-005 | MAJOR | `extra_kwargs` échappatoire | **APPLIQUÉ** — R13 whitelist stricte, `ExtraKwargsForbiddenError`, EC23, `_whitelist.py` | R13, §8, EC23, §5 |
| PB-006 | MAJOR | Secrets dans logs/traces | **APPLIQUÉ** — `__repr__` override + R10b strip `_build_kwargs` + R10c nettoyage `__cause__` | R10/R10b/R10c, §3.3, §3.4 EC15 |
| PB-007 | MAJOR | YAML injection | **APPLIQUÉ v1.1** — R14 `safe_load` + R15 taille max 1 Mo + R16 timeout 2s + EC24. **Révisé v1.2** : R16 retiré (cf. §19.4), R14 + R15 conservés | R14/R15, EC24, §19.4 |
| PB-008 | MAJOR | Path traversal `WINCORP_URD_PATH` | **APPLIQUÉ v1.1** — R17 contrainte racine autorisée + message générique sans chemin tenté. **Renforcé v1.2** : R17 bifurqué dev (auto-détection `.git`) vs installed (env var obligatoire), suppression fallback silencieux vers `$HOME` | R17, EC1, EC27, §14 msg R17 |
| PB-009 | MAJOR | Rétro-compat Phase 1.9 | **APPLIQUÉ** — §18 plan migration complet, wrapper `legacy.py`, liste repos priorisée | §18, §5 arbo, §2 note garantie |
| PB-010 | MAJOR | Perf mtime sur OneDrive | **APPLIQUÉ** — R18 throttle 1/s + §19 benchmarks à compléter | R18, §10.3, §19, test 17 |
| PB-011 | MAJOR | Héritage `KeyError`/`ValueError` | **APPLIQUÉ** — retrait héritages built-in, helpers `is_model_not_found`/`is_capability_mismatch`, renommage `MimirLlmError` → `OdinLlmError` | §8, helpers.py, §3.1 |
| PB-012 | MAJOR | Tests ordre-dépendants `_registry` cache | **APPLIQUÉ** — `_reset_factory_state` invalide aussi `_registry._class_cache`, test dédié | §11.4, §10.5 |
| PB-013 | MAJOR | Timeout `validate_all_models()` | **APPLIQUÉ** — R19 budget env `WINCORP_LLM_VALIDATE_TIMEOUT_S` + EC25 + résolution différée | R19, EC25 |
| PB-014 | MINOR | Scénario DeepSeek pas « zéro ligne » | **APPLIQUÉ** — §15 précisé : ~5 lignes + 3 tests + 1 doc consommateur | §15 |
| PB-015 | MINOR | `reload()` public = DoS | **APPLIQUÉ** — renommé `_reload_for_tests`, hors `__all__`, admin_reload futur | §3.1, §10.5, §2 OUT |

### Ratio spec size

- v1.0.0 : 713 lignes (estimation cat).
- v1.1.0 : ~860 lignes (+21%). Augmentation concentrée §9.2 (conflits OneDrive détaillés), R10-R18-R19 (règles nouvelles), §13.0 (archi), §18 (plan migration), §19 (benchmarks), §20 (ce changelog).

### Points laissés ouverts v1.1 — traités en v1.2

1. ~~**R16 timeout parsing portable**~~ — **RÉSOLU v1.2** : R16 retiré, mitigation basée sur R14+R15 documentée §19.4.
2. ~~**R17 path traversal — détection `project_root`**~~ — **RÉSOLU v1.2** : R17 bifurqué dev (auto-détection `.git`) vs installed (env var **obligatoire**, pas de fallback `$HOME`). Message EC27 FR actionnable.
3. **R19 budget timeout + résolution différée** — le flag `_use_resolution_deferred=True` par modèle n'est pas propagé à `ModelConfig` (dataclass frozen). **Confirmé v1.2** — option B retenue : registre séparé `_deferred_resolutions: set[str]` dans §10.1, purgé par `_reload_for_tests` et par le swap atomique §10.3.

### Points ouverts traités post-v1.2

Aucun — l'architecture copy-on-write (PB-019) a été appliquée, les 2 points ouverts v1.1 sont résolus.

---

## 21. Changelog v1.1 → v1.2 — 3 corrections structurelles post re-review

| Issue | Sévérité | Titre | Statut v1.2 | Section modifiée |
|-------|----------|-------|-------------|-------------------|
| PB-019 | MAJOR | Budget re-validation runtime sous lock → stall 5 s potentiel | **APPLIQUÉ** — stratégie copy-on-write : validation hors lock, swap atomique sous lock court. Budget runtime distinct `WINCORP_LLM_VALIDATE_RUNTIME_TIMEOUT_S` (R19b, défaut 0.5 s) | §10.1 (structures + 2 budgets), §10.3 (pseudo-code complet copy-on-write), R19b (§6), EC26 (§7), §14 msg EC26, §11.2 tests R19b (2 nouveaux) |
| Point ouvert #1 | MAJOR | R16 timeout parsing 2s non-interrompable portable | **APPLIQUÉ — retrait** — R16 supprimé (§6, §9 étape 4, §5 docstring config.py, §11.2 test retiré, EC24 reformulé). R15 (1 Mo) + R14 (safe_load) couvrent le risque réel. Justification §19.4 « Mitigation parsing » | §6 R16, §9.1 étape 4, §5, §11.2 test retiré, EC24, §19.4 (nouvelle section) |
| Point ouvert #2 | MAJOR | R17 fragile en site-packages, fallback silencieux `$HOME` attack surface | **APPLIQUÉ** — R17 bifurqué : mode explicite `WINCORP_URD_PATH` (obligatoire installed) vs mode implicite dev (auto-détection `.git` dans 5 parents). Helper `_assert_under_allowed_root` générique. Message FR actionnable EC27 | §6 R17 (réécrit), §9.1 étape 1 (réécrit), EC27 (§7), §14 msg EC27, §11.2 tests R17 installed + dev + traversal |

### Notes v1.2

- Numérotation des 17 tests figés §11.1 **inchangée** — les 3 nouveaux tests R17/R19b vivent dans §11.2 (complémentaires). Aucune renumérotation nécessaire.
- Structures `_deferred_resolutions: set[str]`, `_STARTUP_TIMEOUT_S`, `_RUNTIME_TIMEOUT_S` ajoutées §10.1.
- `_reload_for_tests()` purge désormais `_deferred_resolutions` (§10.5).
- Pas de rupture de contrat d'interface publique (§3.1 inchangée).
- Un nouveau wrapper interne `_load_and_validate_models(timeout_s: float) -> dict[str, ModelConfig]` pur (sans mutation d'état) est introduit implicitement par §10.3 — doit être extrait de la logique `validate_all_models()` au build.

---

## Changelog

| Version | Date | Modification |
|---------|------|--------------|
| 1.0.0 | 2026-04-20 | Création initiale. Consolidation de 3 specs SDD Opus parallèles (archi / edge cases / testability). Review adversarial lancée. |
| 1.1.0 | 2026-04-20 | 15 corrections PB-001→PB-015 appliquées post review adversarial. Déplacement `wincorp-mimir` → `wincorp-odin` (option B). `MimirLlmError` → `OdinLlmError`. Package `wincorp_odin.llm`. Sections nouvelles §13.0, §16 étendu, §18, §19, §20. 17 tests figés (15 + 2 nouveaux PB-003/PB-010). Spec prête build Phase 1.1. |
| 1.2.0 | 2026-04-20 | 3 corrections structurelles post re-review adversarial (cf. §21) : (1) PB-019 budget runtime sous lock → copy-on-write avec R19b (§10.3 réécrit, nouveau budget runtime 500 ms, EC26, 2 nouveaux tests R19b) ; (2) retrait R16 timeout parsing YAML non-interrompable portable (§19.4 justification, EC24 reformulé) ; (3) R17 bifurqué dev vs installed avec env var obligatoire en installed (EC27, helper `_assert_under_allowed_root`, 3 nouveaux tests R17). Interface publique inchangée. |

---

## @spec

`@spec specs/llm-factory.spec.md v1.2` — à ajouter en en-tête de chaque fichier source `src/wincorp_odin/llm/*.py` dès l'implémentation.
