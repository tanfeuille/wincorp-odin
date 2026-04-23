# valkyries — Specification

> **Statut :** DRAFT
> **Version :** 1.2
> **Niveau :** 2 (standard)
> **Auteur :** Tan Phi HUYNH
> **Date de creation :** 2026-04-24
> **Reference plan amont :** [wincorp-odin/specs/valkyries.plan.md](valkyries.plan.md) (GO recu 2026-04-24 00:30)
> **Changelog vs v1.0 :** 13 corrections audit #1 + 4 corrections audit #1bis (re-review v1.1). Pivot enforcement vers Option 3 (middleware LangChain reel) `feedback_robust_over_temporary`. Fix C1 `MappingProxyType` non-hashable → `tuple[tuple[str, Any], ...]`. Fix C2 `BaseChatModel` Pydantic → `ConfigDict` + `_llm_type`. Swap atomique sans fenetre clear+update. Barrier timeout 5s. Voir §14.

---

## 1. Objectif

Fournir un **registre declaratif des roles produit d'agents** (valkyries) consommable par l'orchestrateur Odin, stocke en source unique dans `wincorp-urd/referentiels/valkyries.yaml` et charge via un loader Python pur dans `wincorp_odin.orchestration.valkyries`.

Chaque valkyrie (brynhildr/sigrun/thor en v1.0) declare ses parametres d'orchestration (`timeout_seconds`, `max_turns`, `max_concurrent`), son modele LLM par reference (`model: "claude-sonnet"` vers `models.yaml`) et ses tools interdits (`blocked_tools`). Le loader retourne une `ValkyrieConfig` frozen dataclass, et une factory `create_valkyrie_chat(role)` compose la config avec le modele LLM via un **middleware LangChain `ValkyrieToolGuard`** qui filtre au runtime les `tool_use` blocks interdits (enforcement reel, pas metadonnees).

Scope v1.0 : 3 roles + loader + middleware + factory + tests comportementaux. L'integration runtime thor Playwright (Phase 3.5) et le bridge asyncio (Phase 2.9) sont hors scope.

## 2. Perimetre

### IN — Ce que le module fait

- Charge `wincorp-urd/referentiels/valkyries.yaml` (path resolu via `_resolve_urd_path` miroir de `llm/config.py`).
- Parse et valide le schema YAML (champs obligatoires, plages numeriques, whitelist tools, existence `model` dans models.yaml).
- Expose `ValkyrieConfig` dataclass `frozen=True` hashable (via `MappingProxyType` pour `extra_kwargs` et `frozenset` pour `blocked_tools`).
- Expose `load_valkyrie(name)`, `list_valkyries()`, `validate_all_valkyries()`.
- Cache thread-safe avec invalidation mtime throttled 1 Hz (copy-on-write PB-019, pattern `llm/factory.py`).
- Expose **`ValkyrieToolGuard(BaseChatModel)`** : middleware langchain-core qui override `_generate/_agenerate/_stream/_astream`, filtre les `tool_use` blocks dont `name ∈ config.blocked_tools`, log WARNING, remplace par bloc texte synthetique informant l'agent.
- Expose **`create_valkyrie_chat(role: str) -> BaseChatModel`** : factory qui compose `load_valkyrie(role)` + `create_model(config.model)` + `ValkyrieToolGuard(wrapped, config)`.
- Expose `ValkyrieConfig.to_dict() -> dict[str, Any]` pour serialisation JSON (consumers heimdall/bifrost).
- API echappatoire tests : `_reload_for_tests()` (non exportee).

### OUT — Ce que le module ne fait PAS

- Ne modifie PAS le contrat de `SubagentExecutor.submit(task: TaskCallable, ...)` — zero cassure orchestration v2.1.1.
- Ne connait PAS les clients (SPINEX/TRIMAT). Roles generiques uniquement.
- Ne fait PAS de tracking tokens/couts (reste dans `llm/tokens.py`).
- Ne modifie PAS `models.yaml` (dependance unidirectionnelle valkyries → models).
- Ne supporte PAS d'override client-specifique (champs fixes par role).
- Ne gere PAS le dispatch multi-valkyries (`ValkyrieRunner` Phase 3.5 futur).
- Ne fait PAS de `raise` sur tool_use bloque — filtre + log + bloc synthetique, l'agent adapte.

## 3. API publique

### 3.1 `ValkyrieConfig` (dataclass frozen, hashable)

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ValkyrieConfig:
    name: str                                  # cle du role (brynhildr, sigrun, thor)
    description: str                           # texte libre < 200 char
    timeout_seconds: int                       # [30, 1800]
    max_turns: int                             # [1, 500]
    max_concurrent: int                        # [1, 20]
    model: str                                 # reference name dans models.yaml (valide au load)
    blocked_tools: frozenset[str]              # immuable, souvent {"task", "shell"}
    extra_kwargs: tuple[tuple[str, Any], ...]  # items tries (k, v), immuable, defaut ()

    def to_dict(self) -> dict[str, Any]:
        """Serialisation JSON-safe : frozenset → list triee, tuple items → dict."""
```

**Hashable reellement** : `frozenset` + `tuple[tuple[str, Any], ...]` (items tries alphabetiquement sur key) + primitives = config utilisable comme cle de dict/set, `hash(config)` OK.

**IMPORTANT `MappingProxyType` rejete** (audit #1bis C1) : `MappingProxyType` n'implemente PAS `__hash__` (CPython issue #87995, intentionnel — le proxy reflete les mutations du dict sous-jacent, ne peut pas garantir `same hash for equal objects`). Choix retenu : `tuple[tuple[str, Any], ...]` trie par key au parsing YAML, vraiment immuable et hashable si les values le sont (caller responsable de ne pas mettre un dict dans extra_kwargs — validation au load).

**Validation hashable des values `extra_kwargs`** au load : si une value n'est pas hashable (dict, list, set muable), `ValkyrieConfigError` "extra_kwargs values doivent etre hashable (primitives OU tuples). Cle fautive : X". Autorise : `str, int, float, bool, None, tuple, frozenset`. Interdit : `dict, list, set`.

### 3.2 Fonctions exportees

| Signature | Role |
|---|---|
| `load_valkyrie(name: str) -> ValkyrieConfig` | Charge la config (cache). Raise `ValkyrieNotFoundError` si absent. |
| `list_valkyries() -> list[str]` | Liste des names disponibles (tries alphabetique). |
| `validate_all_valkyries() -> None` | Valide le YAML au demarrage. Raise `ValkyrieConfigError` si invalide. |
| `create_valkyrie_chat(role: str) -> BaseChatModel` | **Factory principale consumer**. Compose loader + LLM + middleware. Retour : chat model wrapped, pret a l'emploi. |

### 3.3 `ValkyrieToolGuard` — middleware LangChain (enforcement reel)

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from pydantic import ConfigDict
from typing import Any, Iterator, AsyncIterator

class ValkyrieToolGuard(BaseChatModel):
    """Wrapper BaseChatModel qui filtre les tool_use blocks interdits.

    Compose un chat model sous-jacent + ValkyrieConfig. Override _generate,
    _agenerate, _stream, _astream. Parcourt AIMessage.content (list Anthropic
    content blocks), filtre blocs type='tool_use' dont name ∈ config.blocked_tools.

    Strategie : remplacement par bloc texte synthetique, pas raise. L'agent
    recoit feedback et peut adapter sa strategie (re-demander un autre tool).

    Logs : WARNING structure a chaque filtre (role, tool_name, trace_id si present).
    """

    # OBLIGATOIRE (audit #1bis C2) : BaseChatModel herite de Pydantic BaseModel.
    # Sans arbitrary_types_allowed, Pydantic refuse BaseChatModel comme type de
    # champ ET ValkyrieConfig (dataclass non-Pydantic). Lever PydanticUserError
    # a l'instantiation.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseChatModel
    config: ValkyrieConfig

    # OBLIGATOIRE LangChain : BaseChatModel declare _llm_type abstrait.
    # Sans override, ValkyrieToolGuard reste abstract et ne peut s'instancier.
    @property
    def _llm_type(self) -> str:
        return "wincorp-valkyrie-guard"

    def _filter_response(self, response: AIMessage) -> AIMessage: ...
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult: ...
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult: ...
    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]: ...
    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]: ...
```

**Cache factory** (audit #1bis) : `create_valkyrie_chat(role)` ne cache PAS les instances v1.0. Chaque appel reconstruit le guard. Rationale : simplicite + deterministe + cout negligeable (wrap leger). Si profil CPU remonte un hotspot en usage intensif v1.1+, ajouter `lru_cache` sur `(role, model_cache_key)`. Decision differee a usage reel observe.

### 3.4 Exceptions publiques

- `ValkyrieConfigError(ValueError)` — parent, YAML invalide ou schema casse
- `ValkyrieNotFoundError(ValkyrieConfigError)` — name absent du YAML
- `ValkyrieModelRefError(ValkyrieConfigError)` — `model: "xxx"` non resolvable dans models.yaml (2 variantes : inconnu / disabled)
- `ValkyrieRangeError(ValkyrieConfigError)` — champ hors plage [min, max]

Pas de `ValkyrieToolBlockedError` : le middleware filtre, ne raise pas (strategie §5.5).

### 3.5 API echappatoire tests

- `_reload_for_tests() -> None` — vide cache + mtime + **invalide aussi les instances ValkyrieToolGuard connues** (via registry weak ref, cf §5.6). Non exportee.

## 4. Schema YAML

### 4.1 Structure attendue

```yaml
# Roles valkyries — source unique Yggdrasil.
# @consumer wincorp_odin.orchestration.valkyries.load_valkyrie
# @consumer wincorp_odin.orchestration.valkyries.create_valkyrie_chat
# @spec wincorp-odin/specs/valkyries.spec.md v1.1
config_version: 1
source: "Plan DeerFlow Phase 3 — registry valkyries roles produit"
maintainer: "Tan Phi HUYNH"
updated: "2026-04-24"

defaults:
  timeout_seconds: 300
  max_turns: 100
  max_concurrent: 3
  blocked_tools: ["task"]  # pas de recursion subagents par defaut

valkyries:
  brynhildr:
    description: "Valkyrie production Achats — triage fournisseur/date/ref Fulll"
    timeout_seconds: 300
    max_turns: 100
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task", "shell"]
    extra_kwargs: {}

  sigrun:
    description: "Valkyrie production Image — extraction Vision / decideur / builder"
    timeout_seconds: 600
    max_turns: 200
    max_concurrent: 3
    model: "claude-sonnet"
    blocked_tools: ["task"]
    extra_kwargs: {}

  thor:
    description: "Valkyrie Playwright-Fulll — automation DOM (runtime Phase 3.5 differe)"
    timeout_seconds: 900
    max_turns: 150
    max_concurrent: 2
    model: "claude-haiku"
    blocked_tools: ["task", "shell"]
    extra_kwargs: {}
```

### 4.2 Champs — obligatoires vs optionnels

| Champ | Obligatoire | Type | Plage | Defaut si `defaults:` defini |
|---|---|---|---|---|
| `description` | oui | str | len < 200 | — |
| `timeout_seconds` | herite sinon oui | int | [30, 1800] | `defaults.timeout_seconds` |
| `max_turns` | herite sinon oui | int | [1, 500] | `defaults.max_turns` |
| `max_concurrent` | herite sinon oui | int | [1, 20] | `defaults.max_concurrent` |
| `model` | oui | str | name resolu dans models.yaml | — |
| `blocked_tools` | herite sinon optionnel | list[str] | chaque ∈ whitelist (§4.3) | `defaults.blocked_tools` OU `[]` |
| `extra_kwargs` | non | dict | — | `{}` |

### 4.3 `blocked_tools` whitelist statique

Liste close v1.0 : `{"task", "shell", "bash", "write", "edit", "read"}`. Element absent → `ValkyrieConfigError` avec message explicite. Whitelist **elargissable v1.1** si Phase 3.5 introduit de nouveaux tools (migration via `config_version`).

### 4.4 Migration `config_version`

- v1.0 : `config_version: 1`. Loader accepte uniquement `1`.
- Tout autre entier → `ValkyrieConfigError` message : `"config_version {X} non supporte. Mettre a jour wincorp-odin ou downgrade valkyries.yaml."`.
- Migration futur : v2 preservera l'API publique, seule la forme YAML evoluera.

## 5. Comportement runtime

### 5.1 Cache et invalidation mtime (loader)

Pattern inspire de [`llm/factory.py:206-262`](../src/wincorp_odin/llm/factory.py:206), **ameliore** pour eliminer la fenetre `clear()+update()` :

- Cache global `_configs_ref: dict[str, ValkyrieConfig] | None` protege par `threading.Lock` = `_cache_lock`. Note : **variable simple**, pas de mutation via clear/update.
- `_yaml_mtime: float | None`, `_last_mtime_check: float` throttle `stat()` max 1 Hz.
- **Reload strategy swap atomique** (audit #1bis amelioration I1) :
  1. `_load_and_validate_valkyries(timeout_s)` fonction pure, lit YAML, valide **HORS lock valkyries**, retourne `dict[str, ValkyrieConfig]` nouveau.
  2. Swap atomique sous `_cache_lock` court : `_configs_ref = new_dict` + `_yaml_mtime = current`. **Pas de clear+update**.
- **Lecture sans lock** : `load_valkyrie(name)` lit `_configs_ref` (attribution CPython atomique via GIL), puis `.get(name)`. Si miss : fallback sous lock + re-check `_configs_ref` + re-lookup. **Pas de fenetre visible** ou `_configs_ref` serait vide — le swap est une reference unique.
- Si le reload echoue (YAML invalide apres edit live) : log WARNING + `_configs_ref` **conserve sa valeur precedente** (la reference n'est swappee que si `new_dict` est valide), pas de downtime.

**Why swap atomique vs clear+update** : l'attribution d'attribut module en Python est atomique sous GIL (PEP 8 / CPython impl. detail). `clear()+update()` cree une fenetre intermediaire ou `_configs_ref` est vide visible aux lecteurs sans lock. Le swap elimine cette fenetre par design, simplifie les tests (plus besoin d'EC13 qui etait theatral), et reste thread-safe sans lock pour les lecteurs.

### 5.2 Chargement initial

- `_ensure_configs_loaded()` appele au 1er `load_valkyrie()` / `create_valkyrie_chat()`.
- Budget startup `WINCORP_VALKYRIES_VALIDATE_TIMEOUT_S` defaut 5 s, bornes [1, 60].
- Budget depasse : log WARNING + **tout-ou-rien strict** (pas de partial deferred comme factory.py — schema valkyries plus simple, pas de resolution `use:` path differee). EC6 explicite.

### 5.3 Thread-safety

- Mutation globale (`_configs`, `_yaml_mtime`, `_last_mtime_check`) uniquement sous `_cache_lock`.
- Lecture `load_valkyrie()` : lookup sans lock (CPython GIL safe pour dict.get atomique), fallback sous lock si miss.
- Double-check `_ensure_configs_loaded()` : 2 threads simultanes au 1er appel → 1 seul load effectif, l'autre attend sous lock + re-verifie le flag (EC7 testable via `threading.Barrier`, pas de pragma no cover).

### 5.4 Resolution cross-reference `model` — ordre locks

Au load valkyrie : `model: "claude-sonnet"` valide contre `models.yaml` via `wincorp_odin.llm.config.load_models_config()`.

**Ordre locks strict** (pas de deadlock AB-BA) :
- `load_models_config()` appele **HORS** `_cache_lock` valkyries, dans `_load_and_validate_valkyries()` fonction pure.
- `load_models_config()` a son propre lock interne (factory.py) : acquisition independante, sequentielle.
- **INTERDIT** : appeler `load_valkyrie()` depuis sous le lock factory.py (dependance inverse interdite par conception).

Erreurs discriminantes :
- `ValkyrieModelRefError("Modele '{model}' inconnu. Modeles disponibles : {list}.")` — not in models.
- `ValkyrieModelRefError("Modele '{model}' est desactive dans models.yaml (disabled: true). Re-activer ou changer de reference dans valkyries.yaml.")` — disabled.

### 5.5 Middleware `ValkyrieToolGuard` — comportement runtime

**Objectif** : enforcement reel des `blocked_tools` au niveau LLM runtime (couche 2 `feedback_guardrails_theatre_vs_reel`).

**Interception** : override `BaseChatModel._generate`, `_agenerate`, `_stream`, `_astream` :
1. Appel du wrapped chat model → obtient `AIMessage` (ou stream de `AIMessageChunk`).
2. Parcourt `response.content` :
   - Si `content` est str : pass-through integral (pas de tool_use possible).
   - Si `content` est list : pour chaque bloc `{"type": "tool_use", "name": X, ...}` :
     - Si `X ∈ config.blocked_tools` → remplace par `{"type": "text", "text": "[tool_use '{X}' rejete : valkyrie '{config.name}' n'a pas ce tool. Utiliser un autre moyen.]"}`, log WARNING structure `valkyrie={name} tool_blocked={X} trace_id={run_manager.trace_id if present}`.
     - Sinon pass-through bloc intact.
3. Retourne `AIMessage` modifiee.

**Streaming** : chaque `AIMessageChunk` contenant un fragment `tool_use` accumule est evalue au bloc complet (fin de fragment). Si bloc complet bloque → chunk remplacee par text chunk synthetique + WARNING. Partial chunks non-bloquants pass-through immediat.

**Pas de raise** : filtre + log + feedback textuel a l'agent. L'agent peut re-proposer un autre tool dans le tour suivant, comportement naturel langchain.

**Tool blocks malformes** (name absent, type mal formate) : log WARNING `tool_use malforme` + remplace par bloc text `"[tool_use malforme filtre]"`. Pas de crash.

### 5.6 Invalidation middleware au reload

Apres `_reload_for_tests()` ou reload mtime, les `ValkyrieToolGuard` existants detiennent une ref vers l'ancienne `ValkyrieConfig` (frozen immuable). Strategie :
- Les instances existantes **continuent** avec l'ancienne config (snapshot a la creation).
- Nouveaux `create_valkyrie_chat()` post-reload utilisent la nouvelle config.
- Consumer responsable de re-creer le chat si necessaire (documente dans docstring factory).

Alternative rejetee (registry weak ref + invalidation) : complexite non justifiee v1.0, la recreation explicite est plus deterministe.

### 5.7 Logs obligatoires

Format structure (key=value) :

| Evenement | Niveau | Message |
|---|---|---|
| 1er load YAML | INFO | `valkyries_loaded count={N} duration_ms={T} mtime={mtime}` |
| Reload mtime | INFO | `valkyries_reloaded count={N} mtime_old={old} mtime_new={new}` |
| Reload failed | WARNING | `valkyries_reload_failed error={e} cache_preserved=true` |
| Budget timeout | WARNING | `valkyries_load_timeout budget_s={b} elapsed_s={e}` |
| ToolGuard filtre | WARNING | `valkyrie_tool_blocked role={name} tool={tool} trace_id={id}` |

## 6. Validation au load

### 6.1 Ordre des checks (deterministe)

1. `config_version == 1` sinon `ValkyrieConfigError` clair (§4.4).
2. Presence `valkyries:` dict non vide.
3. Chaque cle `valkyries.<name>` : snake_case `^[a-z][a-z0-9_]*$`, sinon erreur.
4. Champs obligatoires presents (appliquer `defaults:` heritage si configure).
5. Plages numeriques : `timeout ∈ [30, 1800]`, `turns ∈ [1, 500]`, `concurrent ∈ [1, 20]`, sinon `ValkyrieRangeError`.
6. `blocked_tools` chaque element ∈ whitelist statique (§4.3), sinon erreur.
7. `extra_kwargs` : dict (converti en `MappingProxyType` immuable ensuite).
8. `model` existe dans `models.yaml` ET non disabled, sinon `ValkyrieModelRefError` discriminant.
9. `description` : str, len ∈ [1, 200).

### 6.2 Messages d'erreur (FR user, EN tech stack, chemin ABSOLU)

Gabarit `ValkyrieRangeError` :
```
[ERREUR] Valkyrie 'sigrun' : timeout_seconds=2000 hors plage [30, 1800].
         Fichier : C:\Users\Tanfeuille\Documents\wincorp-dev\wincorp-urd\referentiels\valkyries.yaml
```

Gabarit `ValkyrieModelRefError` inconnu :
```
[ERREUR] Valkyrie 'brynhildr' : modele 'claude-inexistant' inconnu.
         Modeles disponibles (models.yaml) : ['claude-haiku', 'claude-opus', 'claude-sonnet'].
         Fichier : {resolved_path_absolu}
```

Gabarit `ValkyrieModelRefError` disabled :
```
[ERREUR] Valkyrie 'brynhildr' : modele 'claude-sonnet' est desactive dans models.yaml (disabled: true).
         Reactiver le modele OU changer la reference dans valkyries.yaml.
         Fichier : {resolved_path_absolu}
```

Tous messages incluent `str(resolved_path)` absolu via `_resolve_urd_path()` — debug multi-PC tanfeuille/tanph.

## 7. Regles metier (R1-R18)

### Loader + Config (R1-R12)

| ID | Regle | Test |
|---|---|---|
| R1 | `ValkyrieConfig` est immuable hashable : `frozen=True` + `blocked_tools` frozenset + `extra_kwargs` `tuple[tuple[str, Any], ...]` (items tries). `hash(config)` OK, `config` utilisable comme cle dict/set. | `test_r1_valkyrie_config_immutable_hashable` |
| R2 | `load_valkyrie("brynhildr")` retourne la config cachee sur 2e appel (pas de re-read YAML) | `test_r2_cache_hit` |
| R3 | Edit `valkyries.yaml` + 1.5 s wait + re-load → nouvelle valeur (mtime 1 Hz throttle) | `test_r3_mtime_reload` |
| R4 | `load_valkyrie("inconnue")` → `ValkyrieNotFoundError` avec liste des disponibles | `test_r4_not_found_lists_available` |
| R5 | `timeout_seconds: 2000` au YAML → `ValkyrieRangeError` au `validate_all_valkyries()` avec chemin absolu | `test_r5_range_violation_absolute_path` |
| R6 | `model: "claude-inexistant"` → `ValkyrieModelRefError` variante "inconnu" avec liste disponibles | `test_r6_model_ref_unknown` |
| R7 | `blocked_tools: ["task", "unknown_tool"]` → `ValkyrieConfigError` (whitelist violation) | `test_r7_blocked_tools_whitelist` |
| R8 | YAML invalide apres edit live → log WARNING + cache precedent conserve (pas de downtime) | `test_r8_invalid_reload_keeps_cache` |
| R9 | 100 threads `load_valkyrie()` concurrent → aucune race, toutes lectures coherentes | `test_r9_thread_safety_stress` |
| R10 | `defaults.blocked_tools` heritage : valkyrie sans `blocked_tools` explicite herite des defaults | `test_r10_defaults_inheritance` |
| R11 | `list_valkyries()` retourne noms tries alphabetique | `test_r11_list_sorted` |
| R12 | `ValkyrieConfig.to_dict()` retourne dict JSON-serializable : `frozenset → list triee`, `tuple[tuple[str, Any], ...] → dict`. `json.dumps(config.to_dict())` OK. | `test_r12_to_dict_json_serializable` |

### Migration + Extensibilite (R13-R14)

| ID | Regle | Test |
|---|---|---|
| R13 | `config_version != 1` → `ValkyrieConfigError` clair "non supporte" | `test_r13_config_version_unsupported` |
| R14 | `extra_kwargs: {"foo": "bar"}` passe au load (dict libre v1.0, extensible) | `test_r14_extra_kwargs_passthrough` |

### Middleware ValkyrieToolGuard (R15-R18) — **enforcement reel**

| ID | Regle | Test |
|---|---|---|
| R15 | `ValkyrieToolGuard` filtre tool_use dont `name ∈ blocked_tools` : bloc remplace par text synthetique + WARNING loggue avec role+tool | `test_r15_toolguard_filters_blocked` |
| R16 | `ValkyrieToolGuard` pass-through tool_use dont `name ∉ blocked_tools` (pas de regression) | `test_r16_toolguard_passthrough_allowed` |
| R17 | `ValkyrieToolGuard` mode streaming (`_astream`) : tool_use bloque dans fragment → chunk remplacee par text chunk + WARNING | `test_r17_toolguard_stream_filters` |
| R18 | `create_valkyrie_chat("brynhildr")` retourne `ValkyrieToolGuard` wrapped le chat model resolu via `models.yaml.claude-sonnet` | `test_r18_factory_composes_guard` |

## 8. Edge cases (EC1-EC13)

| ID | Situation | Comportement attendu | Test |
|---|---|---|---|
| EC1 | `valkyries.yaml` absent | `ValkyrieConfigError` clair + chemin absolu attendu | `test_ec1_yaml_absent` |
| EC2 | YAML malforme syntax | `ValkyrieConfigError` wrappe `yaml.YAMLError` | `test_ec2_yaml_malformed` |
| EC3 | Section `defaults:` absente | pas d'heritage, tous champs obligatoires explicites par valkyrie | `test_ec3_no_defaults_section` |
| EC4 | `valkyries:` vide | `ValkyrieConfigError` "aucune valkyrie declaree" | `test_ec4_empty_valkyries` |
| EC5 | `model: null` | `ValkyrieConfigError` champ obligatoire | `test_ec5_model_null` |
| EC6 | Budget timeout startup depasse | log WARNING + **tout-ou-rien** : exception propagée, pas de partial | `test_ec6_timeout_all_or_nothing` |
| EC7 | Race : 2 threads 1er `load_valkyrie()` via `threading.Barrier` | 1 seul load effectif, double-check branch atteinte | `test_ec7_concurrent_first_load_barrier` |
| EC8 | `blocked_tools: []` vide explicite | valide, valkyrie autorise tous tools | `test_ec8_blocked_tools_empty` |
| EC9 | `model` disabled dans models.yaml | `ValkyrieModelRefError` variante "desactive" discriminante | `test_ec9_model_disabled` |
| EC10 | ToolGuard `AIMessage.content = str` (pas list) | pass-through integral (pas de tool_use possible) | `test_ec10_toolguard_content_str` |
| EC11 | ToolGuard bloc tool_use `name` absent/malforme | bloc remplace par text "[tool_use malforme]" + WARNING | `test_ec11_toolguard_malformed_block` |
| EC12 | ToolGuard multiple blocs mixes (text + tool_use autorise + tool_use bloque) | filtre uniquement bloque, preserve ordre autres | `test_ec12_toolguard_mixed_blocks` |
| EC13 | ~~Fenetre `clear()+update()`~~ SUPPRIME v1.2 — swap atomique (§5.1) elimine la fenetre par design, EC13 obsolete | — |

## 9. Tests requis

Couverture attendue : **100% branch** (strict Odin). Zero appel reseau (mocks pour `ChatAnthropic` LLM wrapped).

### 9.1 Localisation

```
wincorp-odin/tests/orchestration/
  test_valkyries_loader.py      # unit loader R1-R14 + EC1-EC9 + EC13
  test_valkyries_toolguard.py   # middleware R15-R17 + EC10-EC12
  test_valkyries_factory.py     # create_valkyrie_chat R18
  test_valkyries_stress.py      # R9 concurrence 100 threads + EC7 barrier
```

### 9.2 Fixtures pytest

- `tmp_valkyries_yaml` : ecrit YAML temporaire dans `tmp_path`.
- `mock_models_yaml` : monkeypatch `load_models_config()` pour isoler des models.yaml reels.
- `mock_chat_model` : `BaseChatModel` (Pydantic avec `arbitrary_types_allowed=True`) qui retourne `AIMessage(content=[...])` parametrable (pour tests ToolGuard).
- `double_check_barrier` : `threading.Barrier(2, timeout=5)` + monkeypatch sur `_load_and_validate_valkyries` pour forcer EC7. Catch `BrokenBarrierError` si scheduler OS bloque — test skippe avec message "scheduler-dependent, retry locally". Pas de flakiness en CI.

### 9.3 Couverture middleware streaming

Tests R17 + EC12 doivent utiliser `AsyncIterator[AIMessageChunk]` mocke avec fragments partiels, pour valider que l'accumulation + filtre au bloc complet fonctionne sans perte de chunks.

## 10. Erreurs et messages

Hierarchie :

```
ValkyrieConfigError (ValueError)
├── ValkyrieNotFoundError
├── ValkyrieModelRefError
│   ├── (variante) "inconnu"
│   └── (variante) "desactive"
└── ValkyrieRangeError
```

Messages FR avec contexte : nom valkyrie, champ fautif, valeur, plage attendue, **chemin absolu** du YAML.

## 11. Dependances

- `pyyaml >=6.0.1` (deja declare)
- `langchain-core >=0.3.0` (deja declare — necessaire pour `BaseChatModel` override)
- `langchain-anthropic >=0.3.0` (deja declare — consume downstream, pas requis pour middleware)
- Import interne `wincorp_odin.llm.config.load_models_config` + `wincorp_odin.llm.factory.create_model`

Pas d'ajout dep externe v1.1.

## 12. Integration downstream

### 12.1 Contrat integration (formalise — audit #1 M3)

**Regle dure** : le loader `load_valkyrie()` et la factory `create_valkyrie_chat()` sont consommes par des **wrappers produit** (brynhildr/sigrun producteurs Python, Phase 3.5 `ValkyrieRunner`, bifrost UI), **JAMAIS par `SubagentExecutor.submit()`** :
- `SubagentExecutor.submit(task: TaskCallable, ...)` garde son contrat v2.1.1 intact.
- Pas de parametre `role: str` ajoute a `submit()`. Zero cassure orchestration.
- Le wrapper produit resout la ValkyrieConfig, cree son chat via `create_valkyrie_chat()`, construit le Callable qui utilise ce chat, puis submit le Callable.

### 12.2 Phase 3.5 — `ValkyrieRunner` (hors scope v1.0)

Documentation uniquement. Composant futur qui orchestrera N valkyries en parallele, lisant la config pour `max_concurrent`, `timeout_seconds`, etc. Composition directe sur v1.1 sans rework middleware (middleware deja fait son travail).

### 12.3 Bifrost UI (futur)

Lecture `list_valkyries()` + `load_valkyrie()` + `ValkyrieConfig.to_dict()` pour afficher roles disponibles dans le dashboard agents. Serialisation JSON deja prete v1.1.

## 13. Decisions structurantes (miroir plan amont + corrections audit #1)

| # | Decision | Version |
|---|---|---|
| D1 | Scope yaml + loader + middleware + factory + tests | v1.0 |
| D2 | Schema generique (brynhildr/sigrun/thor) | v1.0 |
| D3 | Plages [1,20] / [30,1800] / [1,500] defaut 3/300/100 | v1.0 |
| D4 | Decouplage loader ↔ executor (formalise §12.1) | v1.0 + v1.1 §12.1 |
| D5 | **Enforcement reel via middleware LangChain** (Option 3, cf `feedback_robust_over_temporary`) | **v1.1** |
| D6 | Loader pattern factory.py miroir | v1.0 |
| D7 | `model` string ref valide au load + messages discriminants | v1.0 + v1.1 |
| D8 | Frontmatter YAML miroir models.yaml | v1.0 |
| D9 | Pas de pricing v1.0 | v1.0 |
| D10 | 2 passes audit + challenger post-build | v1.0 |
| D11 | `extra_kwargs` hashable via `tuple[tuple[str, Any], ...]` items tries (MappingProxyType non-hashable rejete v1.2 audit #1bis) | **v1.1 → corrige v1.2** |
| D12 | Logs structures obligatoires | **v1.1** |
| D13 | Messages erreur chemin absolu + discriminants | **v1.1** |
| D14 | EC7 force double-check via threading.Barrier | **v1.1** |
| D15 | `to_dict()` serialisation JSON | **v1.1** |
| D16 | Migration `config_version` explicite | **v1.1** |

## 14. Changelog

| Version | Date | Changement |
|---|---|---|
| 1.0 DRAFT | 2026-04-24 | Specification initiale. Plan amont valide 2026-04-24 00:30. |
| 1.1 DRAFT | 2026-04-24 | 13 corrections post audit #1 adversarial : (1) enforcement reel middleware ValkyrieToolGuard Option 3 — directive `feedback_robust_over_temporary.md` ; (2) `extra_kwargs: MappingProxyType` ; (3) formalisation §12.1 wrapper produit ; (4) R13 config_version migration + R14 extra_kwargs passthrough ; (5) EC7 threading.Barrier testable ; (6) §5.1 fenetre clear+update documentee + EC13 dedicated ; (7) §5.4 ordre locks strict ; (8) §6.2 messages discriminants inconnu/disabled + chemin absolu ; (9) `to_dict()` serialisation JSON ; (10) §5.7 logs obligatoires ; (11) EC6 tout-ou-rien explicite ; (12) R15-R18 tests middleware comportementaux ; (13) exports etendus. |
| 1.2 DRAFT | 2026-04-24 | 4 corrections re-review v1.1 (audit #1bis) : (C1) `MappingProxyType` **non-hashable** (CPython #87995) → remplace par `tuple[tuple[str, Any], ...]` items tries, propage R1/R12/D11/§3.1 + validation values hashable au load ; (C2) `BaseChatModel` = Pydantic → ajout `model_config = ConfigDict(arbitrary_types_allowed=True)` + `_llm_type` property obligatoire (sans quoi PydanticUserError + ABC non-instantiable) ; (I1 amelioration) abandon pattern `clear()+update()` → **swap atomique** `_configs_ref = new_dict` via GIL, supprime la fenetre par design, EC13 obsolete ; (M5) `threading.Barrier(2, timeout=5)` + `BrokenBarrierError` skip handling pour eviter flakiness CI EC7. |

## 15. Glossaire

- **Valkyrie** : role produit d'agent specialise (brynhildr=Achats, sigrun=Image, thor=DOM Fulll).
- **ValkyrieConfig** : dataclass frozen hashable retournee par le loader.
- **ValkyrieToolGuard** : middleware LangChain `BaseChatModel` qui filtre au runtime les `tool_use` blocks interdits — **enforcement reel couche 2** (`feedback_guardrails_theatre_vs_reel`).
- **create_valkyrie_chat(role)** : factory principale consumer, compose loader + LLM + guard.
- **blocked_tools** : set immuable de tool names filtres au runtime par le guard. Pas metadonnees informatives : vrai enforcement.
- **Copy-on-write (PB-019)** : strategie invalidation cache, validation hors lock, swap atomique sous lock court.

---

_v1.2 prete pour build TDD. 2 bloquants v1.1 corriges (MappingProxyType, BaseChatModel Pydantic), 1 amelioration (swap atomique), 1 robustesse test (Barrier timeout). Audit #2 interviendra apres build._
