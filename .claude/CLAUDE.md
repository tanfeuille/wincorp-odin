# wincorp-odin — Orchestrateur LLM generique

> **Yggdrasil** : Odin — chef des Ases, strategiste et coordinateur. **Tronc** (transverse). Package Python : `wincorp_odin`.

## ÉTAT DU PROJET

Repo Yggdrasil Odin — orchestrateur LLM générique. Créé 20/04 pour isoler la dette LLM de `wincorp-mimir`.

**Phases livrées** :
- **Phase 1 (20/04)** — llm factory, circuit breaker, retry, tokens middleware, SupabaseSink. 231 tests.
- **Phase 4.1 (22/04)** — sandbox_audit classifier commandes bash (72 tests).
- **Phase 6 partielle (22/04)** — messaging MessageBus + Telegram + WhatsApp (81 tests).
- **Phase 2 (23/04)** — orchestration `SubagentExecutor` non-bloquant + `SubagentResult` frozen + `truncate_task_calls` + `build_initial_state`. 207 tests, 100% branch coverage (512 stmts, 186 branches). 3 passes adversariales sur spec, challenger post-build corrigé 2 races.

**Phase 2.8** (intégration thor Python — scénario X via heimdall REST proxy préconisé) et **Phase 2.9** (asyncio bridge) non-livrées, documentées spec §8.

## RÈGLES D'ISOLATION (DURES — NON NÉGOCIABLES)

1. **Odin n'importe JAMAIS `wincorp_common` (mimir)**. Si un skill Odin a besoin de logique métier, passer en argument via signature fonction. Jamais `from wincorp_common import ...` dans ce repo.
2. **Mimir n'importe JAMAIS `wincorp_odin`**. Ni dans le code, ni dans les tests, ni dans les fixtures. Vérification manuelle à chaque PR mimir.
3. **Namespaces séparés** : `wincorp_common.*` (mimir) et `wincorp_odin.*` (odin). Jamais de namespace-package Python fusionné. Les deux packages s'installent indépendamment via `pip install -e`.

Corollaire : un code métier qui a besoin d'un LLM → le consommateur (heimdall, bifrost, thor) importe les deux et **compose** les deux côtés.

## SCOPE (ce que le repo contient)

- `wincorp_odin.llm` — factory providers LLM (`create_model`, `ModelConfig`, resolution `use:` path)
- `wincorp_odin.llm.circuit_breaker` — closed/half-open/open thread-safe
- `wincorp_odin.llm.retry` — retry exponentiel + parsing Retry-After
- `wincorp_odin.llm.tokens` — usage middleware + sink Supabase
- `wincorp_odin.security.sandbox_audit` — classify bash commands (pattern DeerFlow SandboxAuditMiddleware)
- `wincorp_odin.messaging` — MessageBus asyncio + TelegramChannel + WhatsAppChannel
- `wincorp_odin.orchestration` — `SubagentStatus` / `SubagentResult` / `SubagentExecutor` / `build_initial_state` / `truncate_task_calls` / `SubagentSink` / `LogSink`. API non-bloquante. Spec v2.1.1.

## HORS SCOPE (ne jamais mettre ici)

- Calculs fiscaux, FEC, PCG, actes juridiques — vivent dans `wincorp-mimir`
- Skills métier (`wincorp-fec`, `wincorp-fiscal`) — vivent dans `.claude/skills/`
- Données clients — vivent dans leur repo client
- UI / frontend — vivent dans `wincorp-bifrost`
- Playwright / Fulll automation — vivent dans `wincorp-thor`
- Référentiels YAML (models.yaml, valkyries.yaml) — vivent dans `wincorp-urd`, Odin les consomme uniquement

## CONVENTIONS DE CODE

- **Python 3.12+ strict** (pas de 3.10 pour éviter incohérences mimir)
- Pydantic v2, type hints obligatoires sur toutes les fonctions
- MyPy strict (`disallow_untyped_defs = true`, `strict_optional = true`)
- Ruff lint, ligne max 100 caractères
- Tests pytest, couverture **100% branch** sur module `wincorp_odin.*` (stricter que les 90% mimir)
- **Zéro appel réseau en tests** (mocks obligatoires pour ChatAnthropic, DeepSeek, etc.)
- Messages d'erreur en **français** (Tan est non-dev en runtime), stack trace EN techniques

## Spec-Driven Development (SDD)

Ce projet suit le framework SDD. Voir `wincorp-saga/SPEC-DRIVEN-DEVELOPMENT.md`.
Specs locales dans `specs/`. Template : `specs/_SPEC-TEMPLATE.md`.

**Obligatoire** : `@spec specs/<nom>.spec.md v<version>` en en-tête de chaque fichier source.

## DÉPENDANCES EXTERNES ACCEPTÉES (justifiées par scope)

| Dep | Justification |
|---|---|
| `pydantic >=2.5.0` | Modèles config (ModelConfig, etc.) |
| `pyyaml >=6.0.1` | Lecture models.yaml (wincorp-urd) |
| `langchain-core >=0.3.0` | `BaseChatModel` abstraction (futur multi-providers) |
| `langchain-anthropic >=0.3.0` | `ChatAnthropic` concret (Phase 1.1) |
| `anthropic >=0.40.0` | SDK officiel (transitif mais épinglé pour contrôle versions) |

**Extras optionnels futurs** (non installés par défaut) :
- `providers-deepseek` → `langchain-deepseek` (banc comparaison dossier factice uniquement)
- `providers-openai` → `langchain-openai` (non décidé)

## ERREURS CONNUES

_Aucune (repo neuf)._

## Planification

Ce projet est éligible à `/ultraplan`. Pour chantiers > 30 min de planification : plan mode local → refine with Ultraplan.
