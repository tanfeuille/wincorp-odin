# wincorp-odin

**Yggdrasil** : Odin — chef des Ases, strategiste et coordinateur. Tronc (transverse).

Orchestrateur LLM générique WinCorp. Factory providers, circuit breaker, retry, sub-agents.

## Positionnement

`wincorp-odin` isole la dette LLM de `wincorp-mimir` qui reste sanctuaire métier (calculs fiscaux, FEC, PCG) avec Pydantic comme seule dépendance externe. Les deux repos sont strictement isolés : aucun import croisé.

## Installation dev

```bash
cd wincorp-odin
pip install -e ".[dev]"
```

## Consommation par les autres repos Yggdrasil

```bash
# Dans wincorp-heimdall, wincorp-bifrost (backend), wincorp-thor :
pip install -e ../wincorp-odin
```

## Usage (à venir, Phase 1 DeerFlow)

```python
from wincorp_odin.llm import create_model

model = create_model("claude-sonnet")
response = await model.ainvoke([{"role": "user", "content": "..."}])
```

Configuration via `wincorp-urd/referentiels/models.yaml` (source de vérité Yggdrasil).

## Architecture

Voir `.claude/CLAUDE.md` pour règles d'isolation, scope, conventions.
Voir `specs/` pour specs SDD des modules.

## Phase 1.6b — Activer sink Supabase

Pour persister les événements `TokenUsageEvent` dans la table `llm_usage` Supabase :

1. **Créer la table** — copier `migrations/001_llm_usage.sql` dans Supabase Dashboard > SQL Editor et exécuter.
2. **Définir les variables d'environnement** :
   ```bash
   export SUPABASE_URL="https://<projet>.supabase.co"
   export SUPABASE_SERVICE_ROLE_KEY="<service_role_key>"
   ```
3. **Activer le sink** :
   ```bash
   export WINCORP_LLM_TOKEN_SINK=supabase
   ```
4. **Fonctionnement** : les events sont écrits en batch — flush automatique toutes les 5 secondes ou dès que la queue atteint 10 events. Les erreurs réseau/HTTP sont swallowed (WARNING log), le caller LLM n'est jamais impacté (R28 : observabilité n'interrompt pas la prod).

## Historique

**Créé le 20/04/2026 soir** — Phase 0.5 du plan DeerFlow inspiration Yggdrasil (cf `memory/project_deerflow_inspiration_plan.md`).
