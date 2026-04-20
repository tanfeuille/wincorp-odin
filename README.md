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

## Historique

**Créé le 20/04/2026 soir** — Phase 0.5 du plan DeerFlow inspiration Yggdrasil (cf `memory/project_deerflow_inspiration_plan.md`).
