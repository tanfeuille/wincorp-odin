# valkyries — Plan amont pré-spec SDD

> **Niveau visé :** 2 (standard)
> **Repo :** wincorp-odin (+ asset wincorp-urd)
> **Date :** 2026-04-24
> **Auteur :** HUYNH Tan Phi / agent Claude Code

---

## 1. Grep feedbacks mémoire effectué

**Commande exécutée (verbatim)** :

```bash
MEM="C:/Users/Tanfeuille/.claude/projects/C--Users-Tanfeuille-Documents-wincorp-workspace/memory"
ls "$MEM"/feedback_*.md | xargs -I {} basename {} .md | \
  grep -Ei "valkyrie|yaml|registry|config|cache|parallelism|orchestration|tool|blocked|architect|decoupl|read_complete|theatre|contract_change|plan_amont|no_pruning|fix_at_source|verify_memory|model_selection" | sort
```

**Output brut** (12 feedbacks identifiés) :

```
feedback_agent_model_selection
feedback_agent_parallelism_cap
feedback_architecture_thinking
feedback_commit_before_platform_config
feedback_contract_change_audit
feedback_guardrails_theatre_vs_reel
feedback_nextjs14_data_cache_server_fetch
feedback_plan_amont_spec_sdd
feedback_read_complete_files
feedback_skill_fix_at_source
feedback_skill_no_pruning
feedback_verify_memory_before_proposing
```

**Feedbacks impactant directement le design (7 sur 12)** :

- `feedback_architecture_thinking` — Produit ≠ Client : registry générique (rôles produit brynhildr/sigrun/thor), PAS par client. Impact D2.
- `feedback_agent_parallelism_cap` — défaut prudent max_concurrent 3, plage [1, 20] cohérente Phase 2 orchestration. Impact D3.
- `feedback_guardrails_theatre_vs_reel` — `blocked_tools` enforce via **test d'intégration** (couche 2), pas juste helper static. Impact D5.
- `feedback_read_complete_files` — lire `wincorp-odin/src/wincorp_odin/llm/factory.py` (508 lignes) + `wincorp-urd/referentiels/models.yaml` (114 lignes) **intégralement** AVANT design loader. Fait. Impact D6.
- `feedback_contract_change_audit` — 2 passes audit multi-agent obligatoires : avant code + post-build. Impact D10.
- `feedback_plan_amont_spec_sdd` — ce fichier. Contrat méta.
- `feedback_skill_no_pruning` — contrat workflow pas menu, applicable à la spec si étapes numérotées.

**Feedbacks hors impact design direct (5 sur 12)** :
- `feedback_agent_model_selection` — convention `model: <name>` référencée dans YAML (impact mineur sur schema D9)
- `feedback_commit_before_platform_config`, `feedback_nextjs14_data_cache_server_fetch`, `feedback_skill_fix_at_source`, `feedback_verify_memory_before_proposing` — applicables au process, pas au design.

---

## 2. Patterns refs pré-lus (intégralement)

- [`wincorp-urd/referentiels/models.yaml`](wincorp-urd/referentiels/models.yaml:1) 114 lignes — convention frontmatter YAML WinCorp : `config_version`, `source`, `maintainer`, `updated`, `defaults`, liste d'éléments. À imiter pour valkyries.yaml.
- [`wincorp-odin/src/wincorp_odin/llm/factory.py`](wincorp-odin/src/wincorp_odin/llm/factory.py:1) 508 lignes — loader YAML avec cache mtime throttled 1 Hz + thread-safe + copy-on-write (PB-019) + swap atomique sous lock court + validation hors lock. Pattern à reproduire pour `valkyries.py`. Fonctions clés : [`_load_and_validate_models`](wincorp-odin/src/wincorp_odin/llm/factory.py:157), [`_check_mtime_and_invalidate`](wincorp-odin/src/wincorp_odin/llm/factory.py:206), [`_ensure_configs_loaded`](wincorp-odin/src/wincorp_odin/llm/factory.py:264), [`_reload_for_tests`](wincorp-odin/src/wincorp_odin/llm/factory.py:498).
- [`wincorp-odin/specs/_SPEC-TEMPLATE.md`](wincorp-odin/specs/_SPEC-TEMPLATE.md:1) — convention frontmatter spec SDD (Statut, Version, Niveau 2, Auteur).
- [`wincorp-odin/specs/orchestration.spec.md`](wincorp-odin/specs/orchestration.spec.md:1) 1214 lignes — spec sœur v2.1.1 IMPLEMENTED : `SubagentExecutor` + `SubagentResult` + `build_initial_state` + `truncate_task_calls`. `max_concurrent ∈ [1, 20]` cohérent. Champs valkyries doivent enrichir `SubagentExecutor.submit(...)` sans casser le contrat.
- [`wincorp-odin/.claude/CLAUDE.md`](wincorp-odin/.claude/CLAUDE.md:1) — ÉTAT DU PROJET mentionne explicitement `valkyries.yaml` en scope urd + "Odin les consomme uniquement" (pas de copie locale). Python 3.12+ strict, Pydantic v2, MyPy strict, 100% branch coverage, docstrings FR.
- [`wincorp-urd/.claude/CLAUDE.md`](wincorp-urd/.claude/CLAUDE.md:1) — convention urd : JSON UTF-8 indent 2, source champ obligatoire, snake_case français clés, consumer importe jamais copie.

---

## 3. Décisions structurantes tranchées

| # | Décision | Rationale / feedback aligné |
|---|---|---|
| D1 | **Scope v1.0** : `wincorp-urd/referentiels/valkyries.yaml` + `wincorp-odin/src/wincorp_odin/orchestration/valkyries.py` + tests + exports. Phase 3.5 (intégration runtime thor/sigrun/brynhildr) différée. | Isolation changement minimale, découplage livraison |
| D2 | **Schema YAML générique** : clé top-level `valkyries: {brynhildr: {...}, sigrun: {...}, thor: {...}}`. Rôles produit, PAS par client. Ajouter un rôle ne casse jamais un client. | `feedback_architecture_thinking` P1 Produit ≠ Client |
| D3 | **Plages numériques** : `max_concurrent ∈ [1, 20]` (cohérent Phase 2), défaut 3. `timeout_seconds ∈ [30, 1800]` (cohérent hermod TIMEOUT_SEC_MAX). `max_turns ∈ [1, 500]`. | `feedback_agent_parallelism_cap` + miroir orchestration.spec.md §X |
| D4 | **Découplage loader ↔ executor** : `ValkyrieConfig` dataclass immutable (`frozen=True`) retournée par le loader. Caller (wrapper SubagentExecutor ou agent appelant) l'utilise, pas de mutation auto. | `feedback_architecture_thinking` P2 Dissociabilité + pattern Phase 2 SubagentResult frozen |
| D5 | **`blocked_tools` enforce via test d'intégration** : test qui monte un SubagentExecutor réel, submit avec ValkyrieConfig bloqué sur `["task", "shell"]`, tente l'appel, vérifie rejet runtime. PAS juste helper static qui retourne la liste. | `feedback_guardrails_theatre_vs_reel` couche 2 (test = enforcement réel) |
| D6 | **Loader pattern factory.py miroir** : cache mtime throttled 1 Hz + threading.Lock + copy-on-write PB-019 + `_load_and_validate_valkyries(timeout_s)` fonction pure + swap atomique sous lock court + `_reload_for_tests`. Stdlib + pyyaml uniquement. | `feedback_read_complete_files` + cohérence ecosystem |
| D7 | **Convention `model`** : string référence vers name dans models.yaml (`claude-sonnet`, `claude-opus`, `claude-haiku`). Validation au load : resolve models.yaml et vérifie que le name existe. Erreur claire si typo. | `feedback_agent_model_selection` + cohérence croisée urd |
| D8 | **Frontmatter YAML miroir models.yaml** : `config_version: 1`, `source`, `maintainer`, `updated`, `defaults` (timeout_seconds/max_turns/max_concurrent), puis `valkyries:` flat dict. Pas de `list` comme models.yaml — ici dict car les rôles sont nommés. | Convention urd + discoverability |
| D9 | **Pas de `pricing` v1.0** : tracking tokens reste dans llm/tokens.py (factory middleware). Orchestration = coordination, pas comptage coût. Ajout optionnel futur v1.1 si besoin. | Minimalisme scope, cohérent hors-scope Odin |
| D10 | **2 passes adversariales sur spec DRAFT + 1 challenger post-build** : pattern Phase 2 (résultat 23 corrections cumulées identifiées). Sonnet pour build TDD, Opus pour plan archi final, Haiku pour vérifs. | `feedback_contract_change_audit` + `feedback_agent_model_selection` |

---

## 4. Soumis à validation user

- Date / heure soumission : `2026-04-24 00:15`
- Statut : **GO REÇU** `2026-04-24 00:30` — rédaction spec v1.0 DRAFT autorisée
- Réponses user aux 5 points ambigus : OK pour toutes les positions poussées (reload auto, blocked_tools via whitelist task names, 3 rôles v1.0, validation model strict, extra_kwargs optionnel)

---

## 5. Anti-patterns anticipés

1. **Schema couplé client** : si quelqu'un tente `valkyries.spinex.brynhildr` → rejet design (D2).
2. **Loader couplé executor** : `load_valkyrie("brynhildr")` qui instancie auto un SubagentExecutor → rejet (D4, dataclass uniquement).
3. **`blocked_tools` check fonction pure sans test runtime** : helper `is_blocked(tool, cfg)` facile à gamer → exige test intégration (D5).
4. **Dépendance circulaire valkyries ↔ models** : valkyries.yaml référence models.yaml, models.yaml ne référence PAS valkyries.yaml. Validation unidirectionnelle au load.
5. **Cache non invalidé sur edit yaml** : miroir factory.py mtime throttled 1 Hz (D6).
6. **Race condition load multi-thread** : pattern PB-019 copy-on-write de factory.py (D6).
7. **Defaults trop permissifs** : `max_concurrent: 20` en défaut → préfère 3 prudent, 20 explicite si rôle exigeant (D3).
8. **`model` invalide silencieusement** : erreur au load, pas au runtime (D7).
9. **Spec qui contredit orchestration.spec.md** : `max_concurrent` doit être cohérent, `SubagentExecutor` API enrichie pas cassée (D10 passe adversariale).
10. **Build sans test TDD** : tests avant code (pattern Phase 2).

---

## 6. Scope

**IN v1.0 (livré après GO)** :
- `wincorp-urd/referentiels/valkyries.yaml` (YAML source de vérité, 3 rôles brynhildr/sigrun/thor)
- `wincorp-odin/src/wincorp_odin/orchestration/valkyries.py` (loader + ValkyrieConfig dataclass + validation + cache mtime + thread-safety)
- `wincorp-odin/tests/orchestration/test_valkyries.py` (unit loader + validation + cache + **test d'intégration blocked_tools runtime**)
- Exports via `wincorp-odin/src/wincorp_odin/orchestration/__init__.py` : `ValkyrieConfig`, `load_valkyrie`, `list_valkyries`, `validate_all_valkyries`
- Spec `wincorp-odin/specs/valkyries.spec.md` v1.0 DRAFT → IMPLEMENTED

**OUT v1.0** :
- Intégration runtime wincorp-thor (Phase 3.5, différée)
- Asyncio bridge Phase 2.9
- UI config bifrost (futur)
- Tracking tokens valkyries-level (reste dans llm/tokens)
- Rôles valkyries client-spécifiques (jamais — D2)
- Rôles additionnels au-delà de brynhildr/sigrun/thor v1.0 (ajout cosmétique plus tard)

---

## 7. Audits multi-agent prévus

Conformément à [`feedback_contract_change_audit.md:1`](.claude/projects/C--Users-Tanfeuille-Documents-wincorp-workspace/memory/feedback_contract_change_audit.md:1) :

- **Audit #1 AVANT code (spec DRAFT v1.0)** : 3 agents parallèles
  - `feature-dev:code-architect` (opus) — approches alternatives schema YAML + couplage loader/executor
  - `feature-dev:code-reviewer` (sonnet) — bugs latents cache mtime + race conditions + validation croisée models.yaml
  - `pr-review-toolkit:silent-failure-hunter` (sonnet) — `blocked_tools` enforcement réel vs théâtre
- **Audit #2 AVANT terminé (post-build)** : 2-3 agents
  - `pr-review-toolkit:code-reviewer` (sonnet) — conformité 100% branch coverage + MyPy strict + API publique cohérente
  - `feature-dev:code-explorer` (sonnet) — résidus + régressions sur orchestration + downstream Phase 3.5 fictif
  - `verificateur` (haiku) — format final spec + tests + changelog

---

## 8. Effort estimé

- Plan amont (ce fichier) : ✅ livré
- Audit #1 sur spec DRAFT v1.0 : 15 min (3 agents parallèles)
- Rédaction spec v1.0 DRAFT : 1h
- Passe adversariale #2 sur spec v1.1 : 15 min
- Build TDD (Opus plan → Sonnet build) : 1h30
- Audit #2 post-build : 15 min
- Corrections post-audit : 30 min
- Merge main + housekeeping ERR-001 : 15 min
- **Total** : ~4h

---

## 9. Points ambigus soumis à user

1. **Reload mtime automatique throttled** : j'adopte le pattern factory.py (1 Hz throttle + copy-on-write). Alternative : reload explicite via `reload_valkyries()`. Je pousse l'auto pour cohérence écosystème. OK ?
2. **Test d'intégration blocked_tools** : le test monte un SubagentExecutor réel et tente `task()` → vérifie rejet. Mais Phase 2 orchestration utilise des tasks Python bloqueuses (Callable), pas des tool calls LLM réels. Le "blocked_tools" sera donc une **whitelist** appliquée côté SubagentExecutor qui rejette les tasks dont le nom est dans la liste bloquée. Test ≈ submit une task nommée "task" → assert NotAllowedError. OK comme interprétation, ou tu vises autre chose (blocage LLM runtime via ValkyrieConfig → middleware) ?
3. **Rôles v1.0 (brynhildr/sigrun/thor)** : je garde les 3 dans valkyries.yaml. L'intégration Phase 3.5 (thor automation DOM Playwright) reste différée, mais le rôle dans le YAML est exploitable dès v1.0 pour les producteurs Python. OK ?
4. **`model` string référence vers models.yaml** : validation stricte au load (typo → erreur). Alternative : validation lazy au runtime. Je pousse strict. OK ?
5. **Schema extensible (champs optionnels futurs)** : je laisse `extra_kwargs: {}` optionnel comme models.yaml pour extensibilité sans breaking change. OK ?
