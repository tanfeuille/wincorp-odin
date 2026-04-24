# orchestration — Specification

> **Statut :** IMPLEMENTED
> **Version :** 2.1.2
> **Niveau :** 2 (standard)
> **Auteur :** Tan Phi HUYNH
> **Date de creation :** 2026-04-23
> **Derniere livraison :** 2026-04-24 — CR-025 (shutdown polling deadline-based) + CR-009 (spec bump doc re-check `_uuid_factory`). Tests 207+ verts.
> **Changelog vs v2.1.1 :** CR-025 `shutdown` utilise polling actif avec deadline au lieu de `Event().wait(force_timeout_sec)` passif → exit rapide quand 0 task RUNNING restante. Preservation EC70/EC71. Bump patch 2.1.2.
> **Changelog vs v2.0 :** 23 corrections post re-review adversariale #2 (5 bloquants, 11 majeurs, 7 mineurs). Voir §9.

---

## 1. Objectif

Fournir un socle d'**orchestration de sub-agents** Python pur (zero dep externe : stdlib uniquement) permettant a un orchestrateur (Odin-Spinex, Odin-Trimat, etc.) de dispatcher des taches concurrentes a des agents enfants (valkyries) avec :

1. **API non-bloquante** — `submit()` retourne `task_id` immediatement, `wait(task_id)` bloque pour le terminal, `cancel()` / `get()` non-bloquants.
2. **Statut tracable** (`SubagentStatus` enum 6 etats, str Enum, serialisable JSON).
3. **Resultat structure** (`SubagentResult` dataclass frozen avec trace_id + timestamps UTC distincts `submitted_at`/`started_at`/`completed_at` + messages dedup).
4. **Concurrence bornee** (double `ThreadPoolExecutor` + timeout mecanique + cancel cooperatif).
5. **Contexte heredite selectif** (`build_initial_state` : whitelist parent, pas de cross-contamination).
6. **Fan-out clampe** (`truncate_task_calls` fonction pure, max_concurrent ∈ [1, 20]).
7. **Registre borne** (`max_history`, eviction FIFO terminaux par ordre insertion, hard overflow si aucun terminal evinçable).
8. **Sinks observabilite** (protocol `SubagentSink` + `LogSink` par defaut, errors swallowed sauf KeyboardInterrupt/SystemExit).

Port adapte du pattern DeerFlow Phase 2 (cf `memory/project_deerflow_inspiration_plan.md`). Adresse le chantier Yggdrasil `project_folkvangr_dag_threshold.md` : formaliser le fan-out de valkyries.

**Transformation cle** : `feedback_agent_parallelism_cap.md` (ne pas plafonner arbitrairement) devient une contrainte dure par plage elargie [1, 20] dans `truncate_task_calls`. La plage CLI Agent Teams (5) n'est PAS importee ici — module Python runtime distinct de la couche CLI. Si le besoin est cap 5 plus tard, le caller passe `max_concurrent=5` explicite.

---

## 2. Perimetre

### IN — Ce que le module fait

- Module **exclusivement synchrone** (ThreadPoolExecutor, threading.Event). Bridge asyncio non livre, documente §8.
- `SubagentStatus` enum (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED/TIMED_OUT), str Enum, serialisable JSON.
- `SubagentResult` dataclass `frozen=True, slots=True, eq=True`, **explicitement non-hashable** (`__hash__ = None`), traçable avec `to_dict()` recursif JSON-safe, `is_terminal()`, `duration_ms` property.
- `SubagentExecutor` classe avec API **non-bloquante** : `submit` / `wait` / `cancel` / `get` / `shutdown` / `clear_history` / `stats`. 2 pools (scheduler + exec). Context manager.
- Fonction pure `build_initial_state(parent_state, extra_overrides)` — whitelist heritage.
- Fonction pure `truncate_task_calls(tool_calls, max_concurrent, tool_name)` — fan-out clamp.
- Protocol `SubagentSink` + `LogSink` par defaut (logging structure JSON).
- Exceptions dediees dans `wincorp_odin.orchestration.exceptions` : `SubagentError`, `SubagentExecutorClosedError`, `SubagentExecutorOverflowError`, `SubagentTaskNotFoundError`, `SubagentTaskIdConflictError`, `SubagentCancelledException`.
- Types dedies dans `wincorp_odin.orchestration.types` : `AIMessage` TypedDict, `TaskCallable` alias, `InitialState` alias.
- Helper `_json_safe(obj)` prive (recursion conversion JSON-safe).

### OUT — Ce que le module ne fait PAS

- **Integration LangChain/LangGraph** : module Python pur, zero import `langchain_*`.
- **Integration Anthropic SDK direct** : la `task` recoit un callable opaque, libre au consommateur de composer avec `wincorp_odin.llm`.
- **Support asyncio natif** : executor sync strict. Les consommateurs asyncio (FastAPI heimdall) doivent bridger via `loop.run_in_executor(None, executor.submit, ...)`. Si thor TS doit orchestrer des valkyries Odin → passer par heimdall REST (Scenario X). Support asyncio natif = Phase 2.9 dediee si besoin demontre.
- **Persistance Supabase des SubagentResult** : differe (pourra brancher un `SupabaseSink` analogue a `TokenUsageSink` Phase 1.6 dans une phase ulterieure).
- **Scheduler distribue** : pas de Celery, pas de Redis, pas de queue cross-machine. Mono-process uniquement.
- **Integration wincorp-thor** (Phase 2.8) : decorrelee, livraison dans une session dediee apres validation Phase 2 core.
- **DAG inter-taches explicite** (dependances A→B→C) : differe si le seuil 4+ agents declenche le besoin.
- **Round-trip `from_dict()`** : pas en Phase 2 core. Si besoin persistance → Phase 2.7 dediee.
- **Force-kill de thread zombie** : Python ne le permet pas proprement (pas de `pthread_cancel` expose). Strategie log WARNING + drop (R12).
- **Filtrage automatique des secrets** dans `result` / `ai_messages` : responsabilite caller (R17b).

---

## 3. Interface

### 3.1 Exceptions — module `wincorp_odin.orchestration.exceptions`

```python
class SubagentError(Exception):
    """Racine des exceptions du module orchestration."""

class SubagentExecutorClosedError(SubagentError, RuntimeError):
    """submit() / cancel() / wait() apres shutdown()."""

class SubagentExecutorOverflowError(SubagentError, RuntimeError):
    """submit() lorsqu'aucune entree terminale n'est disponible pour eviction FIFO."""

class SubagentTaskNotFoundError(SubagentError, KeyError):
    """wait() sur task_id inconnu. cancel() renvoie False sans lever."""

class SubagentTaskIdConflictError(SubagentError, ValueError):
    """submit(task_id=X) lorsque X existe deja en PENDING ou RUNNING."""

class SubagentCancelledException(SubagentError):
    """Levee par une task qui observe cancel_event.is_set() == True."""
    def __init__(self, message: str = "Tache annulee cooperativement via cancel_event.") -> None:
        super().__init__(message)
```

### 3.2 Enum et types — module `wincorp_odin.orchestration.types` + `orchestration.result`

```python
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Protocol, TypedDict
import threading

class SubagentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

Role = Literal["user", "assistant", "system", "tool"]

class AIMessage(TypedDict, total=False):
    id: str                  # requis si dedup R20 actif (sinon dict conserve tel quel)
    role: Role
    content: Any             # str, list[dict] blocs tool_use Anthropic, dict
    name: str                # si role=="tool"
    tool_call_id: str        # si role=="tool"

TaskCallable = Callable[[Mapping[str, Any], threading.Event], Any]
"""Signature obligatoire d'une task soumise a SubagentExecutor.

Args:
    initial_state: read-only cote task (Mapping, pas dict).
    cancel_event: threading.Event a checker periodiquement (cooperatif).

Returns:
    Any — caller responsable de la serialisabilite si to_dict() sera appele.
    Si la task raise SubagentCancelledException → status CANCELLED.
    Si la task raise autre Exception → status FAILED, error = repr(exc)[:500].
    KeyboardInterrupt / SystemExit → propagees a l'appelant, pas FAILED.
"""

InitialState = Mapping[str, Any]
"""Alias semantique — read-only cote task. L'executor peut passer dict ou MappingProxyType."""
```

### 3.3 SubagentResult — module `wincorp_odin.orchestration.result`

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass(frozen=True, slots=True, eq=True)
class SubagentResult:
    task_id: str
    trace_id: str
    status: SubagentStatus
    submitted_at: datetime                    # UTC-aware, fige a l'instant submit()
    started_at: datetime | None               # None tant que PENDING, set au pickup RUNNING
    completed_at: datetime | None             # None tant que non terminal
    result: Any                               # None si status != COMPLETED
    error: str | None                         # populate si FAILED/CANCELLED/TIMED_OUT (tronque 500 chars)
    ai_messages: tuple[dict[str, Any], ...]  # tuple (dict pas hashable → hash desactive)

    # Explicitement non-hashable : dict dans tuple = unhashable.
    # Force la non-hashabilite au niveau du type plutot que crash runtime.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Validation ordonnee (R20b) :
        1. Validation tz-aware submitted_at
        2. Validation tz-aware started_at si non None
        3. Validation tz-aware completed_at si non None
        4. Validation type ai_messages (tuple strict, pas list)
        5. Validation chaque msg est dict + id str non vide si present
        6. Dedup via object.__setattr__ si doublons id detectes
        """
        # 1-3 tz-aware
        if self.submitted_at.tzinfo is None:
            raise ValueError("[ERREUR] SubagentResult.submitted_at doit etre tz-aware (UTC).")
        if self.started_at is not None and self.started_at.tzinfo is None:
            raise ValueError("[ERREUR] SubagentResult.started_at doit etre tz-aware (UTC).")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("[ERREUR] SubagentResult.completed_at doit etre tz-aware (UTC).")
        # 4 type ai_messages
        if not isinstance(self.ai_messages, tuple):
            raise TypeError(
                f"[ERREUR] SubagentResult.ai_messages doit etre un tuple "
                f"(recu {type(self.ai_messages).__name__}). Le caller est responsable "
                f"de la conversion list→tuple."
            )
        # 5 type messages + id
        for idx, msg in enumerate(self.ai_messages):
            if not isinstance(msg, dict):
                raise TypeError(
                    f"[ERREUR] SubagentResult.ai_messages[{idx}] doit etre un dict "
                    f"(recu {type(msg).__name__})."
                )
            if "id" in msg and not (isinstance(msg["id"], str) and msg["id"]):
                raise TypeError(
                    f"[ERREUR] SubagentResult.ai_messages[{idx}]: champ 'id' doit etre "
                    f"str non vide (recu {type(msg['id']).__name__})."
                )
        # 6 dedup (dernier gagne, position du dernier — voir R20)
        deduped = _dedup_messages_by_id(self.ai_messages)
        if deduped is not self.ai_messages:
            object.__setattr__(self, "ai_messages", deduped)

    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def duration_ms(self) -> float | None:
        """Duree d'execution en ms depuis le pickup RUNNING, None si pas encore demarre
        ou pas encore termine.

        Calcul : (completed_at - started_at).total_seconds() * 1000.
        `submitted_at` n'est pas utilise ici (temps de file d'attente exclu).
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialisation JSON-safe recursive.

        Normalise :
          - status enum → value string.
          - submitted_at/started_at/completed_at → ISO8601 (None preserve).
          - ai_messages tuple → list, content normalise via _json_safe recursif.
          - result → _json_safe recursif.
          - Champs optionnels None preserves.

        Raises:
            TypeError FR: type non serialisable a chemin JSONPath precis.
            ValueError FR: float non-fini (NaN/Inf) dans result ou ai_messages.
        """
        # Impl : top-level manuel pour datetime/enum + _json_safe pour result/ai_messages.
        ...

_TERMINAL_STATUSES: frozenset[SubagentStatus] = frozenset({
    SubagentStatus.COMPLETED,
    SubagentStatus.FAILED,
    SubagentStatus.CANCELLED,
    SubagentStatus.TIMED_OUT,
})

def _dedup_messages_by_id(
    messages: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Dedup conserve le dernier occurrence de chaque id, a la position du dernier.

    Exemple : (a1, b, a2, c) avec a1.id == a2.id == "a"
             → (b, a2, c) (a1 retire, a2 conserve a sa position originale 2).

    Si aucun doublon → retourne l'input sans copie.
    """
    ...
```

**Note `__hash__ = None`** : force la non-hashabilite au niveau du **type**. Sans cela, `@dataclass(frozen=True)` genere `__hash__` qui leve `TypeError: unhashable type: 'dict'` au premier appel (pytest dedup internes, `set[SubagentResult]`, `functools.lru_cache`). Heisenbug mystere elimine.

### 3.4 SubagentExecutor — module `wincorp_odin.orchestration.executor`

```python
from concurrent.futures import Future, ThreadPoolExecutor

class SubagentExecutor:
    def __init__(
        self,
        *,
        max_workers_scheduler: int = 3,
        max_workers_exec: int = 3,
        max_history: int = 10_000,
        sink: "SubagentSink | None" = None,
        _now_factory: Callable[[], datetime] | None = None,
        _uuid_factory: Callable[[], str] | None = None,
    ) -> None:
        """Construit l'executor. Aucun pool cree (lazy init au 1er submit).

        Args:
            max_workers_scheduler: pool qui execute le wrapper orchestration
                (PENDING→RUNNING→terminal). Minimum 1.
            max_workers_exec: pool qui execute la task utilisateur (payload metier).
                Minimum 1.
            max_history: taille max registre avant tentative eviction FIFO.
                **Doit etre >= 1, sinon ValueError FR.**
            sink: observateur on_start/on_end (defaut LogSink si None).
            _now_factory: override datetime.now(UTC) pour tests (injection).
                **Doit etre thread-safe.** Exception → propagee au caller submit,
                entry non creee (rollback).
            _uuid_factory: override uuid.uuid4() pour tests (injection).
                **Doit etre thread-safe.** Meme regle exception.

        Raises:
            ValueError FR: max_history < 1, max_workers_scheduler < 1,
                           max_workers_exec < 1.
        """

    def submit(
        self,
        task: TaskCallable,
        *,
        initial_state: Mapping[str, Any],
        timeout_sec: float,
        trace_id: str,
        task_id: str | None = None,
    ) -> str:
        """NON-BLOQUANT. Retourne task_id (uuid4 si None).

        Enregistre entree PENDING (submitted_at fige), schedule wrapper dans
        scheduler_pool. Transition PENDING→RUNNING cote wrapper au pickup
        (populate started_at a ce moment).

        Ordre validation (R8) :
            1. timeout_sec type int/float → TypeError sinon
            2. timeout_sec isnan → ValueError "NaN interdit"
            3. timeout_sec <= 0 (inclut -inf, -0.0) → ValueError
               "strictement positif"
            4. timeout_sec float('inf') accepte (no timeout)
            5. trace_id str non vide
            6. task_id str non vide si fourni
            7. task callable
            8. initial_state Mapping

        Raises:
            SubagentExecutorClosedError: shutdown() deja appele.
            SubagentExecutorOverflowError: registre plein ET aucun terminal
                evinçable.
            SubagentTaskIdConflictError: task_id existe deja en PENDING/RUNNING.
            ValueError / TypeError: validations amont.
        """

    def wait(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> SubagentResult:
        """BLOQUANT. Attend l'etat terminal de task_id.

        Args:
            task_id: identifiant de la task.
            timeout: None = attente infinie. Sinon float > 0.

        Returns:
            SubagentResult avec status terminal.

        Raises:
            SubagentTaskNotFoundError: task_id inconnu.
            TimeoutError: timeout expire (builtin Python, PAS
                SubagentStatus.TIMED_OUT — semantiques disjointes).
            KeyboardInterrupt / SystemExit: si task/sink les propagent.
        """

    def cancel(self, task_id: str) -> bool:
        """Non bloquant. Set le cancel_event de task_id.

        Returns:
            True si task_id existe (PENDING/RUNNING/terminal), False sinon
            (pas d'exception si inconnu — wait() leve, cancel() silencieux).
            Si PENDING : set event + force status CANCELLED sous lock (EC32).
            Si RUNNING : set event. Task doit observer coop.
            Si terminal : set event no-op semantique, retourne True.
        """

    def get(self, task_id: str) -> SubagentResult | None:
        """Snapshot courant (PENDING / RUNNING / terminal) ou None si inconnu.

        Lecture atomique sous lock — jamais de tearing partiel entre status
        et timestamps.
        """

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = True,
        force_timeout_sec: float | None = 5.0,
    ) -> None:
        """Ferme l'executor. Idempotent (EC5).

        Ordre strict :
            1. Set flag _closed sous _state_lock.
            2. Force status CANCELLED sur toutes les entrees PENDING.
            3. Set cancel_event sur toutes les entrees RUNNING.
            4. scheduler_pool.shutdown(wait=wait, cancel_futures=cancel_futures).
            5. exec_pool.shutdown(wait=False, cancel_futures=True).
               Note : wait=False pour ne pas bloquer indefiniment sur task zombie
               (Python 3.12 n'a pas ThreadPoolExecutor.shutdown(timeout=...) avant
               3.14). L'executor ne peut pas kill un thread Python de force.
            6. Si force_timeout_sec non None : threading.Event().wait(force_timeout_sec)
               pour laisser le temps aux tasks coop de sortir, puis scan
               threading.enumerate() pour threads `subagent-exec-*` encore vivants.
               Log WARNING avec task_id, thread.ident, sys._current_frames()[thread.ident]
               stack (pas de kill — Python ne le permet pas proprement, les threads
               non-daemon continuent jusqu'a fin coop ou process exit).

        Args:
            wait: si True, attend scheduler_pool.shutdown.
            cancel_futures: si True, cancel les Future scheduler PENDING non picked.
            force_timeout_sec: delai d'attente avant scan zombie.
                None = pas de scan. Plage recommandee [0.1, 300.0], hors plage
                → clip silencieux a 5.0 + log WARNING (pas d'exception pour
                preserver idempotence shutdown).

        Note : shutdown(wait=False) puis shutdown(wait=True) → 2e appel no-op
        (limitation volontaire documentee EC5).
        """

    def clear_history(self) -> int:
        """Supprime les entrees terminales du registre.

        Returns:
            Nombre d'entrees supprimees.
        """

    @property
    def history_size(self) -> int:
        """Nombre total d'entrees dans le registre (terminales + actives)."""

    def stats(self) -> dict[str, int]:
        """Snapshot comptage par status. Thread-safe.

        Returns:
            {
                "total": int,
                "pending": int, "running": int,
                "completed": int, "failed": int,
                "cancelled": int, "timed_out": int,
            }
        """

    def __enter__(self) -> "SubagentExecutor":
        """Context manager : retourne self."""

    def __exit__(self, *exc_info: Any) -> None:
        """Appelle shutdown(wait=True, cancel_futures=True, force_timeout_sec=5.0)."""
```

### 3.5 build_initial_state — module `wincorp_odin.orchestration.state`

```python
def build_initial_state(
    parent_state: Mapping[str, Any],
    *,
    extra_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Herite selectivement du parent_state vers un dict enfant vierge.

    Whitelist : sandbox_state, thread_data, session_id, trace_id (copiees
    profondement via copy.deepcopy). Toute autre cle du parent EXCLUE
    (messages, tool_calls, subagent_results, ...).
    extra_overrides ajoute ou remplace (priorite max).

    Raises:
        TypeError FR: parent_state non Mapping, extra_overrides non Mapping/None.
        ValueError FR: deepcopy leve (valeur non copiable) — message avec cle en cause.
    """
```

### 3.6 truncate_task_calls — module `wincorp_odin.orchestration.middleware`

```python
from collections.abc import Sequence

def truncate_task_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    max_concurrent: int = 3,
    tool_name: str = "task",
) -> list[Mapping[str, Any]]:
    """Tronque les tool_calls de type `tool_name` a `max_concurrent` occurrences.

    Ordre d'entree preserve. Calls `name != tool_name` (y compris `name` absent
    ou None) passent tous. Si count(tool_name) > max_concurrent, log WARNING
    avec le nombre droppe.

    Args:
        tool_calls: Sequence (list, tuple) de tool_calls. Les generateurs sont
            **rejetes** avec TypeError FR (necessite count + filter = 2 passes).
            Caller doit faire list(gen) au prealable.
        max_concurrent: limite stricte ∈ [1, 20]. Type `int` strict (bool rejete).
            Defaut 3. Plage elargie vs [2, 5] pour supporter les consommateurs
            Python runtime (heimdall, bifrost) qui peuvent fan-out 10-20 calls.
            La limite CLI Claude Code Agent Teams (5) n'est PAS importee — le
            caller passe `max_concurrent=5` explicite si besoin matching CLI.
        tool_name: nom du tool a limiter. String non vide.

    Returns:
        Liste tronquee.

    Raises:
        TypeError FR: max_concurrent pas int strict, tool_calls pas Sequence.
        ValueError FR: max_concurrent hors [1, 20], tool_name vide.
    """
```

### 3.7 SubagentSink + LogSink — module `wincorp_odin.orchestration.sinks`

```python
class SubagentSink(Protocol):
    def on_start(self, result: SubagentResult) -> None:
        """Appele apres transition PENDING→RUNNING, avant execution task.
        result.status == RUNNING, result.started_at non-None enforce.
        Si sink raise Exception → log WARNING, continue.
        KeyboardInterrupt / SystemExit : propages (EC25).
        """

    def on_end(self, result: SubagentResult) -> None:
        """Appele apres transition RUNNING→terminal.
        result.status.is_terminal(), result.completed_at non-None enforce.
        Si sink raise Exception → log WARNING, continue.
        KeyboardInterrupt / SystemExit : propages.
        """

class LogSink:
    """Sink par defaut : log structure JSON sur stdout."""
    def __init__(self, *, logger_name: str = "wincorp_odin.orchestration") -> None: ...
    def on_start(self, result: SubagentResult) -> None: ...
    def on_end(self, result: SubagentResult) -> None: ...
```

### 3.8 Helper `_json_safe` — prive `wincorp_odin.orchestration._json_safe`

```python
def _json_safe(obj: Any, *, _path: str = "$") -> Any:
    """Normalise un objet pour serialisation JSON stricte (recursif).

    Conversions supportees :
      - bool/int/float/str/None → passthrough (float non-fini → ValueError)
      - datetime → isoformat() string
      - bytes → base64 str (stdlib base64.b64encode)
      - Path → str (PurePath.__fspath__)
      - Enum → .value (str Enum → str, int Enum → int)
      - dataclass instance → dict recursif (via dataclasses.asdict puis _json_safe)
      - Mapping → dict recursif (cles str uniquement, sinon TypeError FR)
      - tuple/list → list recursif
      - set/frozenset → list recursif (ordre non garanti, documente best-effort)
      - autre → TypeError FR avec chemin JSONPath precis

    Args:
        obj: valeur a normaliser.
        _path: chemin JSONPath interne pour debug (usage recursif).

    Returns:
        Valeur serialisable via json.dumps.

    Raises:
        TypeError FR: type non serialisable.
        ValueError FR: float NaN / Infinity (rejetes par json strict).
    """
```

### 3.9 Synthese erreurs

| Type | Condition | Message FR (template) |
|------|-----------|-----------------------|
| `ValueError` | `max_concurrent` hors [1, 20] | "max_concurrent doit etre un entier entre 1 et 20 (recu {value})." |
| `TypeError` | `max_concurrent` pas int strict (bool, float, str, None) | "max_concurrent doit etre un entier strict (recu {type_name})." |
| `TypeError` | `tool_calls` pas Sequence (generator, set) | "tool_calls doit etre une Sequence (list, tuple) — generateurs non supportes, faire list(gen) au prealable." |
| `ValueError` | `tool_name` vide | "tool_name doit etre une chaine non vide." |
| `TypeError` | `timeout_sec` pas int/float | "timeout_sec doit etre int ou float (recu {type_name})." |
| `ValueError` | `timeout_sec` NaN | "timeout_sec NaN interdit — fournir une valeur numerique finie ou float('inf')." |
| `ValueError` | `timeout_sec` <= 0 (inclut -inf, -0.0) | "timeout_sec doit etre strictement positif (recu {value})." |
| `ValueError` | `trace_id` vide | "trace_id doit etre une chaine non vide." |
| `ValueError` | `task_id` vide (si fourni) | "task_id explicite doit etre une chaine non vide si fourni." |
| `ValueError` | `build_initial_state` deepcopy leve | "Impossible de deep-copier la cle '{key}' de parent_state : {cause}." |
| `TypeError` | `task` non callable, `initial_state` non Mapping | "task doit etre un callable prenant (state: Mapping, cancel_event: Event)." / "initial_state doit etre un Mapping." |
| `ValueError` | `max_history < 1` | "max_history doit etre >= 1 (recu {value})." |
| `ValueError` | `max_workers_scheduler < 1` ou `max_workers_exec < 1` | "max_workers_{pool} doit etre >= 1 (recu {value})." |
| `SubagentExecutorClosedError` | `submit`/`cancel`/`wait` apres `shutdown` | "SubagentExecutor ferme — {method}() refuse apres shutdown()." |
| `SubagentExecutorOverflowError` | Registre plein, aucune entree terminale evinçable | "Limite de taches actives atteinte ({max_history}) — aucune terminale a evincer, attendre une terminaison." |
| `SubagentTaskIdConflictError` | `submit(task_id=X)` avec X en PENDING/RUNNING | "task_id '{task_id}' deja actif (status={status}) — utiliser un id unique." |
| `SubagentTaskNotFoundError` | `wait(task_id)` inconnu | "task_id '{task_id}' inconnu — jamais soumis ou deja purge." |
| `TimeoutError` | `wait(timeout)` expire | "Timeout d'attente depasse ({timeout}s) pour task_id '{task_id}'." |
| `TypeError` | `ai_messages[idx].id` pas `str` non vide | "ai_messages[{idx}]: champ 'id' doit etre str non vide (recu {type_name})." |
| `TypeError` | `ai_messages` pas tuple strict | "SubagentResult.ai_messages doit etre un tuple (recu {type_name}). Le caller est responsable de la conversion list→tuple." |
| `SubagentCancelledException` | Task a vu cancel_event.is_set() | "Tache annulee cooperativement via cancel_event." |
| `TypeError` | `_json_safe` valeur non serialisable | "Type non serialisable JSON a {json_path}: <class '{type_name}'>." |
| `ValueError` | `_json_safe` float non-fini | "Valeur float non-finie a {json_path}: {value}. JSON strict rejette NaN/Infinity." |
| `ValueError` | `SubagentResult` timestamp naive | "SubagentResult.{field} doit etre tz-aware (UTC)." |

---

## 4. Regles metier

### Core enum / result / sink

- **R1** — `SubagentStatus` est `str Enum` : `SubagentStatus.RUNNING.value == "running"`. Permet serialisation JSON triviale (`json.dumps(status.value)`).
- **R2** — `SubagentResult` est `frozen=True, slots=True, eq=True` immutable, `__hash__ = None` explicite (non-hashable au niveau du type). Transitions de statut produisent de **nouvelles instances** cote registre interne (`_TaskEntry` mutable sous lock + `snapshot()` retourne `SubagentResult` frozen).
- **R3** — `SubagentResult.is_terminal()` → `status in {COMPLETED, FAILED, CANCELLED, TIMED_OUT}`. PENDING et RUNNING non-terminaux.
- **R3b** — `SubagentResult.duration_ms` property :
  - `None` si `started_at is None` OU `completed_at is None`.
  - Sinon `(completed_at - started_at).total_seconds() * 1000.0`.
  - Base sur `started_at` (pickup RUNNING), **pas** `submitted_at` (temps file d'attente exclu).
  - **Peut etre negatif** en cas de clock skew systeme (NTP drift, correction horloge pendant run). Pas de clamp automatique — responsabilite caller si monitoring veut garantir `>= 0`. Le cas est assez rare en pratique (UTC monotonic si `_now_factory` par defaut) pour ne pas polluer l'API.
- **R4** — `SubagentResult.to_dict()` :
  - `status` enum → `.value` string.
  - `submitted_at` → ISO8601 avec suffix `+00:00`.
  - `started_at` → ISO8601 si non None, sinon `None`.
  - `completed_at` → ISO8601 si non None, sinon `None`.
  - `ai_messages` tuple → list, **recursivement** normalisee via `_json_safe`.
  - `result` → normalise via `_json_safe`.
  - Aucun champ `cancel_event` (n'existe pas dans le dataclass — vit dans `_TaskEntry` cote executor).
- **R5** — `build_initial_state(parent_state)` copie **uniquement** les cles whitelist : `sandbox_state`, `thread_data`, `session_id`, `trace_id`. Toute autre cle EXCLUE.
- **R6** — `build_initial_state` : `copy.deepcopy` inconditionnel sur les cles whitelist. Si deepcopy raise → `ValueError` FR avec nom de cle en cause.
- **R7** — `extra_overrides` priorite max : injectes apres heritage parent (remplace, pas merge).

### Submit / wait / cancel / get

- **R8** — `SubagentExecutor.submit()` ordre de validation strict :
  1. `task` est callable → `TypeError` sinon.
  2. `initial_state` est `Mapping` → `TypeError` sinon.
  3. `timeout_sec` est `int` ou `float` → `TypeError` sinon.
  4. `math.isnan(timeout_sec)` → `ValueError` "NaN interdit".
  5. `timeout_sec <= 0` (inclut `-inf`, `-0.0`) → `ValueError` "strictement positif".
  6. `float('inf')` accepte (no timeout, `Future.result(timeout=inf)` fonctionne).
  7. `trace_id` str non vide → `ValueError` sinon.
  8. `task_id` str non vide si fourni → `ValueError` sinon.
  9. `_closed` flag lu sous `_state_lock` → `SubagentExecutorClosedError`.
  10. Si `task_id` fourni et existe en PENDING/RUNNING → `SubagentTaskIdConflictError`.
  11. Si `task_id` fourni et existe en terminal → ecrasement silencieux, log DEBUG.
  12. Si `task_id` None → `_uuid_factory()` (defaut `uuid.uuid4()`).
  13. `submitted_at = _now_factory()` (defaut `datetime.now(timezone.utc)`).
  14. Creer `_TaskEntry` avec `cancel_event = threading.Event()`, `status = PENDING`, `started_at = None`, `completed_at = None`.
  15. Inserer sous `_state_lock`. Si `len > max_history` : tentative eviction FIFO terminaux (voir R22). Si aucun terminal → `SubagentExecutorOverflowError` (rollback entree).
  16. Lazy init `_scheduler_pool` et `_exec_pool` sous `_state_lock` (atomique, les 2 ensemble, cf §7.6 pragma cover).
  17. Schedule wrapper dans `_scheduler_pool.submit(self._run_task_wrapper, entry, task, initial_state, timeout_sec)`.
  18. Retour `task_id` immediatement (non-bloquant).
- **R9** — Wrapper scheduler `_run_task_wrapper(entry, task, initial_state, timeout_sec)` :
  - Sous `entry._lock` :
    - Si `entry.status != PENDING` (ex cancel force CANCELLED entre submit et pickup) → sortie immediate.
    - Sinon : `entry.status = RUNNING`, `entry.started_at = _now_factory()`.
  - Notifier `sink.on_start(entry.snapshot())` hors lock. Exceptions capturees sauf KeyboardInterrupt/SystemExit (R16, EC25).
  - Schedule `exec_future = _exec_pool.submit(task, initial_state, entry.cancel_event)`.
  - Appeler `_await_with_precedence(exec_future, entry, timeout_sec)` qui retourne `(status, result, error)`.
  - Sous `entry._lock` :
    - `entry.status = <final>`, `entry.result`, `entry.error`, `entry.completed_at = _now_factory()`.
    - `entry._done_event.set()` (pour `wait()` non-polling).
  - Notifier `sink.on_end(entry.snapshot())` hors lock.
- **R9b** — `_await_with_precedence(exec_future, entry, timeout_sec)` ordre interne :
  1. Try `exec_future.result(timeout=timeout_sec)`.
  2. Si `FuturesTimeoutError` :
     a. Sous `entry._lock` : `entry._timeout_triggered = True`, `entry.cancel_event.set()`.
     b. Ne pas re-attendre (drop le future — continue background).
     c. Retour `(TIMED_OUT, None, "Timeout apres {timeout_sec}s.")`.
  3. Si task raise `SubagentCancelledException` :
     - Si `entry._timeout_triggered` → retour `(TIMED_OUT, None, ...)` (precedence R24).
     - Sinon → retour `(CANCELLED, None, "Tache annulee cooperativement.")`.
  4. Si task raise autre `Exception` :
     - Si `entry._timeout_triggered` → `(TIMED_OUT, None, ...)`.
     - Sinon si `entry.cancel_event.is_set()` → `(CANCELLED, None, repr(exc)[:500])`.
     - Sinon → `(FAILED, None, repr(exc)[:500])`.
  5. `KeyboardInterrupt` / `SystemExit` / `GeneratorExit` : non captures, propages.
  6. Si task return normalement → `(COMPLETED, result, None)` (independant de cancel_event.is_set()).
- **R10** — `cancel(task_id)` non bloquant :
  - Lookup registre sous `_state_lock`.
  - Si inconnu → `False`.
  - Si PENDING : set `cancel_event` + sous `entry._lock` transition PENDING → CANCELLED (`status = CANCELLED`, `completed_at = _now_factory()`, `error = "Annulee avant demarrage."`). Retourne `True`.
  - Si RUNNING : set `cancel_event`. Retourne `True`. Task doit observer coop.
  - Si terminal : set event (no-op semantique), retourne `True`.
- **R11** — `get(task_id)` :
  - Lookup registre sous `_state_lock`.
  - Snapshot `entry.snapshot()` sous `entry._lock` → `SubagentResult` frozen coherent.
  - Retourne snapshot ou `None` si inconnu.
- **R11b** — `wait(task_id, timeout=None)` :
  - Lookup sous `_state_lock`. Inconnu → `SubagentTaskNotFoundError`.
  - `entry._done_event.wait(timeout)` (non-polling). `False` retour → `TimeoutError` (builtin).
  - Retour `entry.snapshot()` (terminal garanti).

### Shutdown

- **R12** — `shutdown()` idempotent :
  - 1er appel sous `_state_lock` : `_closed = True`, force CANCELLED sur tous PENDING (sous `entry._lock` respectifs), set cancel_event sur tous RUNNING.
  - Hors lock : `_scheduler_pool.shutdown(wait=wait, cancel_futures=cancel_futures)`.
  - `_exec_pool.shutdown(wait=False, cancel_futures=True)` — **wait=False force** car `pool.shutdown(wait=True)` bloquerait indefiniment sur task zombie (Python 3.12 n'a pas `timeout=` avant 3.14, aucun moyen stdlib de limiter l'attente avec kill).
  - Si `force_timeout_sec` non None : **polling actif avec deadline** `time.monotonic() + force_timeout_sec` — boucle `running_events = [e._done_event for e in _entries if e.status == RUNNING]` + `wait(timeout=min(remaining, 0.05))` jusqu'a `running_events` vide OU deadline atteinte (v2.1.2, CR-025). Puis scan `threading.enumerate()` pour threads non-daemon `subagent-exec-*` encore vivants → log WARNING avec `task_id`, `thread.ident`, stack.
  - v2.1.1 et avant : `threading.Event().wait(force_timeout_sec)` systematique = shutdown lent meme si toutes tasks coop sortent rapidement. Corrige en v2.1.2.
  - Threads zombies restent vivants (threads non-daemon, process ne peut pas exit avant fin coop). Pas de kill (Python ne le permet pas proprement).
  - 2e appel `shutdown()` : sortie immediate (`_closed == True` deja).
- **R13** — `SubagentExecutor` context manager :
  - `__enter__` retourne self.
  - `__exit__` appelle `shutdown(wait=True, cancel_futures=True, force_timeout_sec=5.0)` y compris sur exception interne du `with` body.

### Validation plages

- **R14** — `truncate_task_calls` :
  - Valide `isinstance(tool_calls, Sequence)` (exclut generators, sets) → `TypeError` FR.
  - Valide `type(max_concurrent) is int` (strict, exclut `bool`) → `TypeError` FR.
  - Valide `max_concurrent ∈ [1, 20]` → `ValueError` FR.
  - Valide `tool_name` string non vide → `ValueError` FR.
  - Defaut `max_concurrent=3`.
  - **Plage [1, 20]** : consommateur Python runtime (heimdall/bifrost) peut fan-out 10-20 sub-agents legitimement. La limite CLI Agent Teams (5) n'est PAS importee — caller passe `max_concurrent=5` explicite si besoin matching CLI.

### Fan-out

- **R15** — `truncate_task_calls(tool_calls, max_concurrent, tool_name)` :
  - Parcourt `tool_calls` en ordre.
  - Conserve tous les calls dont `tc.get("name") != tool_name` (y compris `name` absent ou None).
  - Conserve les `max_concurrent` premiers dont `tc.get("name") == tool_name`. Drop les suivants.
  - Ordre global preserve.
  - Si count(tool_name) > max_concurrent → log WARNING `"truncate_task_calls: {dropped} calls '{tool_name}' droppes (fan-out > {max_concurrent})."`.

### Sinks

- **R16** — Sink errors non-bloquants :
  - `sink.on_start` raise `Exception` → log WARNING avec `task_id`, `trace_id`, type exc, message tronque 200 chars. Continue (task continue).
  - `sink.on_end` raise `Exception` → log WARNING. Continue.
  - `sink.on_start`/`on_end` raise `KeyboardInterrupt` / `SystemExit` / `GeneratorExit` → **propages** (non captures).
  - Sink `None` (param `__init__`) → defaut `LogSink` instancie automatiquement.
  - **Scope `on_end`** : invoque **uniquement** apres transition RUNNING→terminal (fin du wrapper scheduler R9). PAS invoque sur transitions directes PENDING→CANCELLED (R10 cancel pre-RUNNING, R12 shutdown force PENDING, EC25 KI dans on_start). Rationale : `on_end` signale "fin d'execution", or ces 3 cas correspondent a "pas d'execution du tout".

### Observabilite + securite

- **R17** — Aucun log du **contenu complet** de `result` ou `ai_messages` en INFO/WARNING/ERROR :
  - Logs INFO : `task_id`, `trace_id`, `status`, `duration_ms`, `submitted_at` uniquement.
  - Logs DEBUG : troncature 100 chars sur toute valeur dynamique.
  - `error` tronque 500 chars.
  - Messages d'exception : jamais `repr(obj)` user-provided. Toujours `type(obj).__name__` + troncature 200 chars.
- **R17b** — Responsabilite secrets :
  - Le caller de la `task` est responsable de ne pas retourner de secrets dans `result` ou `ai_messages`.
  - Le module **ne filtre pas** automatiquement (zero heuristique secret).
  - Convention : retourner IDs / references (ex session_id Supabase), pas valeurs sensibles (API keys, tokens).

### Timestamps

- **R18** — Timestamps UTC-aware enforce :
  - Executor utilise `_now_factory` (defaut `lambda: datetime.now(timezone.utc)`).
  - `SubagentResult.__post_init__` valide `submitted_at.tzinfo is not None` (obligatoire), `started_at.tzinfo is not None` si non None, `completed_at.tzinfo is not None` si non None.
  - ISO8601 via `.isoformat()` (suffix `+00:00` pour UTC).
  - Pas de `datetime.utcnow()` (deprecated 3.12). Garde-fou grep CI.
- **R18b** — 3 timestamps distincts :
  - `submitted_at`: fige a `submit()` (instant acceptation, inclut temps file d'attente).
  - `started_at`: fige au **pickup RUNNING** (entry._lock, wrapper scheduler). `None` tant que PENDING.
  - `completed_at`: fige a la transition terminale. `None` tant que non terminal.
  - Metrique `duration_ms` = `completed_at - started_at` (exclut file d'attente). `None` si l'un des deux est None.

### Lazy init + registre

- **R19** — Pools lazy :
  - `_scheduler_pool` et `_exec_pool` crees **ensemble** au 1er `submit()`, sous `_state_lock`. Atomic : impossible d'avoir "scheduler seul" ou "exec seul" via API publique.
  - Recrees apres `shutdown()` : **non** (executor ferme).
  - Construction `SubagentExecutor()` ne cree aucun pool, aucun thread.
  - Test branch "un seul pool lazy" : inatteignable par design → `# pragma: no cover` avec justification en §7.6.
- **R20** — Dedup `ai_messages` (applique dans `__post_init__`) :
  - Si 2 messages ont meme `id` (str non vide) → le dernier gagne **a sa position d'origine** (ordre tuple entrant preserve).
  - Exemple : `(msg1_id=a, msg2_id=b, msg3_id=a)` → `(msg2_id=b, msg3_id=a)`. msg1 retire, msg3 conserve en position 1 (apres msg2).
  - Messages sans cle `id` ou avec `id` non-str (voir R2 post-init validation) → conserves sans dedup.
  - Ordre `__post_init__` strict (R20b) : validation type → validation id → dedup.

### Eviction + overflow

- **R21** — `_background_tasks: dict[str, _TaskEntry]` :
  - Ordre insertion preserve (Python 3.7+).
  - Protege par `_state_lock`.
  - Lock par-entree (`entry._lock`) pour transitions de statut.
  - Acquisition toujours dans l'ordre `_state_lock` → `entry._lock` (anti-deadlock).
- **R22** — Eviction FIFO **par ordre insertion** sur terminaux :
  - Trigger : apres insertion dans `submit()`, check `len > max_history`.
  - Si oui : scanner le registre **en ordre d'insertion**, evincer la **premiere entree `is_terminal()`** trouvee. Repeter jusqu'a `len == max_history`.
  - **RUNNING / PENDING jamais evincees**.
  - Note : ordre insertion ≠ ordre `completed_at`. Une task ancienne par insertion mais terminal-recente est evincee avant une task jeune par insertion mais terminal-ancienne. Si besoin ordre `completed_at`, evolution R22c future.
  - Si aucun terminal evinçable (tout actif) → `SubagentExecutorOverflowError` + rollback (entree retiree du registre).
- **R22b** — `max_history` doit etre `>= 1`. `max_history < 1` → `ValueError` FR au constructeur.

### Precedence statut final

- **R23** — Precedence statut final (6 regles deterministes, voir R9b pour ordre exec) :
  - `TIMED_OUT > CANCELLED > FAILED > COMPLETED` (ordre descendant = gagne).
  - `_timeout_triggered` flag sous `entry._lock` differencie "timeout interne" de "cancel manuel".
  - `KeyboardInterrupt` / `SystemExit` / `GeneratorExit` : **non captures**, propages a l'appelant (wait() ou main thread).
- **R24** — Timeout vs cancel manuel :
  - Si `executor.cancel(task_id)` appele **avant** timeout interne : pas de `_timeout_triggered`, task voit cancel_event et raise SubagentCancelledException → `CANCELLED`.
  - Si timeout interne declenche avant cancel manuel : `_timeout_triggered = True` pose avant `cancel_event.set()`, donc task voit event + raise Cancelled, mais wrapper detecte `_timeout_triggered == True` → `TIMED_OUT` prevaut.

### Phase 2.8 future

- **R25** — Integration asyncio differee Phase 2.9 :
  - Pas de `submit_async`, pas de `bridge_cancel` en Phase 2 core.
  - Signatures stubees documentees §8 pour Phase 2.9.
  - Scenario X (thor via heimdall REST proxy) preconise en attendant.

---

## 5. Edge cases

### Core

- **EC1** — Task raise `KeyboardInterrupt` → propage (pas capture, sort de `wait()`).
- **EC2** — Task raise `SubagentCancelledException` sans `_timeout_triggered` → status `CANCELLED`.
- **EC3** — `cancel(task_id)` sur task deja terminale → retourne `True`, cancel_event.set() no-op semantique, pas de changement de statut.
- **EC4** — `cancel("inconnu")` → `False`, pas d'exception.
- **EC5** — `shutdown()` appele 2× → 2e appel no-op (idempotent R12).
- **EC6** — `shutdown(wait=False)` avec tasks RUNNING → retour immediat, scheduler_pool ferme async, exec_pool shutdown(wait=False), tasks continuent en arriere-plan coop.
- **EC7** — `submit()` apres `shutdown()` → `SubagentExecutorClosedError`.
- **EC8** — `truncate_task_calls([])` → `[]`.
- **EC9** — `truncate_task_calls([{name:"task"}×2, {name:"read_file"}, {name:"task"}×2], max=2)` → `[{name:"task"}, {name:"task"}, {name:"read_file"}]` (2 task + 1 other, ordre preserve).
- **EC10** — `truncate_task_calls` avec exactement `max_concurrent` tasks → pas de truncation, pas de log WARNING.
- **EC11** — `build_initial_state({})` → `{}`.
- **EC12** — `build_initial_state({"sandbox_state": {"x": 1}, "messages": [...]})` → `{"sandbox_state": {"x": 1}}`.
- **EC13** — `build_initial_state({"sandbox_state": {"x": 1}}, extra_overrides={"sandbox_state": {"y": 2}})` → `{"sandbox_state": {"y": 2}}` (remplace, pas merge).
- **EC14** — `SubagentResult(status=RUNNING, started_at=datetime.now(UTC), completed_at=None, ...)`, `is_terminal()` → `False`.
- **EC14b** — `SubagentResult.duration_ms` avec `started_at=None` OR `completed_at=None` → `None`.
- **EC14c** — `SubagentResult.duration_ms` avec les deux non None → `(completed_at - started_at).total_seconds() * 1000`.
- **EC15** — `to_dict()` sur result avec `ai_messages=()` → dict contient `"ai_messages": []`.
- **EC15b** — `hash(SubagentResult(...))` → `TypeError: unhashable type: 'SubagentResult'` (force par `__hash__ = None`).
- **EC16** — `submit(task_id=X)` avec X deja present **en terminal** → ecrasement silencieux (log DEBUG).
- **EC17** — `submit(task_id=X)` avec X deja **en PENDING/RUNNING** → `SubagentTaskIdConflictError`.
- **EC17b** — Sequence `submit(task_id="X")` → `cancel("X")` → `submit(task_id="X")` : le 2e submit succeed par ecrasement silencieux (R8 regle 11) car apres `cancel`, X est passe en terminal CANCELLED (R10 transition PENDING→CANCELLED). Le registre contient la nouvelle entree X (remplacement total, pas merge). Log DEBUG seul.
- **EC18** — Timeout tres court (0.01s) sur task `cancel_event.wait(1.0)` → `TIMED_OUT`.

### Concurrence fine

- **EC19** — `submit()` **concurrent** a `shutdown()` : selon ordre atomique `_state_lock`, soit submit OK, soit `SubagentExecutorClosedError`. Jamais crash.
- **EC20** — `cancel(task_id)` pendant RUNNING : task voit `is_set()`, sort → `CANCELLED`.
- **EC21** — Deux `cancel(task_id)` consecutifs : 2 × `True`, event.set() idempotent.
- **EC22** — `shutdown()` avec 100 tasks : force cancel_event sur queued + RUNNING, scheduler.shutdown(wait), exec.shutdown(wait=False, cancel_futures=True).
- **EC23** — `get(task_id)` pendant transition PENDING→RUNNING sous `entry._lock` : soit PENDING (avec `started_at=None`), soit RUNNING (avec `started_at` set). Jamais incoherent.
- **EC24** — `sink.on_start` raise + `sink.on_end` raise sur meme task → 2 logs WARNING distincts, task termine normalement.
- **EC25** — `sink.on_start` raise `KeyboardInterrupt` → **propage**. Task **pas lancee** (exec_pool.submit jamais appele). Status reste RUNNING (`started_at` pose), `cancel_event` non set. **Comportement retenu** : le Future scheduler capture l'exception, `wait(task_id)` la propage a l'appelant (pas de blocage indefini). `get(task_id)` retourne un snapshot RUNNING fige (pas de transition terminale jamais).

### Datetime

- **EC26** — `SubagentResult(submitted_at=datetime(2026,4,23))` (naive) → `ValueError` dans `__post_init__`.
- **EC27** — `SubagentResult(submitted_at=datetime(..., tzinfo=timezone(timedelta(hours=2))))` → accepte (tz-aware).
- **EC28** — `to_dict()` round-trip `from_dict()` : **pas implemente**, OUT §2.
- **EC29** — `datetime.utcnow()` par erreur → naive → ValueError.

### Typing

- **EC30** — `task` retourne `None` → `result.result is None`, status `COMPLETED`.
- **EC31** — `initial_state` passe comme `MappingProxyType` → accepte.
- **EC32** — `timeout_sec=float('inf')` → accepte, no timeout.
- **EC33** — `timeout_sec=0.0` → `ValueError` FR.
- **EC34** — `timeout_sec=-0.0` → `ValueError` FR (`-0.0 > 0` == False).
- **EC35** — `timeout_sec=float('nan')` → `ValueError` FR "NaN interdit".
- **EC35b** — `timeout_sec=float('-inf')` → `ValueError` FR "strictement positif".
- **EC35c** — `timeout_sec="5"` (str) → `TypeError` FR.

### Build_initial_state

- **EC36** — `parent_state["sandbox_state"]` contient objet avec `__deepcopy__` qui raise → `ValueError` FR avec cle.
- **EC37** — Reference circulaire dans thread_data → `copy.deepcopy` gere. OK.
- **EC38** — `Path`, `UUID`, `Enum` dans sandbox_state → deep-copied.
- **EC39** — `extra_overrides={"sandbox_state": None}` → remplace par None.
- **EC40** — `parent_state` non Mapping → `TypeError` FR.

### Truncate

- **EC41** — Dict sans cle `name` → conserve.
- **EC42** — `{"name": None}` → conserve.
- **EC43** — `tool_name="task.*"` regex-like → comparaison `==` strict, pas match.
- **EC44** — `tool_calls` est un generator → `TypeError` FR "Sequence requise".
- **EC44b** — `tool_calls` est un set → `TypeError` FR (set pas Sequence).
- **EC45** — `max_concurrent=True` → `TypeError` FR.
- **EC46** — `max_concurrent=3.5` → `TypeError` FR.
- **EC47** — `max_concurrent="3"` → `TypeError` FR.
- **EC48** — `max_concurrent=0` → `ValueError` FR (hors [1, 20]).
- **EC49** — `max_concurrent=21` → `ValueError` FR.
- **EC49b** — `max_concurrent=1` → accepte (borne basse).
- **EC49c** — `max_concurrent=20` → accepte (borne haute).

### AIMessages

- **EC50** — `ai_messages` passe en `list[dict]` (pas tuple) → **rejete** `TypeError` FR (signature stricte). Caller responsable conversion.
- **EC51** — Message sans `id` → conserve sans dedup.
- **EC52** — Message avec `id=123` → `TypeError` FR dans `__post_init__`.
- **EC53** — Message avec `id=""` → `TypeError` FR.
- **EC54** — Duplicate id : `(msg1_id=a, msg2_id=b, msg3_id=a)` → `(msg2_id=b, msg3_id=a)`. msg3 conserve en position 1. Coherent avec R20.

### Exceptions

- **EC55** — Task raise `asyncio.CancelledError` par erreur → capture comme Exception generique → `FAILED` (sauf si `_timeout_triggered`). Doc : "utiliser `SubagentCancelledException`, pas `asyncio.CancelledError`".
- **EC56** — Task raise `SubagentCancelledException` **sans** que cancel_event soit set (programmatique direct) → status `CANCELLED` (sauf si `_timeout_triggered`).

### Overflow + registre

- **EC57** — Executor construit puis jamais utilise → pas de pool, pas de thread. OK.
- **EC58** — 10 001 submit **sequentiels avec `wait()` entre chaque** (force terminal avant next submit) → registre `<= max_history`, anciennes terminales evincees.
- **EC58b** — 10 001 submit **sans wait** avec pool_size=3 → **invariant final** apres attente de toutes les tasks : `stats()["total"] <= max_history`. Test deterministe sur cet invariant uniquement. La branche "submit individuel leve OverflowError mid-run" n'est pas testee deterministically ici (dependance timing scheduler) — couverture via EC59 dediee.
- **EC59** — `max_history=10000`, 10000 tasks RUNNING + 1 submit (aucun terminal) → `SubagentExecutorOverflowError`.
- **EC59b** — `max_history=10000`, 9999 RUNNING + 1 COMPLETED + 1 submit → OK (terminal evince).
- **EC60** — `wait(task_id="inconnu")` → `SubagentTaskNotFoundError`.
- **EC61** — `wait(task_id, timeout=0.01)` sur task longue → `TimeoutError` builtin.
- **EC62** — `clear_history()` sur 3 tasks (2 terminales + 1 RUNNING) → retourne 2, 1 reste.
- **EC63** — `__exit__` avec exception body → `shutdown(wait=True)` appele, exception body propagee.
- **EC64** — `result.result` contient un `threading.Lock()` → `to_dict()` leve `TypeError` FR "non serialisable a $.result".
- **EC65** — `_json_safe` float `NaN` / `Infinity` → `ValueError` FR.

### Constructor + factories

- **EC66** — `SubagentExecutor(max_history=0)` → `ValueError` FR.
- **EC67** — `SubagentExecutor(max_workers_scheduler=0)` → `ValueError` FR.
- **EC68** — `_now_factory` raise `RuntimeError` pendant submit → propage au caller submit, entry **non** creee (pas d'insertion registre).
- **EC69** — `_uuid_factory` raise → idem.
- **EC70** — `shutdown(force_timeout_sec=-1.0)` → clip silencieux a 5.0 + log WARNING (pas ValueError pour preserver idempotence).
- **EC71** — `shutdown(force_timeout_sec=None)` → pas de scan zombie, retour apres pool shutdown.

---

## 6. Exemples concrets

### Cas nominal — submit + wait

```python
from datetime import datetime, timezone
from wincorp_odin.orchestration import (
    SubagentExecutor, SubagentStatus, build_initial_state, SubagentCancelledException,
)

def my_task(state, cancel_event):
    for i in range(10):
        if cancel_event.is_set():
            raise SubagentCancelledException()
        # ... work ...
    return "done"

parent_state = {"sandbox_state": {"cwd": "/tmp"}, "messages": [{"role": "user"}]}
initial = build_initial_state(parent_state)
# initial == {"sandbox_state": {"cwd": "/tmp"}}

with SubagentExecutor() as ex:
    task_id = ex.submit(
        my_task,
        initial_state=initial,
        timeout_sec=30.0,
        trace_id="trace-abc123",
    )
    snapshot = ex.get(task_id)
    assert snapshot.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING)
    # snapshot.submitted_at non-None, snapshot.started_at peut-etre None (si encore PENDING)

    result = ex.wait(task_id)  # bloquant
    assert result.status == SubagentStatus.COMPLETED
    assert result.result == "done"
    assert result.duration_ms is not None and result.duration_ms >= 0
    print(result.to_dict())
```

### Cas timeout interne

```python
def slow_task(state, cancel_event):
    cancel_event.wait(timeout=5.0)
    return "never"

with SubagentExecutor() as ex:
    tid = ex.submit(slow_task, initial_state={}, timeout_sec=0.1, trace_id="t2")
    result = ex.wait(tid)
    assert result.status == SubagentStatus.TIMED_OUT
    assert "Timeout apres 0.1s" in result.error
```

### Cas cancel manuel

```python
import threading
task_started = threading.Event()

def coop_task(state, cancel_event):
    task_started.set()
    while not cancel_event.is_set():
        cancel_event.wait(timeout=0.05)
    raise SubagentCancelledException()

with SubagentExecutor() as ex:
    tid = ex.submit(coop_task, initial_state={}, timeout_sec=10.0, trace_id="t3")
    task_started.wait(timeout=1.0)
    ex.cancel(tid)
    result = ex.wait(tid)
    assert result.status == SubagentStatus.CANCELLED
```

### Cas truncate fan-out

```python
from wincorp_odin.orchestration import truncate_task_calls

tool_calls = [
    {"name": "task", "args": {"t": 1}},
    {"name": "task", "args": {"t": 2}},
    {"name": "read_file", "args": {"path": "a"}},
    {"name": "task", "args": {"t": 3}},
    {"name": "task", "args": {"t": 4}},
    {"name": "task", "args": {"t": 5}},
]
truncated = truncate_task_calls(tool_calls, max_concurrent=3)
# [t=1, t=2, read_file, t=3] — log WARNING "2 calls 'task' droppes"
```

### Cas d'erreur — task_id deja actif

```python
with SubagentExecutor() as ex:
    ex.submit(slow_task, initial_state={}, timeout_sec=10.0, trace_id="t4", task_id="fixed")
    with pytest.raises(SubagentTaskIdConflictError):
        ex.submit(slow_task, initial_state={}, timeout_sec=10.0, trace_id="t4b", task_id="fixed")
```

### Cas d'erreur — overflow

```python
with SubagentExecutor(max_history=10) as ex:
    # Bloquer les 10 slots avec RUNNING
    for i in range(10):
        ex.submit(slow_task, initial_state={}, timeout_sec=30.0, trace_id=f"t{i}")
    # 11e : aucun terminal → OverflowError
    with pytest.raises(SubagentExecutorOverflowError):
        ex.submit(slow_task, initial_state={}, timeout_sec=30.0, trace_id="t11")
```

---

## 7. Dependances & contraintes

### 7.1 Techniques

- Runtime : Python >= 3.12.
- Deps : **aucune externe**. Stdlib : `concurrent.futures`, `threading`, `dataclasses`, `enum`, `datetime`, `uuid`, `logging`, `copy`, `base64`, `pathlib`, `math`.
- Zero import `langchain_*`, `anthropic`, `httpx`, `asyncio`.
- MyPy strict : `disallow_untyped_defs = true`, `strict_optional = true`.
- Ruff lint ligne max 100 chars.

### 7.2 Performance

- Pas de SLA strict (non testable CI deterministically sans benchmark dedie, hors scope).
- `truncate_task_calls` : O(n).
- `build_initial_state` : O(1) amorti (deepcopy sur whitelist).
- `cancel()` : < 10 µs (lookup + Event.set()).
- Executor mono-process, pas de garanties cross-machine.

### 7.3 Securite

- Pas de secret dans logs (R17, R17b).
- `error` tronque 500 chars.
- Pas de `pickle`, pas de `eval`, `exec`, `importlib` dynamique.
- Pas de `ctypes.pythonapi.PyThreadState_SetAsyncExc` (dangereux, non documente stable).
- Messages exceptions : troncature 200 chars sur valeur user, jamais `repr(obj)`.

### 7.4 Tests — 100% branch coverage

- `ThreadPoolExecutor`, `threading.Event` **reels** (jamais mockes — simuler la coordination = tests inutiles).
- `time.sleep` **interdit** en tests — remplace par `threading.Event.wait(timeout=X)` (sortie immediate sur signal, plus deterministe).
- `datetime.now` : injection `_now_factory` constructor (pas freezegun — incompatible `threading.Event.wait` qui utilise `_thread.allocate_lock()` non patchable freezegun).
- `uuid.uuid4` : injection `_uuid_factory` constructor.
- `logging` : pytest fixture `caplog`.
- Races : reproduction deterministe via `threading.Barrier` + monkeypatch methodes internes.
- Helper `_await_with_precedence` testable par **injection directe** (future pre-resolu, event pre-set). Test via import prive `from wincorp_odin.orchestration.executor import _await_with_precedence` ou via monkeypatch sur wrapper.
- Branches inatteignables par design : `# pragma: no cover` **uniquement** avec commentaire de justification + invariant documente (cf §7.6).

**Branches critiques avec pragma justifie** :
- R19 "un seul pool cree" : impossible par design (atomicite _state_lock). `# pragma: no cover` sur le else branch de `if self._scheduler_pool is None and self._exec_pool is None`.
- R12 "pool.shutdown avec task zombie non-coop" : impossible a tester en CI sans timeout 10s+ (une task qui `time.sleep(3600)` bloque shutdown). Test via task zombie 0.5s + `force_timeout_sec=0.1` → WARNING capture via caplog.

### 7.5 Architecture interne — 2 pools, 2 locks

**Pools** :
| Pool | Role | Justification |
|---|---|---|
| `_scheduler_pool` | Execute `_run_task_wrapper` (orchestration) | Isole orchestration du payload utilisateur |
| `_exec_pool` | Execute la `task` utilisateur | Isolation timeout control |

**Pas de 3e pool** `isolated_loop` : aucun consommateur Yggdrasil n'a besoin d'asyncio natif (Phase 2 core).

**Locks** :
| Lock | Protege | Granularite |
|---|---|---|
| `self._state_lock: threading.Lock` | `_closed`, lazy init pools, registre `_entries`, insertion/eviction/lookup | Court : manipulation structurelle uniquement |
| `entry._lock: threading.Lock` (par-entree) | Transition statut + metadata (status, result, error, started_at, completed_at, _timeout_triggered) | Encore plus court : RMW + release |

**Hierarchie anti-deadlock** : `_state_lock` → `entry._lock`. Toujours dans cet ordre. Documente dans le code.

### 7.6 Structure interne `_TaskEntry` (prive)

```python
# wincorp_odin/orchestration/_entry.py (prive, non exporte)

@dataclass
class _TaskEntry:
    """Mutable, protege par _lock. snapshot() retourne SubagentResult frozen."""
    task_id: str
    trace_id: str
    submitted_at: datetime              # fige au submit(), UTC-aware
    cancel_event: threading.Event       # partage avec task
    future: Future[Any] | None = None   # Future scheduler (pas exec)

    # Champs mutables sous _lock
    status: SubagentStatus = SubagentStatus.PENDING
    started_at: datetime | None = None      # set au pickup RUNNING
    completed_at: datetime | None = None    # set a transition terminale
    result: Any = None
    error: str | None = None
    ai_messages: tuple[dict[str, Any], ...] = ()
    _timeout_triggered: bool = False        # distingue TIMED_OUT vs CANCELLED (R24)

    # Sync primitives
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # _done_event.set() a transition terminale pour wait() non-polling.

    def snapshot(self) -> SubagentResult:
        """Construit un SubagentResult frozen coherent sous _lock.

        Acquiert self._lock le temps de lire les champs mutables, construit
        le SubagentResult hors lock (constructeur __post_init__ peut valider
        + dedup — operations sur un tuple copy, pas sur le registre).
        """
        with self._lock:
            return SubagentResult(
                task_id=self.task_id,
                trace_id=self.trace_id,
                status=self.status,
                submitted_at=self.submitted_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                result=self.result,
                error=self.error,
                ai_messages=self.ai_messages,
            )
```

**Invariants** :
1. `submitted_at` jamais mute apres creation.
2. `status` transitions autorisees : PENDING → RUNNING → {COMPLETED, FAILED, CANCELLED, TIMED_OUT}. PENDING → CANCELLED direct autorise (cancel pre-RUNNING).
3. `started_at` set une seule fois (au pickup RUNNING ou jamais si PENDING→CANCELLED).
4. `completed_at` set une seule fois (transition terminale).
5. `_timeout_triggered` set avant `cancel_event.set()` dans `_await_with_precedence` regle 2a.
6. `_done_event.set()` apres transition terminale. Permet `wait()` non-polling.
7. Acquisition lock : toujours `_state_lock` → `entry._lock`, jamais l'inverse.

---

## 8. Futur Phase 2.9 (documente, NON LIVRE)

### 8.1 `submit_async` (stub documentaire)

```python
# wincorp_odin/orchestration/asyncio_bridge.py (Phase 2.9 — NON LIVRE)
async def submit_async(
    executor: SubagentExecutor,
    coro_factory: Callable[[Mapping[str, Any], asyncio.Event], Awaitable[Any]],
    *,
    initial_state: Mapping[str, Any],
    timeout_sec: float,
    trace_id: str,
) -> str:
    """Variante asyncio via wrapper sync (Phase 2.9)."""
```

### 8.2 `bridge_cancel` helper

```python
def bridge_cancel(
    threading_event: threading.Event,
    asyncio_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Propage threading_event.set() vers asyncio_event.set() cross-thread."""
```

**Decision Phase 2 core** : scenario X (thor via heimdall REST proxy) suffit. Bridge asyncio = Phase 2.9 si consommateur Python async demontre.

---

## 9. Changelog refonte v2.1 (corrections post re-review #2)

### 5 bloquants corriges

| PR | Defaut v2.0 | Fix v2.1 |
|---|---|---|
| R2-001 | Ordre `__post_init__` ambigu dedup vs validation | R20b ordre strict : type → id → dedup via `object.__setattr__` |
| R2-002 | `started_at` fige submit casse `duration_ms` | 3 timestamps distincts : `submitted_at` (fige submit), `started_at: datetime | None` (fige pickup RUNNING), `completed_at`. `duration_ms` property base sur started_at. R18b |
| R2-003 | `force_timeout_sec` irrealisable (pool.shutdown wait=True bloque) | Exec_pool.shutdown(wait=False) **force** + `Event().wait(force_timeout_sec)` + scan threading.enumerate. R12 reecrit. Doc honnete : "threads zombies restent vivants, process continue jusqu'a fin coop" |
| R2-004 | "100% actives" ≠ logique reelle | R22 reecrit : "aucune entree terminale evinçable" = trigger overflow. EC59 clarifie, EC59b ajoute (9999 RUN + 1 terminal OK) |
| R2-005 | `__hash__` heisenbug pytest | `__hash__ = None` explicite sur SubagentResult, EC15b ajoute |

### 11 majeurs corriges

| PR | Fix |
|---|---|
| R2-006 | R22 doc explicite "ordre insertion filtre is_terminal" |
| R2-007 | R20 + EC54 alignes sur "dernier gagne, position du dernier (ordre tuple)" |
| R2-008 | R19 doc `# pragma: no cover` justifie invariant atomicite |
| R2-009 | R8 reecrit liste ordonnee validation timeout_sec + EC35b -inf + EC35c str |
| R2-010 | R9b ordre `_await_with_precedence` + `_timeout_triggered` flag sous entry._lock |
| R2-011 | §3.4 constructor doc thread-safe + EC68/EC69 factory raise |
| R2-012 | EC58 reecrit sequential-with-wait, EC58b ajoute (invariant final) |
| R2-013 | EC25 precise : on_start KI → task pas lancee, propagation via wait |
| R2-014 | **Plage [1, 20]** (option β validee user) — module Python runtime distinct CLI Agent Teams |
| R2-015 | §7.6 structure `_TaskEntry` + invariants 1-7 |
| R2-016 | R14 + EC44b : Sequence check, rejet set/generator avec TypeError FR |

### 7 mineurs corriges

| PR | Fix |
|---|---|
| R2-017 | Typo "Partie 3.3" → "§3.3" |
| R2-018 | §2 IN explicite "exclusivement synchrone" |
| R2-019 | `SubagentResult.duration_ms` property (R3b) + EC14b/EC14c |
| R2-020 | `_json_safe` documente §3.8 avec contrat explicite |
| R2-021 | R22b max_history >= 1 + EC66 |
| R2-022 | `force_timeout_sec` plage + clip silencieux + EC70/EC71 |
| R2-023 | R17b secrets caller responsable |

### Changements structurels v2.0 → v2.1

- **Renommage** `SubagentResult.started_at` figé submit → **3 timestamps distincts** `submitted_at` / `started_at` / `completed_at`. Breaking change interne Phase 2 (module non encore livre, donc pas de casse externe).
- **`__hash__ = None`** ajoute.
- **Plage `max_concurrent`** : [2, 5] → **[1, 20]**.
- **`max_history`** valide >= 1 (R22b).
- **`force_timeout_sec`** comportement corrige : Event.wait + scan, pas pool.shutdown(timeout).
- **Nouvelle propriete** `SubagentResult.duration_ms`.
- **§7.6 nouveau** — structure interne `_TaskEntry` documentee.
- **R9b nouveau** — ordre `_await_with_precedence` explicite.
- **`_timeout_triggered` flag** — ajoute `_TaskEntry`, differencie TIMED_OUT vs CANCELLED.
- **`_dedup_messages_by_id` helper** — documentation comportement "dernier gagne position du dernier".

### Breaking changes externes

Aucun — module non encore livre. Spec v1.0 + v2.0 DRAFT, v2.1 DRAFT finale post-2 reviews adversariales.

---

## 10. Changelog

| Version | Date | Modification |
|---------|------|--------------|
| 1.0 | 2026-04-23 | Creation initiale — Phase 2 DeerFlow (REJETE review adversariale #1 : 23 defauts, 5 bloquants). |
| 2.0 | 2026-04-23 | Refonte complete — 3 agents Opus (archi / testabilite / edge cases). API non-bloquante, 2 pools, [2, 5] max_concurrent, registre borne, exceptions dediees, TypedDict, `_json_safe` recursif, precedence deterministe (REJETE review #2 : 23 defauts, 5 bloquants). |
| 2.1 | 2026-04-23 | Corrections post re-review #2 : 3 timestamps distincts, `__hash__=None`, plage [1, 20], `_TaskEntry` §7.6, `_timeout_triggered`, force_timeout corrige, 70+ EC. |
| 2.1.1 | 2026-04-23 | 5 mineurs post re-review #3 (GO avec corrections) : EC25 trancher, EC58b reformule, R16 scope on_end, R3b clock skew, EC17b submit→cancel→submit. |
