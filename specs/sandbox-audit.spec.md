# sandbox-audit — Specification

> **Statut :** IMPLEMENTED
> **Version :** 1.0
> **Niveau :** 2 (standard)
> **Auteur :** Tan Phi HUYNH
> **Date de creation :** 2026-04-22
> **Implementation :** 72 tests verts / 100% branch coverage / 2026-04-22

---

## 1. Objectif

Classifier chaque commande bash soumise à un sub-agent Odin selon son niveau de risque (block / warn / pass) pour **empêcher les attaques trivialement destructives** (rm -rf /, fork bomb, curl|sh, pipe base64→exec) avant exécution dans un sandbox. Port du pattern DeerFlow `SandboxAuditMiddleware` (cf veille 2026-04-22 Phase 4 DeerFlow).

Ne remplace PAS un sandbox OS-level (Docker, chroot) — c'est une **barrière applicative complémentaire** qui catch les patterns d'attaque connus avant de laisser le shell s'exécuter.

---

## 2. Perimetre

### IN — Ce que le module fait

- Fournit une fonction pure `classify_command(command: str) -> ClassificationResult` (verdict + raison optionnelle).
- Valide l'input (empty, too long, null byte) avant classification regex.
- Split quote-aware les commandes composées (`;`, `&&`, `||`) pour classifier chaque sous-commande indépendamment.
- Applique **2 passes** : (1) scan whole raw command pour patterns multi-statement (fork bomb, while true), (2) classification par sous-commande.
- Fournit `AuditLogger` optionnel pour écrire un audit trail JSONL (format append-only).
- Enum `Verdict` sérialisable (`block` / `warn` / `pass`).
- Messages d'erreur en français (CLAUDE.md odin).

### OUT — Ce que le module ne fait PAS

- N'exécute PAS les commandes (laisse ça au caller).
- Ne remplace PAS un sandbox OS-level (Docker, chroot, namespace).
- Ne log PAS le contenu **complet** des commandes dans le JSONL (tronque à 500 chars pour éviter les blow-ups et les secrets dans les logs).
- Ne gère PAS les prompt injection dans le contenu même des commandes (hors scope — c'est du contenu applicatif).
- N'intègre PAS avec LangChain/LangGraph (module indépendant, consommable par tout orchestrateur).
- Ne chiffre PAS l'audit log (fait partie d'un niveau supérieur si besoin).

---

## 3. Interface

### Fonction principale

```python
def classify_command(command: str) -> ClassificationResult: ...
def validate_input(command: str) -> str | None: ...
```

### Classes publiques

```python
class Verdict(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    PASS = "pass"

@dataclass(frozen=True)
class ClassificationResult:
    verdict: Verdict
    reason: str | None = None  # non-None uniquement si BLOCK pour input sanitisation

@dataclass(frozen=True)
class AuditEvent:
    timestamp: str  # ISO 8601 UTC
    command: str    # tronqué à 500 chars
    verdict: Verdict
    thread_id: str | None = None
    reason: str | None = None

class AuditLogger:
    def __init__(self, log_path: Path | None = None): ...
    def write(self, event: AuditEvent) -> None: ...
    def close(self) -> None: ...
```

### Inputs / Outputs

| Param | Type | Obligatoire | Description | Défaut | Exemple |
|-------|------|:-:|-------------|--------|---------|
| `command` | `str` | Oui | Commande bash brute | — | `"rm -rf /tmp/foo"` |
| `log_path` | `Path \| None` | Non | Chemin du JSONL, None = pas de persistance | `None` | `Path("~/.wincorp/bash-audit.jsonl")` |

### Erreurs

Le module **ne lève jamais** d'exception pour un input utilisateur invalide. Toute anomalie devient un `Verdict.BLOCK` avec `reason` explicite. Les seules exceptions possibles sont :

| Type | Condition | Comportement |
|------|-----------|--------------|
| `PermissionError` | AuditLogger : log_path non accessible en écriture | Lever (erreur config, pas erreur runtime) |
| `OSError` | AuditLogger : disque plein, etc. | Lever |

---

## 4. Regles metier

- **R1** : Command vide ou whitespace only → `BLOCK` (reason="empty command").
- **R2** : `len(command) > 10_000` → `BLOCK` (reason="command too long"). Seuil choisi : ≫ usage légitime, ≪ ARG_MAX Linux. Cible les payload injections / base64.
- **R3** : Null byte (`\x00`) dans command → `BLOCK` (reason="null byte detected").
- **R4** : Pattern high-risk sur **whole raw command** normalisée → `BLOCK`. Priorité sur split car certains patterns (fork bomb, `while true; do bash & done`) perdent leur signature si splitté.
- **R5** : Split compound quote-aware sur `;`, `&&`, `||` **hors** single/double quotes et hors backslash-escape. Les opérateurs à l'intérieur de guillemets ne splittent pas.
- **R6** : Quote non fermée ou backslash dangling → retourne le command entier (fail-closed), classifié comme une seule commande.
- **R7** : Pour chaque sub-command, classification via regex whole + `shlex.split` → join → regex (double passe robuste).
- **R8** : `shlex.split` échoue (unclosed quote détecté dans sub) → `BLOCK` (suspect, fail-closed).
- **R9** : Pattern high-risk match dans une sub-command → `BLOCK` (short-circuit).
- **R10** : Pattern medium-risk match dans une sub-command → `WARN`.
- **R11** : Aucun pattern match → `PASS`.
- **R12** : Verdict global = max du pire des sub-commands (`BLOCK > WARN > PASS`).

### Patterns high-risk (block)

Hérités du port DeerFlow, adaptés :

1. `rm -rf /*`, `rm -rf ~/*`, `rm -rf /home`, `rm -rf /root` — destruction système.
2. `dd if=` — écriture raw device (disque).
3. `mkfs*` — format filesystem.
4. `cat /etc/shadow` — lecture hashes password.
5. `> /etc/*` — overwrite config système.
6. `| sh` / `| bash` — pipe vers shell (curl|sh généralisé).
7. `` `...` `` ou `$(...)` contenant `curl|wget|bash|sh|python|ruby|perl|base64` — command substitution à risque.
8. `base64 -d | ...` — decode base64 piped to exec.
9. `> /usr/bin/*` / `> /bin/*` / `> /sbin/*` — overwrite binaires système.
10. `> ~/.bashrc` / `~/.profile` / `~/.zshrc` / `~/.bash_profile` — persistance shell hijack.
11. `/proc/*/environ` — leak env vars process.
12. `LD_PRELOAD=` / `LD_LIBRARY_PATH=` — dynamic linker hijack.
13. `/dev/tcp/` — bash built-in TCP (bypass tool allowlist).
14. `:(){ :|:& };:` et variantes — fork bomb.
15. `while true; do ... & done` — fork loop.

### Patterns medium-risk (warn)

1. `chmod 777` — permissions ouvertes.
2. `pip install`, `pip3 install`, `apt install`, `apt-get install` — install de packages.
3. `sudo`, `su` — élévation (no-op sous Docker root mais alerte LLM).
4. `PATH=` — modification PATH (attack chain long).

---

## 5. Edge cases

- **EC1** : `command=""` (vide) → `BLOCK` (reason="empty command").
- **EC2** : `command="   "` (whitespace only) → `BLOCK` (reason="empty command").
- **EC3** : `command="a" * 10001` (trop long) → `BLOCK` (reason="command too long").
- **EC4** : `command="ls\x00"` (null byte) → `BLOCK` (reason="null byte detected").
- **EC5** : `command="echo 'unclosed"` (quote non fermée) → classification entière fail-closed. Si pattern détecté → BLOCK. Sinon → PASS (la quote manquante n'est pas en soi un risque).
- **EC6** : `command="safe;rm -rf /*"` (no space autour `;`) → split quote-aware → `safe` PASS + `rm -rf /*` BLOCK → `BLOCK`.
- **EC7** : `command="echo 'rm -rf /' # comment"` — pattern `rm -rf /*` dans literal quoted → whole-command regex MATCH (la regex ne distingue pas quoted/unquoted) → `BLOCK`. **Fail-closed voulu** — mieux bloquer un faux positif que laisser passer une injection.
- **EC8** : `command="echo && chmod 777 file && echo done"` → split → 3 sub-commands → `PASS`, `WARN`, `PASS` → verdict global `WARN`.
- **EC9** : Command avec backslash d'escape non terminé (`echo hello\\`) → return entier fail-closed.
- **EC10** : Command UTF-8 avec emojis / accents → classification sur bytes/str Python (pas de normalisation). Les regex ASCII ne matchent pas → `PASS` sauf si contient pattern ASCII-only (expected).
- **EC11** : Double quote imbriquée (`"a 'b' c"`) → track séparément single/double, ne switche pas quote state à l'intérieur de l'autre.
- **EC12** : Pipe simple (`cat file | grep foo`) → pas un compound operator, pas splitté (ce n'est pas `&&`/`||`/`;`), classifié en entier.

---

## 6. Exemples concrets

### Cas nominal — command safe

```python
result = classify_command("ls -la /tmp")
# ClassificationResult(verdict=Verdict.PASS, reason=None)
```

### Cas nominal — medium-risk warn

```python
result = classify_command("chmod 777 /tmp/foo && ls")
# ClassificationResult(verdict=Verdict.WARN, reason=None)
```

### Cas nominal — high-risk block

```python
result = classify_command("rm -rf /*")
# ClassificationResult(verdict=Verdict.BLOCK, reason=None)
# Le reason est None parce que c'est un match regex, pas une rejet input sanitisation.
```

### Cas input sanitisation

```python
result = classify_command("")
# ClassificationResult(verdict=Verdict.BLOCK, reason="empty command")

result = classify_command("\x00ls")
# ClassificationResult(verdict=Verdict.BLOCK, reason="null byte detected")
```

### Cas fork bomb (multi-statement)

```python
result = classify_command(":(){ :|:& };:")
# ClassificationResult(verdict=Verdict.BLOCK, reason=None)
# Détecté par pass 1 whole-command regex (split sur ; détruirait la signature)
```

### Audit log JSONL

```python
logger = AuditLogger(log_path=Path.home() / ".wincorp/bash-audit.jsonl")
event = AuditEvent(
    timestamp="2026-04-22T14:00:00Z",
    command="rm -rf /",
    verdict=Verdict.BLOCK,
    thread_id="thr_abc123",
)
logger.write(event)
# Append JSON line to log_path (création dossier parent si absent)
logger.close()
```

---

## 7. Dependances & contraintes

### Techniques

- Runtime : **Python >= 3.12** (CLAUDE.md odin).
- Module system : stdlib uniquement (`re`, `shlex`, `json`, `dataclasses`, `enum`, `pathlib`, `datetime`).
- Dependances externes : **aucune** (pas de pydantic pour ce module — overkill pour des dataclasses simples).

### Performance

- `classify_command` < 1 ms pour une commande de 1 KB (regex compilées au import).
- `AuditLogger.write` < 10 ms (append-only, pas de fsync explicite).
- 100% branch coverage requis (CLAUDE.md odin).

### Securite

- **Aucun secret / donnée client en clair** dans les logs.
- Troncature commande à 500 chars dans l'audit event pour éviter fuite de longs payloads.
- Fail-closed partout (erreur de parsing → BLOCK).
- Thread-safe : `AuditLogger` utilise `threading.Lock` sur les writes.

---

## 8. Changelog

| Version | Date | Modification |
|---------|------|--------------|
| 1.0 | 2026-04-22 | Creation initiale — port DeerFlow Phase 4. Regles R1-R12, EC1-EC12. |
