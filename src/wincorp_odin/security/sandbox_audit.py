"""Classification bash + audit log.

@spec specs/sandbox-audit.spec.md v1.0

Port du pattern DeerFlow `SandboxAuditMiddleware` (cf
`memory/project_deerflow_inspiration_plan.md` Phase 4). Classifie une commande
bash selon son niveau de risque (block / warn / pass) avant exécution.

Usage :
    from wincorp_odin.security import classify_command, Verdict

    result = classify_command("rm -rf /*")
    if result.verdict == Verdict.BLOCK:
        raise PermissionError(f"Commande bloquée : {result.reason or 'patron à risque'}")
"""
from __future__ import annotations

import json
import re
import shlex
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TextIO

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_MAX_COMMAND_LENGTH = 10_000
_MAX_LOG_COMMAND_CHARS = 500  # troncature pour audit log (évite fuite payloads)

# Patterns compilés une seule fois au chargement (perf).
_HIGH_RISK_PATTERNS: list[re.Pattern[str]] = [
    # rm destructif (racine, home, root)
    re.compile(r"rm\s+-[^\s]*r[^\s]*\s+(/\*?|~/?\*?|/home\b|/root\b)\s*$"),
    # dd write raw device
    re.compile(r"dd\s+if="),
    # format filesystem
    re.compile(r"mkfs"),
    # lecture hashes passwords
    re.compile(r"cat\s+/etc/shadow"),
    # overwrite config système
    re.compile(r">+\s*/etc/"),
    # pipe to sh/bash (généralisation curl|sh)
    re.compile(r"\|\s*(ba)?sh\b"),
    # command substitution avec executables dangereux
    re.compile(r"[`$]\(?\s*(curl|wget|bash|sh|python|ruby|perl|base64)"),
    # base64 decode piped to execution
    re.compile(r"base64\s+.*-d.*\|"),
    # overwrite binaires système
    re.compile(r">+\s*(/usr/bin/|/bin/|/sbin/)"),
    # overwrite shell startup files
    re.compile(r">+\s*~/?\.(bashrc|profile|zshrc|bash_profile)"),
    # leak env vars process
    re.compile(r"/proc/[^/]+/environ"),
    # dynamic linker hijack
    re.compile(r"\b(LD_PRELOAD|LD_LIBRARY_PATH)\s*="),
    # bash built-in TCP (bypass allowlist)
    re.compile(r"/dev/tcp/"),
    # fork bomb :(){ :|:& };:
    re.compile(r"\S+\(\)\s*\{[^}]*\|\s*\S+\s*&"),
    # fork loop
    re.compile(r"while\s+true.*&\s*done"),
]

_MEDIUM_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"chmod\s+777"),
    re.compile(r"pip3?\s+install"),
    re.compile(r"apt(-get)?\s+install"),
    re.compile(r"\b(sudo|su)\b"),
    re.compile(r"\bPATH\s*="),
]


# ---------------------------------------------------------------------------
# Types publics
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    """Niveau de risque d'une commande bash."""

    BLOCK = "block"
    WARN = "warn"
    PASS = "pass"


@dataclass(frozen=True)
class ClassificationResult:
    """Résultat de la classification d'une commande."""

    verdict: Verdict
    reason: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    """Événement à logger dans le JSONL d'audit."""

    timestamp: str  # ISO 8601 UTC
    command: str  # tronqué à _MAX_LOG_COMMAND_CHARS
    verdict: Verdict
    thread_id: str | None = None
    reason: str | None = None

    def to_json(self) -> str:
        """Sérialise en une ligne JSON."""
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def validate_input(command: str) -> str | None:
    """Valide l'input. Retourne raison de rejet ou None si OK.

    Règles :
    - R1 : vide / whitespace only → "empty command".
    - R2 : > _MAX_COMMAND_LENGTH → "command too long".
    - R3 : null byte → "null byte detected".
    """
    if not command or not command.strip():
        return "empty command"
    if len(command) > _MAX_COMMAND_LENGTH:
        return "command too long"
    if "\x00" in command:
        return "null byte detected"
    return None


def classify_command(command: str) -> ClassificationResult:
    """Classe une commande bash selon son risque.

    Applique :
    1. Validation input (R1-R3) → BLOCK si rejet.
    2. Scan whole raw command pour patterns multi-statement (fork bomb, etc.) (R4).
    3. Split quote-aware en sub-commands (R5-R6).
    4. Classification de chaque sub (R7-R11).
    5. Verdict global = max(BLOCK > WARN > PASS) (R12).
    """
    # Étape 1 : validation input (R1-R3)
    reject_reason = validate_input(command)
    if reject_reason:
        return ClassificationResult(verdict=Verdict.BLOCK, reason=reject_reason)

    # Étape 2 : scan whole raw command (R4)
    normalized = " ".join(command.split())
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(normalized):
            return ClassificationResult(verdict=Verdict.BLOCK)

    # Étape 3 : split quote-aware + classification per-sub (R5-R12)
    sub_commands = _split_compound_command(command)
    worst: Verdict = Verdict.PASS
    for sub in sub_commands:
        verdict = _classify_single_command(sub)
        if verdict == Verdict.BLOCK:
            return ClassificationResult(verdict=Verdict.BLOCK)  # short-circuit R9
        if verdict == Verdict.WARN:
            worst = Verdict.WARN  # on continue pour détecter un BLOCK plus loin
    return ClassificationResult(verdict=worst)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_compound_command(command: str) -> list[str]:
    """Split quote-aware sur ;, &&, || (R5).

    Respecte single/double quotes et backslash-escape. Si quote non fermée ou
    escape dangling → retourne [command] entier (fail-closed, R6).
    """
    parts: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaping = False
    i = 0

    while i < len(command):
        ch = command[i]

        if escaping:
            current.append(ch)
            escaping = False
            i += 1
            continue

        if ch == "\\" and not in_single_quote:
            current.append(ch)
            escaping = True
            i += 1
            continue

        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(ch)
            i += 1
            continue

        if not in_single_quote and not in_double_quote:
            if command.startswith("&&", i) or command.startswith("||", i):
                piece = "".join(current).strip()
                if piece:
                    parts.append(piece)
                current = []
                i += 2
                continue
            if ch == ";":
                piece = "".join(current).strip()
                if piece:
                    parts.append(piece)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    # Quote non fermée ou escape dangling → fail-closed (R6)
    if in_single_quote or in_double_quote or escaping:
        return [command]

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)
    return parts if parts else [command]


def _classify_single_command(command: str) -> Verdict:
    """Classe une sub-command (non-compound). Applique R7-R11."""
    normalized = " ".join(command.split())

    # Pass regex raw (R9 high-risk)
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(normalized):
            return Verdict.BLOCK

    # Pass shlex-parsed tokens (R7) — attrape des formulations avec espaces/quotes
    try:
        tokens = shlex.split(command)
        joined = " ".join(tokens)
        for pattern in _HIGH_RISK_PATTERNS:
            if pattern.search(joined):
                return Verdict.BLOCK
    except ValueError:
        # R8 : unclosed quote → fail-closed BLOCK
        return Verdict.BLOCK

    # Medium-risk (R10)
    for pattern in _MEDIUM_RISK_PATTERNS:
        if pattern.search(normalized):
            return Verdict.WARN

    return Verdict.PASS


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Écrit un audit trail JSONL append-only.

    Usage :
        logger = AuditLogger(Path("~/.wincorp/bash-audit.jsonl").expanduser())
        logger.write(AuditEvent(
            timestamp=datetime.now(UTC).isoformat(),
            command="rm -rf /",
            verdict=Verdict.BLOCK,
        ))
        logger.close()

    Thread-safe : les writes sont sérialisés par un Lock.
    Si `log_path` est None → no-op (logger désactivé).
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: AuditEvent) -> None:
        """Append un event au JSONL. No-op si logger désactivé."""
        if self._fh is None:
            return
        truncated_command = event.command[:_MAX_LOG_COMMAND_CHARS]
        truncated_event = AuditEvent(
            timestamp=event.timestamp,
            command=truncated_command,
            verdict=event.verdict,
            thread_id=event.thread_id,
            reason=event.reason,
        )
        line = truncated_event.to_json() + "\n"
        with self._lock:
            self._fh.write(line)

    def close(self) -> None:
        """Ferme le file handle."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def current_utc_timestamp() -> str:
    """Retourne un timestamp ISO 8601 UTC (helper pour construire AuditEvent)."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
