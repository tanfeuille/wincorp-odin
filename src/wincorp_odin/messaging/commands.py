"""Parsing de commandes (/new, /status, /memory, ...).

@spec specs/messaging.spec.md v1.0
"""
from __future__ import annotations

#: Allowlist centralisée des commandes connues du bus.
#: Étendre ici pour ajouter une commande ; le dispatcher lit ce frozenset.
KNOWN_COMMANDS: frozenset[str] = frozenset({
    "/new",
    "/status",
    "/memory",
    "/models",
    "/help",
})


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Parse un texte en (commande, args) si texte commence par une /commande connue.

    Règles (R6) :
    - Retourne `(cmd, args)` si `text` commence par `/` + mot dans KNOWN_COMMANDS.
    - Retourne `None` si texte vide, ne commence pas par `/`, ou commande inconnue.
    - `args` = liste des tokens séparés par espaces après la commande.

    Exemples :
        parse_command("/status") → ("/status", [])
        parse_command("/new mon sujet") → ("/new", ["mon", "sujet"])
        parse_command("/unknown foo") → None
        parse_command("") → None
        parse_command("hello") → None
    """
    if not text or not text.startswith("/"):
        return None
    tokens = text.strip().split()
    if not tokens:  # pragma: no cover
        # Defensif : text non vide avec "/" → split renvoie forcément ≥1 token.
        return None
    head = tokens[0]
    if head not in KNOWN_COMMANDS:
        return None
    return head, tokens[1:]
