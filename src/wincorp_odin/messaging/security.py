"""Security helpers pour messaging (path traversal guard, etc.).

@spec specs/messaging.spec.md v1.0
"""
from __future__ import annotations

import re
from pathlib import Path

#: Noms de fichiers acceptés pour download (alphanum + underscore + dash + point).
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def safe_download_path(filename: str, base_dir: Path) -> Path:
    """Valide `filename` et retourne le path résolu sous `base_dir`.

    Règles (R5) :
    - `filename` doit matcher `^[A-Za-z0-9_\\-.]+$` (pas de `/`, pas de `..`).
    - Le path résolu doit être strictement dans `base_dir.resolve()`.

    Raises:
        ValueError: si le nom de fichier contient des caractères interdits ou
            si le path résolu sort de `base_dir` (défense en profondeur).
    """
    if not filename or filename in {".", ".."}:
        raise ValueError(f"Nom de fichier invalide : {filename!r}")
    if not _SAFE_FILENAME_RE.match(filename):
        raise ValueError(
            f"Nom de fichier contient des caractères interdits : {filename!r}. "
            "Autorisés : A-Z a-z 0-9 _ - ."
        )

    base_resolved = base_dir.resolve()
    candidate = (base_resolved / filename).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:  # pragma: no cover
        # Défense en profondeur : la regex rejette déjà ../ et /, donc ce chemin
        # ne s'atteint que via symlink hostile ou bug regex. Garde-fou final.
        raise ValueError(
            f"Path traversal détecté : {candidate} sort de {base_resolved}"
        ) from exc
    return candidate
