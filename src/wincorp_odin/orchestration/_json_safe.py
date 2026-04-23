"""Helper prive `_json_safe` : normalisation recursive pour serialisation JSON.

@spec specs/orchestration.spec.md v2.1.1 §3.8

Support :
    - scalars passthrough (bool/int/float/str/None) avec garde NaN/Inf.
    - datetime -> isoformat().
    - bytes -> base64 str.
    - Path -> str via __fspath__.
    - Enum -> .value.
    - dataclass -> dict recursif via dataclasses.asdict.
    - Mapping -> dict recursif, cles str uniquement.
    - tuple/list -> list recursif.
    - set/frozenset -> list recursif best-effort ordre.
    - autre -> TypeError FR avec chemin JSONPath.
"""
from __future__ import annotations

import base64
import dataclasses
import math
import os
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any


def _json_safe(obj: Any, *, _path: str = "$") -> Any:
    """Normalise un objet pour serialisation JSON stricte (recursif).

    Args:
        obj: valeur a normaliser.
        _path: chemin JSONPath interne pour debug (usage recursif).

    Returns:
        Valeur serialisable via json.dumps.

    Raises:
        TypeError: type non serialisable, message FR avec chemin.
        ValueError: float NaN / Infinity (rejetes par json strict).
    """
    # None / bool (bool doit precéder int car True == 1 mais isinstance(True, int)==True)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(
                f"Valeur float non-finie a {_path}: {obj}. JSON strict rejette NaN/Infinity."
            )
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # dataclasses.asdict appelle deep-copy recursif, mais contient pas la conversion
        # enum/datetime/bytes -> on re-passe _json_safe sur la sortie.
        return _json_safe(dataclasses.asdict(obj), _path=_path)
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Type non serialisable JSON a {_path}: cle de type "
                    f"'{type(key).__name__}' (cles str uniquement)."
                )
            out[key] = _json_safe(value, _path=f"{_path}.{key}")
        return out
    if isinstance(obj, (tuple, list)):
        return [_json_safe(item, _path=f"{_path}[{idx}]") for idx, item in enumerate(obj)]
    if isinstance(obj, (set, frozenset)):
        # Ordre best-effort (non garanti documente §3.8).
        return [
            _json_safe(item, _path=f"{_path}[{idx}]")
            for idx, item in enumerate(obj)
        ]
    raise TypeError(
        f"Type non serialisable JSON a {_path}: <class '{type(obj).__name__}'>."
    )
