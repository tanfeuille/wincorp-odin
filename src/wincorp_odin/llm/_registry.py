"""Resolution dynamique `pkg.module:Classe` -> type, avec cache.

@spec specs/llm-factory.spec.md v1.2

R5 : zero import hardcode de ChatAnthropic dans factory.py.
R8 : module prive, non ré-exporte par __init__.py.
PB-012 : cache invalide par `_reload_for_tests` / swap atomique.
"""
from __future__ import annotations

import importlib
from typing import Any

from wincorp_odin.llm.exceptions import ModelConfigError, ProviderNotInstalledError

# Cache classe resolue : {use_string: classe}
_class_cache: dict[str, type[Any]] = {}


def _parse_use(use: str) -> tuple[str, str]:
    """Decompose 'pkg.module:Classe' en (pkg.module, Classe).

    Raises:
        ProviderNotInstalledError: format invalide (pas de ':' ou vide).
    """
    if ":" not in use:
        raise ProviderNotInstalledError(
            f"[ERREUR] Format 'use:' invalide : '{use}'. "
            f"Attendu : 'pkg.module:Classe' (ex 'langchain_anthropic:ChatAnthropic')."
        )
    module_name, _, class_name = use.partition(":")
    if not module_name or not class_name:
        raise ProviderNotInstalledError(
            f"[ERREUR] Format 'use:' invalide : '{use}'. "
            f"Module ou classe vide. Attendu : 'pkg.module:Classe'."
        )
    return module_name, class_name


def validate_use_format(use: str) -> None:
    """Valide uniquement le format `pkg.module:Classe` sans resoudre l'import.

    Utilise par `validate_all_models()` pour verifier les modeles disabled
    (PB-018) sans declencher l'import d'un package potentiellement absent.

    Raises:
        ModelConfigError: format invalide (pas de ':' ou module/classe vide).
    """
    if ":" not in use:
        raise ModelConfigError(
            f"[ERREUR] Format 'use:' invalide : '{use}'. "
            f"Attendu : 'pkg.module:Classe' (ex 'langchain_anthropic:ChatAnthropic')."
        )
    module_name, _, class_name = use.partition(":")
    if not module_name or not class_name:
        raise ModelConfigError(
            f"[ERREUR] Format 'use:' invalide : '{use}'. "
            f"Module ou classe vide. Attendu : 'pkg.module:Classe'."
        )


def resolve_class(use: str) -> type[Any]:
    """Resout 'pkg.module:Classe' en classe Python via importlib.

    Cache la classe resolue. Invalide par `_reload_for_tests` ou swap atomique.

    Args:
        use: Identifiant au format 'pkg.module:Classe'.

    Returns:
        La classe Python resolue (instanciable).

    Raises:
        ProviderNotInstalledError: package non installe, classe absente, ou objet
            non callable.
    """
    cached = _class_cache.get(use)
    if cached is not None:
        return cached

    module_name, class_name = _parse_use(use)

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ProviderNotInstalledError(
            f"[ERREUR] Package '{module_name}' requis par models.yaml "
            f"mais non installe. Executer : uv pip install {module_name.replace('_', '-')}"
        ) from e

    klass = getattr(module, class_name, None)
    if klass is None:
        raise ProviderNotInstalledError(
            f"[ERREUR] Classe '{class_name}' introuvable dans '{module_name}'. "
            f"Verifier le nom dans models.yaml."
        )

    if not callable(klass):
        raise ProviderNotInstalledError(
            f"[ERREUR] '{use}' n'est pas instanciable (objet non-callable)."
        )

    _class_cache[use] = klass
    return klass
