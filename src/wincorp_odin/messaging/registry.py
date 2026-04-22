"""Registry lazy des canaux — import dynamique via chemin `module:ClassName`.

@spec specs/messaging.spec.md v1.0
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wincorp_odin.messaging.base import Channel

#: Registry lazy : nom → path d'import `module:ClassName`.
#: L'import n'a lieu qu'au premier `load_channel(name)`.
CHANNEL_REGISTRY: dict[str, str] = {
    "telegram": "wincorp_odin.messaging.channels.telegram:TelegramChannel",
    "whatsapp": "wincorp_odin.messaging.channels.whatsapp:WhatsAppChannel",
}


def load_channel(name: str, **kwargs: Any) -> Channel:
    """Instancie un canal à partir du registry.

    Args:
        name: clé dans CHANNEL_REGISTRY (ex: "telegram").
        **kwargs: arguments passés au constructeur du canal.

    Raises:
        ValueError: si `name` n'est pas dans CHANNEL_REGISTRY.
        ImportError: si le module/classe pointé n'est pas importable.
    """
    use_path = CHANNEL_REGISTRY.get(name)
    if use_path is None:
        raise ValueError(
            f"Canal '{name}' inconnu. Disponibles : {sorted(CHANNEL_REGISTRY)}"
        )
    module_path, _, class_name = use_path.partition(":")
    if not class_name:
        raise ValueError(
            f"Format registry invalide pour '{name}' : attendu 'module:ClassName', "
            f"reçu '{use_path}'"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)  # type: ignore[no-any-return]
