"""ABC Channel + MessageBus asyncio + dataclasses.

@spec specs/messaging.spec.md v1.0
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ChannelSendError(Exception):
    """Erreur lors de l'envoi d'un message via un canal (HTTP non-2xx, timeout, etc.)."""


class ChannelAuthError(ChannelSendError):
    """Authentification invalide (401/403) — sous-classe de ChannelSendError."""


class ChannelNotFoundError(Exception):
    """Canal demandé pas enregistré dans le bus ou registry."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InboundMessage:
    """Message reçu d'un utilisateur via un canal."""

    channel_name: str
    sender_id: str
    chat_id: str
    text: str
    timestamp: datetime
    thread_id: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundMessage:
    """Message à envoyer via un canal."""

    channel_name: str
    recipient_id: str
    text: str
    reply_to_message_id: str | None = None


# ---------------------------------------------------------------------------
# ABC Channel
# ---------------------------------------------------------------------------


class Channel(ABC):
    """Interface commune pour tous les canaux de messagerie.

    Un canal gère : l'envoi de messages (`send`), le cycle de vie (`start`/`stop`).
    La réception se fait via `parse_webhook` (classmethod, sans state) car la
    plupart des canaux modernes utilisent des webhooks HTTP (Telegram, WhatsApp).
    """

    #: Nom unique du canal (`"telegram"`, `"whatsapp"`, ...).
    name: str

    @abstractmethod
    async def start(self) -> None:
        """Démarre les ressources du canal (ex: httpx.AsyncClient)."""

    @abstractmethod
    async def stop(self) -> None:
        """Arrête les ressources du canal (ferme client HTTP, etc.)."""

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """Envoie un message. Lève ChannelSendError en cas d'échec."""


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------


class MessageBus:
    """Bus asyncio : fan-out inbound vers handlers, routing outbound vers canaux.

    Usage :
        bus = MessageBus()
        bus.register_channel(telegram_channel)
        bus.register_handler(on_inbound_message)
        await bus.start_all()
        ...
        await bus.publish_outbound(OutboundMessage(channel_name="telegram", ...))
        ...
        await bus.stop_all()
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._handlers: list[Callable[[InboundMessage], Awaitable[None]]] = []

    def register_channel(self, channel: Channel) -> None:
        """Enregistre un canal. Remplace silencieusement si `name` déjà présent."""
        self._channels[channel.name] = channel

    def register_handler(
        self, handler: Callable[[InboundMessage], Awaitable[None]]
    ) -> None:
        """Enregistre un handler appelé pour chaque InboundMessage."""
        self._handlers.append(handler)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Dispatch vers tous les handlers (fan-out).

        Un handler qui lève une exception ne bloque pas les autres (R2).
        """
        results = await asyncio.gather(
            *(handler(msg) for handler in self._handlers),
            return_exceptions=True,
        )
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Handler #%d a levé une exception : %s", idx, result
                )

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Route vers le canal dont `name == msg.channel_name`. R1.

        Raises:
            ChannelNotFoundError: si le canal n'est pas enregistré.
        """
        channel = self._channels.get(msg.channel_name)
        if channel is None:
            raise ChannelNotFoundError(
                f"Canal '{msg.channel_name}' non enregistré. "
                f"Canaux dispo : {sorted(self._channels)}"
            )
        await channel.send(msg)

    async def start_all(self) -> None:
        """Démarre tous les canaux enregistrés (en parallèle)."""
        await asyncio.gather(*(c.start() for c in self._channels.values()))

    async def stop_all(self) -> None:
        """Arrête tous les canaux enregistrés (en parallèle)."""
        await asyncio.gather(
            *(c.stop() for c in self._channels.values()),
            return_exceptions=True,
        )

    @property
    def channel_names(self) -> list[str]:
        """Noms des canaux enregistrés."""
        return sorted(self._channels)
