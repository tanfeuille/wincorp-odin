"""Canal WhatsApp — Meta Cloud API (Graph API v21.0, httpx async).

@spec specs/messaging.spec.md v1.0

Ce canal utilise l'API officielle Meta WhatsApp Cloud. Requiert :
- Un Meta Business account avec WhatsApp Business Platform configuré.
- Un `phone_number_id` (disponible dans la console Meta Developers).
- Un `access_token` Bearer (temporaire dev ou permanent production).

JAMAIS de lib non-officielle (whatsapp-web.js, whapi, etc.) — risque ban permanent.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from wincorp_odin.messaging.base import (
    Channel,
    ChannelAuthError,
    ChannelSendError,
    InboundMessage,
    OutboundMessage,
)
from wincorp_odin.messaging.rate_limit import TokenBucket

logger = logging.getLogger(__name__)

WHATSAPP_API_BASE = "https://graph.facebook.com"
WHATSAPP_API_VERSION = "v21.0"


class WhatsAppChannel(Channel):
    """Meta WhatsApp Cloud API.

    Rate limit Meta Tier 1 : 1 000 conversations/jour (limite free).
    Rate limit send : ~80 msg/s recommandé, jusqu'à 250 en Tier 2+.
    """

    name = "whatsapp"

    def __init__(
        self,
        *,
        phone_number_id: str,
        access_token: str,
        allowed_phone_numbers: set[str] | None = None,
        rate_limit: TokenBucket | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not phone_number_id:
            raise ValueError("phone_number_id requis")
        if not access_token:
            raise ValueError("access_token requis")
        self._phone_number_id = phone_number_id
        self._access_token = access_token
        self._allowed_phone_numbers = allowed_phone_numbers
        self._rate_limit = rate_limit
        self._http_client = http_client
        self._owns_client = http_client is None

    async def start(self) -> None:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

    async def stop(self) -> None:
        if self._http_client is not None and self._owns_client:
            await self._http_client.aclose()
            self._http_client = None

    async def send(self, message: OutboundMessage) -> None:
        """POST sur graph.facebook.com/{version}/{phone_id}/messages."""
        if self._rate_limit is not None:
            await self._rate_limit.acquire()

        client = self._ensure_client()
        url = (
            f"{WHATSAPP_API_BASE}/{WHATSAPP_API_VERSION}"
            f"/{self._phone_number_id}/messages"
        )
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": message.recipient_id.lstrip("+"),
            "type": "text",
            "text": {"body": message.text},
        }
        if message.reply_to_message_id is not None:
            body["context"] = {"message_id": message.reply_to_message_id}

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        logger.debug(
            "[WhatsApp] send to=%s len=%d", message.recipient_id, len(message.text)
        )

        try:
            response = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ChannelSendError(f"WhatsApp HTTP error : {exc}") from exc

        if response.status_code in (401, 403):
            raise ChannelAuthError(
                f"WhatsApp auth error {response.status_code} : {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise ChannelSendError(
                f"WhatsApp send failed {response.status_code} : "
                f"{response.text[:200]}"
            )

    @classmethod
    def parse_webhook(
        cls,
        payload: dict[str, Any],
        *,
        allowed_phone_numbers: set[str] | None = None,
    ) -> InboundMessage | None:
        """Parse un webhook WhatsApp Cloud API en InboundMessage.

        Structure attendue :
            {"entry": [{"changes": [{"value": {"messages": [{
                "from": "33671210925",
                "text": {"body": "..."},
                "timestamp": "1729...",
                "id": "wamid...",
            }]}}]}]}

        Règles (R10, R11) :
        - Retourne `None` si pas de `messages` (ex: payload `statuses` delivery).
        - Retourne `None` si message pas de type `text`.
        - Retourne `None` si sender pas dans `allowed_phone_numbers` (si filtré).
        """
        try:
            entry = payload["entry"][0]
            value = entry["changes"][0]["value"]
            messages = value.get("messages")
        except (KeyError, IndexError, TypeError):
            return None

        if not messages:
            return None

        msg = messages[0]
        if msg.get("type") != "text":
            return None

        text = (msg.get("text") or {}).get("body")
        sender_phone = msg.get("from")
        if not isinstance(text, str) or not text or not sender_phone:
            return None

        if (
            allowed_phone_numbers is not None
            and sender_phone not in allowed_phone_numbers
            and f"+{sender_phone}" not in allowed_phone_numbers
        ):
            logger.debug(
                "[WhatsApp] sender %s filtré (pas dans allowed_phone_numbers)",
                sender_phone,
            )
            return None

        timestamp_str = msg.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromtimestamp(int(timestamp_str), tz=UTC)
            except (ValueError, TypeError):
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        return InboundMessage(
            channel_name=cls.name,
            sender_id=sender_phone,
            chat_id=sender_phone,  # WhatsApp 1-1 : chat == sender
            text=text,
            timestamp=timestamp,
            raw_payload=payload,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError(
                "WhatsAppChannel pas démarré. Appeler start() avant send()."
            )
        return self._http_client
