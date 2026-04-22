"""Canal Telegram — Bot API (httpx async).

@spec specs/messaging.spec.md v1.0
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

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramChannel(Channel):
    """Telegram Bot API — send via httpx, webhook parsing en classmethod.

    Auth :
        - `bot_token` obtenu via @BotFather.
        - `allowed_user_ids` (optionnel) : filtre des senders autorisés (R11).

    Rate limit officiel : 30 msg/s global, 1 msg/s par chat. `rate_limit` à
    ajuster selon volumétrie.
    """

    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str,
        allowed_user_ids: set[int] | None = None,
        rate_limit: TokenBucket | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not bot_token:
            raise ValueError("bot_token requis")
        self._bot_token = bot_token
        self._allowed_user_ids = allowed_user_ids
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
        """POST sur /sendMessage. Lève ChannelSendError si non-2xx."""
        if self._rate_limit is not None:
            await self._rate_limit.acquire()

        client = self._ensure_client()
        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        body: dict[str, Any] = {
            "chat_id": message.recipient_id,
            "text": message.text,
        }
        if message.reply_to_message_id is not None:
            body["reply_parameters"] = {"message_id": int(message.reply_to_message_id)}

        logger.debug(
            "[Telegram] send to=%s len=%d", message.recipient_id, len(message.text)
        )

        try:
            response = await client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise ChannelSendError(
                f"Telegram HTTP error : {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise ChannelAuthError(
                f"Telegram auth error {response.status_code} : {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise ChannelSendError(
                f"Telegram send failed {response.status_code} : {response.text[:200]}"
            )

    @classmethod
    def parse_webhook(
        cls,
        payload: dict[str, Any],
        *,
        allowed_user_ids: set[int] | None = None,
    ) -> InboundMessage | None:
        """Parse un payload webhook Telegram en InboundMessage.

        Règles (R9, R11) :
        - Retourne `None` si payload ne contient pas `message.text`.
        - Retourne `None` si sender pas dans `allowed_user_ids` (si filtré).
        """
        msg = payload.get("message") or {}
        text = msg.get("text")
        if not isinstance(text, str) or not text:
            return None

        from_ = msg.get("from") or {}
        chat = msg.get("chat") or {}
        sender_id = from_.get("id")
        chat_id = chat.get("id")
        if sender_id is None or chat_id is None:
            return None

        if allowed_user_ids is not None and int(sender_id) not in allowed_user_ids:
            logger.debug(
                "[Telegram] sender %s filtré (pas dans allowed_user_ids)", sender_id
            )
            return None

        return InboundMessage(
            channel_name=cls.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            text=text,
            timestamp=datetime.now(UTC),
            raw_payload=payload,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError(
                "TelegramChannel pas démarré. Appeler start() avant send()."
            )
        return self._http_client
