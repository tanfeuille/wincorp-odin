"""Tests TelegramChannel — send mocké + parse_webhook."""
from __future__ import annotations

import httpx
import pytest

from wincorp_odin.messaging.base import (
    ChannelAuthError,
    ChannelSendError,
    OutboundMessage,
)
from wincorp_odin.messaging.channels.telegram import TelegramChannel


def _mock_transport(handler: callable) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestTelegramSend:
    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.read()
            return httpx.Response(200, json={"ok": True})

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tg = TelegramChannel(bot_token="TOKEN123", http_client=client)
        await tg.start()

        await tg.send(OutboundMessage(
            channel_name="telegram",
            recipient_id="42",
            text="bonjour",
        ))

        assert "/botTOKEN123/sendMessage" in captured["url"]
        assert b"bonjour" in captured["body"]
        assert b'"chat_id":"42"' in captured["body"]

        await tg.stop()

    @pytest.mark.asyncio
    async def test_send_with_reply(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tg = TelegramChannel(bot_token="T", http_client=client)
        await tg.start()
        await tg.send(OutboundMessage(
            channel_name="telegram",
            recipient_id="42",
            text="hi",
            reply_to_message_id="99",
        ))
        assert b'"reply_parameters"' in captured["body"]
        assert b'"message_id":99' in captured["body"]

    @pytest.mark.asyncio
    async def test_send_auth_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized")

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tg = TelegramChannel(bot_token="BAD", http_client=client)
        await tg.start()
        with pytest.raises(ChannelAuthError, match="401"):
            await tg.send(OutboundMessage(
                channel_name="telegram", recipient_id="42", text="x"
            ))

    @pytest.mark.asyncio
    async def test_send_generic_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tg = TelegramChannel(bot_token="T", http_client=client)
        await tg.start()
        with pytest.raises(ChannelSendError, match="500"):
            await tg.send(OutboundMessage(
                channel_name="telegram", recipient_id="42", text="x"
            ))

    @pytest.mark.asyncio
    async def test_send_http_error_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns fail")

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        tg = TelegramChannel(bot_token="T", http_client=client)
        await tg.start()
        with pytest.raises(ChannelSendError, match="HTTP error"):
            await tg.send(OutboundMessage(
                channel_name="telegram", recipient_id="42", text="x"
            ))

    def test_requires_bot_token(self) -> None:
        with pytest.raises(ValueError, match="bot_token"):
            TelegramChannel(bot_token="")

    @pytest.mark.asyncio
    async def test_send_without_start_raises(self) -> None:
        tg = TelegramChannel(bot_token="T")
        with pytest.raises(RuntimeError, match="démarré"):
            await tg.send(OutboundMessage(
                channel_name="telegram", recipient_id="42", text="x"
            ))

    @pytest.mark.asyncio
    async def test_start_creates_own_client(self) -> None:
        """start() sans http_client fourni → crée un httpx.AsyncClient interne."""
        tg = TelegramChannel(bot_token="T")
        await tg.start()
        # Accès interne minimal pour valider
        assert tg._http_client is not None
        await tg.stop()
        assert tg._http_client is None

    @pytest.mark.asyncio
    async def test_send_with_rate_limit(self) -> None:
        """rate_limit appelé avant send."""
        from wincorp_odin.messaging.rate_limit import TokenBucket

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        bucket = TokenBucket(rate_per_second=100, capacity=10)
        tg = TelegramChannel(bot_token="T", http_client=client, rate_limit=bucket)
        await tg.start()
        await tg.send(OutboundMessage(
            channel_name="telegram", recipient_id="1", text="x"
        ))
        # 1 token consommé
        assert bucket.available_tokens < 10


class TestTelegramParseWebhook:
    def test_message_basic(self) -> None:
        payload = {
            "update_id": 1,
            "message": {
                "from": {"id": 42, "username": "tan"},
                "chat": {"id": 42},
                "text": "/status",
            },
        }
        msg = TelegramChannel.parse_webhook(payload)
        assert msg is not None
        assert msg.channel_name == "telegram"
        assert msg.sender_id == "42"
        assert msg.chat_id == "42"
        assert msg.text == "/status"

    def test_empty_payload(self) -> None:
        """EC10 : payload {} → None."""
        assert TelegramChannel.parse_webhook({}) is None

    def test_callback_query_returns_none(self) -> None:
        """EC11 : callback_query pas message → None."""
        assert TelegramChannel.parse_webhook({"callback_query": {"data": "x"}}) is None

    def test_empty_text_returns_none(self) -> None:
        payload = {"message": {"from": {"id": 1}, "chat": {"id": 1}, "text": ""}}
        assert TelegramChannel.parse_webhook(payload) is None

    def test_missing_from(self) -> None:
        payload = {"message": {"chat": {"id": 1}, "text": "hi"}}
        assert TelegramChannel.parse_webhook(payload) is None

    def test_allowed_user_filter(self) -> None:
        """EC13 / R11 : sender pas dans allowed_user_ids → None."""
        payload = {
            "message": {
                "from": {"id": 99},
                "chat": {"id": 99},
                "text": "/status",
            },
        }
        msg = TelegramChannel.parse_webhook(payload, allowed_user_ids={42})
        assert msg is None

    def test_allowed_user_passes(self) -> None:
        payload = {
            "message": {
                "from": {"id": 42},
                "chat": {"id": 42},
                "text": "/status",
            },
        }
        msg = TelegramChannel.parse_webhook(payload, allowed_user_ids={42})
        assert msg is not None
        assert msg.sender_id == "42"
