"""Tests WhatsAppChannel — send mocké + parse_webhook."""
from __future__ import annotations

import httpx
import pytest

from wincorp_odin.messaging.base import (
    ChannelAuthError,
    ChannelSendError,
    OutboundMessage,
)
from wincorp_odin.messaging.channels.whatsapp import WhatsAppChannel


def _mock_transport(handler: callable) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestWhatsAppSend:
    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.read()
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"messages": [{"id": "wamid.xxx"}]})

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        wa = WhatsAppChannel(
            phone_number_id="PHONE_ID",
            access_token="TOK",
            http_client=client,
        )
        await wa.start()
        await wa.send(OutboundMessage(
            channel_name="whatsapp",
            recipient_id="+33671210925",
            text="test",
        ))

        assert "PHONE_ID/messages" in captured["url"]
        assert captured["auth"] == "Bearer TOK"
        assert b'"messaging_product":"whatsapp"' in captured["body"]
        assert b'"to":"33671210925"' in captured["body"]  # + stripped
        assert b'"body":"test"' in captured["body"]

        await wa.stop()

    @pytest.mark.asyncio
    async def test_send_with_reply(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        wa = WhatsAppChannel(
            phone_number_id="P", access_token="T", http_client=client
        )
        await wa.start()
        await wa.send(OutboundMessage(
            channel_name="whatsapp",
            recipient_id="336",
            text="hi",
            reply_to_message_id="wamid.yyy",
        ))
        assert b'"context"' in captured["body"]
        assert b'"message_id":"wamid.yyy"' in captured["body"]

    @pytest.mark.asyncio
    async def test_send_auth_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden")

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        wa = WhatsAppChannel(
            phone_number_id="P", access_token="BAD", http_client=client
        )
        await wa.start()
        with pytest.raises(ChannelAuthError, match="403"):
            await wa.send(OutboundMessage(
                channel_name="whatsapp", recipient_id="1", text="x"
            ))

    @pytest.mark.asyncio
    async def test_send_generic_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        wa = WhatsAppChannel(
            phone_number_id="P", access_token="T", http_client=client
        )
        await wa.start()
        with pytest.raises(ChannelSendError):
            await wa.send(OutboundMessage(
                channel_name="whatsapp", recipient_id="1", text="x"
            ))

    @pytest.mark.asyncio
    async def test_send_http_error_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns fail")

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        wa = WhatsAppChannel(
            phone_number_id="P", access_token="T", http_client=client
        )
        await wa.start()
        with pytest.raises(ChannelSendError, match="HTTP error"):
            await wa.send(OutboundMessage(
                channel_name="whatsapp", recipient_id="1", text="x"
            ))

    def test_requires_phone_number_id(self) -> None:
        with pytest.raises(ValueError, match="phone_number_id"):
            WhatsAppChannel(phone_number_id="", access_token="t")

    def test_requires_access_token(self) -> None:
        with pytest.raises(ValueError, match="access_token"):
            WhatsAppChannel(phone_number_id="p", access_token="")

    @pytest.mark.asyncio
    async def test_send_without_start_raises(self) -> None:
        wa = WhatsAppChannel(phone_number_id="P", access_token="T")
        with pytest.raises(RuntimeError, match="démarré"):
            await wa.send(OutboundMessage(
                channel_name="whatsapp", recipient_id="1", text="x"
            ))

    @pytest.mark.asyncio
    async def test_start_creates_own_client(self) -> None:
        wa = WhatsAppChannel(phone_number_id="P", access_token="T")
        await wa.start()
        assert wa._http_client is not None
        await wa.stop()
        assert wa._http_client is None

    @pytest.mark.asyncio
    async def test_send_with_rate_limit(self) -> None:
        from wincorp_odin.messaging.rate_limit import TokenBucket

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=_mock_transport(handler))
        bucket = TokenBucket(rate_per_second=100, capacity=10)
        wa = WhatsAppChannel(
            phone_number_id="P",
            access_token="T",
            http_client=client,
            rate_limit=bucket,
        )
        await wa.start()
        await wa.send(OutboundMessage(
            channel_name="whatsapp", recipient_id="1", text="x"
        ))
        assert bucket.available_tokens < 10


class TestWhatsAppParseWebhook:
    def test_message_text(self) -> None:
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "33671210925",
                            "id": "wamid.abc",
                            "timestamp": "1729000000",
                            "type": "text",
                            "text": {"body": "bonjour"},
                        }],
                    },
                }],
            }],
        }
        msg = WhatsAppChannel.parse_webhook(payload)
        assert msg is not None
        assert msg.channel_name == "whatsapp"
        assert msg.sender_id == "33671210925"
        assert msg.text == "bonjour"

    def test_empty_payload(self) -> None:
        assert WhatsAppChannel.parse_webhook({}) is None

    def test_statuses_not_messages(self) -> None:
        """EC12 : payload delivery status → None."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "statuses": [{"status": "delivered"}],
                    },
                }],
            }],
        }
        assert WhatsAppChannel.parse_webhook(payload) is None

    def test_non_text_type_ignored(self) -> None:
        """Type image/audio → None (scope text only)."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "336",
                            "type": "image",
                            "image": {"id": "mid"},
                        }],
                    },
                }],
            }],
        }
        assert WhatsAppChannel.parse_webhook(payload) is None

    def test_allowed_phone_filter_blocks(self) -> None:
        """R11 : sender pas dans allowlist → None."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "33000000000",
                            "type": "text",
                            "text": {"body": "hi"},
                        }],
                    },
                }],
            }],
        }
        msg = WhatsAppChannel.parse_webhook(
            payload,
            allowed_phone_numbers={"33671210925"},
        )
        assert msg is None

    def test_allowed_phone_filter_with_plus(self) -> None:
        """Allowlist avec `+` : match aussi si webhook sans `+`."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "33671210925",
                            "type": "text",
                            "text": {"body": "hi"},
                        }],
                    },
                }],
            }],
        }
        msg = WhatsAppChannel.parse_webhook(
            payload,
            allowed_phone_numbers={"+33671210925"},
        )
        assert msg is not None

    def test_invalid_timestamp_fallback_now(self) -> None:
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "336",
                            "type": "text",
                            "timestamp": "not-a-number",
                            "text": {"body": "hi"},
                        }],
                    },
                }],
            }],
        }
        msg = WhatsAppChannel.parse_webhook(payload)
        assert msg is not None

    def test_missing_timestamp_fallback_now(self) -> None:
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "336",
                            "type": "text",
                            "text": {"body": "hi"},
                        }],
                    },
                }],
            }],
        }
        msg = WhatsAppChannel.parse_webhook(payload)
        assert msg is not None

    def test_missing_body(self) -> None:
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "336",
                            "type": "text",
                            "text": {},
                        }],
                    },
                }],
            }],
        }
        assert WhatsAppChannel.parse_webhook(payload) is None

    def test_malformed_entry(self) -> None:
        """Payload mal formé → None (pas crash)."""
        assert WhatsAppChannel.parse_webhook({"entry": "not-a-list"}) is None
        assert WhatsAppChannel.parse_webhook({"entry": []}) is None
