"""Tests base — MessageBus + dataclasses + exceptions."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wincorp_odin.messaging.base import (
    Channel,
    ChannelAuthError,
    ChannelNotFoundError,
    ChannelSendError,
    InboundMessage,
    MessageBus,
    OutboundMessage,
)


class _FakeChannel(Channel):
    """Channel mock pour tests. Enregistre les sends."""

    def __init__(self, name: str, raise_on_send: Exception | None = None) -> None:
        self.name = name
        self.started = False
        self.stopped = False
        self.sent: list[OutboundMessage] = []
        self._raise = raise_on_send

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, message: OutboundMessage) -> None:
        if self._raise is not None:
            raise self._raise
        self.sent.append(message)


class TestInboundMessage:
    def test_frozen(self) -> None:
        msg = InboundMessage(
            channel_name="telegram",
            sender_id="42",
            chat_id="42",
            text="hi",
            timestamp=datetime.now(UTC),
        )
        with pytest.raises((AttributeError, Exception)):
            msg.text = "mutated"  # type: ignore[misc]

    def test_default_raw_payload(self) -> None:
        msg = InboundMessage(
            channel_name="x", sender_id="1", chat_id="1",
            text="hi", timestamp=datetime.now(UTC),
        )
        assert msg.raw_payload == {}


class TestOutboundMessage:
    def test_basic(self) -> None:
        msg = OutboundMessage(channel_name="telegram", recipient_id="42", text="hi")
        assert msg.reply_to_message_id is None


class TestExceptions:
    def test_auth_is_send_subclass(self) -> None:
        """ChannelAuthError hérite de ChannelSendError."""
        assert issubclass(ChannelAuthError, ChannelSendError)


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_register_and_route_outbound(self) -> None:
        bus = MessageBus()
        ch = _FakeChannel(name="telegram")
        bus.register_channel(ch)
        await bus.publish_outbound(OutboundMessage(
            channel_name="telegram", recipient_id="42", text="hi"
        ))
        assert len(ch.sent) == 1
        assert ch.sent[0].text == "hi"

    @pytest.mark.asyncio
    async def test_publish_outbound_unknown_channel_raises(self) -> None:
        """EC1 / R1 : canal inconnu → ChannelNotFoundError."""
        bus = MessageBus()
        with pytest.raises(ChannelNotFoundError, match="non enregistré"):
            await bus.publish_outbound(OutboundMessage(
                channel_name="slack", recipient_id="x", text="hi"
            ))

    @pytest.mark.asyncio
    async def test_publish_inbound_fan_out(self) -> None:
        bus = MessageBus()
        calls: list[str] = []

        async def h1(msg: InboundMessage) -> None:
            calls.append(f"h1:{msg.text}")

        async def h2(msg: InboundMessage) -> None:
            calls.append(f"h2:{msg.text}")

        bus.register_handler(h1)
        bus.register_handler(h2)
        msg = InboundMessage(
            channel_name="telegram", sender_id="1", chat_id="1",
            text="bonjour", timestamp=datetime.now(UTC),
        )
        await bus.publish_inbound(msg)
        assert set(calls) == {"h1:bonjour", "h2:bonjour"}

    @pytest.mark.asyncio
    async def test_publish_inbound_exception_isolates(self) -> None:
        """R2 / EC2 : un handler qui raise n'empêche pas les autres."""
        bus = MessageBus()
        calls: list[str] = []

        async def broken(msg: InboundMessage) -> None:
            raise RuntimeError("boom")

        async def ok(msg: InboundMessage) -> None:
            calls.append("ok")

        bus.register_handler(broken)
        bus.register_handler(ok)
        msg = InboundMessage(
            channel_name="telegram", sender_id="1", chat_id="1",
            text="t", timestamp=datetime.now(UTC),
        )
        # Ne doit pas raise
        await bus.publish_inbound(msg)
        assert calls == ["ok"]

    @pytest.mark.asyncio
    async def test_start_stop_all(self) -> None:
        bus = MessageBus()
        tg = _FakeChannel(name="telegram")
        wa = _FakeChannel(name="whatsapp")
        bus.register_channel(tg)
        bus.register_channel(wa)
        await bus.start_all()
        assert tg.started and wa.started
        await bus.stop_all()
        assert tg.stopped and wa.stopped

    def test_channel_names(self) -> None:
        bus = MessageBus()
        bus.register_channel(_FakeChannel(name="whatsapp"))
        bus.register_channel(_FakeChannel(name="telegram"))
        assert bus.channel_names == ["telegram", "whatsapp"]

    @pytest.mark.asyncio
    async def test_stop_all_handles_exceptions(self) -> None:
        """stop_all utilise return_exceptions=True pour ne pas crasher."""
        bus = MessageBus()

        class _Broken(_FakeChannel):
            async def stop(self) -> None:
                raise RuntimeError("stop failed")

        bus.register_channel(_Broken(name="x"))
        # Ne doit pas raise
        await bus.stop_all()
