"""Tests registry — R3 lazy import."""
from __future__ import annotations

import pytest

from wincorp_odin.messaging.registry import CHANNEL_REGISTRY, load_channel


class TestLoadChannel:
    def test_registry_contains_telegram_whatsapp(self) -> None:
        assert "telegram" in CHANNEL_REGISTRY
        assert "whatsapp" in CHANNEL_REGISTRY

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="inconnu"):
            load_channel("slack")

    def test_load_telegram_channel(self) -> None:
        channel = load_channel("telegram", bot_token="fake-token")
        assert channel.name == "telegram"

    def test_load_whatsapp_channel(self) -> None:
        channel = load_channel(
            "whatsapp",
            phone_number_id="1234",
            access_token="tok",
        )
        assert channel.name == "whatsapp"

    def test_invalid_registry_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Si une entry du registry est malformée → ValueError."""
        monkeypatch.setitem(CHANNEL_REGISTRY, "bad", "no-colon-here")
        with pytest.raises(ValueError, match="Format registry"):
            load_channel("bad")
