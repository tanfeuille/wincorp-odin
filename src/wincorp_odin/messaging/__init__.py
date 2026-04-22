"""Messaging multi-canal — MessageBus asyncio + canaux Telegram + WhatsApp.

@spec specs/messaging.spec.md v1.0

Phase 6 DeerFlow (partielle, + extension WhatsApp).
"""
from wincorp_odin.messaging.base import (
    Channel,
    ChannelAuthError,
    ChannelNotFoundError,
    ChannelSendError,
    InboundMessage,
    MessageBus,
    OutboundMessage,
)
from wincorp_odin.messaging.channels.telegram import TelegramChannel
from wincorp_odin.messaging.channels.whatsapp import WhatsAppChannel
from wincorp_odin.messaging.commands import KNOWN_COMMANDS, parse_command
from wincorp_odin.messaging.rate_limit import TokenBucket
from wincorp_odin.messaging.registry import CHANNEL_REGISTRY, load_channel
from wincorp_odin.messaging.security import safe_download_path

__all__ = [
    "CHANNEL_REGISTRY",
    "Channel",
    "ChannelAuthError",
    "ChannelNotFoundError",
    "ChannelSendError",
    "InboundMessage",
    "KNOWN_COMMANDS",
    "MessageBus",
    "OutboundMessage",
    "TelegramChannel",
    "TokenBucket",
    "WhatsAppChannel",
    "load_channel",
    "parse_command",
    "safe_download_path",
]
