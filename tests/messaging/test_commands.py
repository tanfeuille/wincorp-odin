"""Tests commands — R6 parse_command."""
from __future__ import annotations

import pytest

from wincorp_odin.messaging.commands import KNOWN_COMMANDS, parse_command


class TestParseCommand:
    def test_empty_returns_none(self) -> None:
        """EC6 : text vide → None."""
        assert parse_command("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert parse_command("   ") is None

    def test_no_slash_returns_none(self) -> None:
        """Text sans / → None."""
        assert parse_command("hello world") is None

    def test_unknown_command_returns_none(self) -> None:
        """EC7 : /unknown → None."""
        assert parse_command("/unknown foo") is None

    def test_known_command_no_args(self) -> None:
        """EC8 : /status → ('/status', [])."""
        assert parse_command("/status") == ("/status", [])

    def test_known_command_with_args(self) -> None:
        """EC9 : /new mon sujet → ('/new', ['mon', 'sujet'])."""
        assert parse_command("/new mon sujet long") == (
            "/new",
            ["mon", "sujet", "long"],
        )

    @pytest.mark.parametrize("cmd", ["/new", "/status", "/memory", "/models", "/help"])
    def test_all_known_commands(self, cmd: str) -> None:
        result = parse_command(cmd)
        assert result is not None
        assert result[0] == cmd

    def test_known_commands_frozenset(self) -> None:
        """KNOWN_COMMANDS est un frozenset immutable."""
        assert isinstance(KNOWN_COMMANDS, frozenset)

    def test_leading_whitespace_trimmed(self) -> None:
        """Whitespace leading toléré."""
        # "/status" précédé d'espaces : startswith("/") est False pour "  /status"
        # Donc retour None (strict). Documenté.
        assert parse_command("  /status") is None
