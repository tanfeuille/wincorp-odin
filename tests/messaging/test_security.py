"""Tests security — R5 safe_download_path."""
from __future__ import annotations

from pathlib import Path

import pytest

from wincorp_odin.messaging.security import safe_download_path


class TestSafeDownloadPath:
    def test_valid_filename(self, tmp_path: Path) -> None:
        """EC5 : filename simple valide → path dans base_dir."""
        result = safe_download_path("report.pdf", tmp_path)
        assert result == (tmp_path / "report.pdf").resolve()

    def test_empty_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalide"):
            safe_download_path("", tmp_path)

    def test_dot_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalide"):
            safe_download_path(".", tmp_path)

    def test_double_dot_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalide"):
            safe_download_path("..", tmp_path)

    def test_path_traversal_explicit(self, tmp_path: Path) -> None:
        """EC4 : ../etc/passwd → ValueError (char interdit)."""
        with pytest.raises(ValueError, match="caractères interdits"):
            safe_download_path("../etc/passwd", tmp_path)

    def test_slash_in_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="caractères interdits"):
            safe_download_path("sub/file.txt", tmp_path)

    def test_backslash_in_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="caractères interdits"):
            safe_download_path("sub\\file.txt", tmp_path)

    def test_special_chars_rejected(self, tmp_path: Path) -> None:
        for bad in ["file;rm.txt", "file|cmd.txt", "file>cmd.txt", "file<cmd.txt"]:
            with pytest.raises(ValueError):
                safe_download_path(bad, tmp_path)

    def test_allowed_chars(self, tmp_path: Path) -> None:
        """Chars allowed : A-Z a-z 0-9 _ - ."""
        for good in ["a.txt", "A-B_C.log", "file.name.ext", "123-456.bin"]:
            result = safe_download_path(good, tmp_path)
            assert result.name == good
