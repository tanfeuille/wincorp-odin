"""Tests helper prive `_json_safe` : couverture exhaustive des conversions.

@spec specs/orchestration.spec.md v2.1.1 §3.8
"""
from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from enum import Enum, IntEnum
from pathlib import PurePosixPath

import pytest

from wincorp_odin.orchestration._json_safe import _json_safe


class _Color(Enum):
    RED = "red"


class _Level(IntEnum):
    LOW = 1


@dataclasses.dataclass
class _Pt:
    x: int
    y: int


def test_json_safe_none() -> None:
    """None passthrough."""
    assert _json_safe(None) is None


def test_json_safe_bool() -> None:
    """bool passthrough (precedent int)."""
    assert _json_safe(True) is True
    assert _json_safe(False) is False


def test_json_safe_int() -> None:
    """int passthrough."""
    assert _json_safe(42) == 42


def test_json_safe_float_finite() -> None:
    """float fini passthrough."""
    assert _json_safe(3.14) == 3.14


def test_json_safe_float_nan_rejects() -> None:
    """float NaN -> ValueError FR avec path."""
    with pytest.raises(ValueError, match="non-finie"):
        _json_safe(float("nan"))


def test_json_safe_float_inf_rejects() -> None:
    """float +inf -> ValueError FR."""
    with pytest.raises(ValueError, match="non-finie"):
        _json_safe(float("inf"))


def test_json_safe_float_ninf_rejects() -> None:
    """float -inf -> ValueError FR."""
    with pytest.raises(ValueError, match="non-finie"):
        _json_safe(float("-inf"))


def test_json_safe_str() -> None:
    """str passthrough."""
    assert _json_safe("hello") == "hello"


def test_json_safe_datetime() -> None:
    """datetime tz-aware -> isoformat."""
    dt = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
    assert _json_safe(dt) == "2026-04-23T14:00:00+00:00"


def test_json_safe_bytes_to_base64() -> None:
    """bytes -> base64 str."""
    out = _json_safe(b"hello")
    assert out == "aGVsbG8="


def test_json_safe_str_enum() -> None:
    """Str Enum -> .value."""
    assert _json_safe(_Color.RED) == "red"


def test_json_safe_int_enum() -> None:
    """IntEnum -> .value (int)."""
    assert _json_safe(_Level.LOW) == 1


def test_json_safe_path() -> None:
    """Path -> fspath str."""
    p = PurePosixPath("/tmp/foo")
    assert _json_safe(p) == "/tmp/foo"


def test_json_safe_dataclass() -> None:
    """Dataclass instance -> dict recursif."""
    assert _json_safe(_Pt(1, 2)) == {"x": 1, "y": 2}


def test_json_safe_mapping_str_keys() -> None:
    """Mapping avec cles str -> dict recursif."""
    assert _json_safe({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_json_safe_mapping_non_str_key_rejects() -> None:
    """Mapping avec cle non-str -> TypeError FR avec path."""
    with pytest.raises(TypeError, match="cles str"):
        _json_safe({1: "x"})


def test_json_safe_tuple_to_list() -> None:
    """tuple -> list recursif."""
    assert _json_safe((1, "a", None)) == [1, "a", None]


def test_json_safe_list() -> None:
    """list -> list recursif."""
    assert _json_safe([1, [2, [3]]]) == [1, [2, [3]]]


def test_json_safe_set_to_list() -> None:
    """set -> list best-effort."""
    out = _json_safe({1, 2, 3})
    assert isinstance(out, list)
    assert sorted(out) == [1, 2, 3]


def test_json_safe_frozenset_to_list() -> None:
    """frozenset -> list best-effort."""
    out = _json_safe(frozenset({1, 2}))
    assert isinstance(out, list)
    assert sorted(out) == [1, 2]


def test_json_safe_unsupported_type_raises_with_path() -> None:
    """Objet non supporte -> TypeError FR avec chemin."""

    class _Unknown:
        pass

    with pytest.raises(TypeError, match=r"\$"):
        _json_safe(_Unknown())


def test_json_safe_nested_path_in_error() -> None:
    """Path JSONPath precis en cas d'erreur imbriquee."""

    class _Unknown:
        pass

    with pytest.raises(TypeError, match=r"\$\.a\[2\]"):
        _json_safe({"a": [1, 2, _Unknown()]})


def test_json_safe_nested_float_nan_in_dict() -> None:
    """ValueError float NaN avec path precis dans dict."""
    with pytest.raises(ValueError, match=r"\$\.metric"):
        _json_safe({"metric": float("nan")})


def test_json_safe_dataclass_with_enum_field() -> None:
    """Dataclass contenant enum -> enum converti via passe recursive."""

    @dataclasses.dataclass
    class Inner:
        kind: _Color

    assert _json_safe(Inner(kind=_Color.RED)) == {"kind": "red"}
