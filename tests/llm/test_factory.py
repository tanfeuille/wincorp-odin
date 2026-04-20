"""Tests factory : create_model, cache, thinking, erreurs runtime, mtime.

@spec specs/llm-factory.spec.md v1.2
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# R1 — Cache et instanciation basique
# ---------------------------------------------------------------------------


def test_r1_create_model_known_name_returns_chat_anthropic(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R1/R5 : create_model("sonnet") instancie via resolution dynamique."""
    from wincorp_odin.llm import create_model

    instance = create_model("sonnet")
    assert instance is not None
    assert mock_chat_anthropic.call_count == 1


def test_r1_create_model_passes_model_id_from_yaml(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R1 : l'argument model_id vient du YAML."""
    from wincorp_odin.llm import create_model

    create_model("sonnet")
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"


def test_r1_create_model_passes_max_tokens_from_yaml(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R1 : max_tokens propage depuis YAML."""
    from wincorp_odin.llm import create_model

    create_model("sonnet")
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert kwargs["max_tokens"] == 8192


def test_r1_create_model_same_key_returns_same_instance(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R1/R7 : deux appels identiques -> MEME instance (identite is)."""
    from wincorp_odin.llm import create_model

    a = create_model("sonnet")
    b = create_model("sonnet")
    assert a is b
    assert mock_chat_anthropic.call_count == 1


def test_r1_create_model_different_names_return_different_instances(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R1 : deux noms distincts -> deux instances distinctes."""
    from wincorp_odin.llm import create_model

    a = create_model("sonnet")
    b = create_model("opus")
    assert a is not b
    assert mock_chat_anthropic.call_count == 2


# ---------------------------------------------------------------------------
# R2 — Thinking enabled variantes
# ---------------------------------------------------------------------------


def test_r2_thinking_enabled_true_applies_when_thinking_enabled_kwargs(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R2/R6 : thinking_enabled=True merge when_thinking_enabled dans kwargs."""
    from wincorp_odin.llm import create_model

    create_model("sonnet", thinking_enabled=True)
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert "thinking" in kwargs
    assert kwargs["thinking"]["type"] == "enabled"
    assert kwargs["thinking"]["budget_tokens"] == 8192


def test_r2_thinking_enabled_false_no_thinking_kwarg(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R2 : thinking_enabled=False n'ajoute pas la cle thinking."""
    from wincorp_odin.llm import create_model

    create_model("sonnet", thinking_enabled=False)
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert "thinking" not in kwargs


def test_r2_cache_distingue_thinking_variants(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R2/R7 : cache keye par (name, thinking_enabled)."""
    from wincorp_odin.llm import create_model

    a = create_model("sonnet", thinking_enabled=False)
    b = create_model("sonnet", thinking_enabled=True)
    assert a is not b
    assert mock_chat_anthropic.call_count == 2


# ---------------------------------------------------------------------------
# R4 — Interpolation ${VAR}
# ---------------------------------------------------------------------------


def test_r4_api_key_interpolated_from_env(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R4 : la cle API est interpolee depuis l'env var."""
    from wincorp_odin.llm import create_model

    create_model("sonnet")
    kwargs = mock_chat_anthropic.call_args.kwargs
    assert kwargs["api_key"] == mock_anthropic_api_key


# ---------------------------------------------------------------------------
# Erreurs runtime
# ---------------------------------------------------------------------------


def test_ec13_unknown_name_raises_model_not_found_error(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC13 : nom inconnu leve ModelNotFoundError avec liste des noms dispo."""
    from wincorp_odin.llm import ModelNotFoundError, create_model

    with pytest.raises(ModelNotFoundError) as excinfo:
        create_model("inconnu")
    msg = str(excinfo.value)
    assert "inconnu" in msg
    assert "sonnet" in msg  # liste des modeles dispo


def test_ec14_thinking_on_non_capable_raises_capability_mismatch(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """EC14/R6 : thinking_enabled=True sur modele non-capable -> CapabilityMismatchError."""
    from wincorp_odin.llm import CapabilityMismatchError, create_model

    with pytest.raises(CapabilityMismatchError) as excinfo:
        create_model("haiku", thinking_enabled=True)
    msg = str(excinfo.value)
    assert "haiku" in msg
    # Liste modeles thinking-compatibles
    assert "sonnet" in msg or "opus" in msg


# ---------------------------------------------------------------------------
# PB-011 — heritage retire
# ---------------------------------------------------------------------------


def test_ec13_model_not_found_not_a_keyerror(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-011 : ModelNotFoundError ne derive PAS de KeyError."""
    from wincorp_odin.llm import ModelNotFoundError, create_model

    try:
        create_model("inconnu")
    except ModelNotFoundError as exc:
        assert not isinstance(exc, KeyError)


def test_ec14_capability_mismatch_not_a_valueerror(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-011 : CapabilityMismatchError ne derive PAS de ValueError."""
    from wincorp_odin.llm import CapabilityMismatchError, create_model

    try:
        create_model("haiku", thinking_enabled=True)
    except CapabilityMismatchError as exc:
        assert not isinstance(exc, ValueError)


def test_pb011_helper_is_model_not_found(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-011 : helper is_model_not_found retourne True sur le bon type."""
    from wincorp_odin.llm import create_model, is_model_not_found

    try:
        create_model("inconnu")
    except Exception as exc:
        assert is_model_not_found(exc) is True
    assert is_model_not_found(RuntimeError("autre")) is False


def test_pb011_helper_is_capability_mismatch(
    mock_anthropic_api_key: str,
    patched_yaml_path_full: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """PB-011 : helper is_capability_mismatch retourne True sur le bon type."""
    from wincorp_odin.llm import create_model, is_capability_mismatch

    try:
        create_model("haiku", thinking_enabled=True)
    except Exception as exc:
        assert is_capability_mismatch(exc) is True
    assert is_capability_mismatch(KeyError("autre")) is False


# ---------------------------------------------------------------------------
# R7 — Thread safety, double-checked locking
# ---------------------------------------------------------------------------


def test_r7_concurrent_create_model_instantiates_once(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R7/EC18 : plusieurs threads en course -> une seule instanciation."""
    from wincorp_odin.llm import create_model

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: list[Any] = [None] * n_threads

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = create_model("sonnet")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Tous les resultats sont la meme instance
    first = results[0]
    for r in results:
        assert r is first
    # Une seule vraie instanciation cote mock
    assert mock_chat_anthropic.call_count == 1


# ---------------------------------------------------------------------------
# R11 — disabled:true exclu
# ---------------------------------------------------------------------------


def test_r11_disabled_model_raises_not_found(
    mock_anthropic_api_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R11/EC21 : disabled:true -> ModelNotFoundError comme si absent."""
    from wincorp_odin.llm import ModelNotFoundError, create_model

    urd = tmp_path / "wincorp-urd" / "referentiels"
    urd.mkdir(parents=True)
    (urd / "models.yaml").write_text(
        """config_version: 1
models:
  - name: "ghost"
    display_name: "Disabled"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
    disabled: true
  - name: "live"
    display_name: "Live"
    use: "langchain_anthropic:ChatAnthropic"
    model: "claude-test"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1024
    supports_thinking: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WINCORP_URD_PATH", str(tmp_path / "wincorp-urd"))

    with pytest.raises(ModelNotFoundError):
        create_model("ghost")
    # Le modele live fonctionne
    assert create_model("live") is not None


# ---------------------------------------------------------------------------
# R10 — Repr secret redacte
# ---------------------------------------------------------------------------


def test_r10_api_key_never_in_repr(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R10 : ModelConfig.__repr__ masque api_key_resolved."""
    from wincorp_odin.llm import load_models_config

    configs = load_models_config()
    cfg = configs["sonnet"]
    r = repr(cfg)
    assert "***REDACTED***" in r
    assert mock_anthropic_api_key not in r


# ---------------------------------------------------------------------------
# R18 — mtime throttled (PB-010 + PB-020)
# ---------------------------------------------------------------------------


def test_r18_mtime_check_throttled(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18/PB-010 : max 1 stat()/s meme sous burst create_model.

    PB-020 : monkeypatch time.monotonic pour tester sans sleep reel.
    """
    from wincorp_odin.llm import create_model, factory

    # Premiere init pour charger le cache
    create_model("sonnet")

    # Compteur d'appels stat
    stat_calls = {"n": 0}
    orig_stat = Path.stat

    def counting_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        stat_calls["n"] += 1
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counting_stat)

    # Simulation temps monotone "fige" sous le throttle
    fake_time = {"t": factory._last_mtime_check + 0.05}

    def fake_monotonic() -> float:
        return fake_time["t"]

    monkeypatch.setattr("wincorp_odin.llm.factory.time.monotonic", fake_monotonic)

    # 10 appels successifs sous throttle
    for _ in range(10):
        create_model("sonnet")
    assert stat_calls["n"] == 0, "stat() doit etre skipp sous throttle 1/s"

    # Avancer au-dela de 1s -> stat reappele au moins une fois
    fake_time["t"] += 1.5
    create_model("sonnet")
    assert stat_calls["n"] >= 1


# ---------------------------------------------------------------------------
# R19b — copy-on-write : reload runtime ne stall pas les lecteurs
# ---------------------------------------------------------------------------


def test_r19b_runtime_reload_failure_keeps_previous_cache(
    mock_anthropic_api_key: str,
    patched_yaml_path: Path,
    mock_chat_anthropic: MagicMock,
) -> None:
    """R19b/EC26 : reload runtime qui echoue -> cache precedent conserve."""
    from wincorp_odin.llm import create_model

    # Init cache
    a = create_model("sonnet")

    # Corruption du YAML -> reload va echouer
    yaml_file = patched_yaml_path / "referentiels" / "models.yaml"
    time.sleep(0.01)
    yaml_file.write_text("@@@ invalid yaml @@@", encoding="utf-8")

    # Forcer le check mtime (bypass throttle en remettant a zero)
    from wincorp_odin.llm import factory
    factory._last_mtime_check = 0.0

    # L'appel ne doit pas lever, cache precedent reutilise
    b = create_model("sonnet")
    assert a is b
