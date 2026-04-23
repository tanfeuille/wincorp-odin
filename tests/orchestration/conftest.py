"""Fixtures partagees pour les tests orchestration.

@spec specs/orchestration.spec.md v2.1.1
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest


@pytest.fixture
def utc_now() -> datetime:
    """datetime UTC-aware fixe, pour tests deterministes."""
    return datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)


@pytest.fixture
def frozen_now(utc_now: datetime) -> Callable[[], datetime]:
    """Factory `_now_factory` qui retourne toujours `utc_now` (monotonic fige)."""

    def _factory() -> datetime:
        return utc_now

    return _factory


@pytest.fixture
def tick_now(utc_now: datetime) -> Callable[[], datetime]:
    """Factory incrementale : premier appel = utc_now, ensuite +10ms a chaque call.

    Utile pour verifier submitted_at vs started_at vs completed_at distincts.
    """
    counter = {"n": 0}

    def _factory() -> datetime:
        counter["n"] += 1
        return utc_now + timedelta(milliseconds=10 * counter["n"])

    return _factory


@pytest.fixture
def uuid_factory_seq() -> Callable[[], str]:
    """Factory UUID deterministe : 'uuid-0001', 'uuid-0002', ..."""
    counter = {"n": 0}

    def _factory() -> str:
        counter["n"] += 1
        return f"uuid-{counter['n']:04d}"

    return _factory


@pytest.fixture
def executor(
    frozen_now: Callable[[], datetime],
    uuid_factory_seq: Callable[[], str],
) -> Iterator[Any]:
    """SubagentExecutor par defaut pour tests basiques.

    Context manager pour garantir shutdown en fin de test.
    """
    from wincorp_odin.orchestration.executor import SubagentExecutor

    ex = SubagentExecutor(
        max_workers_scheduler=3,
        max_workers_exec=3,
        max_history=100,
        _now_factory=frozen_now,
        _uuid_factory=uuid_factory_seq,
    )
    try:
        yield ex
    finally:
        ex.shutdown(wait=True, cancel_futures=True, force_timeout_sec=0.5)


@pytest.fixture
def capture_sink() -> Any:
    """Sink qui capture on_start / on_end dans des listes.

    Attributs :
        started: list[SubagentResult]
        ended: list[SubagentResult]
    """

    class _CaptureSink:
        def __init__(self) -> None:
            self.started: list[Any] = []
            self.ended: list[Any] = []

        def on_start(self, result: Any) -> None:
            self.started.append(result)

        def on_end(self, result: Any) -> None:
            self.ended.append(result)

    return _CaptureSink()


@pytest.fixture
def bad_sink() -> Any:
    """Sink qui raise RuntimeError sur chaque call, pour tester R16."""

    class _BadSink:
        def __init__(self) -> None:
            self.start_called = 0
            self.end_called = 0

        def on_start(self, result: Any) -> None:
            self.start_called += 1
            raise RuntimeError(f"boom start #{self.start_called}")

        def on_end(self, result: Any) -> None:
            self.end_called += 1
            raise RuntimeError(f"boom end #{self.end_called}")

    return _BadSink()


@pytest.fixture
def make_task() -> Callable[..., Any]:
    """Factory qui construit une task simple configurable.

    Usage:
        task = make_task(return_value=42)
        task = make_task(raise_exc=RuntimeError("boom"))
        task = make_task(wait_sec=0.05)
        task = make_task(cooperative_wait=True)  # boucle cancel_event.wait
    """

    def _builder(
        *,
        return_value: Any = None,
        raise_exc: BaseException | None = None,
        wait_sec: float = 0.0,
        cooperative_wait: bool = False,
        cooperative_timeout: float = 2.0,
    ) -> Any:
        def task(state: Any, cancel_event: threading.Event) -> Any:
            if wait_sec > 0:
                cancel_event.wait(wait_sec)
            if cooperative_wait:
                # Boucle cooperative : sort des que event set.
                deadline = cooperative_timeout
                step = 0.01
                while deadline > 0:
                    if cancel_event.wait(step):
                        from wincorp_odin.orchestration.exceptions import (
                            SubagentCancelledException,
                        )

                        raise SubagentCancelledException()
                    deadline -= step
            if raise_exc is not None:
                raise raise_exc
            return return_value

        return task

    return _builder


@pytest.fixture
def barrier_two() -> threading.Barrier:
    """Barrier pour 2 threads (test + task), timeout court."""
    return threading.Barrier(2, timeout=2.0)


@pytest.fixture
def checkpoint() -> threading.Event:
    """Event sync thread de test / worker."""
    return threading.Event()
