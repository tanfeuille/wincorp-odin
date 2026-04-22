"""Tests rate_limit — R4 TokenBucket."""
from __future__ import annotations

import asyncio
import time

import pytest

from wincorp_odin.messaging.rate_limit import TokenBucket


class TestTokenBucket:
    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError, match="rate_per_second"):
            TokenBucket(rate_per_second=-1, capacity=1)

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(rate_per_second=1, capacity=0)

    @pytest.mark.asyncio
    async def test_acquire_single_no_wait(self) -> None:
        """Bucket plein → acquire ne bloque pas."""
        bucket = TokenBucket(rate_per_second=10, capacity=5)
        t0 = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"acquire ne doit pas bloquer (elapsed={elapsed})"

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self) -> None:
        """Bucket vide → acquire attend ~1/rate seconde."""
        bucket = TokenBucket(rate_per_second=10, capacity=1)
        await bucket.acquire()  # vide le bucket
        t0 = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - t0
        # 1 token à 10/s = ~100ms d'attente
        assert 0.05 < elapsed < 0.3, f"attendu ~100ms, elapsed={elapsed}"

    @pytest.mark.asyncio
    async def test_acquire_more_than_capacity_raises(self) -> None:
        bucket = TokenBucket(rate_per_second=1, capacity=2)
        with pytest.raises(ValueError, match="tokens demandés"):
            await bucket.acquire(tokens=3)

    @pytest.mark.asyncio
    async def test_zero_rate_blocks_indefinitely(self) -> None:
        """EC3 : rate=0 + bucket drained → bloque (testé avec timeout)."""
        bucket = TokenBucket(rate_per_second=0, capacity=1)
        await bucket.acquire()  # consomme le seul token
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bucket.acquire(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_available_tokens_refills(self) -> None:
        bucket = TokenBucket(rate_per_second=100, capacity=10)
        await bucket.acquire(tokens=5)  # 5 tokens restants
        await asyncio.sleep(0.05)  # +5 tokens produits théoriques
        # Cap à capacity, donc max 10. Après 50ms * 100/s = 5 tokens ajoutés.
        # 5 restants + 5 ajoutés = 10 (cap).
        assert bucket.available_tokens > 9
