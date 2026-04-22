"""Rate limiter asyncio — token bucket.

@spec specs/messaging.spec.md v1.0
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Token bucket asyncio-compatible pour rate limiting.

    Produit `rate_per_second` tokens par seconde, capé à `capacity`.
    `acquire()` bloque jusqu'à ce qu'un token soit disponible.

    Usage :
        bucket = TokenBucket(rate_per_second=25, capacity=25)
        await bucket.acquire()
        # envoyer le message
    """

    def __init__(self, rate_per_second: float, capacity: int) -> None:
        if rate_per_second < 0:
            raise ValueError("rate_per_second doit être >= 0")
        if capacity <= 0:
            raise ValueError("capacity doit être > 0")
        self._rate = rate_per_second
        self._capacity = capacity
        self._tokens: float = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Bloque jusqu'à `tokens` tokens disponibles (default 1).

        Refill calculé depuis `_last_refill`. Jamais plus de `capacity` en bucket.
        Si `rate == 0`, bloque indéfiniment (useful pour "circuit open").
        """
        if tokens > self._capacity:
            raise ValueError(
                f"tokens demandés ({tokens}) > capacity ({self._capacity})"
            )
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calcul du sleep nécessaire pour avoir `tokens` tokens.
                missing = tokens - self._tokens
                # Bucket mort (rate=0) → attendre durée arbitraire longue puis réessayer.
                sleep_for = 3600.0 if self._rate <= 0 else missing / self._rate
            await asyncio.sleep(sleep_for)

    def _refill(self) -> None:
        """Ajoute les tokens dus depuis le dernier refill. Cap à `capacity`."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0 or self._rate <= 0:
            self._last_refill = now
            return
        added = elapsed * self._rate
        self._tokens = min(self._tokens + added, float(self._capacity))
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Tokens actuellement disponibles (refresh d'abord). Lecture non-bloquante."""
        self._refill()
        return self._tokens
