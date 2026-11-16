"""Cached observation contexts and fixed per-sample Gaussian draws."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..data.records import CalibSample, Pair

__all__ = ["LATENT_STRIDE", "ContextSource", "LatentBank"]

LATENT_STRIDE = 1_000_003


class ContextSource:
    """Encoded observations keyed by ``sample_id``, FIFO-evicted."""

    def __init__(self, policy: Any, samples: Sequence[CalibSample], cache_size: int = 200) -> None:
        self.policy = policy
        self.samples = {sample.record.sample_id: sample for sample in samples}
        self.cache_size = int(cache_size)
        self._cache: dict[str, dict[str, Any]] = {}
        self.misses = 0

    def get(self, sample_id: str) -> dict[str, Any]:
        cached = self._cache.get(sample_id)
        if cached is not None:
            return cached
        sample = self.samples.get(sample_id)
        if sample is None:
            raise KeyError(
                f"no calibration sample for pair {sample_id!r}: the pair set and the "
                "dataset disagree. Rebuild the pairs"
            )
        context = self.policy.prepare_context(sample)
        self.misses += 1
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[sample_id] = context
        return context

    def stats(self) -> dict[str, int]:
        return {
            "cache_size": self.cache_size,
            "cached": len(self._cache),
            "misses": self.misses,
            "samples": len(self.samples),
        }


class LatentBank:
    """Deterministic per-(chunk, latent) noise, cached for one solve.

    Seeds are derived, never stored::

        seed(chunk, i) = base_seed(chunk) + noise_offset + i * LATENT_STRIDE
    """

    def __init__(self, policy: Any, *, noise_offset: int, n_latents: int) -> None:
        if n_latents < 1:
            raise ValueError("n_latents must be >= 1")
        self.policy = policy
        self.noise_offset = int(noise_offset)
        self.n_latents = int(n_latents)
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    @staticmethod
    def base_seed(pair: Pair, context: dict[str, Any] | None = None) -> int:
        """The chunk's own seed, or a stable fallback derived from its id."""
        base = context.get("noise_seed") if context is not None else None
        if base is None:
            base = pair.noise_seed
        if base is None:
            base = int.from_bytes(
                hashlib.sha256(pair.sample_id.encode("utf-8")).digest()[:8], "big"
            )
        return int(base)

    def seed_for(self, pair: Pair, index: int, context: dict[str, Any] | None = None) -> int:
        if not 0 <= index < self.n_latents:
            raise IndexError(f"latent index {index} out of range [0, {self.n_latents})")
        return (self.base_seed(pair, context) + self.noise_offset + index * LATENT_STRIDE) % (2**63)

    def get(self, pair: Pair, index: int, context: dict[str, Any] | None = None) -> np.ndarray:
        key = (pair.sample_id, int(index))
        cached = self._cache.get(key)
        if cached is None:
            cached = self.policy.sample_noise(self.seed_for(pair, index, context))
            self._cache[key] = cached
        return cached

    def clear(self) -> None:
        self._cache.clear()
