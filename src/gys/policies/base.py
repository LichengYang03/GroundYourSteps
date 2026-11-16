"""Frozen-policy interface for schedule optimization.

Adapters receive RGB images and expose velocity on canonical time, where 0 is
noise and 1 is action. Contexts contain normalized demonstrations, fixed noise
seeds, and masked backbone summaries. Rollouts use batched action-expert calls.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["FlowPolicy", "check_state", "default_noise"]


@runtime_checkable
class FlowPolicy(Protocol):
    """A frozen flow-matching policy exposed as a canonical velocity field."""

    device: str
    horizon: int
    latent_dim: int
    supports_time_grad: bool
    quantises_time: bool

    def freeze(self) -> None:
        """Disable grads on every parameter and put the model in eval mode."""

    def prepare_context(self, sample: Any) -> dict[str, Any]:
        """Encode one observation once; reused by every step of every rollout."""

    def stack_contexts(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge ``N`` single-sample contexts into one batched context."""

    def velocity_batch(self, x: Any, t: Any, context: dict[str, Any]) -> Any:
        """Canonical ``v(x, t | o)`` for ``x`` of shape ``(N, H, D)``."""

    def sample_noise(self, seed: int) -> np.ndarray:
        """Deterministic ``Z ~ N(0, I)`` of shape ``(H, D)``."""

    def metadata(self) -> dict[str, Any]:
        """Provenance recorded verbatim into every artifact."""


def default_noise(seed: int, horizon: int, latent_dim: int) -> np.ndarray:
    """Shared deterministic draw, so every adapter seeds identically."""
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((horizon, latent_dim), dtype=np.float32)


def check_state(x: Any, horizon: int, latent_dim: int, *, name: str = "x") -> None:
    if tuple(x.shape[-2:]) != (horizon, latent_dim):
        raise ValueError(f"{name} must end in ({horizon}, {latent_dim}), got {tuple(x.shape)}")
