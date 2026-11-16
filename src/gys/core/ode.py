"""Explicit left-endpoint Euler integration on the noise-to-action axis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["network_evaluations", "rollout_batch"]


def network_evaluations(nodes: Sequence[float] | np.ndarray) -> int:
    """Network evaluations one rollout along ``nodes`` costs."""
    return len(nodes) - 1


def rollout_batch(
    policy: Any,
    context: dict[str, Any],
    x0: Any,
    nodes: Sequence[float] | np.ndarray,
) -> Any:
    """Integrate dx/dt = v(x, t) by left-endpoint Euler.

    x0 and the returned tensor have shape (N, H, D). The caller must disable gradients.
    """
    import torch

    if torch.is_grad_enabled():
        raise RuntimeError("rollout_batch requires torch.no_grad()")
    grid = np.asarray(nodes, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("nodes must be a 1-D grid with at least two entries")

    # Preserve bucket boundaries for policies with discrete time embeddings.
    time_dtype = torch.float64 if getattr(policy, "quantises_time", False) else x0.dtype

    x = x0
    for index in range(grid.size - 1):
        t = torch.as_tensor(float(grid[index]), dtype=time_dtype, device=x.device)
        h = float(grid[index + 1] - grid[index])
        x = x + h * policy.velocity_batch(x, t, context)
    return x
