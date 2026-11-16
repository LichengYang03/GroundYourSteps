"""Canonical Euler nodes and projection onto the minimum-width simplex."""

from __future__ import annotations

import numpy as np

MIN_WIDTH = 1e-3


def validate_nodes(nodes: np.ndarray, budget: int | None = None) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if not np.all(np.isfinite(nodes)):
        raise ValueError("schedule nodes must be finite")
    if nodes.ndim != 1 or nodes.size < 3:
        raise ValueError("a schedule needs at least two steps")
    if budget is not None and nodes.size != budget + 1:
        raise ValueError("node count disagrees with budget")
    if nodes[0] != 0 or nodes[-1] != 1 or np.any(np.diff(nodes) < MIN_WIDTH - 1e-12):
        raise ValueError("nodes must run from 0 to 1 with widths at least 1e-3")
    return nodes


def uniform_nodes(budget: int) -> np.ndarray:
    return validate_nodes(np.linspace(0, 1, budget + 1), budget)


def geometric_nodes(kappa: float, budget: int) -> np.ndarray:
    s = uniform_nodes(budget)
    if kappa == 0:
        return s
    out = np.expm1(kappa * s) / np.expm1(kappa)
    out[0], out[-1] = 0, 1
    return nodes_of_widths(np.diff(out))


def project_widths(widths: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {sum h = 1, h >= 1e-3}."""
    widths = np.asarray(widths, dtype=np.float64)
    if widths.ndim != 1 or not np.all(np.isfinite(widths)):
        raise ValueError("widths must be a finite vector")
    budget = len(widths)
    available = 1 - budget * MIN_WIDTH
    if budget < 2 or available <= 0:
        raise ValueError("budget does not fit the minimum width")
    shifted = widths - MIN_WIDTH
    ranked = np.sort(shifted)[::-1]
    partial = np.cumsum(ranked)
    positive = ranked - (partial - available) / np.arange(1, budget + 1)
    rho = np.flatnonzero(positive > 0)[-1] + 1
    threshold = (partial[rho - 1] - available) / rho
    projected = np.maximum(shifted - threshold, 0) + MIN_WIDTH
    projected /= projected.sum()
    return projected


def nodes_of_widths(widths: np.ndarray) -> np.ndarray:
    widths = project_widths(widths)
    return validate_nodes(np.r_[0.0, np.cumsum(widths)[:-1], 1.0], len(widths))


def snap_nodes(nodes: np.ndarray) -> np.ndarray:
    """Round interior nodes to the 1e-3 lattice with positive step widths."""
    nodes = validate_nodes(nodes)
    ticks = np.rint(nodes * 1000).astype(int)
    ticks[0], ticks[-1] = 0, 1000
    for i in range(1, len(ticks) - 1):
        ticks[i] = max(ticks[i], ticks[i - 1] + 1)
    for i in range(len(ticks) - 2, 0, -1):
        ticks[i] = min(ticks[i], ticks[i + 1] - 1)
    return validate_nodes(ticks / 1000, len(ticks) - 1)


def step_widths(nodes: np.ndarray) -> np.ndarray:
    return np.diff(validate_nodes(nodes))
