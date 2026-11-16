"""Empirical joint energy distance and the two seed-round calibrations."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MethodConfig
from ..core import HorizonMap, resolve_columns, rollout_batch, validate_nodes


def pairwise_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = a is b
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    squared = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    np.maximum(squared, 0, out=squared)
    if same:
        np.fill_diagonal(squared, 0)
    return np.sqrt(squared, out=squared)


def median_pairwise(rows: np.ndarray, scratch: Path) -> float:
    """Exact off-diagonal median, using disk storage to bound RAM."""
    rows = np.asarray(rows, np.float64)
    n = len(rows)
    if n < 2:
        raise ValueError("the median heuristic needs at least two observations")
    count = n * (n - 1) // 2
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="gys-median-", suffix=".dat", dir=scratch)
    os.close(fd)
    try:
        distances = np.memmap(name, dtype=np.float64, mode="w+", shape=(count,))
        cursor = 0
        for start in range(0, n, 256):
            end = min(start + 256, n)
            for other in range(start, n, 256):
                stop = min(other + 256, n)
                block = pairwise_norm(rows[start:end], rows[other:stop])
                values = block[np.triu_indices(end - start, 1)] if other == start else block.ravel()
                distances[cursor : cursor + len(values)] = values
                cursor += len(values)
        assert cursor == count
        middle = count // 2
        if count % 2:
            distances.partition(middle)
            result = float(distances[middle])
        else:
            distances.partition((middle - 1, middle))
            result = float((distances[middle - 1] + distances[middle]) / 2)
        del distances
        return result
    finally:
        Path(name).unlink(missing_ok=True)


def clouds(phi_x: np.ndarray, phi_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = phi_x.shape[1]
    half = n // 2
    cross = phi_x[:, :half] - phi_y[:, None]
    second = phi_x[:, half:]
    self_ = second - np.roll(second, -1, axis=1)
    return cross.reshape(-1, cross.shape[-1]), self_.reshape(-1, self_.shape[-1])


def energy_distance(cross: np.ndarray, self_: np.ndarray) -> float:
    """Empirical energy distance with all diagonal terms."""
    return float(
        2 * pairwise_norm(cross, self_).mean()
        - pairwise_norm(cross, cross).mean()
        - pairwise_norm(self_, self_).mean()
    )


class Objective:
    def __init__(
        self,
        policy: Any,
        pairs: Sequence[Any],
        contexts: Any,
        latents: Any,
        method: MethodConfig,
        budget: int,
        horizon: HorizonMap,
        micro_batch: int,
        scratch: Path,
    ) -> None:
        if not pairs:
            raise ValueError("no calibration pairs; run gys pairs first")
        self.policy, self.pairs = policy, list(pairs)
        self.contexts, self.latents = contexts, latents
        self.method, self.budget, self.horizon = method, budget, horizon
        self.micro_batch, self.scratch = micro_batch, scratch
        self.y = horizon.trim(np.stack([p.y_model for p in pairs]))
        obs = np.stack([np.asarray(p.extras["obs_vector"], np.float64) for p in pairs])
        spread = obs.std(0)
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(self.y)):
            raise ValueError("calibration arrays must be finite")
        self.obs = (obs - obs.mean(0)) / np.where(spread > 0, spread, 1)
        self.lambda_value: float | None = None
        self.lambda_record: dict[str, float] | None = None

    def terminal(self, nodes: np.ndarray) -> np.ndarray:
        import torch

        grid = validate_nodes(nodes, self.budget)
        m, n = len(self.pairs), self.method.n_latents
        out = np.empty((m, n, self.policy.horizon, self.policy.latent_dim), np.float64)
        work = [(i, j) for j in range(n) for i in range(m)]
        with torch.no_grad():
            for start in range(0, len(work), self.micro_batch):
                group = work[start : start + self.micro_batch]
                context_list = [self.contexts.get(self.pairs[i].sample_id) for i, _ in group]
                stacked = self.policy.stack_contexts(context_list)
                noise = np.stack(
                    [
                        self.latents.get(self.pairs[i], j, context_list[k])
                        for k, (i, j) in enumerate(group)
                    ]
                )
                x0 = torch.as_tensor(noise, dtype=self.policy.dtype, device=self.policy.device)
                end = rollout_batch(self.policy, stacked, x0, grid)
                out[[i for i, _ in group], [j for _, j in group]] = (
                    end.detach().to("cpu").to(torch.float64).numpy()
                )
        return self.horizon.trim(out)

    def _clouds(self, terminal: np.ndarray, order: float) -> tuple[np.ndarray, np.ndarray]:
        x = self.horizon.filter(terminal, order).reshape(len(self.pairs), self.method.n_latents, -1)
        y = self.horizon.filter(self.y, order).reshape(len(self.pairs), -1)
        return clouds(x, y)

    def radius_gap(self, terminal: np.ndarray, order: float) -> float:
        cross, self_ = self._clouds(terminal, order)
        return float(pairwise_norm(cross, cross).mean() - pairwise_norm(self_, self_).mean())

    def calibrate_lambda(self, terminals: Sequence[np.ndarray], order: float) -> float:
        rows = []
        for terminal in terminals:
            cross, self_ = self._clouds(terminal, order)
            rows.extend((cross, self_))
        feature_median = median_pairwise(np.concatenate(rows), self.scratch)
        obs_median = median_pairwise(self.obs, self.scratch)
        if feature_median <= 0 or obs_median <= 0:
            raise ValueError("median heuristic is undefined for coincident samples")
        self.lambda_value = feature_median / obs_median
        self.lambda_record = {
            "feature_median": feature_median,
            "observation_median": obs_median,
            "lambda": self.lambda_value,
        }
        return self.lambda_value

    def score(self, terminal: np.ndarray, order: float) -> float:
        if self.lambda_value is None:
            raise RuntimeError("calibrate lambda before scoring")
        cross, self_ = self._clouds(terminal, order)
        repeated = np.repeat(self.obs * self.lambda_value, self.method.n_latents // 2, axis=0)
        return energy_distance(np.hstack((cross, repeated)), np.hstack((self_, repeated)))


def build_objective(
    policy: Any,
    pairs: Sequence[Any],
    contexts: Any,
    latents: Any,
    method: MethodConfig,
    *,
    budget: int,
    micro_batch: int,
    score_horizon: int,
    score_slice: tuple[int, int] | None,
    scratch: Path,
) -> Objective:
    columns = resolve_columns(score_slice, method.gripper_columns, policy.latent_dim)
    horizon = HorizonMap(terms=10, columns=columns, horizon=score_horizon)
    return Objective(
        policy, pairs, contexts, latents, method, budget, horizon, micro_batch, scratch
    )
