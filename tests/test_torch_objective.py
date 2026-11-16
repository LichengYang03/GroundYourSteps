"""A small Euler-to-score smoke test without model weights or a GPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gys.config import MethodConfig  # noqa: E402
from gys.core import uniform_nodes  # noqa: E402
from gys.data.records import Pair  # noqa: E402
from gys.objective import build_objective  # noqa: E402


class FakePolicy:
    horizon = 12
    latent_dim = 14
    dtype = torch.float32
    device = "cpu"

    def stack_contexts(self, contexts):
        return {}

    def velocity_batch(self, x, time, context):
        return torch.ones_like(x) * (time + 0.2)


class Contexts:
    def get(self, sample_id):
        return {}


class Latents:
    n_latents = 8

    def get(self, pair, index, context):
        return np.full((12, 14), 0.01 * (index + 1) + 0.02 * pair.episode_index)


def test_euler_to_joint_energy_distance(tmp_path):
    pairs = [
        Pair(str(i), i, np.zeros((12, 14)), {"obs_vector": np.array([float(i), float(i + 1)])})
        for i in range(2)
    ]
    objective = build_objective(
        FakePolicy(),
        pairs,
        Contexts(),
        Latents(),
        MethodConfig(),
        budget=4,
        micro_batch=4,
        score_horizon=12,
        score_slice=(0, 14),
        scratch=tmp_path,
    )
    terminal = objective.terminal(uniform_nodes(4))
    assert terminal.shape == (2, 8, 12, 12)
    assert np.isfinite(objective.radius_gap(terminal, 0.5))
    weight = objective.calibrate_lambda([terminal], 0.5)
    assert weight > 0
    assert np.isfinite(objective.score(terminal, 0.5))
