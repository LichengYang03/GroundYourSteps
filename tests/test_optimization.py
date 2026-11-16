"""Tests for fractional differencing, distribution matching, and schedule search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gys.config import MethodConfig, SolverConfig, load_config
from gys.core import HorizonMap, project_widths, validate_nodes
from gys.objective import clouds, energy_distance, median_pairwise, pairwise_norm
from gys.search import solve
from gys.search.pso import _root


def test_joint_energy_distance_matches_v_statistic() -> None:
    x = np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3) / 13
    y = np.array([[0.2, 0.1, -0.1], [0.3, -0.2, 0.4]])
    cross, self_ = clouds(x, y)
    expected_cross = (x[:, :4] - y[:, None]).reshape(8, 3)
    expected_self = (x[:, 4:] - x[:, [5, 6, 7, 4]]).reshape(8, 3)
    np.testing.assert_array_equal(cross, expected_cross)
    np.testing.assert_array_equal(self_, expected_self)

    def brute(a, b):
        return np.mean([np.linalg.norm(u - v) for u in a for v in b])

    expected = 2 * brute(cross, self_) - brute(cross, cross) - brute(self_, self_)
    assert np.isclose(energy_distance(cross, self_), expected)


def test_horizon_map_and_simplex() -> None:
    chunk = np.arange(32 * 14, dtype=float).reshape(32, 14)
    keep = [i for i in range(14) if i not in (6, 13)]
    horizon = HorizonMap(terms=10, columns=np.array(keep), horizon=32)
    assert horizon.apply(chunk, 0).shape == (23, 12)
    assert np.array_equal(horizon.apply(chunk, 0), chunk[9:, keep])
    widths = project_widths(np.array([-2.0, 0.0, 0.0, 3.0]))
    assert np.isclose(widths.sum(), 1)
    assert np.all(widths >= 1e-3 - 1e-12)
    validate_nodes(np.r_[0.0, np.cumsum(widths)[:-1], 1.0], 4)


def test_exact_median_uses_all_off_diagonal_pairs(tmp_path: Path) -> None:
    rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    expected = np.median(
        [np.linalg.norm(rows[i] - rows[j]) for i in range(4) for j in range(i + 1, 4)]
    )
    assert median_pairwise(rows, tmp_path) == expected
    assert list(tmp_path.iterdir()) == []


def test_pso_delivers_measured_argmin_in_96_evaluations() -> None:
    class FakeObjective:
        def terminal(self, nodes):
            return np.asarray(nodes)

        def radius_gap(self, terminal, order):
            return order - 0.75

        def calibrate_lambda(self, terminals, order):
            assert len(terminals) == 24
            assert np.isclose(order, 0.75)
            return 1.0

        def score(self, terminal, order):
            return float(np.square(terminal[1:-1] - np.array([0.2, 0.55, 0.8])).sum())

    class Bank:
        def __init__(self):
            self.values = {}

        def get(self, nodes):
            return self.values.get(tuple(nodes))

        def put(self, nodes, terminal):
            self.values[tuple(nodes)] = terminal

    result = solve(FakeObjective(), Bank(), budget=4, method=MethodConfig(), solver=SolverConfig())
    assert result.evaluations == len(result.measured) == 96
    assert result.loss == min(point.loss for point in result.measured)
    assert result.nodes == min(result.measured, key=lambda point: point.loss).nodes


def test_shipped_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("robotwin_pi05", "robotwin_groot"):
        path = root / f"{name}.yaml"
        config = load_config(path, task="lift_pot", budget=4)
        assert config.solver.evaluation_budget() == 96


def test_radius_calibration_rules():
    class Gap:
        def __init__(self, values):
            self.values = values

        def radius_gap(self, terminal, order):
            return self.values[int(order)]

    grid = np.arange(4, dtype=float)
    assert _root(Gap([2, 0, -1, 0]), None, grid) == 1
    assert _root(Gap([2, -2, 2, -2]), None, grid) == 0.5
    with pytest.raises(ValueError, match="no radius-gap root"):
        _root(Gap([1, 2, 3, 4]), None, grid)


def test_pairwise_self_distances_have_zero_diagonal():
    rows = np.random.default_rng(0).normal(size=(12, 276))
    distances = pairwise_norm(rows, rows)
    assert np.array_equal(np.diag(distances), np.zeros(12))
    expected = np.linalg.norm(rows[:, None] - rows[None, :], axis=-1)
    assert np.allclose(distances, expected, atol=1e-12)


def test_delivered_schedule_round_trip_preserves_narrow_steps():
    from gys.core import nodes_of_widths
    from gys.deploy import parse_wire, schedule_document

    nodes = nodes_of_widths(np.array([-1.0, 0.3, -1.0, 0.7]))
    for axis in ("canonical", "flipped"):
        document = schedule_document(nodes, knot_axis=axis, fingerprint="test", provenance={})
        decoded = parse_wire(document["wire"], knot_axis=axis)
        assert np.allclose(decoded, nodes, rtol=0, atol=1e-15)
        assert document["steps"] == 4
