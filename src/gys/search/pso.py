"""Particle swarm optimization of Euler step widths with a ring neighborhood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import MethodConfig, SolverConfig
from ..core import geometric_nodes, nodes_of_widths, project_widths, snap_nodes, uniform_nodes

CHI = 0.7298
C1 = C2 = 2.05


@dataclass(frozen=True)
class MeasuredPoint:
    nodes: tuple[float, ...]
    loss: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes), "loss": self.loss, "source": self.source}


@dataclass(frozen=True)
class Solution:
    nodes: tuple[float, ...]
    loss: float
    order: float
    observation_weight: float
    evaluations: int
    measured: tuple[MeasuredPoint, ...]
    seed_roots: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "loss": self.loss,
            "order": self.order,
            "observation_weight": self.observation_weight,
            "evaluations": self.evaluations,
            "seed_roots": list(self.seed_roots),
            "measured": [point.to_dict() for point in self.measured],
        }


def _root(objective: Any, terminal: np.ndarray, grid: np.ndarray) -> float:
    values = np.array([objective.radius_gap(terminal, float(order)) for order in grid])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite horizon-map radius gap")
    exact = np.flatnonzero(values == 0)
    if exact.size:
        return float(grid[exact[0]])
    crossings = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if not crossings.size:
        raise ValueError("seed schedule has no radius-gap root in the configured interval")
    i = int(crossings[0])
    return float(grid[i] - values[i] * (grid[i + 1] - grid[i]) / (values[i + 1] - values[i]))


def solve(
    objective: Any,
    bank: Any,
    *,
    budget: int,
    method: MethodConfig,
    solver: SolverConfig,
    logger: Any | None = None,
) -> Solution:
    rng = np.random.default_rng(solver.seed)
    seeds = [(uniform_nodes(budget), "uniform")]
    seeds += [
        (geometric_nodes(kappa, budget), f"geometric_{kappa:g}") for kappa in (-2.0, -1.0, 1.0, 2.0)
    ]
    while len(seeds) < solver.particles:
        widths = project_widths(rng.dirichlet(np.ones(budget)))
        seeds.append((nodes_of_widths(widths), "dirichlet"))

    terminals: list[np.ndarray] = []
    nodes_history: list[np.ndarray] = []
    sources: list[str] = []

    def evaluate(nodes: np.ndarray, source: str) -> int:
        terminal = bank.get(nodes)
        if terminal is None:
            terminal = objective.terminal(nodes)
            bank.put(nodes, terminal)
        if not np.all(np.isfinite(terminal)):
            raise FloatingPointError("non-finite policy terminal state")
        if logger:
            logger.write(f"[evaluate {len(terminals) + 1}] {source}\n")
            logger.flush()
        nodes_history.append(nodes.copy())
        terminals.append(terminal)
        sources.append(source)
        return len(terminals) - 1

    for nodes, source in seeds:
        evaluate(nodes, source)
    roots = tuple(_root(objective, terminal, method.band_grid) for terminal in terminals)
    order = float(np.mean(roots))
    weight = objective.calibrate_lambda(terminals, order)

    def score(terminal):
        value = float(objective.score(terminal, order))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite objective score")
        return value

    scores = [score(terminal) for terminal in terminals]
    positions = [np.diff(nodes) for nodes, _ in seeds]
    velocities = [rng.uniform(-1.0, 1.0, budget) * (0.25 / budget) for _ in range(solver.particles)]
    visited = [[i] for i in range(solver.particles)]

    for iteration in range(solver.iterations):
        personal = [min(indices, key=lambda j: scores[j]) for indices in visited]
        for i in range(solver.particles):
            nearby = [
                personal[(i - 1) % solver.particles],
                personal[i],
                personal[(i + 1) % solver.particles],
            ]
            local = min(nearby, key=lambda j: scores[j])
            r1, r2 = rng.random(budget), rng.random(budget)
            velocity = CHI * (
                velocities[i]
                + C1 * r1 * (np.diff(nodes_history[personal[i]]) - positions[i])
                + C2 * r2 * (np.diff(nodes_history[local]) - positions[i])
            )
            nodes = snap_nodes(nodes_of_widths(positions[i] + velocity))
            index = evaluate(nodes, f"round_{iteration + 1}")
            scores.append(score(terminals[index]))
            positions[i], velocities[i] = np.diff(nodes), velocity
            visited[i].append(index)
        if logger:
            logger.write(f"[round {iteration + 1}] best={min(scores):.8g}\n")

    expected = solver.evaluation_budget()
    if len(scores) != expected:
        raise AssertionError(f"expected {expected} evaluations, got {len(scores)}")
    best = int(np.argmin(scores))
    points = tuple(
        MeasuredPoint(tuple(map(float, nodes)), float(loss), source)
        for nodes, loss, source in zip(nodes_history, scores, sources, strict=True)
    )
    if logger:
        logger.write(
            f"[argmin] loss={scores[best]:.8g}, order={order:.6g}, "
            f"lambda={weight:.6g}, nodes={nodes_history[best].tolist()}\n"
        )
    return Solution(
        points[best].nodes, points[best].loss, order, weight, len(scores), points, roots
    )
