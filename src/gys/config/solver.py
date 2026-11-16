"""Particle swarm settings and execution batch sizes."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseConfig, integer


@dataclass(frozen=True)
class SolverConfig(BaseConfig):
    particles: int = 24
    iterations: int = 3
    resolution: float = 1e-3
    micro_batch: int = 32
    context_cache: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("particles", "iterations", "micro_batch", "context_cache", "seed"):
            object.__setattr__(self, name, integer(getattr(self, name), name))
        object.__setattr__(self, "resolution", float(self.resolution))
        if self.particles < 5 or self.iterations < 0:
            raise ValueError("PSO needs at least five seeds and nonnegative iterations")
        if self.resolution != 1e-3:
            raise ValueError("resolution must be 1e-3")
        if self.micro_batch < 1 or self.context_cache < 1:
            raise ValueError("batch size and context cache must be positive")

    def evaluation_budget(self) -> int:
        return self.particles * (self.iterations + 1)
