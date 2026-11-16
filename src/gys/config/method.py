"""Fractional-difference calibration and noise sampling settings."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import BaseConfig, integer


@dataclass(frozen=True)
class MethodConfig(BaseConfig):
    """Ten-tap horizon map and a fixed bank of eight noise draws per pair."""

    band_low: float = -1.0
    band_high: float = 2.0
    band_points: int = 121
    gripper_columns: tuple[int, int] = (6, 13)
    n_latents: int = 8
    noise_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_low", float(self.band_low))
        object.__setattr__(self, "band_high", float(self.band_high))
        object.__setattr__(self, "band_points", integer(self.band_points, "band_points"))
        object.__setattr__(self, "n_latents", integer(self.n_latents, "n_latents"))
        object.__setattr__(self, "noise_offset", integer(self.noise_offset, "noise_offset"))
        object.__setattr__(
            self,
            "gripper_columns",
            tuple(integer(x, "gripper_columns") for x in self.gripper_columns),
        )
        if not math.isfinite(self.band_low) or not math.isfinite(self.band_high):
            raise ValueError("order bounds must be finite")
        if self.band_low < -1 or self.band_high <= self.band_low or self.band_points < 2:
            raise ValueError("invalid horizon-map order interval")
        if self.n_latents < 4 or self.n_latents % 2:
            raise ValueError("n_latents must be even and at least four")
        if len(self.gripper_columns) != 2 or len(set(self.gripper_columns)) != 2:
            raise ValueError("gripper_columns must identify the two grippers")

    @property
    def band_grid(self):
        import numpy as np

        return np.linspace(self.band_low, self.band_high, self.band_points)
