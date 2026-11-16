"""Dataset records, calibration samples, and normalized demonstration pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["CalibRecord", "CalibSample", "Episode", "Pair"]


@dataclass(frozen=True)
class Episode:
    """One demonstration episode of the calibration source."""

    index: int
    length: int
    prompt: str


@dataclass(frozen=True)
class CalibRecord:
    """Addressing information for one calibration chunk.  Pure metadata."""

    sample_id: str
    episode_index: int
    chunk_index: int
    start_frame: int
    valid_length: int
    noise_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_index": self.episode_index,
            "chunk_index": self.chunk_index,
            "start_frame": self.start_frame,
            "valid_length": self.valid_length,
            "noise_seed": self.noise_seed,
        }


@dataclass(frozen=True)
class CalibSample:
    """A materialised calibration sample: one observation and its demonstrated chunk."""

    record: CalibRecord
    observation: dict[str, Any]
    gt_action_raw: np.ndarray


@dataclass(frozen=True)
class Pair:
    """One demonstration in the policy integration space and its observation summary."""

    sample_id: str
    episode_index: int
    y_model: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def noise_seed(self) -> int | None:
        seed = self.extras.get("noise_seed")
        return None if seed is None or int(seed) < 0 else int(seed)
