"""Calibration data: readers, records, and the demonstration targets."""

from __future__ import annotations

from typing import Any

from ..config import DataConfig
from .calibration import (
    assert_pairs_match,
    build_pair,
    build_pair_set,
    load_pairs,
    read_pair,
    read_pair_manifest,
    write_pair,
)
from .records import CalibRecord, CalibSample, Episode, Pair
from .selection import decode_image

__all__ = [
    "CalibRecord",
    "CalibSample",
    "Episode",
    "Pair",
    "assert_pairs_match",
    "build_dataset",
    "build_pair",
    "build_pair_set",
    "decode_image",
    "load_pairs",
    "read_pair",
    "read_pair_manifest",
    "write_pair",
]


def build_dataset(config: DataConfig) -> Any:
    """Open the configured calibration dataset."""
    if config.format == "robotwin_hdf5":
        from .robotwin import RoboTwinHDF5Dataset

        return RoboTwinHDF5Dataset(config)
    if config.format == "lerobot":
        from .lerobot import LeRobotDataset

        return LeRobotDataset(config)
    raise ValueError(f"no reader for data format {config.format!r}")
