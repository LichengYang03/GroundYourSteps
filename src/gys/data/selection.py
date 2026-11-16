"""Deterministic calibration selection and RGB image decoding."""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np

from ..config import DataConfig
from .records import CalibRecord, Episode

__all__ = ["build_records", "decode_image", "noise_seed_for", "pad_chunk", "select_episodes"]


def noise_seed_for(base: int, episode_index: int, chunk_index: int, stride: int) -> int:
    """Derive a chunk seed as base + episode_index * stride + chunk_index."""
    if chunk_index >= stride:
        raise ValueError(
            f"chunk_index {chunk_index} >= noise_episode_stride {stride}; seeds would collide"
        )
    return int(base) + int(episode_index) * int(stride) + int(chunk_index)


def select_episodes(config: DataConfig, episodes: dict[int, Episode]) -> list[int]:
    """Which episodes calibrate.  Stratified, so the set spans the dataset."""
    available = sorted(
        index for index, episode in episodes.items() if episode.length >= config.action_horizon
    )
    if not available:
        raise ValueError(
            f"no episode is at least action_horizon={config.action_horizon} frames long"
        )
    if config.episode_ids is not None:
        missing = sorted(set(config.episode_ids) - set(available))
        if missing:
            raise ValueError(f"requested episodes are unavailable or too short: {missing}")
        return sorted(config.episode_ids)
    count = min(config.episode_count, len(available))
    rng = np.random.default_rng(config.selection_seed)
    chosen = [int(rng.choice(block)) for block in np.array_split(np.asarray(available), count)]
    return sorted(dict.fromkeys(chosen))


def build_records(
    config: DataConfig, episodes: dict[int, Episode], selected: list[int]
) -> Iterator[CalibRecord]:
    """Select chunk starts in episode order with one fixed random generator."""
    rng = np.random.default_rng(config.selection_seed + 1)
    for episode_index in selected:
        episode = episodes[episode_index]
        stop = (
            episode.length - config.action_horizon + 1 if config.complete_only else episode.length
        )
        if config.chunk_selection == "random":
            pool = np.arange(max(stop, 1))
            take = min(int(config.max_chunks_per_episode), len(pool))
            starts = sorted(int(v) for v in rng.choice(pool, size=take, replace=False))
        else:
            starts = list(range(0, max(stop, 0), config.stride))
            if config.max_chunks_per_episode is not None:
                starts = starts[: config.max_chunks_per_episode]
        for chunk_index, start in enumerate(starts):
            valid = min(config.action_horizon, episode.length - start)
            yield CalibRecord(
                sample_id=f"episode_{episode_index:06d}_chunk_{chunk_index:04d}",
                episode_index=episode_index,
                chunk_index=chunk_index,
                start_frame=int(start),
                valid_length=int(valid),
                noise_seed=noise_seed_for(
                    config.noise_seed, episode_index, chunk_index, config.noise_episode_stride
                ),
            )


def pad_chunk(segment: np.ndarray, horizon: int, action_dim: int) -> np.ndarray:
    """Extend a short action chunk with its final command."""
    out = np.empty((horizon, action_dim), dtype=np.float32)
    valid = int(segment.shape[0])
    out[:valid] = segment
    if valid < horizon:
        out[valid:] = segment[-1]
    return out


def decode_image(value: object, *, bgr: bool = False) -> np.ndarray:
    """Decode a stored frame to HWC uint8 RGB; reverse channels when the source is BGR."""
    from PIL import Image

    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(value)
        if array.dtype != np.uint8:
            array = (
                np.clip(array * 255.0, 0, 255).astype(np.uint8)
                if array.max() <= 1.0
                else array.astype(np.uint8)
            )
    if bgr:
        array = np.ascontiguousarray(array[..., ::-1])
    return array
