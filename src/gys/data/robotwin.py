"""Read calibration observations and action chunks from RoboTwin HDF5 datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DataConfig
from .records import CalibRecord, CalibSample, Episode
from .selection import build_records, decode_image, pad_chunk, select_episodes

__all__ = ["RoboTwinHDF5Dataset"]


class RoboTwinHDF5Dataset:
    """Lazy reader over one RoboTwin 2.0 task directory."""

    format = "robotwin_hdf5"

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.root = Path(config.dataset_root)
        self.data_dir = self.root / "data"
        self.instr_dir = self.root / "instructions"
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"missing RoboTwin data directory: {self.data_dir}")

        self._files = {
            int(path.stem.removeprefix("episode")): path
            for path in sorted(self.data_dir.glob("episode*.hdf5"))
        }
        if not self._files:
            raise ValueError(f"no episode*.hdf5 under {self.data_dir}")

        self._handles: dict[int, Any] = {}
        self._task_prompt: str | None = None
        self.episodes = self._read_episodes()
        self.selected = select_episodes(self.config, self.episodes)
        self.records = list(build_records(self.config, self.episodes, self.selected))

    def _read_episodes(self) -> dict[int, Episode]:
        import h5py

        episodes: dict[int, Episode] = {}
        for index, path in self._files.items():
            with h5py.File(path, "r") as handle:
                if self.config.action_key not in handle:
                    raise KeyError(
                        f"{path}: no dataset {self.config.action_key!r}. Set data.action_key "
                        f"to the column your policy was supervised on; this file holds "
                        f"{sorted(handle.keys())}"
                    )
                length = int(handle[self.config.action_key].shape[0])
            episodes[index] = Episode(index=index, length=length, prompt=self._prompt(index))
        return episodes

    def _prompt(self, episode_index: int) -> str:
        if self.config.prompt_override:
            return self.config.prompt_override
        if self.config.prompt_scope == "task":
            if self._task_prompt is None:
                self._task_prompt = self._instruction(min(self._files))
            return self._task_prompt
        return self._instruction(episode_index)

    def _instruction(self, episode_index: int) -> str:
        path = self.instr_dir / f"episode{episode_index}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing instructions for episode {episode_index}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        options = payload.get(self.config.prompt_source) or []
        if not options:
            raise ValueError(f"{path}: no {self.config.prompt_source!r} instructions")
        return str(options[0])

    def _handle(self, episode_index: int) -> Any:
        import h5py

        cached = self._handles.get(episode_index)
        if cached is not None:
            return cached
        if len(self._handles) >= 2:
            self._handles.pop(next(iter(self._handles))).close()
        handle = h5py.File(self._files[episode_index], "r")
        self._handles[episode_index] = handle
        return handle

    def load_sample(self, record: CalibRecord) -> CalibSample:
        cfg = self.config
        handle = self._handle(record.episode_index)
        state_lo, state_hi = cfg.state_slice
        action_lo, action_hi = cfg.action_slice

        state = np.asarray(
            handle[cfg.state_key][record.start_frame, state_lo:state_hi], dtype=np.float32
        )
        segment = np.asarray(
            handle[cfg.action_key][
                record.start_frame : record.start_frame + record.valid_length,
                action_lo:action_hi,
            ],
            dtype=np.float64,
        )
        gt = pad_chunk(segment, cfg.action_horizon, cfg.action_dim)

        observation: dict[str, Any] = {
            "state": state,
            "prompt": self.episodes[record.episode_index].prompt,
        }
        for slot, camera in cfg.cameras.items():
            group = f"observation/{camera}/rgb"
            if group not in handle:
                raise KeyError(
                    f"{self._files[record.episode_index]}: no group {group!r} for camera "
                    f"slot {slot!r}; check data.cameras"
                )
            observation[slot] = decode_image(
                handle[group][record.start_frame], bgr=cfg.needs_bgr_swap
            )
        return CalibSample(record=record, observation=observation, gt_action_raw=gt)

    def __iter__(self) -> Iterator[CalibSample]:
        for record in self.records:
            yield self.load_sample(record)

    def __len__(self) -> int:
        return len(self.records)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "dataset_root": str(self.root),
            "num_episode_files": len(self._files),
            "image_channel_order": self.config.image_channel_order,
            "episodes": sorted(self.selected),
            "num_records": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }
