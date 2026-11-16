"""Read calibration observations and action chunks from LeRobot v2 datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DataConfig
from .records import CalibRecord, CalibSample, Episode
from .selection import build_records, decode_image, pad_chunk, select_episodes

__all__ = ["LeRobotDataset"]


class LeRobotDataset:
    """Lazy reader over a LeRobot v2 dataset directory."""

    format = "lerobot"

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.root = Path(config.dataset_root)
        meta = self.root / "meta"
        self.info_path = meta / "info.json"
        self.episodes_path = meta / "episodes.jsonl"
        for path in (self.info_path, self.episodes_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing LeRobot metadata: {path}")

        self.info = json.loads(self.info_path.read_text(encoding="utf-8"))
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v2"):
            raise ValueError(f"only LeRobot v2 datasets are supported, got {version!r}")
        self.data_path = self.info["data_path"]
        self.video_path = self.info.get("video_path")
        self.chunks_size = int(self.info["chunks_size"])

        self._table_cache: dict[int, dict[str, np.ndarray]] = {}
        self.episodes = self._read_episodes()
        self.selected = select_episodes(self.config, self.episodes)
        self.records = list(build_records(self.config, self.episodes, self.selected))

    def _read_episodes(self) -> dict[int, Episode]:
        episodes: dict[int, Episode] = {}
        with open(self.episodes_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                tasks = payload.get("tasks") or []
                prompt = self.config.prompt_override or (tasks[0] if tasks else "")
                index = int(payload["episode_index"])
                episodes[index] = Episode(index=index, length=int(payload["length"]), prompt=prompt)
        if not episodes:
            raise ValueError(f"{self.episodes_path} is empty")
        return episodes

    def _episode_parquet(self, episode_index: int) -> Path:
        relative = self.data_path.format(
            episode_chunk=episode_index // self.chunks_size, episode_index=episode_index
        )
        return self.root / relative

    def _episode_table(self, episode_index: int) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        cached = self._table_cache.get(episode_index)
        if cached is not None:
            return cached

        path = self._episode_parquet(episode_index)
        table = pq.read_table(path)
        columns: dict[str, np.ndarray] = {}
        for key in (self.config.action_key, self.config.state_key):
            if key not in table.column_names:
                raise KeyError(
                    f"{path}: column {key!r} is missing. Set data.action_key / "
                    f"data.state_key to the columns your policy was supervised on; this "
                    f"file holds {sorted(table.column_names)}"
                )
            columns[key] = np.stack(
                [np.asarray(v, dtype=np.float64) for v in table.column(key).to_pylist()]
            )
        for slot, feature in self.config.cameras.items():
            if feature in table.column_names:
                columns[f"__image__{slot}"] = np.asarray(
                    table.column(feature).to_pylist(), dtype=object
                )
        if len(self._table_cache) >= 2:
            self._table_cache.pop(next(iter(self._table_cache)))
        self._table_cache[episode_index] = columns
        return columns

    def _decode_video_frame(self, episode_index: int, feature: str, frame: int) -> np.ndarray:
        """Decode a video frame to HWC RGB with PyAV."""
        import av

        if self.video_path is None:
            raise ValueError("dataset declares no video_path but a video feature was requested")
        relative = self.video_path.format(
            episode_chunk=episode_index // self.chunks_size,
            video_key=feature,
            episode_index=episode_index,
        )
        path = self.root / relative
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for position, decoded in enumerate(container.decode(stream)):
                if position == frame:
                    return decoded.to_ndarray(format="rgb24")
        raise IndexError(f"frame {frame} not found in {path}")

    def load_sample(self, record: CalibRecord) -> CalibSample:
        cfg = self.config
        columns = self._episode_table(record.episode_index)
        state_lo, state_hi = cfg.state_slice
        action_lo, action_hi = cfg.action_slice

        state = np.ascontiguousarray(
            columns[cfg.state_key][record.start_frame, state_lo:state_hi], dtype=np.float32
        )
        segment = columns[cfg.action_key][
            record.start_frame : record.start_frame + record.valid_length, action_lo:action_hi
        ]
        gt = pad_chunk(segment, cfg.action_horizon, cfg.action_dim)

        observation: dict[str, Any] = {
            "state": state,
            "prompt": self.episodes[record.episode_index].prompt,
        }
        for slot, feature in cfg.cameras.items():
            cached = columns.get(f"__image__{slot}")
            if cached is not None:
                observation[slot] = decode_image(cached[record.start_frame], bgr=cfg.needs_bgr_swap)
            else:
                observation[slot] = self._decode_video_frame(
                    record.episode_index, feature, record.start_frame
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
            "image_channel_order": self.config.image_channel_order,
            "episodes": sorted(self.selected),
            "num_records": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }
