"""Dataset locations, RGB decoding, and calibration selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseConfig, integer

__all__ = ["FORMATS", "DataConfig"]

FORMATS = ("robotwin_hdf5", "lerobot")

CHANNEL_ORDERS = ("rgb", "bgr")


@dataclass(frozen=True)
class DataConfig(BaseConfig):
    """One calibration source: which episodes, which chunks, which columns, which seeds."""

    dataset_root: str = ""
    format: str = "robotwin_hdf5"

    action_horizon: int = 32

    image_channel_order: str = "rgb"

    action_key: str = "joint_action/vector"
    state_key: str = "joint_action/vector"
    action_slice: tuple[int, int] = (0, 14)
    state_slice: tuple[int, int] = (0, 14)

    # Map adapter camera slots to dataset feature names.
    cameras: dict[str, str] = field(default_factory=dict)

    prompt_override: str | None = None
    prompt_source: str = "seen"
    prompt_scope: str = "episode"

    episode_ids: list[int] | None = None
    episode_count: int = 50
    selection_seed: int = 0
    chunk_selection: str = "random"
    chunk_stride: int | None = None
    max_chunks_per_episode: int | None = 4
    complete_only: bool = True

    noise_seed: int = 0
    noise_episode_stride: int = 1000

    def __post_init__(self) -> None:
        for name in (
            "action_horizon",
            "episode_count",
            "selection_seed",
            "noise_seed",
            "noise_episode_stride",
        ):
            object.__setattr__(self, name, integer(getattr(self, name), name))
        for name in ("chunk_stride", "max_chunks_per_episode"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, integer(getattr(self, name), name))
        object.__setattr__(
            self, "action_slice", tuple(integer(v, "action_slice") for v in self.action_slice)
        )
        object.__setattr__(
            self, "state_slice", tuple(integer(v, "state_slice") for v in self.state_slice)
        )
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        if self.image_channel_order not in CHANNEL_ORDERS:
            raise ValueError(
                f"image_channel_order must be one of {CHANNEL_ORDERS}, got "
                f"{self.image_channel_order!r}"
            )
        if not self.dataset_root:
            raise ValueError("dataset_root is required")
        if self.prompt_source not in ("seen", "unseen"):
            raise ValueError("prompt_source must be 'seen' or 'unseen'")
        if self.prompt_scope not in ("episode", "task"):
            raise ValueError("prompt_scope must be 'episode' or 'task'")
        if self.chunk_selection not in ("stride", "random"):
            raise ValueError("chunk_selection must be 'stride' or 'random'")
        if self.chunk_selection == "random" and self.max_chunks_per_episode is None:
            raise ValueError("chunk_selection='random' needs max_chunks_per_episode")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        for name in ("action_slice", "state_slice"):
            lo, hi = getattr(self, name)
            if lo < 0 or hi <= lo:
                raise ValueError(f"{name} must be a non-empty [start, stop)")
        if self.episode_ids is None and self.episode_count <= 0:
            raise ValueError("episode_count must be positive when episode_ids is unset")
        if self.chunk_stride is not None and self.chunk_stride <= 0:
            raise ValueError("chunk_stride must be positive")
        if self.max_chunks_per_episode is not None and self.max_chunks_per_episode < 1:
            raise ValueError("max_chunks_per_episode must be positive")
        if self.selection_seed < 0 or self.noise_seed < 0:
            raise ValueError("seeds must be nonnegative")
        if self.noise_episode_stride <= 0:
            raise ValueError("noise_episode_stride must be positive")
        if (
            self.max_chunks_per_episode is not None
            and self.max_chunks_per_episode >= self.noise_episode_stride
        ):
            raise ValueError(
                f"max_chunks_per_episode={self.max_chunks_per_episode} must stay below "
                f"noise_episode_stride={self.noise_episode_stride}, or two chunks in "
                "different episodes share a starting latent"
            )

    @property
    def action_dim(self) -> int:
        return self.action_slice[1] - self.action_slice[0]

    @property
    def stride(self) -> int:
        return self.chunk_stride or self.action_horizon

    @property
    def needs_bgr_swap(self) -> bool:
        """Do decoded frames need their channels reversed to reach RGB?"""
        return self.image_channel_order == "bgr"

    def for_task(self, task: str | None) -> DataConfig:
        """Substitute ``{task}`` in ``dataset_root``."""
        if "{task}" not in self.dataset_root:
            return self
        if not task:
            raise ValueError(
                f"dataset_root {self.dataset_root!r} is templated on {{task}}; pass --task"
            )
        return self.replace(dataset_root=self.dataset_root.format(task=task))

    def describe_channels(self) -> str:
        """Describe the dataset-to-RGB conversion."""
        if self.needs_bgr_swap:
            return (
                f"images: files store BGR ({self.format}) -> reversed to RGB at decode; "
                "policies receive RGB"
            )
        return f"images: files store RGB ({self.format}) -> passed through unchanged"
