"""Frozen-policy locations, input mappings, and action layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseConfig, ConfigError, load_yaml

__all__ = [
    "GR00TPolicyConfig",
    "OpenPIPolicyConfig",
    "PolicyConfig",
    "load_policy_config",
    "policy_config_from_dict",
]


@dataclass(frozen=True)
class PolicyConfig(BaseConfig):
    """Fields shared by every frozen-policy adapter."""

    EXCLUDE_FROM_FINGERPRINT = ("device",)

    kind: str = ""
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind is required")


@dataclass(frozen=True)
class OpenPIPolicyConfig(PolicyConfig):
    """pi0 / pi0.5 served by the PyTorch branch of ``openpi``."""

    EXCLUDE_FROM_FINGERPRINT = ("device", "attn_implementation")

    kind: str = "openpi"

    openpi_root: str = ""
    checkpoint: str = ""
    config_name: str = ""
    default_prompt: str | None = None
    attn_implementation: str = "eager"

    # Map dataset camera slots to checkpoint input keys.
    camera_keys: dict[str, str] = field(default_factory=dict)

    image_layout: str = "hwc"

    # Apply the upstream dataset-key repack before model preprocessing.
    use_training_repack: bool = True

    # Zero uses the upstream configuration horizon.
    model_action_horizon: int = 0

    model_action_start: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "openpi":
            raise ValueError("kind must be 'openpi'")
        for name in ("openpi_root", "checkpoint", "config_name"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required for the openpi policy")
        if not self.camera_keys:
            raise ValueError("camera_keys is required for the openpi policy")
        if self.image_layout not in ("hwc", "chw"):
            raise ValueError(f"image_layout must be 'hwc' or 'chw', got {self.image_layout!r}")
        if self.model_action_horizon < 0:
            raise ValueError("model_action_horizon must be >= 0")
        if self.model_action_start < 0:
            raise ValueError("model_action_start must be >= 0")


@dataclass(frozen=True)
class GR00TPolicyConfig(PolicyConfig):
    """GR00T checkpoint and embodiment mapping.

    The checkpoint processor supplies the modality layout and normalization.
    """

    kind: str = "groot"

    groot_root: str = ""
    checkpoint: str = ""
    embodiment_tag: str = "new_embodiment"
    arm_joints: int = 6
    # Map dataset camera slots to checkpoint input keys.
    camera_keys: dict[str, str] = field(default_factory=dict)
    # Executed rows to score; zero uses the dataset action horizon.
    score_horizon: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "groot":
            raise ValueError("kind must be 'groot'")
        for name in ("groot_root", "checkpoint", "embodiment_tag"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required for the groot policy")
        if not self.camera_keys:
            raise ValueError("camera_keys is required for the groot policy")
        object.__setattr__(self, "arm_joints", int(self.arm_joints))
        if self.arm_joints < 1:
            raise ValueError(f"arm_joints must be >= 1, got {self.arm_joints}")
        if self.score_horizon < 0:
            raise ValueError(f"score_horizon must be >= 0, got {self.score_horizon}")


_REGISTRY: dict[str, type[PolicyConfig]] = {
    "openpi": OpenPIPolicyConfig,
    "groot": GR00TPolicyConfig,
}


def policy_config_from_dict(payload: dict[str, Any]) -> PolicyConfig:
    kind = str(payload.get("kind", ""))
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ConfigError(f"unknown policy kind {kind!r}, expected one of {sorted(_REGISTRY)}")
    return cls.from_dict(payload)


def load_policy_config(path: str | Path) -> PolicyConfig:
    try:
        return policy_config_from_dict(load_yaml(path))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{Path(path)}: {exc}") from exc
