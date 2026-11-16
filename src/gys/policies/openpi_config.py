"""RoboTwin observation and action transforms for π0.5."""

from __future__ import annotations

from typing import Any

import numpy as np


class RobotWinInputs:
    """Map RGB cameras and absolute joint vectors to openpi's model inputs."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = {
            name: np.asarray(data[key], dtype=np.uint8)
            for name, key in (
                ("base_0_rgb", "head_cam"),
                ("left_wrist_0_rgb", "left_cam"),
                ("right_wrist_0_rgb", "right_cam"),
            )
        }
        result = {
            "state": np.asarray(data["state"]),
            "image": images,
            "image_mask": {name: np.True_ for name in images},
        }
        for key in ("actions", "prompt"):
            if key in data:
                result[key] = data[key]
        return result


class RobotWinOutputs:
    """Select the 14 robot action coordinates from the padded output."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(data["actions"])[..., :14]}


def robotwin_config():
    """Build the 32-step RoboTwin inference configuration."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import SimpleDataConfig, TrainConfig

    return TrainConfig(
        name="gys_robotwin_pi05",
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=32),
        data=SimpleDataConfig(
            data_transforms=lambda _: transforms.Group(
                inputs=[RobotWinInputs()], outputs=[RobotWinOutputs()]
            ),
        ),
    )
