"""GR00T N1.7 adapter with cached conditioning and native noise-to-action time."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from ..config import DataConfig, GR00TPolicyConfig

__all__ = ["GR00TPolicy", "split_dual_arm"]


def split_dual_arm(vector: np.ndarray, arm_joints: int) -> dict[str, np.ndarray]:
    """Split [left arm, left gripper, right arm, right gripper].

    The final dimension must equal 2 * arm_joints + 2.
    """
    joints = int(arm_joints)
    if joints < 1:
        raise ValueError(f"arm_joints must be >= 1, got {arm_joints}")
    width = 2 * joints + 2
    if int(vector.shape[-1]) != width:
        raise ValueError(
            f"dual-arm vector last dim is {vector.shape[-1]}, expected {width} for "
            f"arm_joints={joints} (2x{joints} joints + 2 grippers)"
        )
    return {
        "left_arm": vector[..., 0:joints],
        "left_gripper": vector[..., joints : joints + 1],
        "right_arm": vector[..., joints + 1 : 2 * joints + 1],
        "right_gripper": vector[..., 2 * joints + 1 : 2 * joints + 2],
    }


class GR00TPolicy:
    """Velocity field and conditioning for GR00T N1.7."""

    supports_time_grad = False
    quantises_time = True

    def __init__(
        self,
        config: GR00TPolicyConfig,
        data: DataConfig,
        *,
        device: str | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self.config = config
        self.data = data
        self.device = device or config.device
        self.dtype = torch.float32  # the ODE state; the model itself runs bfloat16

        root = os.path.abspath(config.groot_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.checkpoint,
            device=self.device,
            strict=False,
        )
        if bool(getattr(self.policy, "bgr_to_rgb", False)):
            raise RuntimeError("disable BGR conversion in the GR00T wrapper for RGB inputs")
        self.model = self.policy.model
        self.head = self.model.action_head
        self.model_dtype = next(self.model.parameters()).dtype

        self.horizon = int(self.head.config.action_horizon)
        self.latent_dim = int(self.head.action_dim)
        self.buckets = int(self.head.num_timestep_buckets)

        self.robot_dim = int(data.action_dim)
        self._score_horizon = int(config.score_horizon or data.action_horizon)
        if self._score_horizon > self.horizon:
            raise ValueError(
                f"score_horizon={self._score_horizon} exceeds the checkpoint's "
                f"action_horizon={self.horizon}"
            )
        if self.robot_dim > self.latent_dim:
            raise ValueError(
                f"the robot's action width {self.robot_dim} exceeds the latent width "
                f"{self.latent_dim}"
            )
        self._assert_matches_upstream()
        self.freeze()

    def _assert_matches_upstream(self) -> None:
        """Check the action-head attributes required by the single-step velocity adapter."""
        missing = [
            name
            for name in (
                "action_encoder",
                "action_decoder",
                "model",
                "state_encoder",
                "position_embedding",
                "process_backbone_output",
                "num_timestep_buckets",
                "action_dim",
            )
            if not hasattr(self.head, name)
        ]
        if missing:
            raise RuntimeError(f"GR00T N1.7 action head is missing {missing}")
        if not hasattr(self.head.config, "use_alternate_vl_dit"):
            raise RuntimeError("GR00T action head requires use_alternate_vl_dit")

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def score_slice(self) -> tuple[int, int]:
        return (0, self.robot_dim)

    @property
    def score_horizon(self) -> int:
        return self._score_horizon

    def sample_noise(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return rng.standard_normal((self.horizon, self.latent_dim)).astype(np.float32)

    def _step_data(self, sample: Any, gt_raw: np.ndarray) -> Any:
        """Create one GR00T processor input containing the observation and demonstration."""
        from gr00t.data.types import VLAStepData

        raw = dict(sample.observation)
        state_vector = np.asarray(raw["state"], dtype=np.float32).reshape(-1)
        chunk = np.asarray(gt_raw, dtype=np.float32)
        images = {
            key: np.asarray(raw[slot], dtype=np.uint8)[None, ...]  # (1, H, W, C), already RGB
            for slot, key in self.config.camera_keys.items()
        }
        groups = self.config.arm_joints
        return VLAStepData(
            images=images,
            states=split_dual_arm(state_vector[None, :], groups),
            actions=split_dual_arm(chunk, groups),
            text=str(raw.get("prompt", "")),
            embodiment=self.policy.embodiment_tag,
        )

    def prepare_context(self, sample: Any) -> dict[str, Any]:
        torch = self._torch
        gt_raw = np.asarray(sample.gt_action_raw, dtype=np.float32)
        with torch.no_grad():
            batch = self._processed(self._step_data(sample, gt_raw))
            gt_model = np.asarray(batch["action"][0].float().cpu().numpy(), dtype=np.float32)
            if gt_model.shape != (self.horizon, self.latent_dim):
                raise ValueError(
                    f"the processed demonstration is {gt_model.shape}, expected "
                    f"({self.horizon}, {self.latent_dim})"
                )
            prefix = self._prefix(batch)

        prefix["obs_summary"] = self._pool_prefix(prefix)
        prefix["obs_source"] = "prefix_hidden"
        prefix.update(
            {
                "gt_model": gt_model,
                "model_state": np.asarray(
                    batch["state"][0].float().cpu().numpy(), dtype=np.float32
                ),
                "noise_seed": int(sample.record.noise_seed),
                "sample_id": sample.record.sample_id,
                "episode_index": int(sample.record.episode_index),
            }
        )
        return prefix

    def _pool_prefix(self, prefix: dict[str, Any]) -> np.ndarray:
        """Mean of the action head's backbone features over non-padding tokens."""
        torch = self._torch
        embeds = prefix["vl_embeds"].float()
        mask = prefix.get("backbone_attention_mask")
        if mask is None:
            raise ValueError(
                "the backbone produced no attention mask; pooling would average over padding tokens"
            )
        mask2d = mask.to(embeds.dtype).reshape(1, -1, 1)
        denominator = mask2d.sum()
        if float(denominator) <= 0.0:
            raise ValueError("the backbone attention mask is empty; cannot pool")
        pooled = (embeds * mask2d).sum(dim=1) / denominator
        return pooled.reshape(-1).detach().to("cpu").to(torch.float32).numpy()

    def _processed(self, step: Any) -> dict[str, Any]:
        """Process and collate a single observation and its demonstration."""
        torch = self._torch
        from gr00t.data.types import MessageType

        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        processed = self.policy.processor(messages)
        collated = self.policy.collate_fn([processed])
        batch = collated.get("inputs", collated)
        return {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    def _prefix(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute backbone and state features that are independent of flow time."""
        from transformers.feature_extraction_utils import BatchFeature

        inputs = BatchFeature(data=dict(batch))
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)
        backbone_output = self.model.backbone(backbone_inputs)
        encoded = self.head._encode_features(backbone_output, action_inputs)
        return {
            "vl_embeds": encoded["backbone_features"],
            "state_features": encoded["state_features"],
            "embodiment_id": action_inputs["embodiment_id"].to(self.device),
            "image_mask": getattr(backbone_output, "image_mask", None),
            "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None),
            "batch_size": 1,
        }

    def stack_contexts(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch variable-length conditioning features.

        Pad both image and backbone attention masks with False.
        """
        torch = self._torch
        depths = {tuple(c["vl_embeds"].shape[2:]) for c in contexts}
        if len(depths) != 1:
            raise ValueError(f"prefixes have different embedding dims {sorted(depths)}")
        if any(c["image_mask"] is None or c["backbone_attention_mask"] is None for c in contexts):
            raise ValueError("batching requires image and backbone attention masks")
        longest = max(int(c["vl_embeds"].shape[1]) for c in contexts)

        def pad(value: Any, fill: Any) -> Any:
            short = longest - int(value.shape[1])
            if short == 0:
                return value
            shape = (value.shape[0], short, *value.shape[2:])
            tail = torch.full(shape, fill, dtype=value.dtype, device=value.device)
            return torch.cat([value, tail], dim=1)

        return {
            "vl_embeds": torch.cat([pad(c["vl_embeds"], 0) for c in contexts], dim=0),
            "state_features": torch.cat([c["state_features"] for c in contexts], dim=0),
            "embodiment_id": torch.cat([c["embodiment_id"] for c in contexts], dim=0),
            "image_mask": torch.cat([pad(c["image_mask"], False) for c in contexts], dim=0),
            "backbone_attention_mask": torch.cat(
                [pad(c["backbone_attention_mask"], False) for c in contexts], dim=0
            ),
            "batch_size": len(contexts),
        }

    def _velocity(self, actions: Any, t: Any, context: dict[str, Any]) -> Any:
        """Evaluate the upstream action-head loop body at the requested flow time."""
        torch = self._torch
        head = self.head
        n = int(actions.shape[0])

        bucket = int(float(t) * self.buckets)
        bucket = max(0, min(self.buckets - 1, bucket))
        timesteps = torch.full((n,), bucket, device=actions.device, dtype=torch.long)

        features = head.action_encoder(actions, timesteps, context["embodiment_id"])
        if head.config.add_pos_embed:
            pos_ids = torch.arange(features.shape[1], dtype=torch.long, device=actions.device)
            features = features + head.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((context["state_features"], features), dim=1)
        if head.config.use_alternate_vl_dit:
            output = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=context["vl_embeds"],
                timestep=timesteps,
                image_mask=context["image_mask"],
                backbone_attention_mask=context["backbone_attention_mask"],
            )
        else:
            output = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=context["vl_embeds"],
                timestep=timesteps,
            )
        pred = head.action_decoder(output, context["embodiment_id"])
        return pred[:, -head.config.action_horizon :]

    def velocity_batch(self, x: Any, t: Any, context: dict[str, Any]) -> Any:
        """Evaluate the velocity for a batch of noisy action chunks."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.horizon, self.latent_dim):
            raise ValueError(
                f"x must be (N, {self.horizon}, {self.latent_dim}), got {tuple(x.shape)}"
            )
        n, expected = int(x.shape[0]), int(context["batch_size"])
        if n != expected:
            raise ValueError(f"x has batch {n} but the context was stacked for {expected}")
        native = self._velocity(x.to(dtype=self.model_dtype), t, context)
        return native.to(dtype=x.dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "groot",
            "groot_root": self.config.groot_root,
            "checkpoint": self.config.checkpoint,
            "embodiment_tag": self.config.embodiment_tag,
            "arm_joints": int(self.config.arm_joints),
            "horizon": self.horizon,
            "latent_dim": self.latent_dim,
            "score_slice": list(self.score_slice),
            "score_horizon": self.score_horizon,
            "num_timestep_buckets": self.buckets,
            "native_inference_steps": int(self.head.num_inference_timesteps),
            "model_dtype": str(self.model_dtype),
            "device": self.device,
            "camera_keys": dict(self.config.camera_keys),
            "images_expected": "rgb",
            "wire_knot_axis": "canonical",
            "time_convention": (
                "canonical t (0=noise, 1=action); GR00T trains on (1-t)*noise + t*action "
                f"with target action-noise, so v = v_native and the network sees "
                f"floor(t * {self.buckets})"
            ),
            "obs_source": "prefix_hidden",
        }
