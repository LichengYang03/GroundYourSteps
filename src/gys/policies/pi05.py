"""OpenPI adapter: cached backbone conditioning and velocity v(t) = -v_native(1-t)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DataConfig, OpenPIPolicyConfig
from .base import default_noise

__all__ = ["OpenPIPolicy"]


class OpenPIPolicy:
    """Implements :class:`~gys.policies.base.FlowPolicy` for pi0 / pi0.5."""

    quantises_time = False
    supports_time_grad = True

    def __init__(
        self,
        config: OpenPIPolicyConfig,
        data: DataConfig,
        *,
        device: str | None = None,
    ) -> None:
        _check_camera_slots(config, data)

        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        source_root = str(Path(config.openpi_root) / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        import torch
        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config as training_config
        from openpi.transforms import DeltaActions

        self._torch = torch
        self.config = config
        self.data = data
        self.device = device or config.device
        self.dtype = torch.float32

        checkpoint = Path(config.checkpoint)
        candidates = sorted((checkpoint / "assets").rglob("norm_stats.json"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one norm_stats.json under {checkpoint / 'assets'}, found "
                f"{len(candidates)}"
            )
        self.norm_stats_path = candidates[0]
        norm_stats = normalize.load(self.norm_stats_path.parent)

        if config.config_name == "gys_robotwin_pi05":
            from .openpi_config import robotwin_config

            train_config = robotwin_config()
        else:
            train_config = training_config.get_config(config.config_name)
        if config.model_action_horizon:
            import dataclasses

            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(
                    train_config.model, action_horizon=int(config.model_action_horizon)
                ),
            )
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        _assert_no_channel_flip(data_config)

        self._delta_mask: np.ndarray | None = next(
            (
                np.asarray(t.mask, dtype=bool)
                for t in data_config.data_transforms.inputs
                if isinstance(t, DeltaActions) and t.mask is not None
            ),
            None,
        )

        from openpi import transforms as openpi_transforms

        repack = (
            data_config.repack_transforms
            if config.use_training_repack
            else openpi_transforms.Group()
        )
        self.policy = policy_config.create_trained_policy(
            train_config,
            checkpoint,
            repack_transforms=repack,
            default_prompt=config.default_prompt,
            pytorch_device=self.device,
            norm_stats=norm_stats,
        )
        if not getattr(self.policy, "_is_pytorch_model", False):
            raise TypeError(
                "this package requires a PyTorch openpi checkpoint (JAX is unsupported)"
            )

        self.model = self.policy._model
        self.horizon = int(self.model.config.action_horizon)
        self.latent_dim = int(self.model.config.action_dim)
        self.model_action_start = int(config.model_action_start)
        self.robot_dim = int(data.action_dim)
        if self.model_action_start + self.robot_dim > self.latent_dim:
            raise ValueError(
                f"the scored slice [{self.model_action_start}, "
                f"{self.model_action_start + self.robot_dim}) exceeds the latent width "
                f"{self.latent_dim}"
            )
        if int(data.action_horizon) != self.horizon:
            raise ValueError(
                f"data.action_horizon={data.action_horizon} but the model integrates "
                f"{self.horizon} action steps"
            )
        self.freeze()

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def score_slice(self) -> tuple[int, int]:
        return (self.model_action_start, self.model_action_start + self.robot_dim)

    @property
    def score_horizon(self) -> int:
        return self.horizon

    def sample_noise(self, seed: int) -> np.ndarray:
        return default_noise(seed, self.horizon, self.latent_dim)

    def prepare_context(self, sample: Any) -> dict[str, Any]:
        torch = self._torch
        gt_raw = np.asarray(sample.gt_action_raw, dtype=np.float32)
        observation, gt_model, model_state = self._prepare_inputs(sample.observation, gt_raw)
        with torch.no_grad():
            state, pad_masks, cache, hidden = self._prepare_prefix(observation)

        context: dict[str, Any] = {
            "state": state,
            "pad_masks": pad_masks,
            "cache": cache,
            "gt_model": gt_model,
            "model_state": model_state,
            "noise_seed": int(sample.record.noise_seed),
            "sample_id": sample.record.sample_id,
            "episode_index": int(sample.record.episode_index),
        }
        context["obs_summary"] = self._pool_prefix(pad_masks, hidden)
        context["obs_source"] = "prefix_hidden"
        return context

    def _pool_prefix(self, pad_masks: Any, hidden: Any) -> np.ndarray:
        """Mean of final backbone states over non-padding tokens."""
        torch = self._torch
        hidden = hidden.float()
        mask = pad_masks.to(hidden.dtype).reshape(1, -1, 1)
        denominator = mask.sum()
        if float(denominator) <= 0:
            raise ValueError("the prefix pad mask is empty")
        pooled = ((hidden * mask).sum(dim=1) / denominator).reshape(-1)
        return pooled.detach().to("cpu").to(torch.float32).numpy()

    def stack_contexts(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate fixed-length prefix caches, masks, and states along the batch axis."""
        torch = self._torch
        from transformers.cache_utils import DynamicCache

        caches = [c["cache"] for c in contexts]
        if not all(isinstance(cache, DynamicCache) for cache in caches):
            raise TypeError(
                f"batched rollout needs DynamicCache prefixes, got {type(caches[0]).__name__}; "
                "use the compatible openpi PyTorch environment"
            )
        return {
            "state": torch.cat([c["state"] for c in contexts], dim=0),
            "pad_masks": torch.cat([c["pad_masks"] for c in contexts], dim=0),
            "cache": DynamicCache.from_batch_splits(caches),
            "batch_size": len(contexts),
        }

    def velocity_batch(self, x: Any, t: Any, context: dict[str, Any]) -> Any:
        """Canonical ``v(x, t)`` for ``x`` of shape ``(N, H, D)``; one network call."""
        torch = self._torch
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.horizon, self.latent_dim):
            raise ValueError(
                f"x must be (N, {self.horizon}, {self.latent_dim}), got {tuple(x.shape)}"
            )
        n, expected = int(x.shape[0]), int(context["batch_size"])
        if n != expected:
            raise ValueError(f"x has batch {n} but the context was stacked for {expected}")
        if not torch.is_tensor(t):
            t = torch.as_tensor(float(t), dtype=x.dtype, device=x.device)
        query_time = (1.0 - t).reshape(1).expand(n).to(dtype=self.dtype)
        native = self.model.denoise_step(
            context["state"],
            context["pad_masks"],
            context["cache"],
            x.to(dtype=self.dtype),
            query_time,
        )
        return (-native).to(dtype=x.dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "openpi",
            "checkpoint": str(self.config.checkpoint),
            "config_name": self.config.config_name,
            "norm_stats": str(self.norm_stats_path),
            "horizon": self.horizon,
            "latent_dim": self.latent_dim,
            "score_slice": list(self.score_slice),
            "score_horizon": self.score_horizon,
            "wire_knot_axis": "flipped",
            "delta_action_mask": None if self._delta_mask is None else self._delta_mask.tolist(),
            "camera_keys": dict(self.config.camera_keys),
            "image_layout": self.config.image_layout,
            "images_expected": "rgb",
            "time_convention": "canonical t (0=noise, 1=action); query = 1 - t; v = -v_native",
            "action_space": "the checkpoint's own input transform chain",
            "obs_source": "prefix_hidden",
            "device": self.device,
        }

    def model_actions_to_dataset(
        self, actions_model: np.ndarray, model_state: np.ndarray
    ) -> np.ndarray:
        """Integration space -> dataset units, for reporting.  The exact inverse chain."""
        outputs = self.policy._output_transform(
            {
                "state": np.asarray(model_state, dtype=np.float32),
                "actions": np.asarray(actions_model, dtype=np.float32),
            }
        )
        return np.asarray(outputs["actions"], dtype=np.float64)

    def _prepare_inputs(self, raw_observation: dict[str, Any], gt_raw: np.ndarray):
        """Transform the observation and demonstration into model coordinates."""
        import torch
        from openpi.models import model as openpi_model

        raw: dict[str, Any] = dict(raw_observation)
        for slot, key in self.config.camera_keys.items():
            frame = np.asarray(raw.pop(slot))
            if self.config.image_layout == "chw":
                frame = np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))
            raw[key] = frame
        raw["actions"] = gt_raw

        inputs = self.policy._input_transform(raw)
        if "actions" not in inputs:
            raise RuntimeError(
                f"input transforms for {self.config.config_name!r} must preserve actions"
            )
        gt_model = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if gt_model.shape != (self.horizon, self.latent_dim):
            raise ValueError(
                f"the transformed demonstration is {gt_model.shape}, expected "
                f"({self.horizon}, {self.latent_dim})"
            )
        model_state = np.asarray(inputs["state"], dtype=np.float32)
        inputs = _tree_map(
            lambda value: torch.from_numpy(np.asarray(value)).to(self.device)[None, ...], inputs
        )
        return openpi_model.Observation.from_dict(inputs), gt_model, model_state

    def _prepare_prefix(self, observation: Any):
        import torch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        images, image_masks, language_tokens, language_masks, state = (
            self.model._preprocess_observation(observation, train=False)
        )
        embeddings, pad_masks, attention_masks = self.model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        attention_2d = make_att_2d_masks(pad_masks, attention_masks)
        positions = torch.cumsum(pad_masks, dim=1) - 1
        attention_4d = self.model._prepare_attention_masks_4d(attention_2d)
        language_model = self.model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = self.config.attn_implementation
        hidden, cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        return state, pad_masks, cache, hidden[0]


def _check_camera_slots(config: OpenPIPolicyConfig, data: DataConfig) -> None:
    """Require identical camera slot names in dataset and policy configurations."""
    dataset_slots = set(data.cameras or {})
    policy_slots = set(config.camera_keys)
    if dataset_slots != policy_slots:
        raise ValueError(
            f"camera slots disagree: data.cameras yields {sorted(dataset_slots)}, "
            f"policy.camera_keys names {sorted(policy_slots)}"
        )


def _assert_no_channel_flip(data_config: Any) -> None:
    """Check that policy transforms accept RGB input."""
    flipping = [
        type(t).__name__
        for t in data_config.data_transforms.inputs
        if bool(getattr(t, "bgr_to_rgb", False))
    ]
    if flipping:
        raise RuntimeError(
            f"disable BGR conversion in the checkpoint transforms {flipping} for RGB inputs"
        )


def _tree_map(fn, node):
    """Minimal pytree map over dict / list / tuple, avoiding a JAX import."""
    if isinstance(node, dict):
        return {key: _tree_map(fn, value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        mapped = [_tree_map(fn, value) for value in node]
        return type(node)(mapped) if isinstance(node, tuple) else mapped
    return fn(node)
