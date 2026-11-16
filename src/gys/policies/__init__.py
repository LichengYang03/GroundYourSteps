"""Lazy construction of pi0.5 and GR00T policy adapters."""

from __future__ import annotations

from typing import Any

from ..config import DataConfig, PolicyConfig
from .base import FlowPolicy, check_state, default_noise
from .runtime import ContextSource, LatentBank

__all__ = [
    "ContextSource",
    "FlowPolicy",
    "LatentBank",
    "build_policy",
    "check_state",
    "default_noise",
]


def _build_openpi(config: PolicyConfig, data: DataConfig, device: str | None) -> Any:
    from .pi05 import OpenPIPolicy

    return OpenPIPolicy(config, data, device=device)


def _build_groot(config: PolicyConfig, data: DataConfig, device: str | None) -> Any:
    from .groot import GR00TPolicy

    return GR00TPolicy(config, data, device=device)


_BUILDERS = {
    "openpi": _build_openpi,
    "groot": _build_groot,
}


def build_policy(config: PolicyConfig, data: DataConfig, *, device: str | None = None) -> Any:
    builder = _BUILDERS.get(config.kind)
    if builder is None:
        raise ValueError(f"no adapter for policy kind {config.kind!r}; known: {sorted(_BUILDERS)}")
    return builder(config, data, device)
