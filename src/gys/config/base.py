"""Strict YAML loading, configuration serialization, and fingerprints."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

EXTENDS_KEY = "extends"

T = TypeVar("T", bound="BaseConfig")


class ConfigError(ValueError):
    """A config file is malformed, or names a field that does not exist."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` wins; nested mappings merge, everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _deep_merge(current, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path, *, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML and resolve parent paths relative to each file. Reject inheritance cycles."""
    import yaml

    resolved = Path(path).expanduser().resolve()
    if resolved in _chain:
        loop = " -> ".join(str(p) for p in (*_chain, resolved))
        raise ConfigError(f"extends cycle: {loop}")
    if not resolved.is_file():
        raise FileNotFoundError(f"config not found: {resolved}")

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{resolved}: top level must be a mapping, got {type(payload).__name__}")

    parents = payload.pop(EXTENDS_KEY, None)
    if parents is None:
        return payload
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"{resolved}: '{EXTENDS_KEY}' must be a string or a list of strings")

    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(
            merged, load_yaml(resolved.parent / str(parent), _chain=(*_chain, resolved))
        )
    return _deep_merge(merged, payload)


class BaseConfig:
    """Mixin for the frozen config dataclasses."""

    EXCLUDE_FROM_FINGERPRINT: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ConfigError(
                f"{cls.__name__}: unknown keys {unknown}; known keys are {sorted(known)}"
            )
        return cls(**payload)  # type: ignore[call-arg]

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        try:
            return cls.from_dict(load_yaml(path))
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"{Path(path)}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-compatible mapping."""
        out: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, dict):
                value = {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.items()}
            out[f.name] = value
        return out

    def fingerprint_fields(self) -> dict[str, Any]:
        """Fields used to identify cached results."""
        excluded = set(self.EXCLUDE_FROM_FINGERPRINT)
        return {k: v for k, v in self.to_dict().items() if k not in excluded}

    def replace(self: T, **overrides: Any) -> T:
        """Return a validated copy with non-None overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return type(self).from_dict({**self.to_dict(), **clean})


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(payload: Any, *, length: int = 16) -> str:
    """Return a truncated SHA-256 hash of the canonical JSON configuration."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:length]


def integer(value: Any, name: str) -> int:
    """Reject fractional, boolean, or nonfinite integer settings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be an integer")
    if not math.isfinite(value) or int(value) != value:
        raise ConfigError(f"{name} must be an integer")
    return int(value)
