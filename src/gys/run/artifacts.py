"""Sealed JSON and array artifacts written by atomic replacement."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config.base import canonical_json

__all__ = [
    "HASH_KEY",
    "json_safe",
    "read_json",
    "sha256_text",
    "verify_seal",
    "write_json",
    "write_npz",
]

HASH_KEY = "sha256"


def json_safe(value: Any) -> Any:
    """Recursively make a payload JSON-able: non-finite floats become ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    """Write a sealed artifact; returns its content hash."""
    payload = json_safe(dict(payload))
    payload.pop(HASH_KEY, None)
    digest = sha256_text(canonical_json(payload))
    payload[HASH_KEY] = digest
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        scratch.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    return digest


def verify_seal(payload: dict[str, Any]) -> None:
    """Raise if a payload's recorded hash does not match its contents."""
    stored = payload.get(HASH_KEY)
    if stored is None:
        raise ValueError("missing artifact hash")
    body = {k: v for k, v in payload.items() if k != HASH_KEY}
    actual = sha256_text(canonical_json(body))
    if actual != stored:
        raise ValueError(f"artifact hash mismatch (stored {stored[:12]}, actual {actual[:12]})")


def read_json(path: str | Path, *, check: bool = True) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if check:
        verify_seal(payload)
    return payload


def write_npz(path: str | Path, **arrays: Any) -> str:
    """Write arrays atomically; returns the file's content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        with open(scratch, "wb") as sink:
            np.savez(sink, **arrays)
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
