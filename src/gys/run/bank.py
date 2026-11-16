"""Terminal-state cache bound to one rollout configuration and calibration manifest."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import numpy as np

from .artifacts import read_json, write_json

__all__ = ["EndpointBank"]


class EndpointBank:
    """``put`` / ``get`` terminal arrays by node grid.  Content-addressed, collision-checked."""

    def __init__(self, directory: str | Path, *, identity: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        manifest = self.directory / "manifest.json"
        if manifest.exists():
            if read_json(manifest).get("identity") != identity:
                raise ValueError("endpoint cache belongs to another run; use a fresh --run-root")
        elif any(self.directory.glob("*.npz")):
            raise ValueError("endpoint cache has no provenance; use a fresh --run-root")
        else:
            write_json(manifest, {"schema": "gys-endpoints-v1", "identity": identity})

    @staticmethod
    def key(nodes: np.ndarray) -> str:
        nodes = np.ascontiguousarray(np.asarray(nodes, dtype=np.float64))
        return hashlib.sha256(nodes.tobytes()).hexdigest()[:20]

    def path(self, nodes: np.ndarray) -> Path:
        return self.directory / f"{self.key(nodes)}.npz"

    def get(self, nodes: np.ndarray) -> np.ndarray | None:
        path = self.path(nodes)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                stored, terminal = payload["nodes"], payload["terminal"]
            if not np.all(np.isfinite(terminal)):
                raise ValueError("non-finite terminal array")
        except (zipfile.BadZipFile, EOFError, OSError, KeyError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if not np.array_equal(stored, np.asarray(nodes, dtype=np.float64)):
            raise ValueError(f"{path}: stored nodes disagree with the key -- corrupt bank")
        return terminal

    def put(self, nodes: np.ndarray, terminal: np.ndarray) -> Path:
        path = self.path(nodes)
        if not path.is_file():
            scratch = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            try:
                with open(scratch, "wb") as sink:
                    np.savez(
                        sink,
                        nodes=np.asarray(nodes, dtype=np.float64),
                        terminal=np.asarray(terminal, dtype=np.float64),
                    )
                os.replace(scratch, path)
            finally:
                scratch.unlink(missing_ok=True)
        return path

    def __len__(self) -> int:
        return sum(1 for p in self.directory.glob("*.npz") if not p.name.endswith(".tmp.npz"))
