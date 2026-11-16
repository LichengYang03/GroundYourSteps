"""Strict YAML configuration for a GYS task and NFE budget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ConfigError, fingerprint, load_yaml
from .data import DataConfig
from .method import MethodConfig
from .policy import GR00TPolicyConfig, OpenPIPolicyConfig, PolicyConfig, policy_config_from_dict
from .solver import SolverConfig

__all__ = [
    "ConfigError",
    "DataConfig",
    "GR00TPolicyConfig",
    "MethodConfig",
    "OpenPIPolicyConfig",
    "PolicyConfig",
    "RunConfig",
    "SolverConfig",
    "load_config",
]


@dataclass(frozen=True)
class RunConfig:
    name: str
    run_root: Path
    method: MethodConfig
    policy: PolicyConfig
    data: DataConfig
    solver: SolverConfig
    task: str | None
    budget: int

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "method": self.method.to_dict(),
                "policy": self.policy.fingerprint_fields(),
                "data": self.data.to_dict(),
                "task": self.task,
                "budget": self.budget,
                "solver": self.solver.to_dict(),
            }
        )

    @property
    def data_fingerprint(self) -> str:
        return fingerprint(
            {
                "policy": self.policy.fingerprint_fields(),
                "data": self.data.to_dict(),
                "task": self.task,
            }
        )

    @property
    def task_dir(self) -> Path:
        return self.run_root / self.task if self.task else self.run_root

    @property
    def pairs_dir(self) -> Path:
        return self.task_dir / "pairs"

    @property
    def run_dir(self) -> Path:
        return self.task_dir / f"B{self.budget}"

    @property
    def endpoints_dir(self) -> Path:
        return self.run_dir / "endpoints"

    @property
    def schedule_path(self) -> Path:
        return self.run_dir / "schedule.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_root": str(self.run_root),
            "task": self.task,
            "budget": self.budget,
            "fingerprint": self.fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "method": self.method.to_dict(),
            "policy": self.policy.to_dict(),
            "data": self.data.to_dict(),
            "solver": self.solver.to_dict(),
        }


def _section(value: Any, base: Path) -> dict[str, Any]:
    if isinstance(value, str):
        return load_yaml(base / value)
    if isinstance(value, dict):
        return dict(value)
    raise ConfigError("each of method, policy, data, and solver must be a YAML path or mapping")


def load_config(
    path: str | Path,
    *,
    task: str | None = None,
    budget: int = 4,
    run_root: str | Path | None = None,
    device: str | None = None,
    micro_batch: int | None = None,
) -> RunConfig:
    path = Path(path).expanduser().resolve()
    payload = load_yaml(path)
    allowed = {"name", "run_root", "method", "policy", "data", "solver"}
    if set(payload) - allowed:
        raise ConfigError(f"unknown config keys: {sorted(set(payload) - allowed)}")
    blocks = {
        name: _section(payload[name], path.parent)
        for name in ("method", "policy", "data", "solver")
    }
    if task is not None and (not task or Path(task).name != task or task == ".."):
        raise ConfigError("task must be a directory name")
    if device:
        blocks["policy"]["device"] = device
    if micro_batch is not None:
        blocks["solver"]["micro_batch"] = micro_batch
    root = Path(run_root or payload.get("run_root", "runs"))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    if budget < 2:
        raise ConfigError("budget must be at least two")
    try:
        return RunConfig(
            str(payload.get("name", path.stem)),
            root,
            MethodConfig.from_dict(blocks["method"]),
            policy_config_from_dict(blocks["policy"]),
            DataConfig.from_dict(blocks["data"]).for_task(task),
            SolverConfig.from_dict(blocks["solver"]),
            task,
            int(budget),
        )
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
