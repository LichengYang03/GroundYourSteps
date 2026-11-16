"""Demonstrations and masked backbone summaries in the policy integration space."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..run.artifacts import read_json, write_json, write_npz
from .records import CalibSample, Pair

__all__ = [
    "PAIR_SCHEMA",
    "build_pair",
    "build_pair_set",
    "load_pairs",
    "read_pair",
    "read_pair_manifest",
    "write_pair",
]

PAIR_SCHEMA = "gys-pair-v1"
MANIFEST_SCHEMA = "gys-pair-manifest-v1"


def build_pair(policy: Any, sample: CalibSample) -> Pair:
    """Put one sample's demonstration into the model's integration space."""
    return _pair_from_context(sample, policy.prepare_context(sample))


def _pair_from_context(sample: CalibSample, context: dict[str, Any]) -> Pair:
    gt_model = context.get("gt_model")
    if gt_model is None:
        raise ValueError(f"{sample.record.sample_id}: missing demonstration in model coordinates")
    extras: dict[str, Any] = {"noise_seed": int(context["noise_seed"])}
    summary = context.get("obs_summary")
    if summary is None:
        raise ValueError("the policy must provide a masked final-backbone summary")
    extras["obs_vector"] = np.asarray(summary, dtype=np.float32).reshape(-1)
    extras["obs_source"] = "prefix_hidden"
    return Pair(
        sample_id=sample.record.sample_id,
        episode_index=int(sample.record.episode_index),
        y_model=np.asarray(gt_model, dtype=np.float32),
        extras=extras,
    )


def write_pair(directory: str | Path, pair: Pair) -> str:
    """Serialise one pair to ``<directory>/<sample_id>.npz``; returns its content hash."""
    arrays: dict[str, Any] = {
        "schema": np.asarray(PAIR_SCHEMA),
        "sample_id": np.asarray(pair.sample_id),
        "episode_index": np.asarray(pair.episode_index, dtype=np.int64),
        "noise_seed": np.asarray(int(pair.extras.get("noise_seed", -1)), dtype=np.int64),
        "y_model": np.asarray(pair.y_model, dtype=np.float32),
    }
    obs = pair.extras.get("obs_vector")
    if obs is not None:
        arrays["obs_vector"] = np.asarray(obs, dtype=np.float32)
        arrays["obs_source"] = np.asarray(str(pair.extras.get("obs_source", "")))
    return write_npz(Path(directory) / f"{pair.sample_id}.npz", **arrays)


def read_pair(path: str | Path) -> Pair:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["schema"]) != PAIR_SCHEMA:
            raise ValueError(f"{path}: unsupported pair schema")
        if str(payload["obs_source"]) != "prefix_hidden":
            raise ValueError(f"{path}: expected final-backbone summary")
        return Pair(
            sample_id=str(payload["sample_id"]),
            episode_index=int(payload["episode_index"]),
            y_model=payload["y_model"],
            extras={
                "noise_seed": int(payload["noise_seed"]),
                "obs_vector": payload["obs_vector"],
                "obs_source": "prefix_hidden",
            },
        )


def _pair_path(directory: Path, entry: dict[str, Any]) -> Path:
    sample_id = entry["sample_id"]
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("invalid sample id in calibration manifest")
    path = directory / f"{sample_id}.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        raise ValueError(f"{path}: calibration content hash mismatch; rebuild pairs")
    return path


def load_pairs(directory: str | Path, *, limit: int = 0) -> list[Pair]:
    directory = Path(directory)
    manifest = read_pair_manifest(directory)
    entries = manifest["files"]
    if not manifest.get("complete") or not entries:
        raise ValueError("calibration set is incomplete; run gys pairs first")
    if limit:
        entries = entries[:limit]
    pairs = []
    for entry in entries:
        pair = read_pair(_pair_path(directory, entry))
        if pair.sample_id != entry["sample_id"]:
            raise ValueError("pair sample id disagrees with its manifest")
        pairs.append(pair)
    return pairs


def read_pair_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; rebuild the pairs")
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported calibration manifest schema")
    entries = manifest["files"]
    ids = [entry["sample_id"] for entry in entries]
    if len(ids) != len(set(ids)) or len(ids) != manifest["num_pairs"]:
        raise ValueError("inconsistent calibration manifest")
    return manifest


def build_pair_set(
    policy: Any,
    samples: Iterable[CalibSample],
    out_dir: str | Path,
    *,
    data_fingerprint: str,
    dataset_manifest: dict[str, Any] | None = None,
    overwrite: bool = False,
    progress: Any = None,
    get_context: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and persist the whole target set; returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list(samples)
    ids = [sample.record.sample_id for sample in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("calibration sample ids must be nonempty and unique")
    existing = {}
    manifest_path = out_dir / "manifest.json"
    if not overwrite:
        if manifest_path.exists():
            previous = assert_pairs_match(out_dir, data_fingerprint)
            if previous.get("dataset") != (dataset_manifest or {}):
                raise ValueError("dataset selection changed; rebuild pairs with --overwrite")
            if previous["complete"] and ids == [entry["sample_id"] for entry in previous["files"]]:
                for entry in previous["files"]:
                    _pair_path(out_dir, entry)
                return previous
            existing = {entry["sample_id"]: entry for entry in previous["files"]}
        elif any(out_dir.glob("*.npz")):
            raise ValueError("untracked calibration files; rebuild pairs with --overwrite")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "data_fingerprint": data_fingerprint,
        "policy": policy.metadata(),
        "num_pairs": 0,
        "complete": False,
        "dataset": dataset_manifest or {},
        "files": [],
    }
    write_json(manifest_path, manifest)
    for position, sample in enumerate(samples, start=1):
        sample_id = sample.record.sample_id
        if Path(sample_id).name != sample_id:
            raise ValueError("sample ids must be plain filenames")
        entry = existing.get(sample_id)
        if entry is not None:
            _pair_path(out_dir, entry)
        else:
            context = get_context(sample_id) if get_context else policy.prepare_context(sample)
            pair = _pair_from_context(sample, context)
            entry = {"sample_id": pair.sample_id, "sha256": write_pair(out_dir, pair)}
        manifest["files"].append(entry)
        manifest["num_pairs"] = position
        write_json(manifest_path, manifest)
        if progress is not None:
            progress(f"[pairs] {position}/{len(samples)}  {sample_id}")
    manifest["complete"] = True
    manifest["sha256"] = write_json(manifest_path, manifest)
    return manifest


def assert_pairs_match(directory: str | Path, data_fingerprint: str) -> dict[str, Any]:
    """Check that the calibration set matches the policy and dataset."""
    manifest = read_pair_manifest(directory)
    stored = manifest.get("data_fingerprint")
    if stored != data_fingerprint:
        raise ValueError(
            f"{directory} contains pairs for a different checkpoint or dataset. "
            "Rebuild with gys pairs --overwrite."
        )
    return manifest
