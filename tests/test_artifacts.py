"""Calibration provenance, cache isolation, and numerical input validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from gys.config import MethodConfig, SolverConfig
from gys.core import validate_nodes
from gys.data.calibration import build_pair_set, load_pairs
from gys.run import EndpointBank, read_json


class Policy:
    def metadata(self):
        return {"family": "test"}

    def prepare_context(self, sample):
        return {
            "gt_model": np.zeros((32, 14)),
            "noise_seed": 0,
            "obs_summary": np.arange(4, dtype=float),
        }


def samples():
    return [SimpleNamespace(record=SimpleNamespace(sample_id="sample", episode_index=0))]


def test_pair_resume_rejects_different_checkpoint_and_tampering(tmp_path):
    manifest = build_pair_set(Policy(), samples(), tmp_path, data_fingerprint="checkpoint-a")
    assert manifest == read_json(tmp_path / "manifest.json")
    original = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="different checkpoint"):
        build_pair_set(Policy(), samples(), tmp_path, data_fingerprint="checkpoint-b")
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert len(load_pairs(tmp_path)) == 1
    (tmp_path / "sample.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pairs(tmp_path)


def test_pair_manifest_controls_loading_and_overwrite(tmp_path):
    build_pair_set(Policy(), samples(), tmp_path, data_fingerprint="a")
    (tmp_path / "stale.npz").write_bytes(b"not part of the manifest")
    build_pair_set(Policy(), samples(), tmp_path, data_fingerprint="b", overwrite=True)
    assert [p.sample_id for p in load_pairs(tmp_path)] == ["sample"]
    build_pair_set(Policy(), samples(), tmp_path, data_fingerprint="b")
    assert len(load_pairs(tmp_path)) == 1


def test_endpoint_cache_isolated_by_rollout_identity(tmp_path):
    nodes = np.linspace(0, 1, 5)
    bank = EndpointBank(tmp_path, identity="checkpoint-data-noise-a")
    terminal = np.ones((2, 8, 32, 12))
    bank.put(nodes, terminal)
    assert np.array_equal(bank.get(nodes), terminal)
    with pytest.raises(ValueError, match="another run"):
        EndpointBank(tmp_path, identity="checkpoint-data-noise-b")
    bank.path(nodes).write_bytes(b"incomplete")
    assert bank.get(nodes) is None


@pytest.mark.parametrize("nodes", [[0, float("nan"), 1], [0, float("inf"), 1]])
def test_nonfinite_schedule_rejected(nodes):
    with pytest.raises(ValueError, match="finite"):
        validate_nodes(nodes)


def test_invalid_numeric_configuration_rejected():
    with pytest.raises(ValueError):
        SolverConfig(micro_batch=0)
    with pytest.raises(ValueError):
        SolverConfig(particles=24.5)
    with pytest.raises(ValueError):
        MethodConfig(band_high=float("nan"))
