from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ml.evaluation.phase4_contracts import (
    CLOSEOUT_PATH,
    load_closeout,
    load_phase4_contracts,
    sha256_file,
)

ROOT = Path("experiments/phase4_temporal_analytics")


def _copy_contracts(tmp_path: Path) -> Path:
    for name in ("dataset_contracts.yaml", "model_contracts.yaml", "experiment_plan.yaml"):
        target = tmp_path / name
        target.write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_blocked_contract_is_valid_but_not_runnable(tmp_path: Path) -> None:
    contracts = load_phase4_contracts(_copy_contracts(tmp_path))

    assert contracts.models["chg2cap_levircc"].runnable is False
    assert contracts.models["chg2cap_levircc"].blockers


def test_pass_model_requires_verified_checkpoint_and_license(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "model_contracts.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = payload["models"]["changerex_r18_levircd"]
    model["status"] = "PASS"
    model["source"]["license"] = None
    model["checkpoint"]["locally_verified_sha256"] = None
    model["blockers"] = []
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError), match="license|SHA-256"):
        load_phase4_contracts(root)


def test_unknown_contract_file_key_is_rejected(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "dataset_contracts.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError), match="unexpected"):
        load_phase4_contracts(root)


def test_phase4_closeout_references_existing_hash_matching_artifacts() -> None:
    closeout = load_closeout(CLOSEOUT_PATH)
    for artifact in closeout.artifacts:
        assert artifact.path.is_file()
        assert sha256_file(artifact.path) == artifact.sha256


def test_supported_capability_has_nonzero_measured_evidence() -> None:
    closeout = load_closeout(CLOSEOUT_PATH)
    for capability in closeout.capabilities:
        if capability.status.startswith("SUPPORTED"):
            assert capability.sample_count > 0
            assert capability.metrics


def test_temporal_registry_rejects_missing_domain_contract(tmp_path: Path) -> None:
    from satquery.registry.models import load_model_registry

    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "models": {
                    "change": {
                        "task": "structural_change_segmentation",
                        "provider": "huggingface",
                        "model_id": "org/change",
                        "revision": "0" * 40,
                        "checkpoint_file": "model.pth",
                        "checkpoint_sha256": "0" * 64,
                        "checkpoint_size_bytes": 1,
                        "architecture": "test",
                        "license": "MIT",
                        "preprocessing_profile": "change_v1",
                        "supported_modalities": ["optical_rgb"],
                        "input_shape": [2, 3, 512, 512],
                        "output_semantics": "binary_change",
                        "frozen": True,
                        "allow_remote_code": False,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="training_domain"):
        load_model_registry(path)
