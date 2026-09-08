"""Strict Phase 4 scientific contract loader tests (Task 2 scope only)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ml.evaluation.phase4_contracts import load_phase4_contracts

ROOT = Path("experiments/phase4_temporal_analytics")
AUDIT_FILES = (
    "dataset_contracts.yaml",
    "model_contracts.yaml",
    "experiment_plan.yaml",
)

CAPTION_METRICS = {"BLEU_4", "METEOR", "ROUGE_L", "CIDEr"}


def _copy_contracts(tmp_path: Path) -> Path:
    for name in AUDIT_FILES:
        target = tmp_path / name
        target.write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def _rewrite(root: Path, name: str, payload: dict) -> None:
    (root / name).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _load_model_payload(root: Path) -> dict:
    return yaml.safe_load(
        (root / "model_contracts.yaml").read_text(encoding="utf-8")
    )


def _load_dataset_payload(root: Path) -> dict:
    return yaml.safe_load(
        (root / "dataset_contracts.yaml").read_text(encoding="utf-8")
    )


def _load_plan_payload(root: Path) -> dict:
    return yaml.safe_load(
        (root / "experiment_plan.yaml").read_text(encoding="utf-8")
    )


def _make_pass_model(payload: dict) -> dict:
    """Turn changerex into a fully verified PASS record."""
    model = payload["models"]["changerex_r18_levircd"]
    model["status"] = "PASS"
    model["blockers"] = []
    model["checkpoint"]["locally_verified_sha256"] = "a" * 64
    model["checkpoint"]["license"] = "Apache-2.0"
    model["contract"]["input_shape"] = [2, 3, 512, 512]
    model["dependencies"]["pytorch"] = "2.1.0"
    return payload


def test_current_audit_loads_successfully(tmp_path: Path) -> None:
    contracts = load_phase4_contracts(_copy_contracts(tmp_path))

    assert set(contracts.datasets) == {
        "oscd",
        "sentinel2_l2a_demo",
        "levir_cd",
        "s2looking",
        "levir_cc",
        "sturm_flood",
        "sen1floods11",
        "modified_sen1floods11",
    }
    assert set(contracts.models) == {
        "changerex_r18_levircd",
        "chg2cap_levircc",
        "sturm_sentinel1_unet",
        "rscama_levircc",
    }
    assert set(contracts.experiment_plan.experiments) == {
        "P4-E01",
        "P4-E02",
        "P4-E03",
        "P4-E04",
    }


def test_blocked_contract_is_valid_but_not_runnable(tmp_path: Path) -> None:
    contracts = load_phase4_contracts(_copy_contracts(tmp_path))

    chg2cap = contracts.models["chg2cap_levircc"]
    assert chg2cap.status == "BLOCKED"
    assert chg2cap.blockers
    assert chg2cap.runnable is False
    assert chg2cap.promotable is False


def test_blocked_contract_without_blockers_is_invalid(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_model_payload(root)
    payload["models"]["chg2cap_levircc"]["blockers"] = []
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="blocker"):
        load_phase4_contracts(root)


def _pass_model_root(tmp_path: Path) -> Path:
    root = _copy_contracts(tmp_path)
    payload = _load_model_payload(root)
    _rewrite(root, "model_contracts.yaml", _make_pass_model(payload))
    return root


def test_pass_primary_model_without_sha256_fails(tmp_path: Path) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    payload["models"]["changerex_r18_levircd"]["checkpoint"][
        "locally_verified_sha256"
    ] = None
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="SHA-256"):
        load_phase4_contracts(root)


def test_pass_primary_model_with_malformed_sha256_fails(tmp_path: Path) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    payload["models"]["changerex_r18_levircd"]["checkpoint"][
        "locally_verified_sha256"
    ] = "nothex"
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="SHA-256"):
        load_phase4_contracts(root)


def test_pass_model_without_source_code_license_fails(tmp_path: Path) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    payload["models"]["changerex_r18_levircd"]["source"]["license"] = None
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="license"):
        load_phase4_contracts(root)


def test_pass_model_with_unresolved_checkpoint_license_fails(
    tmp_path: Path,
) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    # checkpoint transport (Hugging Face) differs from the source repository, so a
    # separately resolved checkpoint license is mandatory for PASS
    del payload["models"]["changerex_r18_levircd"]["checkpoint"]["license"]
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="checkpoint license"):
        load_phase4_contracts(root)


def test_pass_model_without_immutable_revision_fails(tmp_path: Path) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    payload["models"]["changerex_r18_levircd"]["source"]["revision"] = "unresolved"
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="revision"):
        load_phase4_contracts(root)


def test_pass_model_without_preprocessing_contract_fails(tmp_path: Path) -> None:
    root = _pass_model_root(tmp_path)
    payload = _load_model_payload(root)
    model = payload["models"]["changerex_r18_levircd"]
    del model["contract"]["data_preprocessor"]
    _rewrite(root, "model_contracts.yaml", payload)

    with pytest.raises(ValueError, match="preprocessing"):
        load_phase4_contracts(root)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_dataset_payload(root)
    payload["datasets"]["oscd"]["guessed_field"] = True
    _rewrite(root, "dataset_contracts.yaml", payload)

    with pytest.raises((ValidationError, ValueError)):
        load_phase4_contracts(root)


def test_unknown_contract_status_rejected(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_dataset_payload(root)
    payload["datasets"]["oscd"]["status"] = "PROBABLY_FINE"
    _rewrite(root, "dataset_contracts.yaml", payload)

    with pytest.raises((ValidationError, ValueError)):
        load_phase4_contracts(root)


def test_robustness_source_cannot_become_development_source(
    tmp_path: Path,
) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E02"]["sample_selection"][
        "development_split"
    ] = "s2looking_split"
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="development"):
        load_phase4_contracts(root)


def test_external_holdout_cannot_become_development_source(
    tmp_path: Path,
) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E04"]["sample_selection"][
        "development_split"
    ] = "sen1floods11_holdout_split"
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="development"):
        load_phase4_contracts(root)


def test_excluded_from_core_cannot_be_runnable(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_dataset_payload(root)
    payload["datasets"]["modified_sen1floods11"]["status"] = "PASS"
    _rewrite(root, "dataset_contracts.yaml", payload)

    with pytest.raises(ValueError, match="excluded_from_core"):
        load_phase4_contracts(root)

    payload = _load_dataset_payload(root)
    payload["datasets"]["modified_sen1floods11"]["status"] = "BLOCKED"
    _rewrite(root, "dataset_contracts.yaml", payload)
    contracts = load_phase4_contracts(root)
    assert contracts.datasets["modified_sen1floods11"].runnable is False


def test_sealed_test_cannot_be_development_split(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_dataset_payload(root)
    payload["datasets"]["levir_cd"]["splits"]["test_sealed"] = False
    _rewrite(root, "dataset_contracts.yaml", payload)

    with pytest.raises(ValueError, match="sealed"):
        load_phase4_contracts(root)


def test_missing_referenced_dataset_id_fails(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E03"]["datasets"] = ["levir_cc", "nonexistent_ds"]
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="nonexistent_ds"):
        load_phase4_contracts(root)


def test_missing_referenced_model_id_fails(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E02"]["model_or_tools"] = [
        "changerex_r18_levircd",
        "ghost_model",
    ]
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="ghost_model"):
        load_phase4_contracts(root)


def test_optional_challenger_cannot_be_experiment_core(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E03"]["model_or_tools"] = [
        "chg2cap_levircc",
        "rscama_levircc",
    ]
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="optional_challenger"):
        load_phase4_contracts(root)


def test_s2looking_blocked_does_not_invalidate_primary_levir_lane(
    tmp_path: Path,
) -> None:
    contracts = load_phase4_contracts(_copy_contracts(tmp_path))

    e02 = contracts.experiment_plan.experiments["P4-E02"]
    assert e02.primary_gate is not None
    assert e02.primary_gate.dataset == "levir_cd"
    assert e02.robustness_lane is not None
    assert e02.robustness_lane.gate_effect.startswith("none")
    assert contracts.datasets["s2looking"].status == "BLOCKED"
    assert contracts.datasets["levir_cd"].role == "primary_benchmark"


def test_conditional_sen1floods11_lane_keeps_sturm_primary_structure(
    tmp_path: Path,
) -> None:
    contracts = load_phase4_contracts(_copy_contracts(tmp_path))

    e04 = contracts.experiment_plan.experiments["P4-E04"]
    assert e04.external_holdout_condition == (
        "conditional_on_audited_input_compatibility"
    )
    assert "sturm_flood" in e04.datasets
    assert contracts.models["sturm_sentinel1_unet"].status == "BLOCKED"


def test_agreement_policy_with_numeric_threshold_rejected(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["shared_policy"]["agreement_policy"] = (
        "Diagnostic; IoU < 0.1 => ABSTAIN, IoU < 0.5 => warn"
    )
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="agreement"):
        load_phase4_contracts(root)


def test_p4_e03_requires_official_caption_metric_names(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_plan_payload(root)
    payload["experiments"]["P4-E03"]["metrics"] = ["accuracy"]
    _rewrite(root, "experiment_plan.yaml", payload)

    with pytest.raises(ValueError, match="caption metric"):
        load_phase4_contracts(root)


def test_loading_contracts_performs_no_registry_mutation(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import (
        DEFAULT_MODEL_REGISTRY,
        DEFAULT_PREPROCESSING_REGISTRY,
    )

    before = (
        DEFAULT_MODEL_REGISTRY.read_bytes(),
        DEFAULT_PREPROCESSING_REGISTRY.read_bytes(),
    )
    load_phase4_contracts(_copy_contracts(tmp_path))
    after = (
        DEFAULT_MODEL_REGISTRY.read_bytes(),
        DEFAULT_PREPROCESSING_REGISTRY.read_bytes(),
    )

    assert before == after


def test_loader_reads_only_the_three_audit_files(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    # an unrelated closeout artifact must be irrelevant to the Task 2 loader
    (root / "PHASE_4_CLOSEOUT.json").write_text("{}", encoding="utf-8")

    contracts = load_phase4_contracts(root)
    assert contracts.experiment_plan.experiments


def test_pass_dataset_requires_license_and_transport(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    payload = _load_dataset_payload(root)
    dataset = payload["datasets"]["levir_cc"]
    dataset["status"] = "PASS"
    dataset["blockers"] = []
    _rewrite(root, "dataset_contracts.yaml", payload)
    contracts = load_phase4_contracts(root)
    assert contracts.datasets["levir_cc"].promotable is True

    payload = _load_dataset_payload(root)
    payload["datasets"]["levir_cc"]["license"] = {"imagery_terms": "only"}
    _rewrite(root, "dataset_contracts.yaml", payload)
    with pytest.raises(ValueError, match="license"):
        load_phase4_contracts(root)


# ---------------------------------------------------------------------------
# Registry schema tests
# ---------------------------------------------------------------------------


def _temporal_registry_yaml(entry: dict) -> str:
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "models": {"test_temporal_model": entry},
        },
        sort_keys=False,
    )


def _temporal_entry(task: str) -> dict:
    base = {
        "task": task,
        "provider": "zenodo",
        "model_id": "org/model",
        "revision": "a" * 40,
        "checkpoint_file": "weights.hdf5",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_size_bytes": 123,
        "architecture": "test-arch",
        "license": "CC-BY-4.0",
        "checkpoint_license": "CC-BY-4.0",
        "preprocessing_profile": "profile_v1",
        "training_domain": "test-domain",
        "supported_modalities": ["sar"],
        "temporal_order": "single_flood_observation",
        "input_shape": [128, 128, 2],
        "output_semantics": "water probability",
        "domain_limitations": ["sentinel-1 only"],
        "frozen": True,
        "allow_remote_code": False,
    }
    if task == "flood_segmentation":
        base.update(
            {
                "required_sensor_names": ["Sentinel-1"],
                "required_polarizations": ["VV", "VH"],
                "required_radiometric_domain": "backscatter_db",
                "expected_resolution_m": 10.0,
            }
        )
    return base


def _write_temporal_registry(tmp_path: Path, entry: dict) -> Path:
    from satquery.registry.models import load_model_registry

    path = tmp_path / "registry.yaml"
    path.write_text(_temporal_registry_yaml(entry), encoding="utf-8")
    return path


def test_existing_phase123_registry_records_still_parse() -> None:
    from satquery.registry.models import (
        load_model_registry,
        load_preprocessing_registry,
    )

    model_registry = load_model_registry()
    preprocessing_registry = load_preprocessing_registry()
    assert model_registry.models
    assert preprocessing_registry.profiles
    temporal_tasks = {
        "structural_change_segmentation",
        "change_captioning",
        "flood_segmentation",
    }
    assert all(model.task not in temporal_tasks for model in model_registry.models.values())


def test_structural_change_registration_rejects_missing_training_domain(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import load_model_registry

    entry = _temporal_entry("structural_change_segmentation")
    entry["supported_modalities"] = ["optical"]
    entry["temporal_order"] = "T1_then_T2"
    del entry["training_domain"]

    with pytest.raises(ValidationError):
        load_model_registry(_write_temporal_registry(tmp_path, entry))


def test_change_caption_registration_rejects_missing_temporal_order(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import load_model_registry

    entry = _temporal_entry("change_captioning")
    entry["supported_modalities"] = ["optical"]
    entry["temporal_order"] = "T2_then_T1_invalid"

    with pytest.raises(ValidationError):
        load_model_registry(_write_temporal_registry(tmp_path, entry))


def test_flood_registration_rejects_missing_sensor_and_radiometric_contract(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import load_model_registry

    entry = _temporal_entry("flood_segmentation")
    del entry["required_sensor_names"]
    del entry["required_radiometric_domain"]

    with pytest.raises(ValidationError):
        load_model_registry(_write_temporal_registry(tmp_path, entry))


def test_flood_registration_rejects_unknown_polarization_order(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import load_model_registry

    entry = _temporal_entry("flood_segmentation")
    entry["required_polarizations"] = ["VH", "VV"]

    with pytest.raises(ValidationError):
        load_model_registry(_write_temporal_registry(tmp_path, entry))


def test_temporal_registration_rejects_unknown_fields(tmp_path: Path) -> None:
    from satquery.registry.models import load_model_registry

    entry = _temporal_entry("flood_segmentation")
    entry["guessed_threshold"] = 0.7

    with pytest.raises(ValidationError):
        load_model_registry(_write_temporal_registry(tmp_path, entry))


def test_valid_temporal_registrations_parse_with_correct_types(
    tmp_path: Path,
) -> None:
    from satquery.registry.models import (
        ChangeCaptionRegistration,
        ChangeDetectionRegistration,
        FloodSegmentationRegistration,
        load_model_registry,
    )

    structural = _temporal_entry("structural_change_segmentation")
    structural["temporal_order"] = "T1_then_T2"
    path = _write_temporal_registry(tmp_path, structural)
    caption_entry = _temporal_entry("change_captioning")
    caption_entry["supported_modalities"] = ["optical"]
    caption_entry["temporal_order"] = "A_pre_then_B_post"
    (tmp_path / "sub").mkdir()
    second = _write_temporal_registry(tmp_path / "sub", caption_entry)

    detected = load_model_registry(path).models["test_temporal_model"]
    captioned = load_model_registry(second).models["test_temporal_model"]
    assert isinstance(detected, ChangeDetectionRegistration)
    assert detected.temporal_order == "T1_then_T2"
    assert isinstance(captioned, ChangeCaptionRegistration)

    (tmp_path / "f").mkdir()
    flood = load_model_registry(
        _write_temporal_registry(tmp_path / "f", _temporal_entry("flood_segmentation"))
    ).models["test_temporal_model"]
    assert isinstance(flood, FloodSegmentationRegistration)
    assert flood.required_polarizations == ("VV", "VH")
    assert flood.expected_resolution_m == 10.0


def test_no_temporal_specialist_registry_entry_exists_yet() -> None:
    import yaml as yaml_module

    from satquery.registry.models import load_model_registry

    registry = load_model_registry()
    temporal_tasks = {
        "structural_change_segmentation",
        "change_captioning",
        "flood_segmentation",
    }
    assert all(model.task not in temporal_tasks for model in registry.models.values())
    raw = yaml_module.safe_load(
        Path("models/registry.yaml").read_text(encoding="utf-8")
    )
    assert raw["models"], "frozen registry entries must remain present"
