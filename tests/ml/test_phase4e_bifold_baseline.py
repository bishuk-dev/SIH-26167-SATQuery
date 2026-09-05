from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.evaluation.phase4e_bifold_baseline import (
    BIGEARTHNET_19_CLASS_ORDER,
    FROZEN_MANIFEST_SHA256,
    BifoldS1Inference,
    BifoldS2Inference,
    Phase4DGatePaths,
    Phase4DNotReadyError,
    Phase4EProvenance,
    assert_phase4d_ready,
    compute_multilabel_metrics,
    evaluate_validation_batches,
    iter_validation_batches,
    logits_to_probabilities,
    probabilities_to_predictions,
    require_validation_split,
    write_validation_artifacts,
)
from satquery.inference.multisensor_preprocessing import SealedTestAccessError


class _FixedModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.observed_shape: tuple[int, ...] | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.observed_shape = tuple(inputs.shape)
        return self.output.expand(inputs.shape[0], -1)


class _DeterminismAssertingModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert torch.are_deterministic_algorithms_enabled()
        return torch.zeros(inputs.shape[0], 19)


@pytest.mark.parametrize(
    ("wrapper_type", "channels"),
    [(BifoldS1Inference, 2), (BifoldS2Inference, 10)],
)
def test_unimodal_wrappers_enforce_channel_count(wrapper_type, channels: int) -> None:
    model = _FixedModel(torch.zeros(1, 19))
    wrapper = wrapper_type(model)

    logits = wrapper.predict_logits(torch.zeros(3, channels, 120, 120))

    assert model.observed_shape == (3, channels, 120, 120)
    assert logits.shape == (3, 19)
    with pytest.raises(ValueError, match="channels"):
        wrapper.predict_logits(torch.zeros(1, channels + 1, 120, 120))


def test_bigearthnet_class_order_is_exact() -> None:
    assert BIGEARTHNET_19_CLASS_ORDER == (
        "Agro-forestry areas",
        "Arable land",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Coastal wetlands",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Industrial or commercial units",
        "Inland waters",
        "Inland wetlands",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Marine waters",
        "Mixed forest",
        "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas",
        "Pastures",
        "Permanent crops",
        "Transitional woodland, shrub",
        "Urban fabric",
    )


def test_wrapper_rejects_any_output_other_than_19_logits() -> None:
    wrapper = BifoldS1Inference(_FixedModel(torch.zeros(1, 18)))
    with pytest.raises(ValueError, match="19 logits"):
        wrapper.predict_logits(torch.zeros(1, 2, 120, 120))


def test_sigmoid_and_threshold_conversion_is_deterministic() -> None:
    logits = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32)
    first = logits_to_probabilities(logits)
    second = logits_to_probabilities(logits.clone())

    assert torch.equal(first, second)
    assert first.tolist()[0] == pytest.approx(
        [0.11920292, 0.5, 0.88079703], abs=1e-7
    )
    assert probabilities_to_predictions(first, threshold=0.5).tolist() == [
        [False, True, True]
    ]


def test_known_multilabel_metric_fixture() -> None:
    targets = np.tile(np.array([[1], [0], [1], [0]], dtype=np.uint8), (1, 19))
    probabilities = np.tile(
        np.array([[0.9], [0.8], [0.7], [0.1]], dtype=np.float64), (1, 19)
    )

    result = compute_multilabel_metrics(targets, probabilities, threshold=0.75)

    assert result.sample_count == 4
    assert result.micro_f1 == pytest.approx(0.5)
    assert result.macro_f1 == pytest.approx(0.5)
    assert result.macro_average_precision == pytest.approx(5 / 6)
    assert result.per_class_f1 == pytest.approx((0.5,) * 19)
    assert result.per_class_average_precision == pytest.approx((5 / 6,) * 19)
    assert result.class_prevalence == pytest.approx((0.5,) * 19)


def _write_gate_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    manifest = root / "split_manifest.json"
    manifest.write_text('{"frozen":true}', encoding="utf-8")
    import hashlib

    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    readiness = root / "phase4e_readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frozen_manifest_sha256": manifest_sha,
                "modalities": {
                    "s1": {
                        "status": "MATERIALIZED_AND_INTEGRITY_VERIFIED",
                        "expected_compressed_bytes": 54_439_153_171,
                        "observed_compressed_bytes": 54_439_153_171,
                        "publisher_md5": "a55eaa2cdf6a917e296bd6601ec1e348",
                        "selected_member_count": 36_002,
                        "missing_member_count": 0,
                        "duplicate_member_count": 0,
                        "unsafe_rejected_member_count": 0,
                        "package_member_count": 36_002,
                        "package_sha256": "c" * 64,
                    },
                    "s2": {
                        "status": "MATERIALIZED_AND_INTEGRITY_VERIFIED",
                        "expected_compressed_bytes": 63_251_710_377,
                        "observed_compressed_bytes": 63_251_710_377,
                        "publisher_md5": "2245ed2d1a93f6ce637d839bc856396e",
                        "selected_member_count": 216_012,
                        "missing_member_count": 0,
                        "duplicate_member_count": 0,
                        "unsafe_rejected_member_count": 0,
                        "package_member_count": 216_012,
                        "package_sha256": "d" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    audit = root / "representative_raster_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "measured_from_three_predeclared_train_pairs",
                "manifest_sha256": manifest_sha,
                "test_pixels_opened": False,
            }
        ),
        encoding="utf-8",
    )
    preprocessing = root / "bifold_contract.json"
    preprocessing.write_text(
        json.dumps(
            {
                "status": "FROZEN_AFTER_NATIVE_TRAIN_RASTER_AUDIT",
                "manifest_sha256": manifest_sha,
            }
        ),
        encoding="utf-8",
    )
    return readiness, audit, preprocessing, manifest


def test_manifest_mismatch_refuses_evaluation(tmp_path: Path) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    manifest.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(Phase4DNotReadyError, match="manifest"):
        assert_phase4d_ready(readiness, audit, preprocessing, manifest)


def test_phase4d_not_frozen_refuses_evaluation(tmp_path: Path) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    payload = json.loads(preprocessing.read_text(encoding="utf-8"))
    payload["status"] = "native_raster_validation_pending"
    preprocessing.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase4DNotReadyError, match="preprocessing contract"):
        assert_phase4d_ready(readiness, audit, preprocessing, manifest)


def test_verified_status_without_integrity_fields_refuses_evaluation(
    tmp_path: Path,
) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    del payload["modalities"]["s2"]["package_sha256"]
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase4DNotReadyError, match="S2 integrity record"):
        assert_phase4d_ready(readiness, audit, preprocessing, manifest)


def test_test_split_remains_sealed() -> None:
    with pytest.raises(SealedTestAccessError, match="sealed"):
        require_validation_split("test")


def test_provenance_serialization_preserves_pins() -> None:
    provenance = Phase4EProvenance(
        experiment_name="phase4e-bifold-s1-validation",
        git_sha="a" * 40,
        modality="s1",
        model_id="BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
        model_revision="d417b3c32f2172cbceb14e5b106dd9aa7b77c647",
        checkpoint_sha256="87990ba9058bc5413b90d7dc955ed1077cf8c54530db393801336209252b3739",
        preprocessing_profile="bifold_resnet50_s1_v020",
        frozen_manifest_sha256=FROZEN_MANIFEST_SHA256,
        materialized_package_sha256="c" * 64,
        threshold=0.5,
    )

    serialized = json.loads(provenance.model_dump_json())
    assert serialized["evaluation_split"] == "validation"
    assert serialized["model_revision"] == (
        "d417b3c32f2172cbceb14e5b106dd9aa7b77c647"
    )
    assert serialized["frozen_manifest_sha256"] == FROZEN_MANIFEST_SHA256


@dataclass(frozen=True)
class _Batch:
    sample_ids: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor


def _provenance(manifest_sha: str) -> Phase4EProvenance:
    return Phase4EProvenance(
        experiment_name="phase4e-bifold-s1-validation",
        git_sha="a" * 40,
        modality="s1",
        model_id="BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
        model_revision="d417b3c32f2172cbceb14e5b106dd9aa7b77c647",
        checkpoint_sha256="87990ba9058bc5413b90d7dc955ed1077cf8c54530db393801336209252b3739",
        preprocessing_profile="bifold_resnet50_s1_v020",
        frozen_manifest_sha256=manifest_sha,
        materialized_package_sha256="c" * 64,
        threshold=0.5,
    )


def test_evaluator_is_gated_and_preserves_logits_and_probabilities(
    tmp_path: Path,
) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    manifest_sha = json.loads(readiness.read_text())["frozen_manifest_sha256"]
    model = _FixedModel(torch.arange(19, dtype=torch.float32).unsqueeze(0) - 9)
    wrapper = BifoldS1Inference(model)
    targets = torch.zeros(2, 19, dtype=torch.bool)
    targets[0, 0] = True
    targets[1, 18] = True
    batch = _Batch(
        sample_ids=("sample-a", "sample-b"),
        inputs=torch.zeros(2, 2, 120, 120),
        targets=targets,
    )

    metrics, predictions = evaluate_validation_batches(
        [batch],
        wrapper,
        provenance=_provenance(manifest_sha),
        gate_paths=Phase4DGatePaths(
            readiness=readiness,
            raster_audit=audit,
            preprocessing_contract=preprocessing,
            manifest=manifest,
        ),
    )

    assert metrics.sample_count == 2
    assert [row.sample_id for row in predictions] == ["sample-a", "sample-b"]
    assert predictions[0].logits[0] == pytest.approx(-9.0)
    assert predictions[0].probabilities[9] == pytest.approx(0.5)

    result = write_validation_artifacts(
        tmp_path / "output",
        provenance=_provenance(manifest_sha),
        metrics=metrics,
        predictions=predictions,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "output/validation_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.logits_and_probabilities_preserved is True
    assert result.prediction_count == 2
    assert len(rows[0]["logits"]) == 19
    assert len(rows[0]["probabilities"]) == 19


def test_evaluator_checks_gate_before_model_execution(tmp_path: Path) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    manifest.write_text('{"changed":true}', encoding="utf-8")
    model = _FixedModel(torch.zeros(1, 19))
    wrapper = BifoldS1Inference(model)
    batch = _Batch(
        sample_ids=("sample-a",),
        inputs=torch.zeros(1, 2, 120, 120),
        targets=torch.zeros(1, 19, dtype=torch.bool),
    )

    with pytest.raises(Phase4DNotReadyError, match="manifest"):
        evaluate_validation_batches(
            [batch],
            wrapper,
            provenance=_provenance("a" * 64),
            gate_paths=Phase4DGatePaths(
                readiness=readiness,
                raster_audit=audit,
                preprocessing_contract=preprocessing,
                manifest=manifest,
            ),
        )
    assert model.observed_shape is None


def test_validation_batch_loader_uses_manifest_order_and_unimodal_channels(
    tmp_path: Path,
) -> None:
    manifest = {
        "samples": [
            {
                "sample_id": "train-is-ignored",
                "official_split": "train",
                "s1_name": "train-s1",
                "patch_id": "train-s2",
                "labels": [BIGEARTHNET_19_CLASS_ORDER[0]],
            },
            {
                "sample_id": "validation-b",
                "official_split": "validation",
                "s1_name": "s1-b",
                "patch_id": "s2-b",
                "labels": [BIGEARTHNET_19_CLASS_ORDER[18]],
            },
            {
                "sample_id": "validation-a",
                "official_split": "validation",
                "s1_name": "s1-a",
                "patch_id": "s2-a",
                "labels": [BIGEARTHNET_19_CLASS_ORDER[1]],
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    batches = list(
        iter_validation_batches(
            manifest_path=manifest_path,
            dataset_root=tmp_path / "data",
            profile=BifoldS1Inference(_FixedModel(torch.zeros(1, 19))).profile,
            batch_size=2,
            device="cpu",
            band_loader=lambda _path: np.zeros((120, 120), dtype=np.float32),
        )
    )

    assert len(batches) == 1
    assert batches[0].sample_ids == ("validation-a", "validation-b")
    assert batches[0].inputs.shape == (2, 2, 120, 120)
    assert torch.where(batches[0].targets[0])[0].tolist() == [1]
    assert torch.where(batches[0].targets[1])[0].tolist() == [18]


def test_evaluator_enables_deterministic_algorithms_only_for_its_run(
    tmp_path: Path,
) -> None:
    readiness, audit, preprocessing, manifest = _write_gate_fixture(tmp_path)
    manifest_sha = json.loads(readiness.read_text())["frozen_manifest_sha256"]
    wrapper = BifoldS1Inference(_DeterminismAssertingModel())
    batch = _Batch(
        sample_ids=("sample-a",),
        inputs=torch.zeros(1, 2, 120, 120),
        targets=torch.zeros(1, 19, dtype=torch.bool),
    )
    was_enabled = torch.are_deterministic_algorithms_enabled()

    evaluate_validation_batches(
        [batch],
        wrapper,
        provenance=_provenance(manifest_sha),
        gate_paths=Phase4DGatePaths(
            readiness=readiness,
            raster_audit=audit,
            preprocessing_contract=preprocessing,
            manifest=manifest,
        ),
    )

    assert torch.are_deterministic_algorithms_enabled() is was_enabled
