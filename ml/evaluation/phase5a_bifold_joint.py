"""Validation-only Phase 5A official BIFOLD S1+S2 baseline and audit."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from pydantic import Field, model_validator

from ml.evaluation.phase4e_bifold_baseline import (
    BIGEARTHNET_19_CLASS_ORDER,
    BifoldJointInference,
    MultilabelMetrics,
    Phase4DGatePaths,
    Phase4DNotReadyError,
    Phase4EPrediction,
    ValidationBatch,
    _deterministic_inference,
    assert_phase4d_ready,
    compute_multilabel_metrics,
    logits_to_probabilities,
    probabilities_to_predictions,
)
from satquery.ingestion.models import ContractModel
from satquery.inference.multisensor_preprocessing import SealedTestAccessError

FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
)
JOINT_MODEL_ID = "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0"
JOINT_MODEL_REVISION = "762acdc186cce6b31ecbc896ddab1721847d4c5e"
JOINT_CHECKPOINT_SHA256 = (
    "b7e5e58d2cf1e384ba9362ac6ce75c117dee159198eac5675927cba3c3960332"
)
JOINT_PREPROCESSING_PROFILE = "bifold_resnet50_all_v020"
JOINT_BAND_ORDER = (
    "VV",
    "VH",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)


class Phase5AProvenance(ContractModel):
    """Execution identity for the official, non-adapted joint baseline."""

    schema_version: Literal[1] = 1
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_worktree: bool
    kaggle_experiment: Literal["phase5a-bifold-joint-validation"]
    kaggle_kernel: Literal["satquery-phase5a-bifold-joint-validation"]
    runtime_seconds: float = Field(gt=0)
    device: str = Field(min_length=1)
    cuda_version: str | None
    torch_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    model_id: Literal[JOINT_MODEL_ID] = JOINT_MODEL_ID
    model_revision: Literal[JOINT_MODEL_REVISION]
    checkpoint_sha256: Literal[JOINT_CHECKPOINT_SHA256]
    s1_materialized_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    s2_materialized_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: Literal[FROZEN_MANIFEST_SHA256]
    preprocessing_profile: Literal[JOINT_PREPROCESSING_PROFILE]
    threshold: Literal[0.5] = 0.5
    evaluation_split: Literal["validation"] = "validation"
    class_order: tuple[str, ...] = BIGEARTHNET_19_CLASS_ORDER
    test_accessed: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_class_order(self) -> Phase5AProvenance:
        if self.class_order != BIGEARTHNET_19_CLASS_ORDER:
            raise ValueError("Phase 5A provenance must use the canonical class order")
        return self


class Phase5AResult(ContractModel):
    """The compact joint-validation artifact written by the Phase 5A notebook."""

    schema_version: Literal[1] = 1
    status: Literal["VALIDATION_EVALUATED"] = "VALIDATION_EVALUATED"
    provenance: Phase5AProvenance
    metrics: MultilabelMetrics
    prediction_count: int = Field(ge=0)
    prediction_artifact: Literal["phase5a_joint_validation_predictions.jsonl"] = (
        "phase5a_joint_validation_predictions.jsonl"
    )
    prediction_format: Literal["jsonl"] = "jsonl"
    logits_and_probabilities_preserved: Literal[True] = True


def require_phase5a_validation_split(split: str) -> None:
    """Preserve the Phase 4 test seal for the controlled joint baseline."""

    if split == "test":
        raise SealedTestAccessError(
            "Phase 4 test access is sealed; Phase 5A accepts validation only"
        )
    if split != "validation":
        raise ValueError("Phase 5A accepts validation only")


def assert_joint_band_order(band_order: Sequence[str]) -> None:
    """Reject any channel permutation or non-model native band."""

    if tuple(band_order) != JOINT_BAND_ORDER:
        raise ValueError("Phase 5A joint semantic channel order does not match")


def assert_official_class_order(class_order: Sequence[str]) -> None:
    """Ensure the external classifier's logits retain the official semantics."""

    if tuple(class_order) != BIGEARTHNET_19_CLASS_ORDER:
        raise ValueError("Phase 5A official class order does not match")


def assert_official_joint_checkpoint(
    model_id: str, revision: str, checkpoint_sha256: str
) -> None:
    """Refuse adapted, S1-only, or S2-only checkpoints in the control arm."""

    if (model_id, revision, checkpoint_sha256) != (
        JOINT_MODEL_ID,
        JOINT_MODEL_REVISION,
        JOINT_CHECKPOINT_SHA256,
    ):
        raise ValueError("Phase 5A requires the exact official joint checkpoint")


def assert_joint_package_provenance(
    readiness: Mapping[str, Any],
    expected_s1_sha256: str,
    expected_s2_sha256: str,
) -> None:
    """Require both independently verified materialization package identities."""

    modalities = readiness.get("modalities")
    if not isinstance(modalities, Mapping):
        raise ValueError("Phase 5A joint package provenance is missing")
    s1, s2 = modalities.get("s1"), modalities.get("s2")
    if not isinstance(s1, Mapping) or s1.get("package_sha256") != expected_s1_sha256:
        raise ValueError("Phase 5A S1 package provenance does not match")
    if not isinstance(s2, Mapping) or s2.get("package_sha256") != expected_s2_sha256:
        raise ValueError("Phase 5A S2 package provenance does not match")


def evaluate_joint_validation_batches(
    batches: Iterable[ValidationBatch],
    wrapper: BifoldJointInference,
    *,
    provenance: Phase5AProvenance,
    gate_paths: Phase4DGatePaths,
) -> tuple[MultilabelMetrics, tuple[Phase4EPrediction, ...]]:
    """Evaluate only the official 12-channel checkpoint after both package gates."""

    require_phase5a_validation_split(provenance.evaluation_split)
    assert_official_joint_checkpoint(
        wrapper.registration.model_id,
        wrapper.registration.revision,
        wrapper.registration.checkpoint_sha256,
    )
    assert_joint_band_order(wrapper.profile.band_order)
    assert_phase4d_ready(
        gate_paths.readiness,
        gate_paths.raster_audit,
        gate_paths.preprocessing_contract,
        gate_paths.manifest,
    )
    readiness = _read_json(gate_paths.readiness, "Phase 4 materialization readiness")
    if provenance.manifest_sha256 != _sha256(gate_paths.manifest):
        raise Phase4DNotReadyError("Phase 5A provenance manifest does not match")
    assert_joint_package_provenance(
        readiness,
        provenance.s1_materialized_package_sha256,
        provenance.s2_materialized_package_sha256,
    )
    with _deterministic_inference():
        return _collect_joint_predictions(batches, wrapper, threshold=provenance.threshold)


def write_joint_validation_artifacts(
    output_dir: Path,
    *,
    provenance: Phase5AProvenance,
    metrics: MultilabelMetrics,
    predictions: Sequence[Phase4EPrediction],
) -> Phase5AResult:
    """Write the exact configured Phase 5A result filenames once."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "phase5a_joint_validation_predictions.jsonl"
    result_path = output_dir / "phase5a_joint_validation_result.json"
    if prediction_path.exists() or result_path.exists():
        raise FileExistsError("Phase 5A joint validation artifacts already exist")
    prediction_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in predictions), encoding="utf-8"
    )
    result = Phase5AResult(
        provenance=provenance,
        metrics=metrics,
        prediction_count=len(predictions),
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def build_complementarity_report(
    *,
    s1_predictions: Sequence[Phase4EPrediction],
    s2_predictions: Sequence[Phase4EPrediction],
    joint_predictions: Sequence[Phase4EPrediction],
    metrics: Mapping[str, MultilabelMetrics],
) -> dict[str, Any]:
    """Compare all official outputs without causal claims about either sensor."""

    ordered = {"s1": s1_predictions, "s2": s2_predictions, "joint": joint_predictions}
    _validate_complementarity_inputs(ordered, metrics)
    targets = _targets(s1_predictions)
    predicted = {name: _predictions(rows) for name, rows in ordered.items()}
    per_class = _per_class_report(metrics)
    s1_s2 = _paired_correctness(targets, predicted["s1"], predicted["s2"])
    joint_s2 = _paired_correctness(targets, predicted["joint"], predicted["s2"])

    return {
        "schema_version": 1,
        "status": "COMPLEMENTARITY_ANALYZED",
        "scope": "frozen_validation_only",
        "sample_count": int(targets.shape[0]),
        "class_order": list(BIGEARTHNET_19_CLASS_ORDER),
        "threshold": 0.5,
        "global_metrics": _global_metrics(metrics),
        "per_class": per_class,
        "classes_where_joint_improves_over_s2_ap": [
            row["class_name"] for row in per_class if row["joint_minus_s2_ap"] > 0
        ],
        "classes_where_joint_degrades_vs_s2_ap": [
            row["class_name"] for row in per_class if row["joint_minus_s2_ap"] < 0
        ],
        "classes_where_s1_exceeds_s2_ap": [
            row["class_name"] for row in per_class if row["s1_ap"] > row["s2_ap"]
        ],
        "classes_where_joint_exceeds_both_ap": [
            row["class_name"]
            for row in per_class
            if row["joint_ap"] > row["s1_ap"] and row["joint_ap"] > row["s2_ap"]
        ],
        "paired_label_correctness": {
            "s1_s2": {
                "s1_correct_s2_correct": s1_s2["left_correct_right_correct"],
                "s1_correct_s2_wrong": s1_s2["left_correct_right_wrong"],
                "s1_wrong_s2_correct": s1_s2["left_wrong_right_correct"],
                "s1_wrong_s2_wrong": s1_s2["left_wrong_right_wrong"],
            },
            "joint_s2": {
                "joint_rescue_over_s2": joint_s2["left_correct_right_wrong"],
                "joint_harm_vs_s2": joint_s2["left_wrong_right_correct"],
                "both_correct": joint_s2["left_correct_right_correct"],
                "both_wrong": joint_s2["left_wrong_right_wrong"],
            },
        },
        "sample_level_score": {
            "definition": "Per-sample multilabel F1 over the 19 threshold-0.5 binary decisions; empty-target/empty-prediction samples score 0.",
            "s1_vs_s2": _sample_score_comparison(
                targets, predicted["s1"], predicted["s2"], "s1", "s2"
            ),
            "joint_vs_s2": _sample_score_comparison(
                targets, predicted["joint"], predicted["s2"], "joint", "s2"
            ),
        },
        "interpretation_caution": "A joint-model rescue is associated with additional S1 input; output comparisons alone do not establish that SAR caused a correction.",
    }


def load_prediction_artifact(path: Path) -> tuple[Phase4EPrediction, ...]:
    """Load a compact frozen JSONL prediction artifact without raster access."""

    try:
        rows = tuple(
            Phase4EPrediction.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load prediction artifact: {path}") from exc
    if not rows:
        raise ValueError("Prediction artifact is empty")
    return rows


def _validate_complementarity_inputs(
    predictions: Mapping[str, Sequence[Phase4EPrediction]],
    metrics: Mapping[str, MultilabelMetrics],
) -> None:
    if set(predictions) != {"s1", "s2", "joint"} or set(metrics) != {
        "s1",
        "s2",
        "joint",
    }:
        raise ValueError("Complementarity requires S1, S2, and joint predictions/metrics")
    counts = {name: len(rows) for name, rows in predictions.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise ValueError("Complementarity prediction counts must be equal and non-zero")
    reference = predictions["s1"]
    reference_ids = tuple(row.sample_id for row in reference)
    reference_targets = tuple(row.target_indices for row in reference)
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Complementarity sample IDs must be unique")
    for name, rows in predictions.items():
        if tuple(row.sample_id for row in rows) != reference_ids:
            raise ValueError(f"Complementarity sample IDs differ for {name}")
        if tuple(row.target_indices for row in rows) != reference_targets:
            raise ValueError(f"Complementarity targets differ for {name}")
        metric = metrics[name]
        if (
            metric.sample_count != len(rows)
            or metric.class_order != BIGEARTHNET_19_CLASS_ORDER
            or metric.threshold != 0.5
        ):
            raise ValueError(f"Complementarity metric contract differs for {name}")


def _targets(rows: Sequence[Phase4EPrediction]) -> np.ndarray:
    matrix = np.zeros((len(rows), 19), dtype=bool)
    for row_index, row in enumerate(rows):
        matrix[row_index, list(row.target_indices)] = True
    return matrix


def _predictions(rows: Sequence[Phase4EPrediction]) -> np.ndarray:
    matrix = np.zeros((len(rows), 19), dtype=bool)
    for row_index, row in enumerate(rows):
        matrix[row_index, list(row.predicted_indices)] = True
    return matrix


def _global_metrics(metrics: Mapping[str, MultilabelMetrics]) -> dict[str, Any]:
    def summary(metric: MultilabelMetrics) -> dict[str, float]:
        return {
            "macro_average_precision": metric.macro_average_precision,
            "micro_f1": metric.micro_f1,
            "macro_f1": metric.macro_f1,
        }

    s1, s2, joint = (metrics[name] for name in ("s1", "s2", "joint"))
    return {
        "s1": summary(s1),
        "s2": summary(s2),
        "joint": summary(joint),
        "joint_minus_s2": _metric_deltas(joint, s2),
        "joint_minus_s1": _metric_deltas(joint, s1),
    }


def _metric_deltas(
    left: MultilabelMetrics, right: MultilabelMetrics
) -> dict[str, float]:
    return {
        "macro_average_precision": left.macro_average_precision
        - right.macro_average_precision,
        "micro_f1": left.micro_f1 - right.micro_f1,
        "macro_f1": left.macro_f1 - right.macro_f1,
    }


def _per_class_report(metrics: Mapping[str, MultilabelMetrics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(BIGEARTHNET_19_CLASS_ORDER):
        s1, s2, joint = (metrics[name] for name in ("s1", "s2", "joint"))
        rows.append(
            {
                "class_name": class_name,
                "s1_ap": s1.per_class_average_precision[index],
                "s2_ap": s2.per_class_average_precision[index],
                "joint_ap": joint.per_class_average_precision[index],
                "joint_minus_s2_ap": joint.per_class_average_precision[index]
                - s2.per_class_average_precision[index],
                "joint_minus_s1_ap": joint.per_class_average_precision[index]
                - s1.per_class_average_precision[index],
                "s1_f1": s1.per_class_f1[index],
                "s2_f1": s2.per_class_f1[index],
                "joint_f1": joint.per_class_f1[index],
                "joint_minus_s2_f1": joint.per_class_f1[index]
                - s2.per_class_f1[index],
                "joint_minus_s1_f1": joint.per_class_f1[index]
                - s1.per_class_f1[index],
            }
        )
    return rows


def _paired_correctness(
    targets: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, int]:
    left_correct = left == targets
    right_correct = right == targets
    return {
        "left_correct_right_correct": int(np.sum(left_correct & right_correct)),
        "left_correct_right_wrong": int(np.sum(left_correct & ~right_correct)),
        "left_wrong_right_correct": int(np.sum(~left_correct & right_correct)),
        "left_wrong_right_wrong": int(np.sum(~left_correct & ~right_correct)),
    }


def _sample_score_comparison(
    targets: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> dict[str, int]:
    left_scores = _sample_f1(targets, left)
    right_scores = _sample_f1(targets, right)
    return {
        f"{left_name}_score_gt_{right_name}": int(np.sum(left_scores > right_scores)),
        f"{right_name}_score_gt_{left_name}": int(np.sum(right_scores > left_scores)),
        "tie": int(np.sum(left_scores == right_scores)),
    }


def _sample_f1(targets: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true_positive = np.sum(targets & predicted, axis=1)
    false_positive = np.sum(~targets & predicted, axis=1)
    false_negative = np.sum(targets & ~predicted, axis=1)
    denominator = 2 * true_positive + false_positive + false_negative
    return np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def _collect_joint_predictions(
    batches: Iterable[ValidationBatch],
    wrapper: BifoldJointInference,
    *,
    threshold: float,
) -> tuple[MultilabelMetrics, tuple[Phase4EPrediction, ...]]:
    rows: list[Phase4EPrediction] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in batches:
        if len(batch.sample_ids) != batch.inputs.shape[0] or batch.targets.shape != (
            batch.inputs.shape[0],
            19,
        ):
            raise ValueError("Joint validation batch identities, inputs, and targets do not align")
        logits = wrapper.predict_logits(batch.inputs)
        probs = logits_to_probabilities(logits)
        predicted = probabilities_to_predictions(probs, threshold=threshold)
        targets_cpu = batch.targets.detach().to(dtype=torch.bool, device="cpu")
        for index, sample_id in enumerate(batch.sample_ids):
            rows.append(
                Phase4EPrediction(
                    sample_id=sample_id,
                    target_indices=tuple(
                        int(value) for value in torch.where(targets_cpu[index])[0]
                    ),
                    logits=tuple(float(value) for value in logits[index]),
                    probabilities=tuple(float(value) for value in probs[index]),
                    predicted_indices=tuple(
                        int(value) for value in torch.where(predicted[index])[0]
                    ),
                )
            )
        targets.append(targets_cpu.numpy())
        probabilities.append(probs.numpy())
    if not rows:
        raise ValueError("Joint validation iterator produced no samples")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Joint validation iterator contains duplicate sample IDs")
    return (
        compute_multilabel_metrics(
            np.concatenate(targets), np.concatenate(probabilities), threshold=threshold
        ),
        tuple(rows),
    )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4DNotReadyError(f"Cannot read {description}") from exc
    if not isinstance(payload, dict):
        raise Phase4DNotReadyError(f"{description} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
