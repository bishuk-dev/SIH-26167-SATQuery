from __future__ import annotations

import importlib.util

import pytest
import torch

from ml.evaluation import phase5a_bifold_joint
from ml.evaluation.phase4e_bifold_baseline import (
    BIGEARTHNET_19_CLASS_ORDER,
    MultilabelMetrics,
    Phase4EPrediction,
)
from satquery.inference.multisensor_preprocessing import SealedTestAccessError


def test_phase5a_joint_evaluator_module_exists() -> None:
    assert importlib.util.find_spec("ml.evaluation.phase5a_bifold_joint") is not None


def _metrics(value: float) -> MultilabelMetrics:
    return MultilabelMetrics(
        sample_count=2,
        class_order=BIGEARTHNET_19_CLASS_ORDER,
        threshold=0.5,
        micro_f1=value,
        macro_f1=value,
        per_class_f1=(value,) * 19,
        macro_average_precision=value,
        per_class_average_precision=(value,) * 19,
        class_prevalence=(0.5,) * 19,
    )


def _prediction(sample_id: str, target: tuple[int, ...], predicted: tuple[int, ...]) -> Phase4EPrediction:
    probabilities = tuple(0.9 if index in predicted else 0.1 for index in range(19))
    return Phase4EPrediction(
        sample_id=sample_id,
        target_indices=target,
        logits=probabilities,
        probabilities=probabilities,
        predicted_indices=predicted,
    )


def test_joint_contract_requires_exact_12_semantic_bands() -> None:
    phase5a_bifold_joint.assert_joint_band_order(
        ("VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    )

    with pytest.raises(ValueError, match="semantic channel order"):
        phase5a_bifold_joint.assert_joint_band_order(
            ("VH", "VV", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
        )

    phase5a_bifold_joint.assert_official_class_order(BIGEARTHNET_19_CLASS_ORDER)
    with pytest.raises(ValueError, match="class order"):
        phase5a_bifold_joint.assert_official_class_order(
            tuple(reversed(BIGEARTHNET_19_CLASS_ORDER))
        )


class _FixedJointModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape[0], 19)


def test_joint_wrapper_requires_exactly_12_channels() -> None:
    wrapper = phase5a_bifold_joint.BifoldJointInference(_FixedJointModel())
    assert wrapper.predict_logits(torch.zeros(1, 12, 120, 120)).shape == (1, 19)
    with pytest.raises(ValueError, match="channels"):
        wrapper.predict_logits(torch.zeros(1, 10, 120, 120))


def test_phase5a_refuses_test_split() -> None:
    with pytest.raises(SealedTestAccessError, match="sealed"):
        phase5a_bifold_joint.require_phase5a_validation_split("test")


def test_phase5a_refuses_adapted_checkpoint_for_official_baseline() -> None:
    with pytest.raises(ValueError, match="official joint"):
        phase5a_bifold_joint.assert_official_joint_checkpoint(
            "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
            "f5a590cd5876845f62702260ff62ea5a325433bb",
            "f55f8a5416d326c7e3139a97c6866e606dfc2e5cde30d52e2ccc47ce5f8b9be5",
        )


def test_joint_package_provenance_requires_both_verified_hashes() -> None:
    readiness = {
        "modalities": {
            "s1": {"package_sha256": "a" * 64},
            "s2": {"package_sha256": "b" * 64},
        }
    }
    phase5a_bifold_joint.assert_joint_package_provenance(
        readiness, "a" * 64, "b" * 64
    )
    with pytest.raises(ValueError, match="S2 package"):
        phase5a_bifold_joint.assert_joint_package_provenance(
            readiness, "a" * 64, "c" * 64
        )


def test_complementarity_fails_when_paired_targets_differ() -> None:
    s1 = (_prediction("a", (0,), ()), _prediction("b", (), (0,)))
    s2 = (_prediction("a", (0,), (0,)), _prediction("b", (), ()))
    joint = (_prediction("a", (), (0,)), _prediction("b", (), ()))

    with pytest.raises(ValueError, match="targets"):
        phase5a_bifold_joint.build_complementarity_report(
            s1_predictions=s1,
            s2_predictions=s2,
            joint_predictions=joint,
            metrics={"s1": _metrics(0.4), "s2": _metrics(0.5), "joint": _metrics(0.6)},
        )


def test_complementarity_fails_when_prediction_counts_differ() -> None:
    row = _prediction("a", (), ())
    with pytest.raises(ValueError, match="prediction counts"):
        phase5a_bifold_joint.build_complementarity_report(
            s1_predictions=(row,),
            s2_predictions=(row,),
            joint_predictions=(row, row),
            metrics={"s1": _metrics(0.4), "s2": _metrics(0.5), "joint": _metrics(0.6)},
        )


def test_complementarity_counts_label_level_rescue_and_harm() -> None:
    s1 = (_prediction("a", (0,), ()), _prediction("b", (), (0,)))
    s2 = (_prediction("a", (0,), (0,)), _prediction("b", (), ()))
    joint = (_prediction("a", (0,), (0,)), _prediction("b", (), (0,)))

    report = phase5a_bifold_joint.build_complementarity_report(
        s1_predictions=s1,
        s2_predictions=s2,
        joint_predictions=joint,
        metrics={"s1": _metrics(0.4), "s2": _metrics(0.5), "joint": _metrics(0.6)},
    )

    assert report["paired_label_correctness"]["s1_s2"] == {
        "s1_correct_s2_correct": 36,
        "s1_correct_s2_wrong": 0,
        "s1_wrong_s2_correct": 2,
        "s1_wrong_s2_wrong": 0,
    }
    assert report["paired_label_correctness"]["joint_s2"] == {
        "joint_rescue_over_s2": 0,
        "joint_harm_vs_s2": 1,
        "both_correct": 37,
        "both_wrong": 0,
    }
    assert report["global_metrics"]["joint_minus_s2"]["macro_average_precision"] == pytest.approx(0.1)


def _provenance() -> phase5a_bifold_joint.Phase5AProvenance:
    return phase5a_bifold_joint.Phase5AProvenance(
        git_sha="a" * 40,
        dirty_worktree=False,
        kaggle_experiment="phase5a-bifold-joint-validation",
        kaggle_kernel="satquery-phase5a-bifold-joint-validation",
        runtime_seconds=1.0,
        device="Tesla T4",
        cuda_version="12.8",
        torch_version="2.10.0+cu128",
        python_version="3.12.13",
        peak_gpu_memory_bytes=1,
        model_revision="762acdc186cce6b31ecbc896ddab1721847d4c5e",
        checkpoint_sha256="b7e5e58d2cf1e384ba9362ac6ce75c117dee159198eac5675927cba3c3960332",
        s1_materialized_package_sha256="c" * 64,
        s2_materialized_package_sha256="d" * 64,
        manifest_sha256="615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6",
        preprocessing_profile="bifold_resnet50_all_v020",
        test_accessed=False,
    )
def test_phase5a_provenance_serializes_both_packages() -> None:
    assert _provenance().model_id == "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0"


def test_phase5a_writer_uses_the_registered_artifact_names(tmp_path) -> None:
    result = phase5a_bifold_joint.write_joint_validation_artifacts(
        tmp_path,
        provenance=_provenance(),
        metrics=_metrics(0.6),
        predictions=(_prediction("a", (), ()), _prediction("b", (0,), (0,))),
    )

    assert result.prediction_artifact == "phase5a_joint_validation_predictions.jsonl"
    assert (tmp_path / "phase5a_joint_validation_result.json").is_file()
    assert (tmp_path / "phase5a_joint_validation_predictions.jsonl").is_file()


def test_phase5a_loads_json_metrics_with_serialized_lists(tmp_path) -> None:
    result_path = tmp_path / "validation_result.json"
    result_path.write_text(
        __import__("json").dumps({"metrics": _metrics(0.6).model_dump(mode="json")}),
        encoding="utf-8",
    )

    metrics = phase5a_bifold_joint.load_metrics_artifact(result_path)

    assert metrics.class_order == BIGEARTHNET_19_CLASS_ORDER
    assert metrics.per_class_f1 == (0.6,) * 19
