from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from ml.evaluation.run_phase3_final_grounding_test import run_final_test
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.grounding import (
    GroundingBackendResult,
    RawGroundingDetection,
)


class GuardrailBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        return GroundingBackendResult(
            image.width,
            image.height,
            (
                RawGroundingDetection(
                    "whole scene", 0.95, 0, 0, image.width, image.height
                ),
                RawGroundingDetection(
                    "target",
                    0.80,
                    image.width * 0.1,
                    image.height * 0.1,
                    image.width * 0.3,
                    image.height * 0.3,
                ),
            ),
        )


class NoCallBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        raise AssertionError("Inference must not run when the policy guard fails")


def _frozen_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    image_path = data_root / "scene.png"
    Image.new("RGB", (512, 512), (40, 80, 120)).save(image_path)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = []
    for scene_index in range(8):
        for reference_index in range(2):
            samples.append(
                {
                    "sample_id": f"sample-{scene_index}-{reference_index}",
                    "scene_id": f"scene-{scene_index}",
                    "source_image_id": "scene.png",
                    "image_path": "unused/scene.png",
                    "image_sha256": image_hash,
                    "query": "the target",
                    "ground_truth_normalized_xyxy": [0.1, 0.1, 0.3, 0.3],
                    "split": "test",
                }
            )
    manifest = {
        "dataset": {"id": "test/frozen"},
        "counts": {"test": {"scenes": 8, "references": 16}},
        "samples": samples,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = json.loads(
        Path("experiments/phase3b_grounding_threshold_calibration/decision.json")
        .read_text(encoding="utf-8")
    )
    # Unit tests exercise the pre-execution guard state without touching the real test.
    decision["test_split_evaluated"] = False
    decision["final_test_executed_once"] = False
    decision.pop("final_test", None)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return manifest_path, data_root, decision_path


def test_final_test_uses_production_guardrail_and_refuses_rerun(
    tmp_path: Path,
) -> None:
    manifest, data_root, decision = _frozen_inputs(tmp_path)
    output = tmp_path / "results"
    clock_values = iter((10.0, 12.5))
    settings = GroundingRuntimeSettings(model_root=tmp_path, device="cpu")

    metrics = run_final_test(
        manifest,
        data_root,
        output,
        decision,
        settings=settings,
        confirmed=True,
        backend=GuardrailBackend(),
        clock=lambda: next(clock_values),
    )

    assert metrics["mean_iou"] == 1.0
    assert metrics["acc_at_0_5_iou"] == 1.0
    assert metrics["no_detection_count"] == 0
    assert metrics["detected_reference_count"] == 16
    assert metrics["huge_selected_box_count"] == 0
    assert metrics["runtime"]["elapsed_seconds"] == 2.5
    predictions = [
        json.loads(line)
        for line in (output / "final_test_predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 16
    assert all(row["raw_model_score"] == 0.80 for row in predictions)
    assert all(row["predicted_normalized_xyxy"] == [0.1, 0.1, 0.3, 0.3] for row in predictions)

    with pytest.raises(FileExistsError, match="refusing rerun"):
        run_final_test(
            manifest,
            data_root,
            output,
            decision,
            settings=settings,
            confirmed=True,
            backend=NoCallBackend(),
        )


def test_final_test_requires_explicit_confirmation(tmp_path: Path) -> None:
    manifest, data_root, decision = _frozen_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="explicit confirmation"):
        run_final_test(
            manifest,
            data_root,
            tmp_path / "results",
            decision,
            settings=GroundingRuntimeSettings(model_root=tmp_path, device="cpu"),
            confirmed=False,
            backend=NoCallBackend(),
        )


def test_final_test_rejects_policy_drift_before_inference(tmp_path: Path) -> None:
    manifest, data_root, decision_path = _frozen_inputs(tmp_path)
    decision = json.loads(decision_path.read_text())
    decision["frozen_production_policy"]["box_threshold"] = 0.31
    decision_path.write_text(json.dumps(decision))

    with pytest.raises(ValueError, match="policy changed"):
        run_final_test(
            manifest,
            data_root,
            tmp_path / "results",
            decision_path,
            settings=GroundingRuntimeSettings(model_root=tmp_path, device="cpu"),
            confirmed=True,
            backend=NoCallBackend(),
        )


def test_completed_final_test_artifacts_match_recorded_decision() -> None:
    experiment = Path("experiments/phase3b_grounding_threshold_calibration")
    decision = json.loads((experiment / "decision.json").read_text(encoding="utf-8"))
    final_test = decision["final_test"]
    metrics_path = experiment / final_test["metrics_artifact"]
    predictions_path = experiment / final_test["predictions_artifact"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == final_test[
        "metrics_artifact_sha256"
    ]
    assert hashlib.sha256(predictions_path.read_bytes()).hexdigest() == final_test[
        "predictions_artifact_sha256"
    ]
    assert metrics["prediction_file_sha256"] == final_test[
        "predictions_artifact_sha256"
    ]
    for key in (
        "mean_iou",
        "acc_at_0_5_iou",
        "no_detection_count",
        "detected_reference_count",
        "detected_only_mean_iou",
        "huge_selected_box_count",
    ):
        assert final_test[key] == metrics[key]
    assert len(predictions) == 16
    assert len({row["scene_id"] for row in predictions}) == 8
    assert decision["test_split_evaluated"] is True
    assert decision["final_test_executed_once"] is True
    assert decision["phase3_validation_tuning_closed"] is True
    assert decision["phase3_status"] == "complete"
