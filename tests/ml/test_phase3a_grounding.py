from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.evaluation.prepare_vrsbench_grounding import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    parse_vrsbench_box,
    prepare_subset,
)
from ml.evaluation.run_phase3a_grounding import intersection_over_union
from ml.evaluation.run_phase3b_grounding_calibration import (
    choose_threshold,
    diagnose_semantic_failures,
    summarize_threshold,
    validate_phase3a_anchor,
)


def test_committed_vrsbench_manifest_is_scene_safe_and_reproducible() -> None:
    before = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    manifest = prepare_subset(DEFAULT_MANIFEST, DEFAULT_DATA_ROOT, allow_download=False)
    after = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()

    validation_scenes = {
        sample["source_image_id"]
        for sample in manifest["samples"]
        if sample["split"] == "validation"
    }
    test_scenes = {
        sample["source_image_id"]
        for sample in manifest["samples"]
        if sample["split"] == "test"
    }
    assert manifest["counts"] == {
        "validation": {"scenes": 12, "references": 24},
        "test": {"scenes": 8, "references": 16},
    }
    assert validation_scenes.isdisjoint(test_scenes)
    assert before == after


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{<25><40><33><60>}", [0.25, 0.4, 0.33, 0.6]),
        ("{<0><0><100><100>}", [0.0, 0.0, 1.0, 1.0]),
        ("{<25><40><25><60>}", None),
        ("not-a-box", None),
    ],
)
def test_vrsbench_box_parser(raw: str, expected: list[float] | None) -> None:
    assert parse_vrsbench_box(raw) == expected


def test_grounding_metrics_use_actual_top_prediction() -> None:
    expected = [0.25, 0.25, 0.75, 0.75]
    assert intersection_over_union(expected, expected) == 1.0
    assert intersection_over_union(None, expected) == 0.0
    assert intersection_over_union([0.0, 0.0, 0.5, 0.5], expected) == pytest.approx(
        1 / 7
    )


def test_calibration_keeps_highest_score_instead_of_ground_truth_oracle() -> None:
    records = [
        {
            "sample_id": "sample-1",
            "query": "the ship on the water",
            "object_class": "ship",
            "ground_truth_normalized_xyxy": [0.7, 0.7, 0.9, 0.9],
            "candidates_at_0_15": [
                {
                    "normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "raw_model_score": 0.40,
                    "phrase": "water",
                    "box_area_fraction": 1.0,
                },
                {
                    "normalized_xyxy": [0.7, 0.7, 0.9, 0.9],
                    "raw_model_score": 0.30,
                    "phrase": "ship",
                    "box_area_fraction": 0.04,
                },
            ],
        }
    ]

    row = summarize_threshold(records, 0.25)
    diagnostics = diagnose_semantic_failures(records, 0.25)

    assert row["mean_iou"] == pytest.approx(0.04)
    assert row["acc_at_0_5_iou"] == 0
    assert row["huge_box_count"] == 1
    assert diagnostics["phrase_mismatch_count"] == 1
    assert diagnostics["phrase_mismatches"][0]["model_phrase"] == "water"


def test_threshold_selection_follows_predeclared_metric_order() -> None:
    rows = [
        {
            "box_threshold": 0.20,
            "acc_at_0_5_iou": 0.25,
            "mean_iou": 0.30,
            "no_detection_count": 2,
        },
        {
            "box_threshold": 0.25,
            "acc_at_0_5_iou": 0.50,
            "mean_iou": 0.31,
            "no_detection_count": 4,
        },
        {
            "box_threshold": 0.30,
            "acc_at_0_5_iou": 0.50,
            "mean_iou": 0.35,
            "no_detection_count": 6,
        },
        {
            "box_threshold": 0.35,
            "acc_at_0_5_iou": 0.50,
            "mean_iou": 0.35,
            "no_detection_count": 5,
        },
    ]

    assert choose_threshold(rows)["box_threshold"] == 0.35


def test_phase3b_requires_phase3a_anchor_to_reproduce() -> None:
    matching = {
        "box_threshold": 0.40,
        "mean_iou": 0.1286,
        "acc_at_0_5_iou": 0.1667,
        "no_detection_count": 17,
        "detected_reference_count": 7,
    }
    validate_phase3a_anchor([matching])

    with pytest.raises(RuntimeError, match="detection-count anchor"):
        validate_phase3a_anchor([{**matching, "no_detection_count": 16}])


def test_final_guardrail_artifact_matches_frozen_decision() -> None:
    experiment_root = Path("experiments/phase3b_grounding_threshold_calibration")
    decision = json.loads((experiment_root / "decision.json").read_text("utf-8"))
    artifact_path = experiment_root / decision["final_validation"]["artifact"]
    artifact = json.loads(artifact_path.read_text("utf-8"))

    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert artifact_hash == decision["final_validation"]["artifact_sha256"]
    assert artifact["validation"] == {
        "reference_count": 24,
        "mean_iou": 0.20581098,
        "acc_at_0_5_iou": 0.25,
        "no_detection_count": 9,
        "detected_reference_count": 15,
        "detected_only_mean_iou": 0.32929757,
        "huge_selected_box_count": 0,
    }
    assert artifact["guardrail_accepted"] is True
    assert artifact["phase3_validation_tuning_closed"] is True
    assert artifact["inference_rerun"] is False
    assert artifact["test_split_evaluated"] is False
