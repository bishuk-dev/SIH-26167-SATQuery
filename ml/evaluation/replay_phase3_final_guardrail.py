"""Replay the frozen grounding guardrail from stored validation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.evaluation.run_phase3a_grounding import intersection_over_union
from satquery.inference.grounding import (
    DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
    GroundingSelectionCandidate,
    select_grounding_candidate,
)
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments/phase3b_grounding_threshold_calibration"
)
DEFAULT_CALIBRATION = EXPERIMENT_ROOT / "results/calibration.json"
DEFAULT_CANDIDATES = EXPERIMENT_ROOT / "results/validation_candidates.jsonl"
DEFAULT_DECISION = EXPERIMENT_ROOT / "decision.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results/final_guardrail_validation.json"


def replay_guardrail(
    calibration_path: Path,
    candidates_path: Path,
    decision_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _validate_sources(calibration, decision, calibration_path, candidates_path)
    registration = load_model_registry().models[DEFAULT_GROUNDING_MODEL_REGISTRY_ID]
    if not isinstance(registration, GroundingModelRegistration):
        raise ValueError("Frozen production grounding registration is invalid")
    profile = load_preprocessing_registry().profiles[
        registration.preprocessing_profile
    ]
    if not isinstance(profile, GroundingPreprocessingProfile):
        raise ValueError("Frozen production grounding profile is invalid")
    if profile.box_threshold != 0.30 or profile.text_threshold != 0.30:
        raise ValueError("Frozen grounding thresholds do not match the final policy")
    if profile.max_normalized_box_area != 0.80:
        raise ValueError("Frozen grounding area guardrail does not match the decision")

    records = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != calibration["sample_count"]:
        raise ValueError("Stored validation candidates are incomplete")

    predictions = []
    for record in records:
        raw_candidates = record["candidates_at_0_15"]
        _validate_candidate_order(raw_candidates)
        candidates = tuple(
            GroundingSelectionCandidate(
                candidate_index=index,
                raw_score=float(candidate["raw_model_score"]),
                normalized_xyxy=tuple(candidate["normalized_xyxy"]),
            )
            for index, candidate in enumerate(raw_candidates)
            if float(candidate["raw_model_score"]) >= profile.box_threshold
        )
        selected = select_grounding_candidate(
            candidates, max_area=profile.max_normalized_box_area
        )
        selected_record = (
            raw_candidates[selected.candidate_index] if selected is not None else None
        )
        expected = record["ground_truth_normalized_xyxy"]
        iou = intersection_over_union(
            selected_record["normalized_xyxy"] if selected_record else None,
            expected,
        )
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "selected": selected_record,
                "iou": round(iou, 8),
                "acc_at_0_5": iou >= 0.5,
                "abstained": selected_record is None,
            }
        )

    metrics = _summarize(predictions, profile.max_normalized_box_area)
    phase3a = next(
        row
        for row in calibration["threshold_results"]
        if row["box_threshold"] == 0.40
    )
    phase3b = calibration["chosen"]
    accepted = (
        metrics["mean_iou"] > phase3b["mean_iou"]
        and metrics["acc_at_0_5_iou"] > phase3b["acc_at_0_5_iou"]
    )
    artifact = {
        "schema_version": 1,
        "phase": "3_final_grounding_validation",
        "split": "validation",
        "inference_rerun": False,
        "test_split_evaluated": False,
        "production_policy": {
            "model_registry_id": DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
            "box_threshold": profile.box_threshold,
            "text_threshold": profile.text_threshold,
            "max_normalized_box_area_exclusive": profile.max_normalized_box_area,
            "selection": "highest raw model score among remaining detections",
            "empty_selection": "abstain with valid empty evidence",
        },
        "validation": metrics,
        "comparisons": {
            "phase3a_threshold_0_40": _comparison(metrics, phase3a),
            "phase3b_threshold_0_30": _comparison(metrics, phase3b),
        },
        "decision": (
            "freeze_threshold_0_30_with_oversized_box_guardrail"
            if accepted
            else "freeze_plain_threshold_0_30"
        ),
        "guardrail_accepted": accepted,
        "phase3_validation_tuning_closed": True,
        "source_evidence": {
            "calibration_sha256": _sha256(calibration_path),
            "validation_candidates_sha256": _sha256(candidates_path),
        },
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write canonical LF bytes so the frozen artifact hash is stable on Windows and Linux.
    output_path.write_bytes((json.dumps(artifact, indent=2) + "\n").encode("utf-8"))
    return artifact


def _summarize(
    predictions: list[dict[str, Any]], max_area: float
) -> dict[str, Any]:
    ious = [float(prediction["iou"]) for prediction in predictions]
    detected = [prediction for prediction in predictions if not prediction["abstained"]]
    detected_ious = [float(prediction["iou"]) for prediction in detected]
    huge_count = sum(
        (
            prediction["selected"]["normalized_xyxy"][2]
            - prediction["selected"]["normalized_xyxy"][0]
        )
        * (
            prediction["selected"]["normalized_xyxy"][3]
            - prediction["selected"]["normalized_xyxy"][1]
        )
        >= max_area
        for prediction in detected
    )
    return {
        "reference_count": len(predictions),
        "mean_iou": round(sum(ious) / len(ious), 8),
        "acc_at_0_5_iou": round(sum(iou >= 0.5 for iou in ious) / len(ious), 8),
        "no_detection_count": len(predictions) - len(detected),
        "detected_reference_count": len(detected),
        "detected_only_mean_iou": round(
            sum(detected_ious) / len(detected_ious), 8
        ),
        "huge_selected_box_count": huge_count,
    }


def _comparison(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous": {
            "mean_iou": previous["mean_iou"],
            "acc_at_0_5_iou": previous["acc_at_0_5_iou"],
            "no_detection_count": previous["no_detection_count"],
            "detected_reference_count": previous["detected_reference_count"],
            "detected_only_mean_iou": previous["detected_only_mean_iou"],
            "huge_selected_box_count": previous["huge_box_count"],
        },
        "delta": {
            "mean_iou": round(current["mean_iou"] - previous["mean_iou"], 8),
            "acc_at_0_5_iou": round(
                current["acc_at_0_5_iou"] - previous["acc_at_0_5_iou"], 8
            ),
            "no_detection_count": (
                current["no_detection_count"] - previous["no_detection_count"]
            ),
            "detected_reference_count": (
                current["detected_reference_count"]
                - previous["detected_reference_count"]
            ),
            "detected_only_mean_iou": round(
                current["detected_only_mean_iou"]
                - previous["detected_only_mean_iou"],
                8,
            ),
            "huge_selected_box_count": (
                current["huge_selected_box_count"] - previous["huge_box_count"]
            ),
        },
    }


def _validate_sources(
    calibration: dict[str, Any],
    decision: dict[str, Any],
    calibration_path: Path,
    candidates_path: Path,
) -> None:
    if calibration.get("split") != "validation":
        raise ValueError("Guardrail replay accepts validation evidence only")
    if calibration.get("chosen", {}).get("box_threshold") != 0.30:
        raise ValueError("Phase 3B did not select threshold 0.30")
    evidence = decision["evidence"]
    if _sha256(calibration_path) != evidence["calibration_sha256"]:
        raise ValueError("Calibration artifact checksum changed")
    if _sha256(candidates_path) != evidence["validation_candidates_sha256"]:
        raise ValueError("Validation candidate artifact checksum changed")
    if decision.get("test_split_evaluated") is not False:
        raise ValueError("Phase 3B decision does not preserve the test split")


def _validate_candidate_order(candidates: list[dict[str, Any]]) -> None:
    scores = [float(candidate["raw_model_score"]) for candidate in candidates]
    if scores != sorted(scores, reverse=True):
        raise ValueError("Stored detections are not score-sorted")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = replay_guardrail(
        args.calibration.resolve(),
        args.candidates.resolve(),
        args.decision.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(artifact["validation"], indent=2))
    print(artifact["decision"])


if __name__ == "__main__":
    main()
