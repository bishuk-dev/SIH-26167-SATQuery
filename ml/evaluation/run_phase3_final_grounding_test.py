"""Execute the frozen Phase 3 grounding policy on the untouched test split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ml.evaluation.prepare_vrsbench_grounding import DEFAULT_DATA_ROOT, DEFAULT_MANIFEST
from ml.evaluation.run_phase3a_grounding import intersection_over_union
from ml.evaluation.run_phase3b_grounding_calibration import _resolve_verified_image
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.grounding import (
    DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
    GroundingBackend,
    GroundingDinoBackend,
    GroundingSelectionCandidate,
    select_grounding_candidate,
)
from satquery.inference.grounding_preprocessing import GroundingImagePreprocessor
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/phase3b_grounding_threshold_calibration"
DEFAULT_DECISION = EXPERIMENT_ROOT / "decision.json"
METRICS_FILENAME = "final_test_metrics.json"
PREDICTIONS_FILENAME = "final_test_predictions.jsonl"
EXPECTED_TEST_SCENES = 8
EXPECTED_TEST_REFERENCES = 16


class Clock(Protocol):
    def __call__(self) -> float: ...


def run_final_test(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    decision_path: Path,
    *,
    settings: GroundingRuntimeSettings,
    confirmed: bool,
    backend: GroundingBackend | None = None,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Run the locked test once; existing artifacts or policy drift abort the run."""

    if not confirmed:
        raise RuntimeError("The one-time test requires explicit confirmation")
    output_dir = output_dir.expanduser().resolve()
    metrics_path = output_dir / METRICS_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Final grounding test artifacts already exist; refusing rerun")

    manifest_path = manifest_path.expanduser().resolve()
    decision_path = decision_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    registration, profile = _validate_frozen_policy(decision)
    samples = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    scene_count = len({sample["scene_id"] for sample in samples})
    if len(samples) != EXPECTED_TEST_REFERENCES or scene_count != EXPECTED_TEST_SCENES:
        raise ValueError("Frozen VRSBench test split must contain 8 scenes / 16 references")
    if manifest["counts"]["test"] != {
        "scenes": EXPECTED_TEST_SCENES,
        "references": EXPECTED_TEST_REFERENCES,
    }:
        raise ValueError("VRSBench manifest test counts do not match its samples")

    import torch

    detector = backend or GroundingDinoBackend(registration, profile, settings)
    preprocessor = GroundingImagePreprocessor(profile)
    if settings.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Inference and frozen score/geometry selection complete before ground truth is read.
    blind_predictions: list[dict[str, Any]] = []
    started = clock()
    for sample in samples:
        image_path = _resolve_verified_image(sample, data_root.expanduser().resolve())
        with Image.open(image_path) as image:
            prepared = preprocessor.from_pil(image)
        result = detector.detect(prepared.image, str(sample["query"]))
        if (result.input_width, result.input_height) != prepared.image.size:
            raise RuntimeError("Grounding backend returned inconsistent dimensions")
        candidates = tuple(
            GroundingSelectionCandidate(
                candidate_index=index,
                raw_score=detection.score,
                normalized_xyxy=(
                    detection.x_min / result.input_width,
                    detection.y_min / result.input_height,
                    detection.x_max / result.input_width,
                    detection.y_max / result.input_height,
                ),
            )
            for index, detection in enumerate(result.detections)
            if detection.score >= profile.box_threshold
        )
        selected = select_grounding_candidate(
            candidates, max_area=profile.max_normalized_box_area
        )
        detection = (
            result.detections[selected.candidate_index]
            if selected is not None
            else None
        )
        blind_predictions.append(
            {
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "source_image_id": sample["source_image_id"],
                "query": sample["query"],
                "predicted_normalized_xyxy": (
                    list(selected.normalized_xyxy) if selected is not None else None
                ),
                "raw_model_score": detection.score if detection is not None else None,
                "model_phrase": detection.phrase if detection is not None else None,
                "abstained": selected is None,
            }
        )
    elapsed = clock() - started

    predictions = []
    for sample, prediction in zip(samples, blind_predictions, strict=True):
        expected = [float(value) for value in sample["ground_truth_normalized_xyxy"]]
        iou = intersection_over_union(
            prediction["predicted_normalized_xyxy"], expected
        )
        predictions.append(
            {
                **prediction,
                "ground_truth_normalized_xyxy": expected,
                "iou": round(iou, 8),
                "acc_at_0_5": iou >= 0.5,
            }
        )

    metrics = _summarize(predictions, profile.max_normalized_box_area)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_bytes(
        "".join(json.dumps(row) + "\n" for row in predictions).encode("utf-8")
    )
    artifact = {
        "schema_version": 1,
        "phase": "3_final_grounding_test",
        "status": "final_observational_evidence",
        "split": "test",
        "test_execution_ordinal": 1,
        "validation_tuning_closed": True,
        "phase3_tuning_reopened": False,
        **metrics,
        "dataset": manifest["dataset"],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "frozen_decision_sha256": _sha256(decision_path),
        "model": {
            "registry_id": DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
            "model_id": registration.model_id,
            "revision": registration.revision,
            "checkpoint_file": registration.checkpoint_file,
            "checkpoint_sha256": registration.checkpoint_sha256,
            "preprocessing_profile": registration.preprocessing_profile,
            "preprocessing_version": profile.version,
            "box_threshold": profile.box_threshold,
            "text_threshold": profile.text_threshold,
            "max_normalized_box_area_exclusive": profile.max_normalized_box_area,
            "selection": "highest raw model score among remaining detections",
            "empty_selection": "abstain with valid empty evidence",
        },
        "runtime": {
            "elapsed_seconds": round(elapsed, 4),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": settings.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if settings.device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
        "provenance": _git_provenance(output_dir),
        "prediction_file": PREDICTIONS_FILENAME,
        "prediction_file_sha256": _sha256(predictions_path),
    }
    metrics_path.write_bytes((json.dumps(artifact, indent=2) + "\n").encode("utf-8"))
    return artifact


def _validate_frozen_policy(
    decision: dict[str, Any],
) -> tuple[GroundingModelRegistration, GroundingPreprocessingProfile]:
    if decision.get("phase3_validation_tuning_closed") is not True:
        raise ValueError("Phase 3 validation tuning is not frozen")
    if decision.get("test_split_evaluated") is not False:
        raise RuntimeError("The frozen decision records that the test was already executed")
    frozen = decision["frozen_production_policy"]
    if frozen != {
        "model_registry_id": "grounding_dino_tiny_phase3_final_v1",
        "box_threshold": 0.3,
        "text_threshold": 0.3,
        "reject_normalized_box_area_greater_than_or_equal_to": 0.8,
        "selection": "highest raw model score among remaining detections",
        "all_rejected": "abstain with valid empty evidence",
    }:
        raise ValueError("Frozen decision policy changed")
    registration = load_model_registry().models[DEFAULT_GROUNDING_MODEL_REGISTRY_ID]
    if not isinstance(registration, GroundingModelRegistration):
        raise ValueError("Frozen grounding model registration is invalid")
    profile = load_preprocessing_registry().profiles[registration.preprocessing_profile]
    if not isinstance(profile, GroundingPreprocessingProfile):
        raise ValueError("Frozen grounding preprocessing registration is invalid")
    if (
        profile.box_threshold,
        profile.text_threshold,
        profile.max_normalized_box_area,
    ) != (0.3, 0.3, 0.8):
        raise ValueError("Production grounding registry does not match the decision")
    return registration, profile


def _summarize(predictions: list[dict[str, Any]], max_area: float) -> dict[str, Any]:
    ious = [float(prediction["iou"]) for prediction in predictions]
    detected = [row for row in predictions if not row["abstained"]]
    detected_ious = [float(row["iou"]) for row in detected]
    huge_count = sum(
        (box[2] - box[0]) * (box[3] - box[1]) >= max_area
        for row in detected
        if (box := row["predicted_normalized_xyxy"]) is not None
    )
    return {
        "scene_count": len({row["scene_id"] for row in predictions}),
        "reference_count": len(predictions),
        "mean_iou": round(sum(ious) / len(ious), 8),
        "acc_at_0_5_iou": round(sum(iou >= 0.5 for iou in ious) / len(ious), 8),
        "no_detection_count": len(predictions) - len(detected),
        "detected_reference_count": len(detected),
        "detected_only_mean_iou": (
            round(sum(detected_ious) / len(detected_ious), 8) if detected else None
        ),
        "huge_selected_box_count": huge_count,
    }


def _git_provenance(output_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    runner_meta_path = output_dir / "runner_meta.json"
    runner_meta = (
        json.loads(runner_meta_path.read_text(encoding="utf-8"))
        if runner_meta_path.is_file()
        else None
    )
    return {
        "commit": os.environ.get("SATQUERY_GIT_REF") or git("rev-parse", "HEAD"),
        "dirty_worktree": bool(status),
        "runner_experiment": os.environ.get("SATQUERY_EXPERIMENT_NAME"),
        "runner_metadata": runner_meta,
        "production_grounding_sha256": _sha256(
            PROJECT_ROOT / "satquery/inference/grounding.py"
        ),
        "model_registry_sha256": _sha256(PROJECT_ROOT / "models/registry.yaml"),
        "preprocessing_registry_sha256": _sha256(
            PROJECT_ROOT / "satquery/registry/preprocessing.yaml"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--confirm-one-time-test", action="store_true")
    args = parser.parse_args()
    result = run_final_test(
        args.manifest,
        args.data_root,
        args.output_dir,
        args.decision,
        settings=GroundingRuntimeSettings(
            model_root=Path("./models"),
            allow_remote_network=args.allow_download,
            device=args.device,
        ),
        confirmed=args.confirm_one_time_test,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
