"""Validation-only box-threshold calibration for the frozen Phase 3A model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ml.evaluation.prepare_vrsbench_grounding import DEFAULT_DATA_ROOT, DEFAULT_MANIFEST
from ml.evaluation.run_phase3a_grounding import intersection_over_union
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.grounding import (
    GroundingBackend,
    GroundingDinoBackend,
    RawGroundingDetection,
)
from satquery.inference.grounding_preprocessing import GroundingImagePreprocessor
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

MODEL_REGISTRY_ID = "grounding_dino_tiny_v1"
BOX_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
TEXT_THRESHOLD = 0.30
HUGE_BOX_AREA_FRACTION = 0.80
PHASE3A_ANCHOR = {
    "box_threshold": 0.40,
    "mean_iou": 0.1286,
    "acc_at_0_5_iou": 0.1667,
    "no_detection_count": 17,
    "detected_reference_count": 7,
}


class Clock(Protocol):
    def __call__(self) -> float: ...


def run_calibration(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    settings: GroundingRuntimeSettings,
    backend: GroundingBackend | None = None,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate fixed thresholds on validation; the test split is inaccessible."""

    import torch

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [
        sample for sample in manifest["samples"] if sample["split"] == "validation"
    ]
    if not samples:
        raise ValueError("VRSBench manifest has no validation samples")

    registration, registered_profile = _registered_grounding_configuration()
    if registered_profile.text_threshold != TEXT_THRESHOLD:
        raise ValueError("Registered text threshold changed from the Phase 3B contract")
    sweep_profile = registered_profile.model_copy(
        update={"box_threshold": min(BOX_THRESHOLDS)}
    )
    detector = backend or GroundingDinoBackend(
        registration, sweep_profile, settings
    )
    preprocessor = GroundingImagePreprocessor(registered_profile)
    data_root = data_root.expanduser().resolve()

    records: list[dict[str, Any]] = []
    inference_started = clock()
    for index, sample in enumerate(samples, start=1):
        image_path = _resolve_verified_image(sample, data_root)
        with Image.open(image_path) as image:
            prepared = preprocessor.from_pil(image)
        result = detector.detect(prepared.image, str(sample["query"]))
        if (result.input_width, result.input_height) != prepared.image.size:
            raise RuntimeError("Grounding backend returned inconsistent dimensions")
        candidates = [
            _normalized_detection(detection, result.input_width, result.input_height)
            for detection in result.detections
            if detection.score >= min(BOX_THRESHOLDS)
        ]
        candidates.sort(key=lambda candidate: candidate["raw_model_score"], reverse=True)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "source_image_id": sample["source_image_id"],
                "query": sample["query"],
                "object_class": sample["object_class"],
                "ground_truth_normalized_xyxy": sample[
                    "ground_truth_normalized_xyxy"
                ],
                "candidates_at_0_15": candidates,
            }
        )
        print(
            f"validation inference {index}/{len(samples)}: {sample['sample_id']}",
            flush=True,
        )
    shared_inference_seconds = clock() - inference_started

    rows = []
    for threshold in BOX_THRESHOLDS:
        postprocess_started = clock()
        row = summarize_threshold(records, threshold)
        row["selection_runtime_seconds"] = round(
            clock() - postprocess_started, 6
        )
        row["effective_runtime_seconds"] = round(
            shared_inference_seconds + row["selection_runtime_seconds"], 4
        )
        rows.append(row)
    validate_phase3a_anchor(rows)
    chosen = choose_threshold(rows)
    semantic_diagnostics = diagnose_semantic_failures(records, chosen["box_threshold"])

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "validation_candidates.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    artifact = {
        "schema_version": 1,
        "phase": "3B",
        "task": "validation_only_grounding_threshold_calibration",
        "split": "validation",
        "selection_policy": (
            "max Acc@0.5 IoU, then max mean IoU, then min no-detection count; "
            "final exact tie prefers the higher threshold"
        ),
        "phase3a_validation_anchor": PHASE3A_ANCHOR,
        "prediction_policy": "highest model score only; no ground-truth oracle",
        "sample_count": len(records),
        "scene_count": len({record["scene_id"] for record in records}),
        "box_thresholds": list(BOX_THRESHOLDS),
        "text_threshold": TEXT_THRESHOLD,
        "huge_box_definition": (
            f"selected normalized box area >= {HUGE_BOX_AREA_FRACTION:.2f}"
        ),
        "threshold_results": rows,
        "chosen": chosen,
        "semantic_diagnostics_at_chosen_threshold": semantic_diagnostics,
        "shared_inference_seconds": round(shared_inference_seconds, 4),
        "runtime_method": (
            "one forward pass per reference at threshold 0.15; higher thresholds "
            "filter the same score-sorted detections"
        ),
        "dataset": manifest["dataset"],
        "manifest_sha256": _sha256(manifest_path),
        "model": {
            "registry_id": MODEL_REGISTRY_ID,
            "model_id": registration.model_id,
            "revision": registration.revision,
            "checkpoint_sha256": registration.checkpoint_sha256,
            "preprocessing_profile": registration.preprocessing_profile,
            "preprocessing_version": registered_profile.version,
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": settings.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "predictions_file": str(predictions_path),
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    return artifact


def summarize_threshold(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    ious: list[float] = []
    detected_ious: list[float] = []
    huge_box_count = 0
    for record in records:
        selected = _highest_scoring_detection(record["candidates_at_0_15"], threshold)
        predicted = selected["normalized_xyxy"] if selected is not None else None
        iou = intersection_over_union(
            predicted, record["ground_truth_normalized_xyxy"]
        )
        ious.append(iou)
        if selected is not None:
            detected_ious.append(iou)
            x_min, y_min, x_max, y_max = predicted
            if (x_max - x_min) * (y_max - y_min) >= HUGE_BOX_AREA_FRACTION:
                huge_box_count += 1
    detected_count = len(detected_ious)
    return {
        "box_threshold": threshold,
        "mean_iou": round(sum(ious) / len(ious), 8),
        "acc_at_0_5_iou": round(sum(iou >= 0.5 for iou in ious) / len(ious), 8),
        "no_detection_count": len(records) - detected_count,
        "detected_reference_count": detected_count,
        "detected_only_mean_iou": (
            round(sum(detected_ious) / detected_count, 8) if detected_count else None
        ),
        "huge_box_count": huge_box_count,
    }


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one threshold result is required")
    return max(
        rows,
        key=lambda row: (
            row["acc_at_0_5_iou"],
            row["mean_iou"],
            -row["no_detection_count"],
            row["box_threshold"],
        ),
    ).copy()


def validate_phase3a_anchor(rows: list[dict[str, Any]]) -> None:
    anchor = next(
        (row for row in rows if row["box_threshold"] == PHASE3A_ANCHOR["box_threshold"]),
        None,
    )
    if anchor is None:
        raise RuntimeError("Phase 3B sweep omitted the Phase 3A anchor threshold")
    count_fields = ("no_detection_count", "detected_reference_count")
    if any(anchor[field] != PHASE3A_ANCHOR[field] for field in count_fields):
        raise RuntimeError("Phase 3A detection-count anchor was not reproduced")
    metric_fields = ("mean_iou", "acc_at_0_5_iou")
    if any(
        abs(anchor[field] - PHASE3A_ANCHOR[field]) > 0.0001
        for field in metric_fields
    ):
        raise RuntimeError("Phase 3A metric anchor was not reproduced")


def diagnose_semantic_failures(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    mismatches = []
    for record in records:
        selected = _highest_scoring_detection(record["candidates_at_0_15"], threshold)
        if selected is None:
            continue
        phrase_matches = _phrase_matches_class(
            selected["phrase"], record["object_class"]
        )
        if not phrase_matches:
            mismatches.append(
                {
                    "sample_id": record["sample_id"],
                    "query": record["query"],
                    "object_class": record["object_class"],
                    "model_phrase": selected["phrase"],
                    "raw_model_score": selected["raw_model_score"],
                    "normalized_xyxy": selected["normalized_xyxy"],
                    "box_area_fraction": selected["box_area_fraction"],
                }
            )
    return {
        "method": (
            "diagnostic only: normalized selected phrase contains the annotated "
            "object-class tokens; object_class is never model input or selection input"
        ),
        "phrase_mismatch_count": len(mismatches),
        "phrase_mismatches": mismatches,
    }


def _highest_scoring_detection(
    candidates: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate["raw_model_score"] >= threshold
        ),
        None,
    )


def _normalized_detection(
    detection: RawGroundingDetection, width: int, height: int
) -> dict[str, Any]:
    box = [
        detection.x_min / width,
        detection.y_min / height,
        detection.x_max / width,
        detection.y_max / height,
    ]
    return {
        "normalized_xyxy": box,
        "raw_model_score": detection.score,
        "phrase": detection.phrase,
        "box_area_fraction": (box[2] - box[0]) * (box[3] - box[1]),
    }


def _phrase_matches_class(phrase: str, object_class: str) -> bool:
    phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase.casefold()))
    class_tokens = set(re.findall(r"[a-z0-9]+", object_class.casefold()))
    return bool(class_tokens) and class_tokens.issubset(phrase_tokens)


def _registered_grounding_configuration() -> tuple[
    GroundingModelRegistration, GroundingPreprocessingProfile
]:
    registration = load_model_registry().models[MODEL_REGISTRY_ID]
    if not isinstance(registration, GroundingModelRegistration):
        raise ValueError("Registered Phase 3 model is not a grounding model")
    profile = load_preprocessing_registry().profiles[
        registration.preprocessing_profile
    ]
    if not isinstance(profile, GroundingPreprocessingProfile):
        raise ValueError("Registered Phase 3 preprocessing is not grounding-compatible")
    return registration, profile


def _resolve_verified_image(sample: dict[str, Any], data_root: Path) -> Path:
    candidates = [
        data_root / str(sample["source_image_id"]),
        Path(__file__).resolve().parents[2] / str(sample["image_path"]),
    ]
    for candidate in candidates:
        if candidate.is_file():
            if _sha256(candidate) != sample["image_sha256"]:
                raise RuntimeError(
                    f"VRSBench subset image checksum changed: {sample['source_image_id']}"
                )
            return candidate
    raise FileNotFoundError(
        f"VRSBench subset image is unavailable: {sample['source_image_id']}"
    )


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    artifact = run_calibration(
        args.manifest,
        args.data_root,
        args.output_dir,
        settings=GroundingRuntimeSettings(
            model_root=Path("./models"),
            allow_remote_network=args.allow_download,
            device=args.device,
        ),
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
