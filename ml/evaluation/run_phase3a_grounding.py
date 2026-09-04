"""Evaluate the frozen Grounding DINO baseline on the VRSBench subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ml.evaluation.prepare_vrsbench_grounding import DEFAULT_DATA_ROOT, DEFAULT_MANIFEST
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.grounding import GroundingDinoBackend
from satquery.inference.grounding_preprocessing import GroundingImagePreprocessor
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

DEFAULT_MODEL_REGISTRY_ID = "grounding_dino_tiny_v1"


def intersection_over_union(
    predicted: list[float] | None, expected: list[float]
) -> float:
    if predicted is None:
        return 0.0
    intersection_width = max(
        0.0, min(predicted[2], expected[2]) - max(predicted[0], expected[0])
    )
    intersection_height = max(
        0.0, min(predicted[3], expected[3]) - max(predicted[1], expected[1])
    )
    intersection = intersection_width * intersection_height
    predicted_area = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    union = predicted_area + expected_area - intersection
    return intersection / union if union > 0 else 0.0


def run_benchmark(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    split: Literal["validation", "test"],
    settings: GroundingRuntimeSettings,
) -> dict[str, Any]:
    import torch

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest_path)
    samples = [sample for sample in manifest["samples"] if sample["split"] == split]
    if not samples:
        raise ValueError(f"VRSBench subset has no {split} samples")
    registration = load_model_registry().models[DEFAULT_MODEL_REGISTRY_ID]
    if not isinstance(registration, GroundingModelRegistration):
        raise ValueError("Registered Phase 3A model is not a grounding model")
    profile = load_preprocessing_registry().profiles[
        registration.preprocessing_profile
    ]
    if not isinstance(profile, GroundingPreprocessingProfile):
        raise ValueError("Registered Phase 3A preprocessing is not grounding-compatible")
    backend = GroundingDinoBackend(registration, profile, settings)
    preprocessor = GroundingImagePreprocessor(profile)
    data_root = data_root.expanduser().resolve()
    predictions = []
    started = time.perf_counter()
    for sample in samples:
        image_path = _resolve_image(sample, data_root)
        if _sha256(image_path) != sample["image_sha256"]:
            raise RuntimeError(
                f"VRSBench subset image checksum changed: {sample['source_image_id']}"
            )
        with Image.open(image_path) as image:
            prepared = preprocessor.from_pil(image)
        result = backend.detect(prepared.image, str(sample["query"]))
        boxes = [
            [
                detection.x_min / result.input_width,
                detection.y_min / result.input_height,
                detection.x_max / result.input_width,
                detection.y_max / result.input_height,
            ]
            for detection in result.detections
        ]
        expected = [float(value) for value in sample["ground_truth_normalized_xyxy"]]
        top_box = boxes[0] if boxes else None
        iou = intersection_over_union(top_box, expected)
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "source_image_id": sample["source_image_id"],
                "query": sample["query"],
                "object_class": sample["object_class"],
                "unique": sample["unique"],
                "ground_truth_normalized_xyxy": expected,
                "predicted_normalized_xyxy": top_box,
                "raw_model_score": (
                    result.detections[0].score if result.detections else None
                ),
                "model_phrase": (
                    result.detections[0].phrase if result.detections else None
                ),
                "all_model_detections": [
                    {
                        "box": box,
                        "raw_score": detection.score,
                        "phrase": detection.phrase,
                    }
                    for box, detection in zip(boxes, result.detections, strict=True)
                ],
                "iou": round(iou, 8),
                "acc_at_0_5": iou >= 0.5,
            }
        )
    elapsed = time.perf_counter() - started
    ious = [float(prediction["iou"]) for prediction in predictions]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "task": "text_guided_grounding",
        "split": split,
        "sample_count": len(samples),
        "scene_count": len({sample["scene_id"] for sample in samples}),
        "mean_iou": round(sum(ious) / len(ious), 8),
        "acc_at_0_5_iou": round(
            sum(iou >= 0.5 for iou in ious) / len(ious), 8
        ),
        "no_detection_count": sum(
            prediction["predicted_normalized_xyxy"] is None
            for prediction in predictions
        ),
        "elapsed_seconds": round(elapsed, 4),
        "dataset": manifest["dataset"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model": {
            "registry_id": DEFAULT_MODEL_REGISTRY_ID,
            "model_id": registration.model_id,
            "revision": registration.revision,
            "checkpoint_sha256": registration.checkpoint_sha256,
            "preprocessing_profile": registration.preprocessing_profile,
            "preprocessing_version": profile.version,
            "box_threshold": profile.box_threshold,
            "text_threshold": profile.text_threshold,
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": settings.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "prediction_file": str(predictions_path),
    }
    (output_dir / f"{split}_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _resolve_image(sample: dict[str, Any], data_root: Path) -> Path:
    candidates = [
        data_root / str(sample["source_image_id"]),
        (Path(__file__).resolve().parents[2] / str(sample["image_path"])),
    ]
    for candidate in candidates:
        if candidate.is_file():
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
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        args.manifest,
        args.data_root,
        args.output_dir,
        split=args.split,
        settings=GroundingRuntimeSettings(
            model_root=Path("./models"),
            allow_remote_network=args.allow_download,
            device=args.device,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
