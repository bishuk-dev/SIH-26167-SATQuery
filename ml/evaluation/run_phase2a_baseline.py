"""Run the frozen Phase 2A model on the held-out grouped subset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from satquery.inference.config import VqaRuntimeSettings
from satquery.inference.preprocessing import FrozenImagePreprocessor
from satquery.inference.vqa import DEFAULT_MODEL_REGISTRY_ID, SmolVlmBackend
from satquery.registry import load_model_registry, load_preprocessing_registry

DEFAULT_MANIFEST = Path(
    "experiments/phase2a_smolvlm_rsvqa_lr/split_manifest.json"
)
DEFAULT_RESULTS = Path("experiments/phase2a_smolvlm_rsvqa_lr/results.json")
DEFAULT_PREDICTIONS = Path(
    "experiments/phase2a_smolvlm_rsvqa_lr/predictions.jsonl"
)


def run(
    manifest_path: Path,
    results_path: Path,
    predictions_path: Path,
    *,
    allow_download: bool,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_registration = load_model_registry().models[DEFAULT_MODEL_REGISTRY_ID]
    profile = load_preprocessing_registry().profiles[
        model_registration.preprocessing_profile
    ]
    settings = VqaRuntimeSettings.from_env().model_copy(
        update={"allow_remote_network": allow_download}
    )
    backend = SmolVlmBackend(model_registration, profile, settings)
    preprocessor = FrozenImagePreprocessor(profile)
    test_samples = [
        sample for sample in manifest["samples"] if sample["split"] == "test"
    ]
    train_answers = [
        _normalize(sample["answer"])
        for sample in manifest["samples"]
        if sample["split"] == "train"
    ]
    majority_answer = Counter(train_answers).most_common(1)[0][0]

    predictions = []
    started = time.perf_counter()
    for sample in test_samples:
        sample_started = time.perf_counter()
        with Image.open(sample["image_path"]) as source_image:
            image = preprocessor.from_pil(source_image)
        predicted = backend.answer(image, sample["question"])
        expected_normalized = _normalize(sample["answer"])
        predicted_normalized = _normalize(predicted)
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "scene_id": sample["scene_id"],
                "question": sample["question"],
                "expected": sample["answer"],
                "predicted": predicted,
                "exact_match": predicted_normalized == expected_normalized,
                "majority_exact_match": majority_answer == expected_normalized,
                "latency_seconds": round(time.perf_counter() - sample_started, 4),
            }
        )
    elapsed = time.perf_counter() - started
    count = len(predictions)
    results = {
        "schema_version": 1,
        "run_id": "phase2a_smolvlm_256m_rsvqa_lr_grouped_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "registry_id": DEFAULT_MODEL_REGISTRY_ID,
            "model_id": model_registration.model_id,
            "revision": model_registration.revision,
            "checkpoint_sha256": model_registration.checkpoint_sha256,
            "preprocessing_profile": model_registration.preprocessing_profile,
            "preprocessing_version": profile.version,
            "frozen": True,
        },
        "dataset": manifest["dataset"],
        "split_manifest": manifest_path.as_posix(),
        "evaluated_split": "test",
        "sample_count": count,
        "scene_count": len({item["scene_id"] for item in predictions}),
        "metrics": {
            "normalized_exact_match": sum(
                item["exact_match"] for item in predictions
            )
            / count,
            "majority_answer": majority_answer,
            "majority_normalized_exact_match": sum(
                item["majority_exact_match"] for item in predictions
            )
            / count,
        },
        "runtime": {
            "device": settings.device,
            "cpu_threads": settings.cpu_threads,
            "total_seconds": round(elapsed, 4),
            "mean_seconds_per_sample": round(elapsed / count, 4),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "torch": torch.__version__,
        },
        "limitations": [
            "Smoke-scale held-out subset; not a publishable benchmark estimate.",
            "Generic frozen model has not been adapted to remote-sensing imagery.",
            "Generated answers provide no calibrated confidence score.",
        ],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    predictions_path.write_text(
        "".join(json.dumps(item) + "\n" for item in predictions),
        encoding="utf-8",
    )
    return results


def _normalize(value: str) -> str:
    lowered = value.casefold().strip()
    without_punctuation = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(without_punctuation.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    results = run(
        args.manifest,
        args.results,
        args.predictions,
        allow_download=args.allow_download,
    )
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
