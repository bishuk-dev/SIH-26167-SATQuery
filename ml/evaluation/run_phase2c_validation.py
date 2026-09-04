"""Validation-only comparison of Phase 2B and visual-contrast adapters."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Literal

from ml.evaluation.phase2c_diagnostics import (
    accuracy,
    per_type_accuracy,
    question_only_predictions,
)
from ml.evaluation.run_phase2b_comparison import (
    _predict,
    _scene_balanced_subset,
    _shuffled_scene_paths,
)
from ml.training.config import load_training_config
from ml.training.phase2b import (
    PROJECT_ROOT,
    VqaSample,
    hardware_report,
    load_manifest_samples,
    model_cache_dir,
    sha256_file,
)
from ml.training.precision import select_precision
from ml.training.sampling import normalize_text
from satquery.inference.preprocessing import FrozenImagePreprocessor
from satquery.registry import load_model_registry, load_preprocessing_registry

DEFAULT_CONFIG = PROJECT_ROOT / "ml/configs/phase2c_smolvlm_visual_contrast.yaml"


def run_validation_comparison(
    config_path: Path,
    output_path: Path,
    *,
    phase2b_adapter: Path,
    phase2c_adapter: Path,
    data_root: Path | None,
    precision_mode: Literal["auto", "fp32"],
) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2C validation comparison requires a CUDA GPU")
    config = load_training_config(config_path)
    registration = load_model_registry().models[config.model_registry_id]
    profile = load_preprocessing_registry().profiles[config.preprocessing_profile]
    manifest_path = (
        config.manifest.resolve()
        if config.manifest.is_absolute()
        else (PROJECT_ROOT / config.manifest).resolve()
    )
    train, validation, manifest = load_manifest_samples(
        manifest_path,
        data_root=data_root,
        train_split=config.train_split,
        validation_split=config.validation_split,
    )
    validation = _scene_balanced_subset(validation, config.evaluation_max_samples)
    snapshot = Path(
        snapshot_download(
            repo_id=registration.model_id,
            revision=registration.revision,
            cache_dir=model_cache_dir(),
        )
    )
    if sha256_file(snapshot / registration.checkpoint_file) != (
        registration.checkpoint_sha256
    ):
        raise RuntimeError("Base-model checkpoint hash is invalid")
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=registration.allow_remote_code,
    )
    precision = select_precision(torch, force_fp32=precision_mode == "fp32")
    preprocessor = FrozenImagePreprocessor(profile)
    shuffled_paths = _shuffled_scene_paths(validation)

    baseline = _evaluate_adapter(
        snapshot,
        phase2b_adapter,
        registration.allow_remote_code,
        precision.torch_dtype(torch),
        processor,
        preprocessor,
        validation,
        shuffled_paths,
        config.max_new_tokens,
    )
    candidate = _evaluate_adapter(
        snapshot,
        phase2c_adapter,
        registration.allow_remote_code,
        precision.torch_dtype(torch),
        processor,
        preprocessor,
        validation,
        shuffled_paths,
        config.max_new_tokens,
    )
    validation_dicts = [_sample_dict(sample) for sample in validation]
    train_dicts = [_sample_dict(sample) for sample in train]
    question_only = question_only_predictions(train_dicts, validation_dicts)
    question_only_score = accuracy(question_only, validation_dicts)
    baseline_result = _score_adapter(baseline, validation_dicts)
    candidate_result = _score_adapter(candidate, validation_dicts)
    gap_gain = (
        candidate_result["visual_dependence_gap"]
        - baseline_result["visual_dependence_gap"]
    )
    result = {
        "schema_version": 1,
        "evaluation_split": "validation",
        "test_samples_used": 0,
        "sample_count": len(validation),
        "scene_count": len({sample.scene_id for sample in validation}),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path.resolve()),
        "dataset": manifest["dataset"],
        "runtime": hardware_report(torch, precision),
        "question_only": {
            "normalized_exact_match": round(question_only_score, 6),
            "per_question_type": per_type_accuracy(
                question_only, validation_dicts
            ),
        },
        "phase2b": {
            "adapter": str(phase2b_adapter.resolve()),
            "adapter_sha256": _adapter_sha256(phase2b_adapter),
            **baseline_result,
        },
        "phase2c": {
            "adapter": str(phase2c_adapter.resolve()),
            "adapter_sha256": _adapter_sha256(phase2c_adapter),
            **candidate_result,
        },
        "visual_dependence_gap_gain": round(gap_gain, 6),
        "materially_improved": gap_gain >= 0.05,
        "material_improvement_threshold": 0.05,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path.with_name("validation_predictions.jsonl")
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    **validation_dicts[index],
                    "question_only": question_only[index],
                    "phase2b_correct": baseline["correct"][index],
                    "phase2b_blank": baseline["blank"][index],
                    "phase2b_shuffled": baseline["shuffled"][index],
                    "phase2b_correct_differs_from_shuffled": normalize_text(
                        baseline["correct"][index]
                    )
                    != normalize_text(baseline["shuffled"][index]),
                    "phase2c_correct": candidate["correct"][index],
                    "phase2c_blank": candidate["blank"][index],
                    "phase2c_shuffled": candidate["shuffled"][index],
                    "phase2c_correct_differs_from_shuffled": normalize_text(
                        candidate["correct"][index]
                    )
                    != normalize_text(candidate["shuffled"][index]),
                }
            )
            + "\n"
            for index in range(len(validation))
        ),
        encoding="utf-8",
    )
    result["predictions"] = str(predictions_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _evaluate_adapter(
    snapshot: Path,
    adapter_path: Path,
    allow_remote_code: bool,
    dtype: Any,
    processor: Any,
    preprocessor: FrozenImagePreprocessor,
    samples: list[VqaSample],
    shuffled_paths: list[Path],
    max_new_tokens: int,
) -> dict[str, list[str]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    adapter_path = adapter_path.expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    base = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=allow_remote_code,
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to(
        "cuda"
    )
    model.eval()
    predictions = {
        "correct": _predict(
            model,
            processor,
            preprocessor,
            samples,
            image_paths=[sample.image_path for sample in samples],
            max_new_tokens=max_new_tokens,
        ),
        "blank": _predict(
            model,
            processor,
            preprocessor,
            samples,
            image_paths=[None] * len(samples),
            max_new_tokens=max_new_tokens,
        ),
        "shuffled": _predict(
            model,
            processor,
            preprocessor,
            samples,
            image_paths=shuffled_paths,
            max_new_tokens=max_new_tokens,
        ),
    }
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def _score_adapter(
    predictions: dict[str, list[str]], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    scores = {
        name: accuracy(values, samples) for name, values in predictions.items()
    }
    visual_gap = scores["correct"] - max(scores["blank"], scores["shuffled"])
    differing = sum(
        normalize_text(correct) != normalize_text(shuffled)
        for correct, shuffled in zip(
            predictions["correct"], predictions["shuffled"], strict=True
        )
    )
    correct_only_wins = sum(
        normalize_text(correct) == normalize_text(str(sample["answer"]))
        and normalize_text(shuffled) != normalize_text(str(sample["answer"]))
        for correct, shuffled, sample in zip(
            predictions["correct"], predictions["shuffled"], samples, strict=True
        )
    )
    shuffled_only_wins = sum(
        normalize_text(correct) != normalize_text(str(sample["answer"]))
        and normalize_text(shuffled) == normalize_text(str(sample["answer"]))
        for correct, shuffled, sample in zip(
            predictions["correct"], predictions["shuffled"], samples, strict=True
        )
    )
    return {
        "normalized_exact_match": {
            name: round(value, 6) for name, value in scores.items()
        },
        "visual_dependence_gap": round(visual_gap, 6),
        "correct_vs_shuffled_predictions_differ": differing,
        "correct_vs_shuffled_difference_share": round(differing / len(samples), 6),
        "correct_image_only_wins": correct_only_wins,
        "shuffled_image_only_wins": shuffled_only_wins,
        "per_question_type": {
            name: per_type_accuracy(values, samples)
            for name, values in predictions.items()
        },
    }


def _sample_dict(sample: VqaSample) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "question": sample.question,
        "answer": sample.answer,
    }


def _adapter_sha256(adapter_path: Path) -> str:
    checkpoint = adapter_path.expanduser().resolve() / "adapter_model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint}")
    return sha256_file(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase2b-adapter", type=Path, required=True)
    parser.add_argument("--phase2c-adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--precision", choices=("auto", "fp32"), default="auto")
    args = parser.parse_args()
    result = run_validation_comparison(
        args.config,
        args.output,
        phase2b_adapter=args.phase2b_adapter,
        phase2c_adapter=args.phase2c_adapter,
        data_root=args.data_root,
        precision_mode=args.precision,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
