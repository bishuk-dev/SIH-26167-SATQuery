"""Compare frozen and Phase 2B adapted VQA models with shortcut controls."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ml.training.config import load_training_config
from ml.training.phase2b import (
    PROJECT_ROOT,
    VqaSample,
    load_manifest_samples,
    model_cache_dir,
    resolve_image_path,
    sha256_file,
)
from satquery.inference.preprocessing import FrozenImagePreprocessor
from satquery.registry import load_model_registry, load_preprocessing_registry

DEFAULT_CONFIG = PROJECT_ROOT / "ml/configs/phase2b_smolvlm_lora.yaml"
DEFAULT_OUTPUT = Path("outputs/phase2b_smolvlm_lora")


def run_comparison(
    config_path: Path,
    output_dir: Path,
    *,
    adapter_dir: Path,
    data_root: Path | None,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2B comparison requires a CUDA GPU")
    config = load_training_config(config_path)
    registration = load_model_registry().models[config.model_registry_id]
    profile = load_preprocessing_registry().profiles[config.preprocessing_profile]
    manifest_path = (
        config.manifest.resolve()
        if config.manifest.is_absolute()
        else (PROJECT_ROOT / config.manifest).resolve()
    )
    train_samples, _, manifest = load_manifest_samples(
        manifest_path,
        data_root=data_root,
        train_split=config.train_split,
        validation_split=config.validation_split,
    )
    test_samples = _scene_balanced_subset(
        _load_test_samples(manifest, data_root),
        config.evaluation_max_samples,
    )
    if not test_samples:
        raise ValueError("Phase 2B manifest has no test samples")

    output_dir = output_dir.expanduser().resolve()
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
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
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    preprocessor = FrozenImagePreprocessor(profile)

    started = time.perf_counter()
    frozen = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=registration.allow_remote_code,
        dtype=dtype,
    ).to("cuda")
    frozen.eval()
    frozen_answers = _predict(
        frozen,
        processor,
        preprocessor,
        test_samples,
        image_paths=[sample.image_path for sample in test_samples],
        max_new_tokens=config.max_new_tokens,
    )
    del frozen
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=registration.allow_remote_code,
        dtype=dtype,
    )
    adapted = PeftModel.from_pretrained(
        base,
        adapter_dir.expanduser().resolve(),
        is_trainable=False,
    ).to("cuda")
    adapted.eval()
    correct_paths = [sample.image_path for sample in test_samples]
    shuffled_paths = _shuffled_scene_paths(test_samples)
    adapted_answers = _predict(
        adapted,
        processor,
        preprocessor,
        test_samples,
        image_paths=correct_paths,
        max_new_tokens=config.max_new_tokens,
    )
    blank_answers = _predict(
        adapted,
        processor,
        preprocessor,
        test_samples,
        image_paths=[None] * len(test_samples),
        max_new_tokens=config.max_new_tokens,
    )
    shuffled_answers = _predict(
        adapted,
        processor,
        preprocessor,
        test_samples,
        image_paths=shuffled_paths,
        max_new_tokens=config.max_new_tokens,
    )

    majority_answer = Counter(_normalize(item.answer) for item in train_samples).most_common(1)[0][0]
    question_answers: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in train_samples:
        question_answers[_normalize(sample.question)][_normalize(sample.answer)] += 1
    question_only = [
        question_answers[_normalize(sample.question)].most_common(1)[0][0]
        if question_answers[_normalize(sample.question)]
        else majority_answer
        for sample in test_samples
    ]
    expected = [_normalize(sample.answer) for sample in test_samples]
    predictions = []
    for index, sample in enumerate(test_samples):
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "scene_id": sample.scene_id,
                "question": sample.question,
                "expected": sample.answer,
                "frozen": frozen_answers[index],
                "adapted": adapted_answers[index],
                "majority": majority_answer,
                "question_only": question_only[index],
                "blank_image": blank_answers[index],
                "shuffled_image": shuffled_answers[index],
            }
        )
    metrics = {
        "frozen_normalized_exact_match": _accuracy(frozen_answers, expected),
        "adapted_normalized_exact_match": _accuracy(adapted_answers, expected),
        "majority_normalized_exact_match": _accuracy(
            [majority_answer] * len(expected), expected
        ),
        "question_only_normalized_exact_match": _accuracy(question_only, expected),
        "blank_image_normalized_exact_match": _accuracy(blank_answers, expected),
        "shuffled_image_normalized_exact_match": _accuracy(
            shuffled_answers, expected
        ),
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest["dataset"],
        "manifest": str(manifest_path),
        "split": "test",
        "sample_count": len(test_samples),
        "scene_count": len({sample.scene_id for sample in test_samples}),
        "adapter": str(adapter_dir.resolve()),
        "metrics": metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "warning": (
            "Use this test comparison only after training and validation choices are frozen."
        ),
    }
    (evaluation_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions),
        encoding="utf-8",
    )
    (evaluation_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _load_test_samples(
    manifest: dict[str, Any], data_root: Path | None
) -> list[VqaSample]:
    samples = []
    for item in manifest["samples"]:
        if item["split"] != "test":
            continue
        samples.append(
            VqaSample(
                sample_id=str(item["sample_id"]),
                scene_id=str(item["scene_id"]),
                image_path=resolve_image_path(
                    str(item["image_path"]), data_root=data_root
                ),
                question=str(item["question"]),
                answer=str(item["answer"]),
            )
        )
    return samples


def _predict(
    model: Any,
    processor: Any,
    preprocessor: FrozenImagePreprocessor,
    samples: list[VqaSample],
    *,
    image_paths: list[Path | None],
    max_new_tokens: int,
) -> list[str]:
    import torch

    answers = []
    for sample, image_path in zip(samples, image_paths, strict=True):
        if image_path is None:
            image = Image.new(
                "RGB",
                (preprocessor.profile.width, preprocessor.profile.height),
                preprocessor.profile.padding_rgb,
            )
        else:
            with Image.open(image_path) as source:
                image = preprocessor.from_pil(source)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": preprocessor.profile.prompt_template.format(
                            question=sample.question
                        ),
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
            do_resize=False,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        input_length = inputs["input_ids"].shape[-1]
        answers.append(
            processor.decode(
                generated[0, input_length:], skip_special_tokens=True
            ).strip()
        )
    return answers


def _shuffled_scene_paths(samples: list[VqaSample]) -> list[Path]:
    scene_paths: dict[str, Path] = {}
    for sample in samples:
        scene_paths.setdefault(sample.scene_id, sample.image_path)
    scene_ids = sorted(scene_paths)
    if len(scene_ids) < 2:
        raise ValueError("Shuffled-image control requires at least two scenes")
    shuffled = scene_ids.copy()
    random.Random(42).shuffle(shuffled)
    if any(left == right for left, right in zip(scene_ids, shuffled, strict=True)):
        shuffled = scene_ids[1:] + scene_ids[:1]
    mapping = {
        scene_id: scene_paths[other]
        for scene_id, other in zip(scene_ids, shuffled, strict=True)
    }
    return [mapping[sample.scene_id] for sample in samples]


def _scene_balanced_subset(
    samples: list[VqaSample], maximum_samples: int
) -> list[VqaSample]:
    by_scene: dict[str, list[VqaSample]] = defaultdict(list)
    for sample in samples:
        by_scene[sample.scene_id].append(sample)
    selected: list[VqaSample] = []
    depth = 0
    while len(selected) < maximum_samples:
        added = False
        for scene_id in sorted(by_scene):
            scene_samples = by_scene[scene_id]
            if depth < len(scene_samples):
                selected.append(scene_samples[depth])
                added = True
                if len(selected) == maximum_samples:
                    break
        if not added:
            break
        depth += 1
    return selected


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _accuracy(predicted: list[str], expected: list[str]) -> float:
    return sum(
        _normalize(prediction) == target
        for prediction, target in zip(predicted, expected, strict=True)
    ) / len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    adapter_dir = args.adapter_dir or args.output_dir / "adapter"
    result = run_comparison(
        args.config,
        args.output_dir,
        adapter_dir=adapter_dir,
        data_root=args.data_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
