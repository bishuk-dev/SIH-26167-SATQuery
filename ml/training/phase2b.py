"""Phase 2B remote-sensing VQA adaptation with a small LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ml.training.config import TrainingConfig, load_training_config
from ml.training.precision import PrecisionSelection, select_precision
from ml.training.stability import NonFiniteTrainingError, StabilityMonitorCallback
from satquery.inference.preprocessing import FrozenImagePreprocessor
from satquery.registry import load_model_registry, load_preprocessing_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "ml" / "configs" / "phase2b_smolvlm_lora.yaml"
DEFAULT_OUTPUT = Path("outputs/phase2b_smolvlm_lora")


@dataclass(frozen=True)
class VqaSample:
    sample_id: str
    scene_id: str
    image_path: Path
    question: str
    answer: str


class ManifestDataset:
    def __init__(self, samples: Sequence[VqaSample]) -> None:
        self._samples = tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> VqaSample:
        return self._samples[index]


class VqaTrainingCollator:
    """Build supervised batches while masking prompt and image tokens from loss."""

    def __init__(
        self,
        processor: Any,
        preprocessor: FrozenImagePreprocessor,
        prompt_template: str,
    ) -> None:
        self.processor = processor
        self.preprocessor = preprocessor
        self.prompt_template = prompt_template
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    def __call__(self, samples: Sequence[VqaSample]) -> dict[str, Any]:
        images = [self._load_image(sample.image_path) for sample in samples]
        prompt_texts = [self._prompt(sample.question) for sample in samples]
        full_texts = [
            self._conversation(sample.question, sample.answer) for sample in samples
        ]
        image_batches = [[image] for image in images]
        prompt_batch = self.processor(
            text=prompt_texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        batch = self.processor(
            text=full_texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels
        return batch

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return self.preprocessor.from_pil(image)

    def _prompt(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._question_text(question)},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    def _conversation(self, question: str, answer: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._question_text(question)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
        ).strip()

    def _question_text(self, question: str) -> str:
        return self.prompt_template.format(question=question)


def run_training(
    config_path: Path,
    output_dir: Path,
    *,
    data_root: Path | None,
    resume_from_checkpoint: Path | None,
    smoke_test: bool,
    stability_smoke: bool,
    precision_mode: Literal["auto", "fp32"],
    allow_cpu: bool,
) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "Phase 2B requires a CUDA GPU; use --allow-cpu only for local debugging"
        )
    if smoke_test and stability_smoke:
        raise ValueError("Choose either --smoke-test or --stability-smoke")
    if resume_from_checkpoint is not None and (smoke_test or stability_smoke):
        raise ValueError("Smoke modes cannot resume an existing training run")
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_from_checkpoint}"
        )

    config = load_training_config(config_path)
    models = load_model_registry()
    profiles = load_preprocessing_registry()
    registration = models.models[config.model_registry_id]
    profile = profiles.profiles[config.preprocessing_profile]
    if registration.preprocessing_profile != config.preprocessing_profile:
        raise ValueError("Training config and model preprocessing profiles differ")

    manifest_path = _resolve_project_path(config.manifest)
    train_samples, validation_samples, manifest = load_manifest_samples(
        manifest_path,
        data_root=data_root,
        train_split=config.train_split,
        validation_split=config.validation_split,
    )
    if smoke_test:
        train_samples = train_samples[:2]
        validation_samples = validation_samples[:1]
    elif stability_smoke:
        stability_microbatches = 8 * config.gradient_accumulation_steps
        stability_samples = (
            stability_microbatches * config.per_device_train_batch_size
        )
        train_samples = train_samples[:stability_samples]
        validation_samples = validation_samples[:1]

    output_dir = output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    adapter_dir = output_dir / "adapter"
    for path in (checkpoint_dir, metrics_dir, logs_dir, adapter_dir):
        path.mkdir(parents=True, exist_ok=True)

    precision = select_precision(torch, force_fp32=precision_mode == "fp32")
    resume_guard = {
        "config_sha256": sha256_file(config_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "model_registry_id": config.model_registry_id,
        "model_revision": registration.revision,
        "preprocessing_profile": config.preprocessing_profile,
        "preprocessing_version": profile.version,
        "selected_precision": precision.name,
    }
    guard_path = output_dir / "resume_guard.json"
    if resume_from_checkpoint is not None:
        _validate_resume_guard(guard_path, resume_guard)
    else:
        _write_json(guard_path, resume_guard)

    set_seed(config.seed)
    random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = precision.torch_dtype(torch)
    snapshot = Path(
        snapshot_download(
            repo_id=registration.model_id,
            revision=registration.revision,
            cache_dir=model_cache_dir(),
        )
    )
    checkpoint = snapshot / registration.checkpoint_file
    if sha256_file(checkpoint) != registration.checkpoint_sha256:
        raise RuntimeError("Downloaded base-model checkpoint hash is invalid")

    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=registration.allow_remote_code,
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=registration.allow_remote_code,
        dtype=dtype,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            inference_mode=False,
        ),
    )
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    if trainable <= 0 or trainable >= total:
        raise RuntimeError("LoRA did not produce a bounded trainable parameter set")

    collator = VqaTrainingCollator(
        processor,
        FrozenImagePreprocessor(profile),
        profile.prompt_template,
    )
    max_steps = 1 if smoke_test else (8 if stability_smoke else config.max_steps)
    is_short_run = smoke_test or stability_smoke
    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        run_name=(
            config.run_name
            + ("_smoke" if smoke_test else "_stability" if stability_smoke else "")
        ),
        seed=config.seed,
        data_seed=config.seed,
        num_train_epochs=1 if is_short_run else config.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=(
            1 if smoke_test else config.gradient_accumulation_steps
        ),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=0 if smoke_test else config.warmup_ratio,
        logging_steps=1 if is_short_run else config.logging_steps,
        save_strategy="no" if stability_smoke else "steps",
        save_steps=1 if smoke_test else config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="no" if is_short_run else "epoch",
        fp16=precision.trainer_fp16,
        bf16=precision.trainer_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=device == "cuda",
        logging_dir=str(logs_dir),
        report_to=["tensorboard"],
    )
    monitor = StabilityMonitorCallback(torch, model, precision.name)

    class MonitoredTrainer(Trainer):
        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            loss = super().training_step(*args, **kwargs)
            monitor.record_loss(loss)
            return loss

    trainer = MonitoredTrainer(
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=ManifestDataset(train_samples),
        eval_dataset=ManifestDataset(validation_samples),
        callbacks=[monitor],
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(resume_from_checkpoint.resolve())
                if resume_from_checkpoint is not None
                else None
            )
        )
        parameters_changed = monitor.verify_parameters_changed(model)
    except NonFiniteTrainingError as exc:
        elapsed = time.perf_counter() - started
        failure = {
            "status": "FAIL",
            "error": str(exc),
            "selected_precision": precision.name,
            "gradient_accumulation_steps": (
                1 if smoke_test else config.gradient_accumulation_steps
            ),
            "runtime_seconds": round(elapsed, 4),
            **monitor.report(parameters_changed=False),
        }
        if device == "cuda":
            failure["peak_gpu_memory_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            )
        _write_json(metrics_dir / "numerical_failure.json", failure)
        raise
    finally:
        monitor.close()
    elapsed = time.perf_counter() - started
    trainer.save_model(adapter_dir)
    processor.save_pretrained(adapter_dir)
    adapter_checkpoint = adapter_dir / "adapter_model.safetensors"
    adapter_sha256 = sha256_file(adapter_checkpoint)

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "elapsed_seconds": round(elapsed, 4),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
            "selected_precision": precision.name,
        }
    )
    if device == "cuda":
        metrics["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        )
    _write_json(metrics_dir / "train_metrics.json", metrics)
    if is_short_run:
        _write_json(
            metrics_dir / "smoke_prediction.json",
            _smoke_inference(
                model,
                processor,
                FrozenImagePreprocessor(profile),
                validation_samples[0],
                config.max_new_tokens,
                device,
            ),
        )
    if stability_smoke:
        stability_report = {
            "status": "PASS",
            "selected_precision": precision.name,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "expected_optimizer_steps": 8,
            "runtime_seconds": round(elapsed, 4),
            **monitor.report(parameters_changed=parameters_changed),
        }
        if monitor.optimizer_steps != 8:
            raise RuntimeError(
                f"Stability smoke expected 8 optimizer steps, got {monitor.optimizer_steps}"
            )
        if device == "cuda":
            stability_report["peak_gpu_memory_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            )
        _write_json(metrics_dir / "stability_smoke.json", stability_report)

    run_record = {
        "schema_version": 1,
        "run_name": config.run_name,
        "smoke_test": smoke_test,
        "stability_smoke": stability_smoke,
        "precision_mode": precision_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": str(config_path.resolve()),
        "config_sha256": resume_guard["config_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": resume_guard["manifest_sha256"],
        "dataset": manifest["dataset"],
        "base_model": {
            "registry_id": config.model_registry_id,
            "model_id": registration.model_id,
            "revision": registration.revision,
            "checkpoint_sha256": registration.checkpoint_sha256,
            "preprocessing_profile": config.preprocessing_profile,
            "preprocessing_version": profile.version,
        },
        "runtime": hardware_report(torch, precision),
        "artifacts": {
            "adapter": str(adapter_dir),
            "adapter_checkpoint_sha256": adapter_sha256,
            "checkpoints": str(checkpoint_dir),
            "metrics": str(metrics_dir),
            "logs": str(logs_dir),
        },
    }
    _write_json(output_dir / "run.json", run_record)
    (output_dir / "training_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return run_record


def load_manifest_samples(
    manifest_path: Path,
    *,
    data_root: Path | None,
    train_split: str,
    validation_split: str,
) -> tuple[list[VqaSample], list[VqaSample], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_scene: dict[str, str] = {}
    selected: dict[str, list[VqaSample]] = {
        train_split: [],
        validation_split: [],
    }
    for item in manifest["samples"]:
        split = str(item["split"])
        scene_id = str(item["scene_id"])
        previous = by_scene.setdefault(scene_id, split)
        if previous != split:
            raise ValueError(f"Scene {scene_id} crosses split boundaries")
        if split not in selected:
            continue
        image_path = resolve_image_path(
            str(item["image_path"]),
            data_root=data_root,
        )
        selected[split].append(
            VqaSample(
                sample_id=str(item["sample_id"]),
                scene_id=scene_id,
                image_path=image_path,
                question=str(item["question"]),
                answer=str(item["answer"]),
            )
        )
    train_scenes = {sample.scene_id for sample in selected[train_split]}
    validation_scenes = {
        sample.scene_id for sample in selected[validation_split]
    }
    if not selected[train_split] or not selected[validation_split]:
        raise ValueError("Manifest must contain non-empty train and validation splits")
    if train_scenes & validation_scenes:
        raise ValueError("Train and validation scenes overlap")
    return selected[train_split], selected[validation_split], manifest


def hardware_report(
    torch: Any,
    precision: PrecisionSelection | None = None,
) -> dict[str, Any]:
    selection = precision or select_precision(torch)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda": torch.version.cuda,
        "selected_precision": selection.name,
        "compute_capability": (
            list(selection.compute_capability)
            if selection.compute_capability is not None
            else None
        ),
        "bf16_runtime_reported": selection.bf16_runtime_reported,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu_name": properties.name,
                "gpu_vram_bytes": properties.total_memory,
                "gpu_vram_gib": round(properties.total_memory / 1024**3, 2),
                "bf16_selected": selection.trainer_bf16,
            }
        )
    return report


def _smoke_inference(
    model: Any,
    processor: Any,
    preprocessor: FrozenImagePreprocessor,
    sample: VqaSample,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch

    collator = VqaTrainingCollator(
        processor,
        preprocessor,
        preprocessor.profile.prompt_template,
    )
    image = collator._load_image(sample.image_path)
    prompt = collator._prompt(sample.question)
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
        do_resize=False,
    ).to(device)
    model.config.use_cache = True
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    input_length = inputs["input_ids"].shape[-1]
    prediction = processor.decode(
        generated[0, input_length:],
        skip_special_tokens=True,
    ).strip()
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "expected": sample.answer,
        "predicted": prediction,
    }


def _resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def model_cache_dir() -> Path:
    model_root = Path(os.environ.get("MODEL_ROOT", str(PROJECT_ROOT / "models")))
    return model_root.expanduser().resolve() / "cache"


def resolve_image_path(raw_path: str, *, data_root: Path | None) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path]
    if data_root is not None:
        root = data_root.expanduser().resolve()
        candidates = [root / path, root / path.name, *candidates]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Manifest image is unavailable: {raw_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _validate_resume_guard(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file():
        raise RuntimeError("Cannot resume without the original resume_guard.json")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(
            "Resume refused because config, manifest, model, or preprocessing changed"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stability-smoke", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32"),
        default="auto",
        help="Use capability-gated automatic precision or explicitly force FP32",
    )
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    record = run_training(
        args.config,
        args.output_dir,
        data_root=args.data_root,
        resume_from_checkpoint=args.resume_from_checkpoint,
        smoke_test=args.smoke_test,
        stability_smoke=args.stability_smoke,
        precision_mode=args.precision,
        allow_cpu=args.allow_cpu,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
