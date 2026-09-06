"""Controlled, validation-only S2 head adaptation for Phase 4F.

The official 19-class classifier is retained and trained while every ResNet-50
backbone parameter remains frozen. This module never opens the sealed test split.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import torch
from pydantic import Field

from ml.evaluation.phase4e_bifold_baseline import (
    BIGEARTHNET_19_CLASS_ORDER,
    compute_multilabel_metrics,
)
from satquery.ingestion.models import ContractModel
from satquery.inference.multisensor_preprocessing import SealedTestAccessError
from satquery.inference.multisensor_preprocessing import (
    materialized_band_paths,
    preprocess_bifold_bands,
)
from satquery.registry.models import BifoldPreprocessingProfile, load_preprocessing_registry

FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
)
FROZEN_PREPROCESSING_PROFILE = "bifold_resnet50_s2_v020"
FROZEN_PHASE4E_S2_BASELINE: Mapping[str, object] = {
    "model_id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
    "sample_count": 3000,
    "threshold": 0.5,
    "micro_f1": 0.7325822442541685,
    "macro_f1": 0.6452843453433793,
    "macro_average_precision": 0.7420321532834496,
    "validation_result_sha256": (
        "82f72da05442ef50d8a5bb5cc59fb3dea2d2346e70b6222fe06c11bf988a1bd4"
    ),
}


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    frozen_parameter_count: int


class Phase4FProvenance(ContractModel):
    """Runtime provenance required for every Phase 4F validation result."""

    schema_version: Literal[1] = 1
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_worktree: bool
    kaggle_experiment: str = Field(min_length=1)
    kaggle_kernel: str = Field(min_length=1)
    runtime_seconds: float = Field(gt=0)
    device: str = Field(min_length=1)
    cuda_version: str | None
    torch_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0)
    model_id: Literal["BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"]
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: Literal[FROZEN_MANIFEST_SHA256]
    preprocessing_profile: Literal[FROZEN_PREPROCESSING_PROFILE]
    test_accessed: Literal[False] = False


class Phase4FTrainingConfig(ContractModel):
    """Frozen primary-run hyperparameters; tests may reduce only resource sizes."""

    optimizer: Literal["AdamW"] = "AdamW"
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    batch_size: int = Field(default=128, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_epochs: int = Field(default=10, gt=0)
    early_stopping_metric: Literal["validation_macro_average_precision"] = (
        "validation_macro_average_precision"
    )
    early_stopping_mode: Literal["max"] = "max"
    early_stopping_patience: int = Field(default=2, ge=0)
    early_stopping_min_delta: Literal[0.0] = 0.0
    scheduler: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    scheduler_eta_min: float = Field(default=1e-4, ge=0)
    random_seed: int = 20260906
    threshold: Literal[0.5] = 0.5


@dataclass(frozen=True, slots=True)
class FitResult:
    epochs_completed: int
    best_epoch: int
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    validation_macro_average_precision: tuple[float, ...]


class Phase4FS2Dataset(torch.utils.data.Dataset[tuple[str, torch.Tensor, torch.Tensor]]):
    """Lazy native-raster dataset constrained to the frozen S2 profile."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        band_loader: Callable[[Path], np.ndarray] | None = None,
    ) -> None:
        require_phase4f_split(split)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            raise ValueError("Frozen manifest samples are missing")
        self.samples = sorted(
            (
                sample
                for sample in raw_samples
                if isinstance(sample, Mapping)
                and sample.get("official_split") == split
            ),
            key=lambda sample: str(sample.get("sample_id", "")),
        )
        if not self.samples:
            raise ValueError(f"Frozen manifest contains no {split} samples")
        profile = load_preprocessing_registry().profiles[FROZEN_PREPROCESSING_PROFILE]
        if not isinstance(profile, BifoldPreprocessingProfile):
            raise ValueError("Frozen Phase 4F S2 preprocessing profile is unavailable")
        self.profile = profile
        self.dataset_root = dataset_root
        self.split = split
        self.band_loader = band_loader or _read_native_band
        self.class_indices = {
            name: index for index, name in enumerate(BIGEARTHNET_19_CLASS_ORDER)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        sample_id = sample.get("sample_id")
        labels = sample.get("labels")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Phase 4F sample has an invalid sample_id")
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or label not in self.class_indices
            for label in labels
        ):
            raise ValueError(f"Phase 4F sample {sample_id} has invalid labels")
        paths = materialized_band_paths(
            self.dataset_root,
            sample,
            split=self.split,
            allow_sealed_test=False,
        )
        bands = {
            name: self.band_loader(paths[name]) for name in self.profile.band_order
        }
        target = torch.zeros(19, dtype=torch.float32)
        for label in labels:
            target[self.class_indices[label]] = 1.0
        return sample_id, preprocess_bifold_bands(bands, self.profile), target


def require_phase4f_split(split: str) -> None:
    """Allow only the frozen TRAIN and validation partitions."""

    if split == "test":
        raise SealedTestAccessError(
            "Phase 4 test access is sealed; Phase 4F is train/validation-only"
        )
    if split not in {"train", "validation"}:
        raise ValueError("Phase 4F accepts only train or validation")


def freeze_s2_classifier_head(
    model: torch.nn.Module,
    *,
    official_class_order: tuple[str, ...] = BIGEARTHNET_19_CLASS_ORDER,
) -> ParameterSummary:
    """Freeze the exact ConfigILM ResNet-50 except its pretrained 19-class fc."""

    config = getattr(model, "config", None)
    if (
        getattr(config, "timm_model_name", None) != "resnet50"
        or getattr(config, "classes", None) != 19
    ):
        raise ValueError("Phase 4F requires the pinned 19-class ResNet-50 checkpoint")
    # The pinned HF config stores numeric output placeholders. Their semantics
    # come from ConfigILM NEW_LABELS, which the caller reads from the exact
    # pinned source and supplies here for an independent index-order check.
    if tuple(getattr(config, "class_names", ())) != tuple(map(str, range(19))):
        raise ValueError("Phase 4F checkpoint class order placeholders changed")
    if official_class_order != BIGEARTHNET_19_CLASS_ORDER:
        raise ValueError("Phase 4F checkpoint class order does not match the evaluator")

    inner = getattr(model, "model", None)
    encoder = getattr(inner, "vision_encoder", None)
    classifier = getattr(encoder, "fc", None)
    if not isinstance(classifier, torch.nn.Linear) or classifier.out_features != 19:
        raise ValueError("Pinned ConfigILM ResNet-50 must expose a 19-class fc head")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)

    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    expected = (
        "model.vision_encoder.fc.weight",
        "model.vision_encoder.fc.bias",
    )
    if trainable != expected:
        raise ValueError(f"Unexpected Phase 4F trainable parameter set: {trainable}")
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    return ParameterSummary(trainable, trainable_count, frozen_count)


def validate_phase4e_baseline_linkage(closeout: Mapping[str, Any]) -> None:
    """Fail closed if the Phase 4F comparison baseline differs from Phase 4E."""

    if closeout.get("s2") != FROZEN_PHASE4E_S2_BASELINE:
        raise ValueError("Phase 4E S2 baseline linkage does not match the frozen result")


def fit_head_only(
    model: torch.nn.Module,
    *,
    train_batches: Iterable[tuple[Any, ...]],
    validation_batches: Iterable[tuple[Any, ...]],
    config: Phase4FTrainingConfig,
    device: str,
) -> FitResult:
    """Fit only the already-unfrozen classifier and select by validation mAP."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Phase 4F has no trainable classifier parameters")
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable_names != (
        "model.vision_encoder.fc.weight",
        "model.vision_encoder.fc.bias",
    ):
        raise ValueError("Phase 4F trainable parameter set is not classifier-only")

    _seed_everything(config.random_seed)
    model.to(device)
    model.eval()  # Frozen BatchNorm statistics must not drift during head adaptation.
    execution_model: torch.nn.Module = model
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        execution_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.scheduler_eta_min,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    map_history: list[float] = []
    best_map = float("-inf")
    best_epoch = 0
    best_head: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        for batch_count, batch in enumerate(train_batches, start=1):
            inputs, targets = _batch_tensors(batch)
            logits = execution_model(inputs.to(device))
            loss = loss_fn(logits, targets.to(device=device, dtype=torch.float32))
            (loss / config.gradient_accumulation_steps).backward()
            if batch_count % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
        if batch_count == 0:
            raise ValueError("Phase 4F training iterator produced no samples")
        if batch_count % config.gradient_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_history.append(running_loss / batch_count)

        validation_loss, validation_map = _validation_epoch(
            execution_model, validation_batches, loss_fn, device, config.threshold
        )
        validation_history.append(validation_loss)
        map_history.append(validation_map)
        scheduler.step()
        if validation_map > best_map + config.early_stopping_min_delta:
            best_map = validation_map
            best_epoch = epoch
            stale_epochs = 0
            classifier = model.model.vision_encoder.fc
            best_head = copy.deepcopy(classifier.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs > config.early_stopping_patience:
                break

    if best_head is None:
        raise RuntimeError("Phase 4F did not produce a selectable checkpoint")
    model.model.vision_encoder.fc.load_state_dict(best_head)
    return FitResult(
        epochs_completed=len(train_history),
        best_epoch=best_epoch,
        train_loss=tuple(train_history),
        validation_loss=tuple(validation_history),
        validation_macro_average_precision=tuple(map_history),
    )


def _validation_epoch(
    model: torch.nn.Module,
    batches: Iterable[tuple[Any, ...]],
    loss_fn: torch.nn.Module,
    device: str,
    threshold: float,
) -> tuple[float, float]:
    losses: list[float] = []
    targets_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in batches:
            inputs, targets = _batch_tensors(batch)
            targets_device = targets.to(device=device, dtype=torch.float32)
            logits = model(inputs.to(device))
            losses.append(float(loss_fn(logits, targets_device).cpu()))
            targets_all.append(targets.detach().cpu().numpy())
            probabilities_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not losses:
        raise ValueError("Phase 4F validation iterator produced no samples")
    metrics = compute_multilabel_metrics(
        np.concatenate(targets_all),
        np.concatenate(probabilities_all),
        threshold=threshold,
    )
    return float(np.mean(losses)), metrics.macro_average_precision


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _batch_tensors(batch: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    values = batch[-2:]
    if len(values) != 2 or not all(isinstance(value, torch.Tensor) for value in values):
        raise ValueError("Phase 4F batches must end with input and target tensors")
    return values[0], values[1]


def _read_native_band(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(path, sharing=False) as dataset:
            if dataset.count != 1:
                raise ValueError(f"Expected a single-band raster: {path}")
            return dataset.read(1, masked=True)
