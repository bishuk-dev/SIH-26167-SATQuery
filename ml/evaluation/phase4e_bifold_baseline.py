"""Frozen, validation-only infrastructure for Phase 4E BIFOLD baselines.

This module deliberately does not start evaluation on import. Real execution is
guarded by the Phase 4D readiness contract and accepts only the validation split.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np
import torch
from pydantic import Field, model_validator

from satquery.ingestion.models import ContractModel
from satquery.inference.multisensor_preprocessing import (
    SealedTestAccessError,
    materialized_band_paths,
    preprocess_bifold_bands,
)
from satquery.registry.models import (
    BifoldPreprocessingProfile,
    MultisensorModelRegistration,
    load_model_registry,
    load_preprocessing_registry,
)

FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
)
VERIFIED_STATUS = "MATERIALIZED_AND_INTEGRITY_VERIFIED"
FROZEN_PREPROCESSING_STATUS = "FROZEN_AFTER_NATIVE_TRAIN_RASTER_AUDIT"
COMPLETED_RASTER_AUDIT_STATUS = "measured_from_three_predeclared_train_pairs"
_MATERIALIZATION_EXPECTATIONS = {
    "s1": (54_439_153_171, "a55eaa2cdf6a917e296bd6601ec1e348", 36_002),
    "s2": (63_251_710_377, "2245ed2d1a93f6ce637d839bc856396e", 216_012),
}

BIGEARTHNET_19_CLASS_ORDER = (
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
)


class Phase4DNotReadyError(RuntimeError):
    """Raised before model or raster access when a Phase 4D gate is not satisfied."""


class Phase4EProvenance(ContractModel):
    """Frozen identities required to reproduce one unimodal validation run."""

    schema_version: Literal[1] = 1
    experiment_name: str = Field(min_length=1)
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    modality: Literal["s1", "s2"]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_profile: str = Field(min_length=1)
    frozen_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(ge=0.0, le=1.0)
    evaluation_split: Literal["validation"] = "validation"
    class_order: tuple[str, ...] = BIGEARTHNET_19_CLASS_ORDER
    test_accessed: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_classes(self) -> Phase4EProvenance:
        if self.class_order != BIGEARTHNET_19_CLASS_ORDER:
            raise ValueError("Phase 4E provenance must use the canonical class order")
        return self


class MultilabelMetrics(ContractModel):
    """Complete validation metrics in canonical class order."""

    sample_count: int = Field(ge=0)
    class_order: tuple[str, ...]
    threshold: float = Field(ge=0.0, le=1.0)
    micro_f1: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_class_f1: tuple[float, ...]
    macro_average_precision: float = Field(ge=0.0, le=1.0)
    per_class_average_precision: tuple[float, ...]
    class_prevalence: tuple[float, ...]

    @model_validator(mode="after")
    def require_class_aligned_values(self) -> MultilabelMetrics:
        count = len(self.class_order)
        aligned = (
            self.per_class_f1,
            self.per_class_average_precision,
            self.class_prevalence,
        )
        if not self.class_order or any(len(values) != count for values in aligned):
            raise ValueError("Metric vectors must align with class_order")
        return self


class Phase4EPrediction(ContractModel):
    """One prediction row with unrounded model outputs."""

    sample_id: str = Field(min_length=1)
    target_indices: tuple[int, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    predicted_indices: tuple[int, ...]

    @model_validator(mode="after")
    def require_19_outputs(self) -> Phase4EPrediction:
        if len(self.logits) != 19 or len(self.probabilities) != 19:
            raise ValueError("Prediction rows must preserve exactly 19 model outputs")
        return self


class Phase4EResult(ContractModel):
    """Top-level deterministic validation artifact contract."""

    schema_version: Literal[1] = 1
    status: Literal["VALIDATION_EVALUATED"] = "VALIDATION_EVALUATED"
    provenance: Phase4EProvenance
    metrics: MultilabelMetrics
    prediction_count: int = Field(ge=0)
    prediction_artifact: str = Field(min_length=1)
    prediction_format: Literal["jsonl"] = "jsonl"
    logits_and_probabilities_preserved: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Phase4DGatePaths:
    readiness: Path
    raster_audit: Path
    preprocessing_contract: Path
    manifest: Path


class BifoldBatch(Protocol):
    sample_ids: Sequence[str]
    inputs: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class ValidationBatch:
    sample_ids: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor


class _BifoldUnimodalInference:
    registry_id: str
    expected_channels: int

    def __init__(self, model: torch.nn.Module) -> None:
        models = load_model_registry().models
        profiles = load_preprocessing_registry().profiles
        registration = models.get(self.registry_id)
        if not isinstance(registration, MultisensorModelRegistration):
            raise ValueError("BIFOLD registry entry is missing or has the wrong task")
        profile = profiles.get(registration.preprocessing_profile)
        if not isinstance(profile, BifoldPreprocessingProfile):
            raise ValueError("BIFOLD preprocessing profile is missing or invalid")
        if registration.input_channels != self.expected_channels:
            raise ValueError("BIFOLD registry channel count changed")
        self.registration = registration
        self.profile = profile
        self.model = model.eval()

    def predict_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.expected_channels:
            raise ValueError(
                f"Expected N x {self.expected_channels} channels x H x W input"
            )
        if tuple(inputs.shape[-2:]) != (self.profile.height, self.profile.width):
            raise ValueError("BIFOLD inputs must use the frozen 120 x 120 profile")
        with torch.inference_mode():
            output = self.model(inputs)
        logits = _extract_logits(output)
        if logits.ndim != 2 or logits.shape != (inputs.shape[0], 19):
            raise ValueError("BIFOLD model must return exactly 19 logits per sample")
        if not torch.isfinite(logits).all():
            raise ValueError("BIFOLD model returned non-finite logits")
        return logits.detach().to(dtype=torch.float32, device="cpu")

    @classmethod
    def from_pretrained(
        cls,
        model_class: type[Any],
        *,
        cache_dir: Path,
        allow_network: bool,
    ) -> _BifoldUnimodalInference:
        """Load the audited external architecture from the exact pinned snapshot."""

        from huggingface_hub import snapshot_download

        registration = load_model_registry().models[cls.registry_id]
        if not isinstance(registration, MultisensorModelRegistration):
            raise ValueError("BIFOLD registry entry has the wrong task")
        snapshot = Path(
            snapshot_download(
                repo_id=registration.model_id,
                revision=registration.revision,
                cache_dir=cache_dir,
                local_files_only=not allow_network,
                allow_patterns=(registration.checkpoint_file, "config.json"),
                max_workers=1,
            )
        )
        checkpoint = snapshot / registration.checkpoint_file
        if _sha256(checkpoint) != registration.checkpoint_sha256:
            raise ValueError("Pinned BIFOLD checkpoint SHA-256 mismatch")
        model = model_class.from_pretrained(str(snapshot))
        if not isinstance(model, torch.nn.Module):
            raise TypeError("BIFOLD model loader did not return a torch module")
        return cls(model)


class BifoldS1Inference(_BifoldUnimodalInference):
    """Frozen 2-channel VV/VH BIFOLD inference wrapper."""

    registry_id = "bifold_resnet50_s1_v020"
    expected_channels = 2


class BifoldS2Inference(_BifoldUnimodalInference):
    """Frozen 10-channel BIFOLD optical inference wrapper."""

    registry_id = "bifold_resnet50_s2_v020"
    expected_channels = 10


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Apply the fixed multilabel sigmoid conversion without calibration."""

    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite floating-point values")
    return torch.sigmoid(logits)


def probabilities_to_predictions(
    probabilities: torch.Tensor, *, threshold: float
) -> torch.Tensor:
    """Apply an inclusive deterministic threshold selected on validation only."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if not torch.isfinite(probabilities).all() or torch.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return probabilities >= threshold


def compute_multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    class_order: tuple[str, ...] = BIGEARTHNET_19_CLASS_ORDER,
) -> MultilabelMetrics:
    """Compute deterministic dependency-free multilabel F1 and average precision."""

    truth = np.asarray(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or scores.shape != truth.shape:
        raise ValueError("targets and probabilities must be aligned 2D arrays")
    if truth.shape[1] != len(class_order):
        raise ValueError("Metric arrays must align with class_order")
    if truth.shape[0] == 0:
        raise ValueError("At least one validation sample is required")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("targets must contain only zero and one")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    truth_bool = truth.astype(bool)
    predicted = scores >= threshold
    tp = np.sum(predicted & truth_bool, axis=0)
    fp = np.sum(predicted & ~truth_bool, axis=0)
    fn = np.sum(~predicted & truth_bool, axis=0)
    per_class_f1 = _safe_f1(tp, fp, fn)
    micro_f1 = float(
        _safe_f1(
            np.array([tp.sum()]),
            np.array([fp.sum()]),
            np.array([fn.sum()]),
        )[0]
    )
    per_class_ap = tuple(
        _binary_average_precision(truth_bool[:, index], scores[:, index])
        for index in range(scores.shape[1])
    )
    return MultilabelMetrics(
        sample_count=int(truth.shape[0]),
        class_order=class_order,
        threshold=threshold,
        micro_f1=micro_f1,
        macro_f1=float(per_class_f1.mean()),
        per_class_f1=tuple(float(value) for value in per_class_f1),
        macro_average_precision=float(np.mean(per_class_ap)),
        per_class_average_precision=per_class_ap,
        class_prevalence=tuple(
            float(value) for value in truth_bool.mean(axis=0, dtype=np.float64)
        ),
    )


def require_validation_split(split: str) -> None:
    if split == "test":
        raise SealedTestAccessError(
            "Phase 4 test access is sealed; Phase 4E baselines are validation-only"
        )
    if split != "validation":
        raise ValueError("Phase 4E baseline evaluator accepts only validation")


def iter_validation_batches(
    *,
    manifest_path: Path,
    dataset_root: Path,
    profile: BifoldPreprocessingProfile,
    batch_size: int,
    device: str,
    band_loader: Callable[[Path], np.ndarray] | None = None,
) -> Iterable[ValidationBatch]:
    """Yield validation samples in stable sample-id order from native bands."""

    require_validation_split("validation")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = _read_json(manifest_path, "frozen split manifest")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Frozen manifest samples are missing")
    samples = sorted(
        (
            sample
            for sample in raw_samples
            if isinstance(sample, Mapping)
            and sample.get("official_split") == "validation"
        ),
        key=lambda sample: str(sample.get("sample_id", "")),
    )
    load_band = band_loader or _read_native_band
    class_indices = {
        class_name: index
        for index, class_name in enumerate(BIGEARTHNET_19_CLASS_ORDER)
    }
    batch_ids: list[str] = []
    batch_inputs: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    for sample in samples:
        sample_id = sample.get("sample_id")
        labels = sample.get("labels")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Validation sample has an invalid sample_id")
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or label not in class_indices for label in labels
        ):
            raise ValueError(f"Validation sample {sample_id} has invalid labels")
        paths = materialized_band_paths(
            dataset_root, sample, split="validation", allow_sealed_test=False
        )
        bands = {name: load_band(paths[name]) for name in profile.band_order}
        target = torch.zeros(19, dtype=torch.bool)
        for label in labels:
            target[class_indices[label]] = True
        batch_ids.append(sample_id)
        batch_inputs.append(preprocess_bifold_bands(bands, profile))
        batch_targets.append(target)
        if len(batch_ids) == batch_size:
            yield ValidationBatch(
                sample_ids=tuple(batch_ids),
                inputs=torch.stack(batch_inputs).to(device),
                targets=torch.stack(batch_targets),
            )
            batch_ids, batch_inputs, batch_targets = [], [], []
    if batch_ids:
        yield ValidationBatch(
            sample_ids=tuple(batch_ids),
            inputs=torch.stack(batch_inputs).to(device),
            targets=torch.stack(batch_targets),
        )


def assert_phase4d_ready(
    readiness_path: Path,
    raster_audit_path: Path,
    preprocessing_contract_path: Path,
    manifest_path: Path,
) -> None:
    """Fail before model loading unless every Phase 4D acceptance gate is proven."""

    readiness = _read_json(readiness_path, "Phase 4D readiness")
    raster_audit = _read_json(raster_audit_path, "native TRAIN raster audit")
    preprocessing = _read_json(
        preprocessing_contract_path, "Phase 4D preprocessing contract"
    )
    actual_manifest_sha = _sha256(manifest_path)
    recorded_sha = readiness.get("frozen_manifest_sha256")
    if not isinstance(recorded_sha, str) or actual_manifest_sha != recorded_sha:
        raise Phase4DNotReadyError(
            "Phase 4D not ready: frozen manifest SHA-256 does not match"
        )
    modalities = readiness.get("modalities")
    if not isinstance(modalities, Mapping):
        raise Phase4DNotReadyError(
            "Phase 4D not ready: modality integrity records are missing"
        )
    for modality in ("s1", "s2"):
        record = modalities.get(modality)
        if not isinstance(record, Mapping) or record.get("status") != VERIFIED_STATUS:
            raise Phase4DNotReadyError(
                f"Phase 4D not ready: {modality.upper()} materialization is not verified"
            )
        expected_bytes, expected_md5, expected_members = _MATERIALIZATION_EXPECTATIONS[
            modality
        ]
        required_values = {
            "expected_compressed_bytes": expected_bytes,
            "observed_compressed_bytes": expected_bytes,
            "publisher_md5": expected_md5,
            "selected_member_count": expected_members,
            "missing_member_count": 0,
            "duplicate_member_count": 0,
            "unsafe_rejected_member_count": 0,
            "package_member_count": expected_members,
        }
        package_sha = record.get("package_sha256")
        if any(record.get(key) != value for key, value in required_values.items()) or (
            not isinstance(package_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", package_sha) is None
        ):
            raise Phase4DNotReadyError(
                f"Phase 4D not ready: {modality.upper()} integrity record is incomplete"
            )
    if (
        raster_audit.get("status") != COMPLETED_RASTER_AUDIT_STATUS
        or raster_audit.get("manifest_sha256") != actual_manifest_sha
        or raster_audit.get("test_pixels_opened") is not False
    ):
        raise Phase4DNotReadyError(
            "Phase 4D not ready: representative native TRAIN raster audit is incomplete"
        )
    if (
        preprocessing.get("status") != FROZEN_PREPROCESSING_STATUS
        or preprocessing.get("manifest_sha256") != actual_manifest_sha
    ):
        raise Phase4DNotReadyError(
            "Phase 4D not ready: preprocessing contract is not frozen"
        )


def evaluate_validation_batches(
    batches: Iterable[BifoldBatch],
    wrapper: _BifoldUnimodalInference,
    *,
    provenance: Phase4EProvenance,
    gate_paths: Phase4DGatePaths,
) -> tuple[MultilabelMetrics, tuple[Phase4EPrediction, ...]]:
    """Evaluate an ordered validation iterator after the complete Phase 4D gate."""

    require_validation_split(provenance.evaluation_split)
    assert_phase4d_ready(
        gate_paths.readiness,
        gate_paths.raster_audit,
        gate_paths.preprocessing_contract,
        gate_paths.manifest,
    )
    if _sha256(gate_paths.manifest) != provenance.frozen_manifest_sha256:
        raise Phase4DNotReadyError(
            "Phase 4D not ready: evaluator provenance manifest does not match"
        )
    readiness = _read_json(gate_paths.readiness, "Phase 4D readiness")
    modality_record = readiness["modalities"][provenance.modality]
    if (
        not isinstance(modality_record, Mapping)
        or modality_record.get("package_sha256")
        != provenance.materialized_package_sha256
    ):
        raise Phase4DNotReadyError(
            "Phase 4D not ready: evaluator package provenance does not match"
        )
    with _deterministic_inference():
        return _collect_validation_predictions(batches, wrapper, provenance)


def _collect_validation_predictions(
    batches: Iterable[BifoldBatch],
    wrapper: _BifoldUnimodalInference,
    provenance: Phase4EProvenance,
) -> tuple[MultilabelMetrics, tuple[Phase4EPrediction, ...]]:
    rows: list[Phase4EPrediction] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch in batches:
        if len(batch.sample_ids) != batch.inputs.shape[0] or batch.targets.shape != (
            batch.inputs.shape[0],
            19,
        ):
            raise ValueError("Validation batch identities, inputs, and targets do not align")
        logits = wrapper.predict_logits(batch.inputs)
        probs = logits_to_probabilities(logits)
        predicted = probabilities_to_predictions(probs, threshold=provenance.threshold)
        targets_cpu = batch.targets.detach().to(dtype=torch.bool, device="cpu")
        for index, sample_id in enumerate(batch.sample_ids):
            rows.append(
                Phase4EPrediction(
                    sample_id=sample_id,
                    target_indices=tuple(
                        int(value) for value in torch.where(targets_cpu[index])[0]
                    ),
                    logits=tuple(float(value) for value in logits[index]),
                    probabilities=tuple(float(value) for value in probs[index]),
                    predicted_indices=tuple(
                        int(value) for value in torch.where(predicted[index])[0]
                    ),
                )
            )
        targets.append(targets_cpu.numpy())
        probabilities.append(probs.numpy())
    if not rows:
        raise ValueError("Validation iterator produced no samples")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Validation iterator contains duplicate sample IDs")
    metrics = compute_multilabel_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        threshold=provenance.threshold,
    )
    return metrics, tuple(rows)


@contextmanager
def _deterministic_inference() -> Iterable[None]:
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    cudnn_available = hasattr(torch.backends, "cudnn")
    previous_benchmark = torch.backends.cudnn.benchmark if cudnn_available else False
    previous_cudnn_deterministic = (
        torch.backends.cudnn.deterministic if cudnn_available else False
    )
    torch.use_deterministic_algorithms(True)
    if cudnn_available:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        if cudnn_available:
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.use_deterministic_algorithms(algorithms_enabled)


def write_validation_artifacts(
    output_dir: Path,
    *,
    provenance: Phase4EProvenance,
    metrics: MultilabelMetrics,
    predictions: Sequence[Phase4EPrediction],
) -> Phase4EResult:
    """Write JSONL predictions and a result contract without overwriting a run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "validation_predictions.jsonl"
    result_path = output_dir / "validation_result.json"
    if prediction_path.exists() or result_path.exists():
        raise FileExistsError("Phase 4E validation artifacts already exist")
    prediction_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in predictions),
        encoding="utf-8",
    )
    result = Phase4EResult(
        provenance=provenance,
        metrics=metrics,
        prediction_count=len(predictions),
        prediction_artifact=prediction_path.name,
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise ValueError("BIFOLD model output does not expose logits")


def _safe_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def _binary_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(targets.sum())
    if positive_count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    sorted_scores = scores[order]
    cumulative_tp = np.cumsum(sorted_targets, dtype=np.int64)
    cumulative_fp = np.cumsum(~sorted_targets, dtype=np.int64)
    threshold_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    precision = cumulative_tp[threshold_ends] / (
        cumulative_tp[threshold_ends] + cumulative_fp[threshold_ends]
    )
    recall = cumulative_tp[threshold_ends] / positive_count
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase4DNotReadyError(
            f"Phase 4D not ready: cannot read {description}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase4DNotReadyError(
            f"Phase 4D not ready: {description} must be a JSON object"
        )
    return payload


def _read_native_band(path: Path) -> np.ndarray:
    import rasterio

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(path, sharing=False) as dataset:
                if dataset.count != 1:
                    raise ValueError(f"Expected a single-band raster: {path}")
                return dataset.read(1, masked=True)
    except OSError as exc:
        raise ValueError(f"Cannot read required native raster: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Phase4DNotReadyError(
            f"Phase 4D not ready: cannot hash required artifact {path}"
        ) from exc
    return digest.hexdigest()
