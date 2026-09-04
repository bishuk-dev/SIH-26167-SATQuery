"""Registered text-guided grounding with coordinate-preserving evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image
from pydantic import ValidationError

from satquery.evidence.models import (
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    GroundingDetection,
    GroundingEvidence,
    NormalizedBoundingBox,
    PixelBoundingBox,
    WorldBoundingPolygon,
)
from satquery.geo.coordinates import pixel_to_world
from satquery.geo.exceptions import CoordinateTransformError
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.exceptions import (
    EvidenceGeometryError,
    ModelExecutionError,
    ModelInputUnsupportedError,
    ModelUnavailableError,
)
from satquery.inference.grounding_preprocessing import (
    GroundingImagePreprocessor,
    PreparedGroundingImage,
)
from satquery.ingestion.models import Modality, ObservationState
from satquery.ingestion.storage import FilesystemObservationStore
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

DEFAULT_GROUNDING_MODEL_REGISTRY_ID = "grounding_dino_tiny_phase3_final_v1"


@dataclass(frozen=True)
class RawGroundingDetection:
    phrase: str
    score: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class GroundingBackendResult:
    input_width: int
    input_height: int
    detections: tuple[RawGroundingDetection, ...]


@dataclass(frozen=True)
class GroundingSelectionCandidate:
    candidate_index: int
    raw_score: float
    normalized_xyxy: tuple[float, float, float, float]

    @property
    def normalized_area(self) -> float:
        x_min, y_min, x_max, y_max = self.normalized_xyxy
        return (x_max - x_min) * (y_max - y_min)


class GroundingBackend(Protocol):
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult: ...


class GroundingDinoBackend:
    """Lazy Transformers backend for the pinned Grounding DINO Tiny model."""

    def __init__(
        self,
        registration: GroundingModelRegistration,
        profile: GroundingPreprocessingProfile,
        settings: GroundingRuntimeSettings,
    ) -> None:
        self.registration = registration
        self.profile = profile
        self.settings = settings
        self._processor = None
        self._model = None
        self._torch = None

    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        model_query = _format_query(query)
        try:
            inputs = self._processor(
                images=image,
                text=model_query,
                return_tensors="pt",
                do_resize=False,
            ).to(self.settings.device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.profile.box_threshold,
                text_threshold=self.profile.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            phrases = result.get("text_labels", result.get("labels"))
            if phrases is None or not all(isinstance(value, str) for value in phrases):
                raise ValueError("Grounding processor did not return text phrases")
            detections = []
            for score, box, phrase in zip(
                result["scores"].tolist(),
                result["boxes"].tolist(),
                phrases,
                strict=True,
            ):
                clipped = _clip_box(box, image.width, image.height)
                clean_phrase = phrase.strip()
                if (
                    clipped is None
                    or not clean_phrase
                    or not math.isfinite(float(score))
                    or not 0 <= float(score) <= 1
                ):
                    continue
                detections.append(
                    RawGroundingDetection(
                        phrase=clean_phrase,
                        score=float(score),
                        x_min=clipped[0],
                        y_min=clipped[1],
                        x_max=clipped[2],
                        y_max=clipped[3],
                    )
                )
        except Exception as exc:
            raise ModelExecutionError("Grounding DINO execution failed") from exc
        detections.sort(key=lambda detection: detection.score, reverse=True)
        return GroundingBackendResult(
            input_width=image.width,
            input_height=image.height,
            detections=tuple(detections),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailableError(
                "The local grounding runtime dependencies are unavailable"
            ) from exc
        if self.settings.device != "cpu" and not (
            self.settings.device.startswith("cuda") and torch.cuda.is_available()
        ):
            raise ModelUnavailableError(
                f"Configured grounding device {self.settings.device!r} is unavailable"
            )
        cache_dir = (self.settings.model_root / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=self.registration.model_id,
                    revision=self.registration.revision,
                    cache_dir=cache_dir,
                    local_files_only=not self.settings.allow_remote_network,
                    allow_patterns=(
                        self.registration.checkpoint_file,
                        "*.json",
                        "*.txt",
                    ),
                    max_workers=1,
                )
            )
            checkpoint = snapshot / self.registration.checkpoint_file
            if _sha256(checkpoint) != self.registration.checkpoint_sha256:
                raise ModelUnavailableError(
                    "Cached grounding checkpoint hash is invalid"
                )
            processor = AutoProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
            )
            _validate_processor_profile(processor, self.profile)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
                dtype=torch.float32,
            )
            torch.set_num_threads(self.settings.cpu_threads)
            model.to(self.settings.device)
            model.eval()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "The registered grounding checkpoint is not available locally"
            ) from exc
        self._torch = torch
        self._processor = processor
        self._model = model


class TextGuidedGroundingService:
    def __init__(
        self,
        store: FilesystemObservationStore,
        *,
        settings: GroundingRuntimeSettings | None = None,
        backend: GroundingBackend | None = None,
        model_registry_id: str = DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
    ) -> None:
        self.settings = settings or GroundingRuntimeSettings.from_env()
        models = load_model_registry()
        profiles = load_preprocessing_registry()
        try:
            registration = models.models[model_registry_id]
            profile = profiles.profiles[registration.preprocessing_profile]
        except KeyError as exc:
            raise ValueError("Grounding registry references an unknown entry") from exc
        if not isinstance(registration, GroundingModelRegistration) or not isinstance(
            profile, GroundingPreprocessingProfile
        ):
            raise ValueError("Grounding registry task or preprocessing type is invalid")
        self.registration = registration
        self.profile = profile
        self.model_registry_id = model_registry_id
        self._store = store
        self._preprocessor = GroundingImagePreprocessor(profile)
        self._backend = backend or GroundingDinoBackend(
            registration, profile, self.settings
        )

    def ground(self, observation_id: str, query: str) -> GroundingEvidence:
        clean_query = query.strip()
        if (
            not clean_query
            or not any(character.isalnum() for character in clean_query)
            or len(clean_query) > self.settings.max_query_characters
        ):
            raise ModelExecutionError("Grounding query is empty or exceeds the limit")
        registered, visualization_path = self._store.load_registration(observation_id)
        observation = registered.observation
        visualization = registered.visualization
        if (
            not visualization.source_grid_preserved
            or visualization.width != observation.raster.width
            or visualization.height != observation.raster.height
        ):
            raise ModelInputUnsupportedError(
                "Grounding requires a source-grid-preserving visualization"
            )
        prepared = self._preprocessor.from_visualization(
            visualization_path, visualization
        )
        result = self._backend.detect(prepared.image, clean_query)
        if (result.input_width, result.input_height) != prepared.image.size:
            raise EvidenceGeometryError(
                "Grounding backend returned an inconsistent input coordinate space"
            )
        try:
            mapped_detections = tuple(
                self._to_evidence_detection(raw, prepared, observation)
                for raw in result.detections
            )
        except (CoordinateTransformError, ValidationError, ValueError) as exc:
            raise EvidenceGeometryError(
                "Grounding geometry could not be mapped to the source raster"
            ) from exc
        detections, oversized_count = self._select_detections(mapped_detections)
        warnings = [
            "FROZEN_GENERIC_GROUNDING_BASELINE",
            "DISPLAY_DERIVATIVE_USED_FOR_GROUNDING",
        ]
        reasons = ["MODEL_NOT_REMOTE_SENSING_ADAPTED"]
        if observation.geo.crs is None or observation.geo.transform is None:
            warnings.append("GEOREFERENCE_UNAVAILABLE")
        if observation.sensor.modality is Modality.SAR:
            warnings.append("SAR_INTERPRETATION_NOT_SPECIALIZED")
            reasons.append("GENERIC_GROUNDING_SAR_INPUT")
        if oversized_count:
            warnings.append("OVERSIZED_GROUNDING_DETECTIONS_DISCARDED")
        if mapped_detections and not detections and oversized_count:
            warnings.append("GROUNDING_ABSTAINED_OVERSIZED_BOXES")
        if not detections:
            warnings.append("NO_GROUNDING_DETECTIONS")
        return GroundingEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            query=clean_query,
            detections=detections,
            source_observations=(observation.observation_id,),
            source_modalities=(observation.sensor.modality,),
            model=EvidenceModelProvenance(
                registry_id=self.model_registry_id,
                model_id=self.registration.model_id,
                revision=self.registration.revision,
                checkpoint_sha256=self.registration.checkpoint_sha256,
                preprocessing_profile=self.registration.preprocessing_profile,
                preprocessing_version=self.profile.version,
            ),
            domain=DomainAssessment(status=DomainStatus.SHIFTED, reasons=tuple(reasons)),
            warnings=tuple(warnings),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id="text_guided_grounding",
                input_asset_id=visualization.asset_id,
            ),
        )

    def _select_detections(
        self, detections: tuple[GroundingDetection, ...]
    ) -> tuple[tuple[GroundingDetection, ...], int]:
        max_area = self.profile.max_normalized_box_area
        if max_area is None:
            return detections, 0
        candidates = tuple(
            GroundingSelectionCandidate(
                candidate_index=index,
                raw_score=detection.raw_score,
                normalized_xyxy=(
                    detection.normalized_box.x_min,
                    detection.normalized_box.y_min,
                    detection.normalized_box.x_max,
                    detection.normalized_box.y_max,
                ),
            )
            for index, detection in enumerate(detections)
            if detection.raw_score >= self.profile.box_threshold
        )
        selected = select_grounding_candidate(candidates, max_area=max_area)
        oversized_count = sum(
            candidate.normalized_area >= max_area for candidate in candidates
        )
        if selected is None:
            return (), oversized_count
        return (detections[selected.candidate_index],), oversized_count

    @staticmethod
    def _to_evidence_detection(
        raw: RawGroundingDetection,
        prepared: PreparedGroundingImage,
        observation: ObservationState,
    ) -> GroundingDetection:
        model_box = PixelBoundingBox(
            coordinate_space="model_input",
            x_min=raw.x_min,
            y_min=raw.y_min,
            x_max=raw.x_max,
            y_max=raw.y_max,
            image_width=prepared.image.width,
            image_height=prepared.image.height,
        )
        source_values = (
            raw.x_min * prepared.scale_to_source_x,
            raw.y_min * prepared.scale_to_source_y,
            raw.x_max * prepared.scale_to_source_x,
            raw.y_max * prepared.scale_to_source_y,
        )
        source_box = PixelBoundingBox(
            coordinate_space="source_image",
            x_min=source_values[0],
            y_min=source_values[1],
            x_max=source_values[2],
            y_max=source_values[3],
            image_width=prepared.source_width,
            image_height=prepared.source_height,
        )
        normalized = NormalizedBoundingBox(
            x_min=source_box.x_min / source_box.image_width,
            y_min=source_box.y_min / source_box.image_height,
            x_max=source_box.x_max / source_box.image_width,
            y_max=source_box.y_max / source_box.image_height,
        )
        world = _world_polygon(observation, source_box)
        return GroundingDetection(
            detection_id=f"detection_{uuid4().hex}",
            phrase=raw.phrase,
            raw_score=raw.score,
            model_input_box=model_box,
            source_pixel_box=source_box,
            normalized_box=normalized,
            world_polygon=world,
        )


def _world_polygon(
    observation: ObservationState, box: PixelBoundingBox
) -> WorldBoundingPolygon | None:
    if observation.geo.crs is None or observation.geo.transform is None:
        return None
    corners = (
        (box.x_min, box.y_min),
        (box.x_max, box.y_min),
        (box.x_max, box.y_max),
        (box.x_min, box.y_max),
    )
    coordinates = tuple(
        pixel_to_world(
            observation.geo.transform,
            column,
            row,
            offset="upper_left",
        )
        for column, row in corners
    )
    return WorldBoundingPolygon(crs=observation.geo.crs, coordinates=coordinates)


def select_grounding_candidate(
    candidates: tuple[GroundingSelectionCandidate, ...],
    *,
    max_area: float,
) -> GroundingSelectionCandidate | None:
    """Select the highest model score among boxes below the frozen area cap."""

    if not 0 < max_area <= 1:
        raise ValueError("max_area must be in the interval (0, 1]")
    valid = (
        candidate
        for candidate in candidates
        if candidate.normalized_area < max_area
    )
    return max(valid, key=lambda candidate: candidate.raw_score, default=None)


def _format_query(query: str) -> str:
    normalized = " ".join(query.casefold().split()).rstrip(".")
    return f"{normalized}."


def _validate_processor_profile(
    processor: object, profile: GroundingPreprocessingProfile
) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ModelUnavailableError("Grounding checkpoint has no image processor")
    checks = {
        "image_mean": list(profile.image_mean),
        "image_std": list(profile.image_std),
        "rescale_factor": profile.rescale_factor,
    }
    for attribute, expected in checks.items():
        actual = getattr(image_processor, attribute, None)
        if actual is None or not _profile_value_matches(actual, expected):
            raise ModelUnavailableError(
                f"Grounding processor does not match registered {attribute}"
            )


def _profile_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, list):
        try:
            values = list(actual)  # type: ignore[arg-type]
        except TypeError:
            return False
        return len(values) == len(expected) and all(
            math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
            for left, right in zip(values, expected, strict=True)
        )
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _clip_box(
    values: list[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
        return None
    x_min = min(max(float(values[0]), 0.0), float(width))
    y_min = min(max(float(values[1]), 0.0), float(height))
    x_max = min(max(float(values[2]), 0.0), float(width))
    y_max = min(max(float(values[3]), 0.0), float(height))
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
