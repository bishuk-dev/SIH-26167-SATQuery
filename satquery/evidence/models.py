"""Structured evidence returned by deterministic inference adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from satquery.ingestion.models import AffineTransform, ContractModel, GeoBounds, Modality


class DomainStatus(StrEnum):
    IN_DOMAIN = "in_domain"
    SHIFTED = "shifted"
    UNKNOWN = "unknown"


class VqaPrediction(ContractModel):
    answer: str = Field(min_length=1)
    raw_score: float | None = None


class EvidenceModelProvenance(ContractModel):
    registry_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_profile: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)


class DomainAssessment(ContractModel):
    status: DomainStatus
    reasons: tuple[str, ...] = ()


class EvidenceProvenance(ContractModel):
    created_at: datetime
    operation_id: str = Field(min_length=1)
    input_asset_id: str = Field(min_length=1)
    parent_evidence_ids: tuple[str, ...] = ()

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class VqaEvidence(ContractModel):
    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["single_image_vqa"] = "single_image_vqa"
    prediction: VqaPrediction
    source_observations: tuple[str, ...]
    source_modalities: tuple[Modality, ...]
    model: EvidenceModelProvenance
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance


class PixelBoundingBox(ContractModel):
    coordinate_space: Literal["model_input", "source_image"]
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(ge=0)
    y_max: float = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> PixelBoundingBox:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box must have positive area")
        if self.x_max > self.image_width or self.y_max > self.image_height:
            raise ValueError("bounding box exceeds its image coordinate space")
        return self


class NormalizedBoundingBox(ContractModel):
    coordinate_space: Literal["source_normalized"] = "source_normalized"
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> NormalizedBoundingBox:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("normalized bounding box must have positive area")
        return self


class WorldBoundingPolygon(ContractModel):
    crs: str = Field(min_length=1)
    coordinates: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    @field_validator("coordinates")
    @classmethod
    def require_finite_coordinates(
        cls,
        value: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        if not all(math.isfinite(number) for point in value for number in point):
            raise ValueError("world coordinates must be finite")
        return value


class GroundingDetection(ContractModel):
    detection_id: str = Field(pattern=r"^detection_[0-9a-f]{32}$")
    phrase: str = Field(min_length=1)
    raw_score: float = Field(ge=0, le=1)
    model_input_box: PixelBoundingBox
    source_pixel_box: PixelBoundingBox
    normalized_box: NormalizedBoundingBox
    world_polygon: WorldBoundingPolygon | None = None

    @model_validator(mode="after")
    def validate_coordinate_spaces(self) -> GroundingDetection:
        if self.model_input_box.coordinate_space != "model_input":
            raise ValueError("model_input_box has the wrong coordinate space")
        if self.source_pixel_box.coordinate_space != "source_image":
            raise ValueError("source_pixel_box has the wrong coordinate space")
        return self


class GroundingEvidence(ContractModel):
    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["text_guided_grounding"] = "text_guided_grounding"
    query: str = Field(min_length=1)
    detections: tuple[GroundingDetection, ...]
    source_observations: tuple[str, ...]
    source_modalities: tuple[Modality, ...]
    model: EvidenceModelProvenance
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance


class MaskAsset(ContractModel):
    """A saved binary mask asset and the grid it exists on.

    ``source_grid_observation_id`` identifies the observation grid the mask
    maps to; it is recorded, never inferred. Pixel-space masks (benchmark
    PNG/JPEG imagery without georeferencing) are valid with all three
    georeferencing fields unset; partial georeferencing is not.
    """

    asset_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crs: str | None = None
    transform: AffineTransform | None = None
    bounds: GeoBounds | None = None
    source_grid_observation_id: str = Field(min_length=1)
    value_semantics: Literal["binary_0_1"] = "binary_0_1"
    immutable: Literal[True] = True

    @model_validator(mode="after")
    def validate_georeferencing_coherence(self) -> MaskAsset:
        present = (self.crs is not None, self.transform is not None, self.bounds is not None)
        if any(present) and not all(present):
            raise ValueError(
                "georeferenced masks require crs, transform, and bounds together"
            )
        return self


class TemporalPairEvidence(ContractModel):
    """Ordered temporal pair provenance; the order source is explicit."""

    t1_observation_id: str = Field(min_length=1)
    t2_observation_id: str = Field(min_length=1)
    order_source: Literal[
        "metadata",
        "explicit_user_mapping",
        "frozen_dataset_contract",
    ]

    @model_validator(mode="after")
    def validate_distinct_observations(self) -> TemporalPairEvidence:
        if self.t1_observation_id == self.t2_observation_id:
            raise ValueError("T1 and T2 must be distinct observations")
        return self


class ChangeMaskEvidence(ContractModel):
    """Change-localization evidence; target class and domain limits stay
    explicit. This is never a generic "change truth" record."""

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["change_localize"] = "change_localize"
    target_class: str = Field(min_length=1)
    change_kind: Literal["gain", "loss", "symmetric_change"]
    temporal: TemporalPairEvidence
    mask: MaskAsset
    # model-dependent score semantics; not guaranteed to be a probability
    raw_model_score: float | None = None
    model: EvidenceModelProvenance | None = None
    tool_id: str | None = None
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance

    @field_validator("raw_model_score")
    @classmethod
    def require_finite_raw_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("raw_model_score must be finite")
        return value

    @model_validator(mode="after")
    def require_producer(self) -> ChangeMaskEvidence:
        if self.model is None and self.tool_id is None:
            raise ValueError("evidence requires a model or tool producer")
        return self


class FloodMaskEvidence(ContractModel):
    """Single-observation water/flood segmentation evidence.

    Sensor-agnostic at the evidence layer; sensor and preprocessing
    restrictions live in the producing service and model registration.
    """

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["flood_segmentation"] = "flood_segmentation"
    source_observation_id: str = Field(min_length=1)
    source_modality: Modality
    target_class: Literal[
        "water_extent",
        "flood_extent",
        "water_or_flood_extent",
    ]
    mask: MaskAsset
    raw_model_score: float | None = None
    model: EvidenceModelProvenance | None = None
    tool_id: str | None = None
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance

    @field_validator("raw_model_score")
    @classmethod
    def require_finite_raw_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("raw_model_score must be finite")
        return value

    @model_validator(mode="after")
    def require_producer(self) -> FloodMaskEvidence:
        if self.model is None and self.tool_id is None:
            raise ValueError("evidence requires a model or tool producer")
        return self


class ChangeCaptionEvidence(ContractModel):
    """Text-only change description evidence.

    Carries no mask, measurement, pixel count, area, or numeric confidence:
    captions never support spatial or quantitative claims.
    """

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["change_captioning"] = "change_captioning"
    temporal: TemporalPairEvidence
    caption: str
    model: EvidenceModelProvenance
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance

    @field_validator("caption")
    @classmethod
    def require_visible_caption(cls, value: str) -> str:
        caption = value.strip()
        if not caption:
            raise ValueError("caption must contain non-whitespace text")
        return caption


class MeasurementEvidence(ContractModel):
    """Deterministic GIS measurement of a mask evidence result.

    Phase 4 supports AREA only; other measurement types are added when
    implemented scientifically. Values come from registered tools, never
    from caption or VQA output.
    """

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["measurement"] = "measurement"
    measurement_type: Literal["area"] = "area"
    source_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    value: float = Field(ge=0)
    unit: Literal["m2", "ha", "km2"]
    method: str = Field(min_length=1)
    calculation_crs: str = Field(min_length=1)
    positive_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(ge=0)
    tool_id: str = Field(min_length=1)
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance

    @field_validator("value")
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("measurement value must be finite")
        return value

    @model_validator(mode="after")
    def validate_pixel_counts(self) -> MeasurementEvidence:
        if self.positive_pixel_count > self.valid_pixel_count:
            raise ValueError(
                "positive_pixel_count cannot exceed valid_pixel_count"
            )
        return self


class AgreementEvidence(ContractModel):
    """Diagnostic agreement between two mask evidences.

    Agreement is not accuracy and must not drive ALLOW/WARN/ABSTAIN policy:
    this contract intentionally has no confidence, outcome, or threshold
    fields.
    """

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["mask_agreement"] = "mask_agreement"
    first_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    second_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    metric: Literal["mask_iou"]
    value: float | None = None
    intersection_pixel_count: int = Field(ge=0)
    union_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(ge=0)
    interpretation: Literal["agreement_not_accuracy"] = "agreement_not_accuracy"
    tool_id: str = Field(min_length=1)
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def validate_agreement(self) -> AgreementEvidence:
        if self.first_evidence_id == self.second_evidence_id:
            raise ValueError("agreement requires two distinct evidence IDs")
        if self.intersection_pixel_count > self.union_pixel_count:
            raise ValueError("intersection cannot exceed union")
        if self.union_pixel_count > self.valid_pixel_count:
            raise ValueError("union cannot exceed the valid pixel count")
        if self.union_pixel_count == 0:
            if self.value is not None:
                raise ValueError("an empty union has no defined IoU; value must be null")
        else:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("a non-empty union requires a finite IoU value")
            if not 0.0 <= self.value <= 1.0:
                raise ValueError("IoU must be within [0, 1]")
        return self
