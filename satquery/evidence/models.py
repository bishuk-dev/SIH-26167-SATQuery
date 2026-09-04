"""Structured evidence returned by deterministic inference adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from satquery.ingestion.models import ContractModel, Modality


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
