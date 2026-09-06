"""Typed domain contracts for temporal observation pairs, ROIs, and change results."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from satquery.evidence.models import EvidenceModelProvenance
from satquery.ingestion.models import ContractModel


class AnalysisROI(ContractModel):
    """Region of interest defined in spatial or pixel coordinates."""

    roi_id: str = Field(min_length=1)
    crs: str = Field(min_length=1)
    geometry_type: Literal["bbox", "polygon"]
    coordinates: tuple[tuple[float, float], ...]
    pixel_bounds: tuple[int, int, int, int] | None = None  # (min_x, min_y, max_x, max_y)

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, coords: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        if len(coords) < 3:
            raise ValueError("ROI coordinates must contain at least 3 points")
        for pt in coords:
            if len(pt) != 2 or not all(math.isfinite(val) for val in pt):
                raise ValueError(f"Invalid coordinate point: {pt}")
        return coords

    @model_validator(mode="after")
    def validate_pixel_bounds(self) -> AnalysisROI:
        if self.pixel_bounds is not None:
            min_x, min_y, max_x, max_y = self.pixel_bounds
            if min_x >= max_x or min_y >= max_y or min_x < 0 or min_y < 0:
                raise ValueError(f"Invalid pixel bounds: {self.pixel_bounds}")
        return self


class TemporalObservationPair(ContractModel):
    """Temporal observation pair contract enforcing chronological ordering and spatial overlap."""

    pair_id: str = Field(min_length=1)
    pre_observation_id: str = Field(min_length=1)
    post_observation_id: str = Field(min_length=1)
    pre_acquisition_time: datetime | None = None
    post_acquisition_time: datetime | None = None
    delta_seconds: float | None = None
    overlap_fraction: float = Field(ge=0.0, le=1.0)
    grid_aligned: bool = False
    crs: str = Field(min_length=1)
    modality_pair: str = Field(min_length=1)
    is_valid_temporal_order: bool = True

    @model_validator(mode="after")
    def validate_pair_consistency(self) -> TemporalObservationPair:
        if self.pre_observation_id == self.post_observation_id:
            raise ValueError("Temporal pair must contain distinct observations; duplicate ID provided")
        if (
            self.pre_acquisition_time is not None
            and self.post_acquisition_time is not None
            and self.is_valid_temporal_order
        ):
            if self.pre_acquisition_time >= self.post_acquisition_time:
                raise ValueError("Pre-observation acquisition time must precede post-observation")
        return self


class TemporalChangeResult(ContractModel):
    """Structured result of a deterministic bi-temporal change detection computation."""

    change_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    change_type: str = Field(min_length=1)
    delta_stats: dict[str, float]
    change_area_m2: float = Field(ge=0.0)
    change_pixel_count: int = Field(ge=0)
    baseline_area_m2: float | None = None
    percent_change: float | None = None
    evidence_mask_id: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ChangeVQAResult(ContractModel):
    """Answer and evidence lineage for bi-temporal visual question answering."""

    query: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    confidence: float | None = None
    pair_id: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    model_provenance: EvidenceModelProvenance | None = None
    limitations: tuple[str, ...] = ()
