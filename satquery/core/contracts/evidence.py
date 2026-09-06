"""Structured evidence contracts for raster indices, masks, measurements, and change results."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, field_validator, model_validator

from satquery.ingestion.models import ContractModel


class RasterStatistic(ContractModel):
    """Statistical summary of valid pixels in a raster layer."""

    min_value: float
    max_value: float
    mean_value: float
    std_value: float = Field(ge=0.0)
    valid_pixels: int = Field(ge=0)
    nodata_pixels: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_stats(self) -> RasterStatistic:
        if not all(
            math.isfinite(val)
            for val in (self.min_value, self.max_value, self.mean_value, self.std_value)
        ):
            raise ValueError("All statistic values must be finite numbers")
        if self.min_value > self.max_value:
            raise ValueError(f"min_value ({self.min_value}) cannot exceed max_value ({self.max_value})")
        return self


class IndexRasterEvidence(ContractModel):
    """Evidence artifact generated from a computed spectral index."""

    evidence_id: str = Field(min_length=1)
    index_name: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    stats: RasterStatistic
    crs: str = Field(min_length=1)
    transform: tuple[float, float, float, float, float, float]
    shape: tuple[int, int]
    provenance: dict[str, Any] = Field(default_factory=dict)


class MaskEvidence(ContractModel):
    """Boolean mask evidence derived from deterministic thresholds."""

    evidence_id: str = Field(min_length=1)
    mask_type: str = Field(min_length=1)
    source_observation_id: str = Field(min_length=1)
    positive_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0)
    threshold_criteria: dict[str, float] = Field(default_factory=dict)

    @property
    def positive_fraction(self) -> float:
        return float(self.positive_pixels / self.total_pixels)


class ChangeMaskEvidence(ContractModel):
    """Boolean change mask evidence derived from bi-temporal differencing."""

    evidence_id: str = Field(min_length=1)
    change_type: str = Field(min_length=1)
    pre_observation_id: str = Field(min_length=1)
    post_observation_id: str = Field(min_length=1)
    changed_pixels: int = Field(ge=0)
    total_pixels: int = Field(gt=0)
    change_fraction: float = Field(ge=0.0, le=1.0)
    threshold: float
    provenance: dict[str, Any] = Field(default_factory=dict)


class MeasurementEvidence(ContractModel):
    """Physical GIS measurement evidence calculated from raster geometry."""

    evidence_id: str = Field(min_length=1)
    measurement_type: str = Field(min_length=1)
    area_value: float = Field(ge=0.0)
    unit: str = Field(min_length=1)
    pixel_count: int = Field(ge=0)
    pixel_area_m2: float = Field(ge=0.0)
    crs: str = Field(min_length=1)
    calculation_path: str = Field(min_length=1)


class SpectralAnalysisResult(ContractModel):
    """Complete output of a deterministic spectral analysis operation."""

    analysis_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    index_name: str = Field(min_length=1)
    stats: RasterStatistic
    measurement: MeasurementEvidence | None = None
    warnings: tuple[str, ...] = ()


class SarChangeResult(ContractModel):
    """Complete output of a deterministic SAR temporal backscatter change computation."""

    analysis_id: str = Field(min_length=1)
    pre_observation_id: str = Field(min_length=1)
    post_observation_id: str = Field(min_length=1)
    polarization: str = Field(min_length=1)
    flood_detected: bool
    flood_area_m2: float = Field(ge=0.0)
    flood_pixel_count: int = Field(ge=0)
    backscatter_decrease_db_threshold: float
    mean_backscatter_delta_db: float
    post_event_water_area_m2: float | None = None
    post_event_water_pixel_count: int | None = None
    flood_expansion_area_m2: float | None = None
    flood_expansion_pixel_count: int | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

