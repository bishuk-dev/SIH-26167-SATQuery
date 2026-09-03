"""Typed contracts for observation-pair compatibility."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from satquery.ingestion.models import ContractModel


class CompatibilityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class RegistrationStatus(StrEnum):
    VERIFIED = "verified"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class ModalityPairType(StrEnum):
    OPTICAL_SAR = "optical_sar"
    TEMPORAL_SAME_MODALITY = "temporal_same_modality"
    TEMPORAL_CROSS_MODAL = "temporal_cross_modal"
    UNKNOWN = "unknown"


class OverlapCompatibility(ContractModel):
    known: bool
    overlap_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    sufficient: bool | None = None

    @model_validator(mode="after")
    def validate_known_state(self) -> Self:
        if self.known and (self.overlap_fraction is None or self.sufficient is None):
            raise ValueError("known overlap requires a fraction and sufficiency result")
        if not self.known and (
            self.overlap_fraction is not None or self.sufficient is not None
        ):
            raise ValueError("unknown overlap cannot include calculated values")
        return self


class CrsCompatibility(ContractModel):
    equal: bool | None = None
    transformable: bool | None = None


class GridCompatibility(ContractModel):
    same_shape: bool
    aligned: bool | None = None
    same_resolution: bool | None = None


class TemporalCompatibility(ContractModel):
    order_known: bool
    first: str | None = None
    second: str | None = None
    time_delta_seconds: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        ordered_values = (self.first, self.second, self.time_delta_seconds)
        if self.order_known and any(value is None for value in ordered_values):
            raise ValueError("known temporal order requires both IDs and a time delta")
        if not self.order_known and any(value is not None for value in ordered_values):
            raise ValueError("unknown temporal order cannot name an ordering")
        return self


class ModalityCompatibility(ContractModel):
    pair_type: ModalityPairType
    compatible: bool | None = None


class PairResult(ContractModel):
    status: CompatibilityStatus
    reasons: tuple[str, ...] = ()


class PairCompatibility(ContractModel):
    observation_a: str
    observation_b: str
    overlap: OverlapCompatibility
    crs: CrsCompatibility
    grid: GridCompatibility
    temporal: TemporalCompatibility
    modality: ModalityCompatibility
    registration: RegistrationStatus
    result: PairResult


class PixelWindow(ContractModel):
    """A source-raster window expressed in pixel coordinates."""

    column_offset: float = Field(ge=0.0)
    row_offset: float = Field(ge=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)

