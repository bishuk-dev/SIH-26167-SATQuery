"""Typed contracts for inspected remote-sensing observations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """Immutable base for domain records exchanged across SatQuery boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class MetadataQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SourceAsset(ContractModel):
    asset_id: str = Field(min_length=1)
    original_name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    immutable: Literal[True] = True


class RasterMetadata(ContractModel):
    driver: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtypes: tuple[str, ...]
    nodata: tuple[float | int | None, ...]
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_band_aligned_fields(self) -> RasterMetadata:
        if len(self.dtypes) != self.band_count:
            raise ValueError("dtypes must contain one value per raster band")
        if len(self.nodata) != self.band_count:
            raise ValueError("nodata must contain one value per raster band")
        return self


class BandMetadata(ContractModel):
    index: int = Field(gt=0)
    description: str | None = None
    dtype: str = Field(min_length=1)
    nodata: float | int | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class SensorMetadata(ContractModel):
    modality: Modality = Modality.UNKNOWN
    sensor_name: str | None = None
    platform: str | None = None
    product_level: str | None = None
    bands: tuple[BandMetadata, ...]
    polarizations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_band_indexes(self) -> SensorMetadata:
        expected = tuple(range(1, len(self.bands) + 1))
        actual = tuple(band.index for band in self.bands)
        if actual != expected:
            raise ValueError("bands must be ordered with contiguous one-based indexes")
        return self


class AffineTransform(ContractModel):
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


class GeoBounds(ContractModel):
    left: float
    bottom: float
    right: float
    top: float

    @model_validator(mode="after")
    def validate_order(self) -> GeoBounds:
        if self.left > self.right or self.bottom > self.top:
            raise ValueError("bounds must be ordered left/bottom/right/top")
        return self


class GeoMetadata(ContractModel):
    crs: str | None = None
    transform: AffineTransform | None = None
    bounds: GeoBounds | None = None
    native_gsd_x: float | None = Field(default=None, gt=0)
    native_gsd_y: float | None = Field(default=None, gt=0)
    units: str | None = None

    @model_validator(mode="after")
    def validate_transform_dependent_fields(self) -> GeoMetadata:
        derived_values = (self.bounds, self.native_gsd_x, self.native_gsd_y)
        if self.transform is None and any(value is not None for value in derived_values):
            raise ValueError("bounds and native GSD require an affine transform")
        return self


class TemporalMetadata(ContractModel):
    acquisition_time: datetime | None = None

    @field_validator("acquisition_time")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("acquisition_time must include a timezone")
        return value


class ValidityMetadata(ContractModel):
    has_crs: bool
    has_transform: bool
    has_nodata: bool
    metadata_quality: MetadataQuality = MetadataQuality.UNKNOWN
    warnings: tuple[str, ...] = ()


class ObservationProvenance(ContractModel):
    created_at: datetime
    ingestion_version: str = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class ObservationState(ContractModel):
    observation_id: str = Field(min_length=1)
    source_asset: SourceAsset
    raster: RasterMetadata
    sensor: SensorMetadata
    geo: GeoMetadata
    temporal: TemporalMetadata
    validity: ValidityMetadata
    provenance: ObservationProvenance

    @model_validator(mode="after")
    def validate_band_count(self) -> ObservationState:
        if len(self.sensor.bands) != self.raster.band_count:
            raise ValueError("sensor bands must match raster band_count")
        expected_flags = {
            "has_crs": self.geo.crs is not None,
            "has_transform": self.geo.transform is not None,
            "has_nodata": any(value is not None for value in self.raster.nodata),
        }
        actual_flags = {
            "has_crs": self.validity.has_crs,
            "has_transform": self.validity.has_transform,
            "has_nodata": self.validity.has_nodata,
        }
        if actual_flags != expected_flags:
            raise ValueError("validity flags must match the observation metadata")
        return self
