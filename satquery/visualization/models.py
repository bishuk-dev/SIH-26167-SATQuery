"""Typed visualization asset and provenance records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from satquery.ingestion.models import (
    AffineTransform,
    ContractModel,
    GeoBounds,
    ObservationState,
)


class RenderingMode(StrEnum):
    RGB = "rgb"
    GRAYSCALE = "grayscale"


class TileScheme(StrEnum):
    WEB_MERCATOR = "web_mercator"
    PIXEL = "pixel"


class BandStretch(ContractModel):
    source_band: int = Field(gt=0)
    lower: float
    upper: float
    scale: Literal["linear", "log1p", "magnitude", "magnitude_log1p"]


class VisualizationAsset(ContractModel):
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    parent_asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    kind: Literal["visualization"] = "visualization"
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    immutable: Literal[True] = True
    format: Literal["COG"] = "COG"
    media_type: Literal["image/tiff"] = "image/tiff"
    rendering: RenderingMode
    source_band_indexes: tuple[int, ...]
    stretches: tuple[BandStretch, ...]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    color_band_count: Literal[1, 3]
    alpha_band: int = Field(gt=0)
    crs: str | None = None
    transform: AffineTransform | None = None
    source_bounds: GeoBounds | None = None
    source_grid_preserved: Literal[True] = True
    tile_scheme: TileScheme
    tile_crs: str | None = None
    tile_extent: GeoBounds
    created_at: datetime
    generator_version: str = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class ObservationRegistration(ContractModel):
    observation: ObservationState
    visualization: VisualizationAsset
