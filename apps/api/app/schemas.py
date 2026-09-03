"""Public HTTP schemas for observation ingestion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from satquery.ingestion.models import (
    GeoMetadata,
    ObservationProvenance,
    RasterMetadata,
    SensorMetadata,
    TemporalMetadata,
    ValidityMetadata,
)
from satquery.visualization.models import (
    ObservationRegistration,
    RenderingMode,
    TileScheme,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetResponse(ApiModel):
    asset_id: str
    original_name: str
    sha256: str
    immutable: Literal[True]
    kind: Literal["original"] = "original"


class TileExtentResponse(ApiModel):
    left: float
    bottom: float
    right: float
    top: float


class VisualizationAssetResponse(ApiModel):
    asset_id: str
    parent_asset_id: str
    kind: Literal["visualization"]
    sha256: str
    immutable: Literal[True]
    format: Literal["COG"]
    rendering: RenderingMode
    source_band_indexes: tuple[int, ...]
    source_grid_preserved: Literal[True]
    tile_url_template: str
    tile_scheme: TileScheme
    tile_crs: str | None
    tile_extent: TileExtentResponse
    pixel_y_axis: Literal["down"] | None = None


class ObservationMetadataResponse(ApiModel):
    raster: RasterMetadata
    sensor: SensorMetadata
    geo: GeoMetadata
    temporal: TemporalMetadata
    provenance: ObservationProvenance


class ObservationUploadResponse(ApiModel):
    observation_id: str
    status: Literal["READY"] = "READY"
    asset: AssetResponse
    visualization: VisualizationAssetResponse
    metadata: ObservationMetadataResponse
    validity: ValidityMetadata
    warnings: tuple[str, ...]

    @classmethod
    def from_registration(
        cls, registration: ObservationRegistration
    ) -> ObservationUploadResponse:
        state = registration.observation
        visualization = registration.visualization
        return cls(
            observation_id=state.observation_id,
            asset=AssetResponse(
                asset_id=state.source_asset.asset_id,
                original_name=state.source_asset.original_name,
                sha256=state.source_asset.sha256,
                immutable=state.source_asset.immutable,
            ),
            visualization=VisualizationAssetResponse(
                asset_id=visualization.asset_id,
                parent_asset_id=visualization.parent_asset_id,
                kind=visualization.kind,
                sha256=visualization.sha256,
                immutable=visualization.immutable,
                format=visualization.format,
                rendering=visualization.rendering,
                source_band_indexes=visualization.source_band_indexes,
                source_grid_preserved=visualization.source_grid_preserved,
                tile_url_template=(
                    f"/tiles/{visualization.asset_id}/{{z}}/{{x}}/{{y}}.png"
                ),
                tile_scheme=visualization.tile_scheme,
                tile_crs=visualization.tile_crs,
                tile_extent=TileExtentResponse(
                    **visualization.tile_extent.model_dump()
                ),
                pixel_y_axis=(
                    "down" if visualization.tile_scheme is TileScheme.PIXEL else None
                ),
            ),
            metadata=ObservationMetadataResponse(
                raster=state.raster,
                sensor=state.sensor,
                geo=state.geo,
                temporal=state.temporal,
                provenance=state.provenance,
            ),
            validity=state.validity,
            warnings=state.validity.warnings,
        )


class FailureDetail(ApiModel):
    code: str
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    outcome: Literal["ALLOW", "ALLOW_WITH_WARNING", "REQUEST_INPUT", "ABSTAIN", "REJECT"]
    user_message: str
    technical_message: str
    affected_requirement: str | None = None
    recoverable: bool
    required_action: dict[str, object] | None = None
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ErrorResponse(ApiModel):
    error: FailureDetail
