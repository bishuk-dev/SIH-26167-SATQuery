"""Failures at the visualization derivative and tile boundaries."""

from satquery.ingestion.exceptions import IngestionError


class VisualizationError(IngestionError):
    code = "VISUALIZATION_FAILED"


class VisualizationGenerationError(VisualizationError):
    code = "VISUALIZATION_GENERATION_FAILED"


class VisualizationResourceLimitError(VisualizationError):
    code = "VISUALIZATION_RESOURCE_LIMIT_EXCEEDED"


class VisualizationAssetNotFoundError(VisualizationError):
    code = "VISUALIZATION_ASSET_NOT_FOUND"


class InvalidTileRequestError(VisualizationError):
    code = "INVALID_TILE_REQUEST"


class TileRenderingError(VisualizationError):
    code = "TILE_RENDERING_FAILED"
