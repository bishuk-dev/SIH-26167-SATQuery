"""Failures raised at the raster inspection boundary."""


class IngestionError(Exception):
    """Base class for safe, user-mappable ingestion failures."""

    code = "INVALID_UPLOAD"


class RasterInspectionError(IngestionError):
    """Base class for raster metadata inspection failures."""

    code = "RASTER_METADATA_INVALID"


class InvalidRasterError(RasterInspectionError):
    code = "INVALID_RASTER"


class UnsupportedRasterDriverError(RasterInspectionError):
    code = "UNSUPPORTED_RASTER_DRIVER"


class RasterDriverMismatchError(RasterInspectionError):
    code = "RASTER_DRIVER_MISMATCH"


class RasterResourceLimitError(RasterInspectionError):
    code = "RASTER_RESOURCE_LIMIT_EXCEEDED"


class InvalidUploadError(IngestionError):
    code = "INVALID_UPLOAD"


class AssetStorageError(IngestionError):
    code = "ASSET_STORAGE_FAILED"
