"""Domain exceptions for deterministic analytics operations."""

from __future__ import annotations


class AnalyticsError(ValueError):
    """Base class for deterministic analytics failures."""


class MissingRequiredBandError(AnalyticsError):
    """Raised when an index lacks a semantically identified input band."""


class InvalidRasterArrayError(AnalyticsError):
    """Raised when array inputs violate shape, dimensionality, or value rules."""


class GridPreparationError(AnalyticsError):
    """Raised when a temporal pair cannot be placed on a common grid."""


class NoSpatialOverlapError(GridPreparationError):
    """Raised when a temporal pair has known zero spatial overlap."""


class SingularTransformError(InvalidRasterArrayError):
    """Raised when an affine transform is non-invertible or has zero area."""


class CrsRequiredForMeasurementError(AnalyticsError):
    """Raised when an area cannot be computed without a valid spatial reference."""


class UnsupportedCrsMeasurementError(AnalyticsError):
    """Raised when a CRS cannot support a confident deterministic measurement."""
