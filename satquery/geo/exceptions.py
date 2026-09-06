"""Exceptions raised by geospatial coordinate operations."""

from __future__ import annotations


class GeospatialError(ValueError):
    """Base class for invalid geospatial operations or inputs."""


class CoordinateTransformError(GeospatialError):
    """Raised when coordinates cannot be transformed safely."""


class MeasurementError(GeospatialError):
    """Raised when deterministic GIS measurements cannot be computed."""


class CrsMeasurementError(MeasurementError):
    """Raised when CRS is invalid, angular, or missing for metric measurement."""


class SpectralIndexError(GeospatialError):
    """Raised when spectral index calculation fails or is invalid."""


class MissingBandError(SpectralIndexError):
    """Raised when required spectral bands are missing."""

