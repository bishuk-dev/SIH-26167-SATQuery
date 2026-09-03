"""Exceptions raised by geospatial coordinate operations."""

from __future__ import annotations


class GeospatialError(ValueError):
    """Base class for invalid geospatial operations or inputs."""


class CoordinateTransformError(GeospatialError):
    """Raised when coordinates cannot be transformed safely."""

