"""Geospatial domain contracts and deterministic coordinate operations."""

from satquery.geo.coordinates import (
    crop_to_source,
    pixel_to_world,
    transform_bounds,
    world_to_pixel,
)
from satquery.geo.models import (
    CompatibilityStatus,
    PairCompatibility,
    PixelWindow,
    RegistrationStatus,
)
from satquery.geo.pairing import PairValidator

__all__ = [
    "CompatibilityStatus",
    "PairCompatibility",
    "PairValidator",
    "PixelWindow",
    "RegistrationStatus",
    "crop_to_source",
    "pixel_to_world",
    "transform_bounds",
    "world_to_pixel",
]
