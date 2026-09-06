"""Sensor definitions and semantic band mapping for SatQuery."""

from satquery.sensors.semantics import (
    SemanticBandRole,
    get_semantic_band,
    resolve_band_for_role,
)

__all__ = [
    "SemanticBandRole",
    "get_semantic_band",
    "resolve_band_for_role",
]
