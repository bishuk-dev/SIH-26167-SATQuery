"""Deterministic remote-sensing analytics."""

from satquery.analytics.measurement import (
    CrsRequiredForMeasurementError,
    MeasurementResult,
    measure_mask_area,
)
from satquery.analytics.spectral import (
    MissingRequiredBandError,
    compute_index,
    normalized_difference,
)
from satquery.analytics.temporal import (
    AlignedPair,
    prepare_common_grid,
    threshold_temporal_difference,
)

__all__ = [
    "AlignedPair",
    "CrsRequiredForMeasurementError",
    "MeasurementResult",
    "MissingRequiredBandError",
    "compute_index",
    "measure_mask_area",
    "normalized_difference",
    "prepare_common_grid",
    "threshold_temporal_difference",
]
