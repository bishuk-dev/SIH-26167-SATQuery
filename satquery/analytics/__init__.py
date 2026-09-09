"""Deterministic scientific analytics: spectral indices, temporal grids and
differences, and CRS-safe mask-area measurement."""

from satquery.analytics.exceptions import (
    AnalyticsError,
    CrsRequiredForMeasurementError,
    GridPreparationError,
    InvalidRasterArrayError,
    MissingRequiredBandError,
    NoSpatialOverlapError,
    SingularTransformError,
    UnsupportedCrsMeasurementError,
)
from satquery.analytics.measurement import MeasurementResult, measure_mask_area
from satquery.analytics.sar import (
    AgreementResult,
    SarChangeResult,
    SarInputContract,
    mask_agreement,
    sar_temporal_change,
)
from satquery.analytics.spectral import (
    SpectralBandRole,
    SpectralIndex,
    compute_index,
    normalized_difference,
)
from satquery.analytics.temporal import (
    AlignedPair,
    prepare_common_grid,
    temporal_difference,
    threshold_temporal_difference,
)

__all__ = [
    "AgreementResult",
    "AlignedPair",
    "AnalyticsError",
    "CrsRequiredForMeasurementError",
    "GridPreparationError",
    "InvalidRasterArrayError",
    "MeasurementResult",
    "MissingRequiredBandError",
    "NoSpatialOverlapError",
    "SarChangeResult",
    "SarInputContract",
    "SingularTransformError",
    "SourceAssetIntegrityError",
    "SpectralBandRole",
    "SpectralIndex",
    "UnsupportedCrsMeasurementError",
    "compute_index",
    "mask_agreement",
    "measure_mask_area",
    "normalized_difference",
    "prepare_common_grid",
    "sar_temporal_change",
    "temporal_difference",
    "threshold_temporal_difference",
]
