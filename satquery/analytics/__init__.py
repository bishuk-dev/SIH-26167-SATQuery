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
from satquery.analytics.sar import (
    AgreementResult,
    SarChangeResult,
    SarInputContract,
    UnknownSarSemanticsError,
    mask_agreement,
    sar_temporal_change,
)
from satquery.analytics.temporal import (
    AlignedPair,
    prepare_common_grid,
    threshold_temporal_difference,
)

__all__ = [
    "AgreementResult",
    "AlignedPair",
    "CrsRequiredForMeasurementError",
    "MeasurementResult",
    "MissingRequiredBandError",
    "SarChangeResult",
    "SarInputContract",
    "UnknownSarSemanticsError",
    "compute_index",
    "mask_agreement",
    "measure_mask_area",
    "normalized_difference",
    "prepare_common_grid",
    "sar_temporal_change",
    "threshold_temporal_difference",
]
