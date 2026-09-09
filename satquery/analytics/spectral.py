"""Deterministic semantic spectral-index calculations.

Bands are consumed by semantic role only; mapping sensor bands to roles
belongs to dataset/sensor ingestion adapters, never to this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import numpy as np

from satquery.analytics.exceptions import (
    InvalidRasterArrayError,
    MissingRequiredBandError,
)


class SpectralBandRole(str, Enum):
    RED = "RED"
    GREEN = "GREEN"
    NIR = "NIR"
    SWIR1 = "SWIR1"


class SpectralIndex(str, Enum):
    NDVI = "NDVI"
    NDWI = "NDWI"
    MNDWI = "MNDWI"


_FORMULAS: dict[SpectralIndex, tuple[SpectralBandRole, SpectralBandRole]] = {
    SpectralIndex.NDVI: (SpectralBandRole.NIR, SpectralBandRole.RED),
    SpectralIndex.NDWI: (SpectralBandRole.GREEN, SpectralBandRole.NIR),
    SpectralIndex.MNDWI: (SpectralBandRole.GREEN, SpectralBandRole.SWIR1),
}


def normalized_difference(
    numerator_band: np.ndarray,
    denominator_band: np.ndarray,
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    """Return ``(numerator - denominator) / (numerator + denominator)``.

    Invalid cells — outside ``valid``, non-finite inputs, or a zero
    denominator — stay masked and are never coerced into numeric values.
    """

    numerator = np.asarray(numerator_band, dtype="float64")
    denominator_band_array = np.asarray(denominator_band, dtype="float64")
    valid_array = np.asarray(valid, dtype=bool)
    if numerator.ndim != 2 or denominator_band_array.ndim != 2 or valid_array.ndim != 2:
        raise InvalidRasterArrayError("spectral arrays must be two-dimensional")
    if (
        numerator.shape != denominator_band_array.shape
        or numerator.shape != valid_array.shape
    ):
        raise InvalidRasterArrayError("spectral arrays and valid mask must have equal shapes")

    denominator = numerator + denominator_band_array
    invalid = (
        ~valid_array
        | ~np.isfinite(numerator)
        | ~np.isfinite(denominator_band_array)
        | ~np.isfinite(denominator)
        | (denominator == 0)
    )
    values = np.zeros(numerator.shape, dtype="float64")
    np.divide(
        numerator - denominator_band_array,
        denominator,
        out=values,
        where=~invalid,
    )
    return np.ma.array(values, mask=invalid)


def compute_index(
    name: SpectralIndex,
    bands: Mapping[SpectralBandRole, np.ndarray],
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    """Compute one frozen normalized-difference index from semantic bands."""

    try:
        index = SpectralIndex(name)
    except ValueError as exc:
        raise ValueError(f"Unsupported spectral index: {name!r}") from exc
    numerator_role, denominator_role = _FORMULAS[index]
    missing = [
        role for role in (numerator_role, denominator_role) if role not in bands
    ]
    if missing:
        raise MissingRequiredBandError(
            f"{index.value} requires semantic bands: "
            f"{', '.join(role.value for role in missing)}"
        )
    return normalized_difference(bands[numerator_role], bands[denominator_role], valid)
