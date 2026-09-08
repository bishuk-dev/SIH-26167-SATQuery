"""Deterministic spectral-index calculations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class MissingRequiredBandError(ValueError):
    """Raised when an index lacks a semantically identified input band."""


def compute_index(
    name: str,
    bands: Mapping[str, np.ndarray],
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    """Compute one registered normalized-difference index."""

    formulas = {
        "ndvi": ("NIR", "RED"),
        "ndwi": ("GREEN", "NIR"),
        "mndwi": ("GREEN", "SWIR1"),
    }
    try:
        numerator_band, denominator_band = formulas[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported spectral index: {name}") from exc
    missing = [
        band
        for band in (numerator_band, denominator_band)
        if band not in bands
    ]
    if missing:
        raise MissingRequiredBandError(
            f"{name.upper()} requires semantic bands: {', '.join(missing)}"
        )
    return normalized_difference(
        np.asarray(bands[numerator_band]),
        np.asarray(bands[denominator_band]),
        valid,
    )


def normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    """Return ``(first - second) / (first + second)`` with invalid pixels masked."""

    first_array, second_array = (
        np.asarray(first, dtype="float64"),
        np.asarray(second, dtype="float64"),
    )
    valid_array = np.asarray(valid, dtype=bool)
    if first_array.shape != second_array.shape or first_array.shape != valid_array.shape:
        raise ValueError("spectral arrays and valid mask must have equal shapes")
    denominator = first_array + second_array
    invalid = (
        ~valid_array
        | ~np.isfinite(first_array)
        | ~np.isfinite(second_array)
        | ~np.isfinite(denominator)
        | (denominator == 0)
    )
    values = np.zeros(first_array.shape, dtype="float64")
    np.divide(
        first_array - second_array,
        denominator,
        out=values,
        where=~invalid,
    )
    return np.ma.array(values, mask=invalid)
