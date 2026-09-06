"""Deterministic remote-sensing spectral index computation and validation."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from satquery.geo.exceptions import MissingBandError, SpectralIndexError
from satquery.geo.models import SpectralIndex, SpectralIndexStats

# Common band aliases across sensors (Sentinel-2, Landsat, generic multispectral)
BAND_ALIASES: dict[str, tuple[str, ...]] = {
    "nir": ("nir", "b08", "b8", "b8a", "band8", "near_infrared", "b08_10m", "b8a_20m"),
    "red": ("red", "b04", "b4", "band4", "b04_10m"),
    "green": ("green", "b03", "b3", "band3", "b03_10m"),
    "blue": ("blue", "b02", "b2", "band2", "b02_10m"),
    "swir": ("swir", "swir1", "b11", "b12", "band11", "band12", "b11_20m", "b12_20m"),
}

INDEX_REQUIRED_BANDS: dict[SpectralIndex, tuple[str, str]] = {
    SpectralIndex.NDVI: ("nir", "red"),
    SpectralIndex.NDWI: ("green", "nir"),
    SpectralIndex.NDWI_GAO: ("nir", "swir"),
    SpectralIndex.NBR: ("nir", "swir"),
}


def resolve_band(bands: Mapping[str, np.ndarray], role: str) -> tuple[str, np.ndarray]:
    """Find a band matching a semantic role from available raster bands.

    Returns (matched_key, band_array).
    Raises MissingBandError if no matching band is found.
    """
    role_lower = role.lower()
    aliases = BAND_ALIASES.get(role_lower, (role_lower,))

    # Direct match or case-insensitive match
    for key, val in bands.items():
        key_norm = key.strip().lower()
        if key_norm in aliases:
            return key, val

    # Substring match (e.g. "B04" in "Sentinel2_B04")
    for key, val in bands.items():
        key_norm = key.strip().lower()
        for alias in aliases:
            if alias in key_norm:
                return key, val

    available = list(bands.keys())
    raise MissingBandError(
        f"Required band for role {role!r} not found. Available bands: {available}"
    )


def compute_spectral_index(
    bands: Mapping[str, np.ndarray],
    index: SpectralIndex | str,
    *,
    nodata_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, SpectralIndexStats]:
    """Compute a normalized difference spectral index deterministically.

    Enforces physical range [-1.0, 1.0] and excludes NoData / non-finite pixels.
    Returns (index_array_float32, stats).
    """
    try:
        spec_index = SpectralIndex(index)
    except ValueError as exc:
        raise SpectralIndexError(f"Unsupported spectral index: {index!r}") from exc

    band_a_role, band_b_role = INDEX_REQUIRED_BANDS[spec_index]
    name_a, arr_a = resolve_band(bands, band_a_role)
    name_b, arr_b = resolve_band(bands, band_b_role)

    data_a = np.asarray(arr_a, dtype=np.float32)
    data_b = np.asarray(arr_b, dtype=np.float32)

    if data_a.ndim != 2 or data_b.ndim != 2:
        raise SpectralIndexError("Bands must be two-dimensional arrays")
    if data_a.shape != data_b.shape:
        raise SpectralIndexError(
            f"Band shape mismatch for {spec_index.value}: {name_a} {data_a.shape} vs {name_b} {data_b.shape}"
        )

    # Combine non-finite and user NoData
    invalid = ~np.isfinite(data_a) | ~np.isfinite(data_b)
    if nodata_mask is not None:
        invalid |= np.asarray(nodata_mask, dtype=bool)

    num = data_a - data_b
    den = data_a + data_b

    # Division by zero or near-zero check
    near_zero = np.abs(den) < 1e-7
    invalid |= near_zero

    result = np.full(data_a.shape, np.nan, dtype=np.float32)
    valid_idx = ~invalid
    np.divide(num, den, out=result, where=valid_idx)

    # Physical range clamp [-1.0, 1.0] for valid pixels
    np.clip(result, -1.0, 1.0, out=result, where=valid_idx)

    valid_count = int(np.count_nonzero(valid_idx))
    nodata_count = int(data_a.size - valid_count)

    if valid_count > 0:
        valid_vals = result[valid_idx]
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
    else:
        min_val = 0.0
        max_val = 0.0
        mean_val = 0.0
        std_val = 0.0

    stats = SpectralIndexStats(
        index_name=spec_index.value,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
        valid_pixels=valid_count,
        nodata_pixels=nodata_count,
    )

    return result, stats


def create_index_mask(
    index_data: np.ndarray,
    *,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
) -> np.ndarray:
    """Create a boolean mask indicating pixels within specified index thresholds."""
    data = np.asarray(index_data)
    valid = np.isfinite(data)
    mask = valid.copy()

    if min_threshold is not None:
        mask &= (data >= min_threshold)
    if max_threshold is not None:
        mask &= (data <= max_threshold)

    return mask
