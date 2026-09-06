"""Tests for deterministic remote-sensing spectral index computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from satquery.geo.exceptions import MissingBandError, SpectralIndexError
from satquery.geo.models import SpectralIndex
from satquery.geo.spectral import (
    compute_spectral_index,
    create_index_mask,
    resolve_band,
)


def test_resolve_band_aliases() -> None:
    bands = {
        "B04": np.ones((5, 5)),
        "B08": np.ones((5, 5)) * 2,
        "B03": np.ones((5, 5)) * 3,
    }

    # "red" resolves to B04
    name_red, _ = resolve_band(bands, "red")
    assert name_red == "B04"

    # "nir" resolves to B08
    name_nir, _ = resolve_band(bands, "nir")
    assert name_nir == "B08"

    # "green" resolves to B03
    name_green, _ = resolve_band(bands, "green")
    assert name_green == "B03"

    # Missing band raises MissingBandError
    with pytest.raises(MissingBandError, match="role 'swir' not found"):
        resolve_band(bands, "swir")


def test_compute_ndvi_deterministic() -> None:
    # Dense vegetation: NIR high (0.8), Red low (0.1)
    # NDVI = (0.8 - 0.1) / (0.8 + 0.1) = 0.7 / 0.9 = 0.7777...
    nir = np.full((4, 4), 0.8, dtype=np.float32)
    red = np.full((4, 4), 0.1, dtype=np.float32)

    bands = {"nir": nir, "red": red}
    ndvi_arr, stats = compute_spectral_index(bands, SpectralIndex.NDVI)

    expected_val = (0.8 - 0.1) / (0.8 + 0.1)
    assert np.allclose(ndvi_arr, expected_val, atol=1e-5)
    assert stats.index_name == "ndvi"
    assert math.isclose(stats.mean_value, expected_val, rel_tol=1e-5)
    assert stats.valid_pixels == 16
    assert stats.nodata_pixels == 0


def test_compute_ndwi_and_nbr() -> None:
    green = np.full((3, 3), 0.6, dtype=np.float32)
    nir = np.full((3, 3), 0.2, dtype=np.float32)
    swir = np.full((3, 3), 0.1, dtype=np.float32)

    bands = {"green": green, "nir": nir, "swir": swir}

    # NDWI = (green - nir) / (green + nir) = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    ndwi_arr, stats_ndwi = compute_spectral_index(bands, SpectralIndex.NDWI)
    assert np.allclose(ndwi_arr, 0.5, atol=1e-5)
    assert math.isclose(stats_ndwi.mean_value, 0.5, rel_tol=1e-5)

    # NBR = (nir - swir) / (nir + swir) = (0.2 - 0.1) / (0.2 + 0.1) = 0.1 / 0.3 = 0.3333...
    nbr_arr, stats_nbr = compute_spectral_index(bands, SpectralIndex.NBR)
    assert np.allclose(nbr_arr, 1.0 / 3.0, atol=1e-5)


def test_compute_spectral_index_safeguards() -> None:
    # Zero denominator (0 + 0) must be handled safely as NaN and excluded from valid_pixels
    nir = np.zeros((2, 2), dtype=np.float32)
    red = np.zeros((2, 2), dtype=np.float32)
    nir[0, 0] = 0.5
    red[0, 0] = 0.1

    bands = {"nir": nir, "red": red}
    ndvi_arr, stats = compute_spectral_index(bands, SpectralIndex.NDVI)

    assert not np.isnan(ndvi_arr[0, 0])
    assert np.isnan(ndvi_arr[0, 1])
    assert stats.valid_pixels == 1
    assert stats.nodata_pixels == 3

    # Mismatched shapes
    bad_bands = {
        "nir": np.zeros((2, 2)),
        "red": np.zeros((3, 3)),
    }
    with pytest.raises(SpectralIndexError, match="Band shape mismatch"):
        compute_spectral_index(bad_bands, "ndvi")


def test_create_index_mask() -> None:
    data = np.array([[-0.2, 0.1, 0.4], [0.6, 0.8, np.nan]], dtype=np.float32)
    # Threshold for vegetation: ndvi >= 0.3
    mask = create_index_mask(data, min_threshold=0.3)
    assert mask[0, 0] is False or mask[0, 0] == 0
    assert mask[0, 1] is False or mask[0, 1] == 0
    assert mask[0, 2] is True or mask[0, 2] == 1
    assert mask[1, 0] is True or mask[1, 0] == 1
    assert mask[1, 1] is True or mask[1, 1] == 1
    assert mask[1, 2] is False or mask[1, 2] == 0  # NaN must be False
