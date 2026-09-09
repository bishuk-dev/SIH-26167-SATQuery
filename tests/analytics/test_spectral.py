"""Deterministic spectral-index contract tests (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.analytics.exceptions import (
    InvalidRasterArrayError,
    MissingRequiredBandError,
)
from satquery.analytics.spectral import (
    SpectralBandRole,
    SpectralIndex,
    compute_index,
    normalized_difference,
)


def test_ndvi_matches_hand_calculation_and_masks_invalid() -> None:
    red = np.array([[1.0, 2.0], [0.0, 4.0]])
    nir = np.array([[3.0, 2.0], [0.0, 0.0]])
    valid = np.array([[True, True], [False, True]])

    ndvi = compute_index(
        SpectralIndex.NDVI,
        {SpectralBandRole.RED: red, SpectralBandRole.NIR: nir},
        valid,
    )

    np.testing.assert_allclose(ndvi.compressed(), [0.5, 0.0, -1.0])
    assert ndvi.mask[1, 0]


def test_ndwi_matches_hand_calculation() -> None:
    green = np.array([[4.0]])
    nir = np.array([[2.0]])
    valid = np.array([[True]])

    ndwi = compute_index(
        SpectralIndex.NDWI,
        {SpectralBandRole.GREEN: green, SpectralBandRole.NIR: nir},
        valid,
    )

    np.testing.assert_allclose(ndwi.compressed(), [(4.0 - 2.0) / (4.0 + 2.0)])


def test_mndwi_matches_hand_calculation() -> None:
    green = np.array([[6.0]])
    swir1 = np.array([[2.0]])
    valid = np.array([[True]])

    mndwi = compute_index(
        SpectralIndex.MNDWI,
        {SpectralBandRole.GREEN: green, SpectralBandRole.SWIR1: swir1},
        valid,
    )

    np.testing.assert_allclose(mndwi.compressed(), [(6.0 - 2.0) / (6.0 + 2.0)])


def test_zero_denominator_is_masked_not_coerced() -> None:
    first = np.array([[5.0]])
    second = np.array([[-5.0]])
    valid = np.array([[True]])

    result = normalized_difference(first, second, valid)

    assert result.mask[0, 0]
    assert result.count() == 0


def test_nonfinite_inputs_are_masked() -> None:
    first = np.array([[np.nan], [np.inf]])
    second = np.array([[1.0], [1.0]])
    valid = np.array([[True], [True]])

    result = normalized_difference(first, second, valid)

    assert result.count() == 0
    assert result.mask.all()


def test_missing_required_semantic_band_rejects() -> None:
    with pytest.raises(MissingRequiredBandError):
        compute_index(
            SpectralIndex.NDVI,
            {SpectralBandRole.RED: np.ones((2, 2))},
            np.ones((2, 2), dtype=bool),
        )


def test_unknown_index_name_rejects_instead_of_guessing() -> None:
    with pytest.raises(ValueError):
        compute_index(
            "b4_b8_guess",  # type: ignore[arg-type]
            {
                SpectralBandRole.RED: np.ones((2, 2)),
                SpectralBandRole.NIR: np.ones((2, 2)),
            },
            np.ones((2, 2), dtype=bool),
        )


def test_mismatched_shapes_reject() -> None:
    with pytest.raises(InvalidRasterArrayError):
        normalized_difference(
            np.ones((2, 2)),
            np.ones((3, 2)),
            np.ones((2, 2), dtype=bool),
        )

    with pytest.raises(InvalidRasterArrayError):
        normalized_difference(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((3, 3), dtype=bool),
        )


def test_non_2d_inputs_reject() -> None:
    with pytest.raises(InvalidRasterArrayError):
        normalized_difference(
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2), dtype=bool),
        )


def test_inputs_are_never_mutated() -> None:
    first = np.array([[1.0, 2.0], [0.0, 4.0]])
    second = np.array([[3.0, 2.0], [0.0, 0.0]])
    valid = np.array([[True, True], [False, True]])
    first_before = first.copy()
    second_before = second.copy()
    valid_before = valid.copy()

    normalized_difference(first, second, valid)

    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)
    np.testing.assert_array_equal(valid, valid_before)


def test_output_dtype_is_float64_and_masked() -> None:
    result = normalized_difference(
        np.array([[1, 3]], dtype=np.uint8),
        np.array([[3, 1]], dtype=np.uint8),
        np.array([[True, True]]),
    )

    assert result.dtype == np.float64
    np.testing.assert_allclose(result.compressed(), [-0.5, 0.5])
