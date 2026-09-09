"""CRS-safe deterministic mask-area measurement tests (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from satquery.analytics.exceptions import (
    CrsRequiredForMeasurementError,
    InvalidRasterArrayError,
    UnsupportedCrsMeasurementError,
)
from satquery.analytics.measurement import measure_mask_area
from satquery.ingestion.models import AffineTransform


def _three_positive_mask() -> np.ndarray:
    return np.array([[1, 0], [1, 1]], dtype=bool)


def test_projected_north_up_metric_grid_area() -> None:
    result = measure_mask_area(
        _three_positive_mask(),
        Affine(10, 0, 0, 0, -10, 100),
        "EPSG:32643",
        unit="m2",
    )

    assert result.value == pytest.approx(300.0)
    assert result.unit == "m2"
    assert result.positive_pixel_count == 3
    assert result.valid_pixel_count == 4
    assert result.method == "projected_affine_determinant"
    assert result.calculation_crs == "EPSG:32643"


def test_rotated_affine_uses_full_determinant() -> None:
    transform = Affine(10, 2, 0, 1, -10, 0)

    result = measure_mask_area(
        _three_positive_mask(), transform, "EPSG:32643", unit="m2"
    )

    assert result.value == pytest.approx(306.0)


def test_hectare_and_km2_conversion() -> None:
    transform = Affine(10, 2, 0, 1, -10, 0)

    ha = measure_mask_area(_three_positive_mask(), transform, "EPSG:32643", unit="ha")
    km2 = measure_mask_area(_three_positive_mask(), transform, "EPSG:32643", unit="km2")

    assert ha.value == pytest.approx(0.0306)
    assert km2.value == pytest.approx(0.000306)


def test_valid_mask_excludes_pixels_from_area_and_counts() -> None:
    mask = np.array([[1, 1], [1, 0]], dtype=bool)
    valid = np.array([[True, False], [True, True]])

    result = measure_mask_area(
        mask, Affine(10, 0, 0, 0, -10, 100), "EPSG:32643", unit="m2", valid=valid
    )

    assert result.positive_pixel_count == 2
    assert result.valid_pixel_count == 3
    assert result.value == pytest.approx(200.0)


def test_missing_crs_rejects() -> None:
    with pytest.raises(CrsRequiredForMeasurementError):
        measure_mask_area(
            _three_positive_mask(), Affine(10, 0, 0, 0, -10, 100), None, unit="m2"
        )


def test_malformed_crs_rejects() -> None:
    with pytest.raises(CrsRequiredForMeasurementError):
        measure_mask_area(
            _three_positive_mask(), Affine(10, 0, 0, 0, -10, 100), "NOT_A_CRS", unit="m2"
        )


def test_geographic_crs_uses_geodesic_area_not_degree_squaring() -> None:
    transform = Affine(0.001, 0, 0, 0, -0.001, 0)

    result = measure_mask_area(
        np.array([[True]], dtype=bool), transform, "EPSG:4326", unit="m2"
    )

    # a 0.001° x 0.001° cell at the equator on WGS84 is about 111.32 m per
    # side; degrees squared would be ~1e-6 and a constant approximation
    # would not vary with latitude
    assert 11_000.0 < result.value < 13_000.0
    assert result.method.startswith("geodesic")


def test_geographic_area_varies_with_latitude() -> None:
    transform_at_equator = Affine(1.0, 0, 0.0, 0, -1.0, 1.0)
    transform_at_60 = Affine(1.0, 0, 0.0, 0, -1.0, -59.0)

    equator = measure_mask_area(
        np.array([[True]], dtype=bool), transform_at_equator, "EPSG:4326", unit="m2"
    )
    north = measure_mask_area(
        np.array([[True]], dtype=bool), transform_at_60, "EPSG:4326", unit="m2"
    )

    assert north.value < 0.75 * equator.value
    assert north.value > 0.25 * equator.value


def test_geographic_multi_pixel_sum_matches_single_cells() -> None:
    transform = Affine(0.001, 0, 0.0, 0, -0.001, 0.0)
    mask = np.array([[True, True], [False, True]], dtype=bool)

    result = measure_mask_area(mask, transform, "EPSG:4326", unit="m2")

    single = measure_mask_area(
        np.array([[True]], dtype=bool), transform, "EPSG:4326", unit="m2"
    )
    # same row band cells have equal areas; the third cell sits one row down
    # so its area differs only marginally at this latitude
    assert result.value == pytest.approx(3 * single.value, rel=0.01)


def test_projected_non_metre_crs_applies_declared_unit_factor() -> None:
    # EPSG:2263 (New York Long Island) uses US survey feet
    result = measure_mask_area(
        _three_positive_mask(), Affine(1, 0, 0, 0, -1, 100), "EPSG:2263", unit="m2"
    )

    factor = 0.3048006096012192
    assert result.value == pytest.approx(3 * factor * factor)


def test_singular_or_zero_area_transform_rejects() -> None:
    with pytest.raises(InvalidRasterArrayError):
        measure_mask_area(
            _three_positive_mask(), Affine(0, 0, 0, 0, 0, 0), "EPSG:32643", unit="m2"
        )

    with pytest.raises(InvalidRasterArrayError):
        measure_mask_area(
            _three_positive_mask(), Affine(10, 0, 0, 0, 0, 100), "EPSG:32643", unit="m2"
        )


def test_nonbinary_numeric_masks_reject() -> None:
    with pytest.raises(InvalidRasterArrayError):
        measure_mask_area(
            np.array([[2.0, 0.0], [0.0, 1.0]]),
            Affine(10, 0, 0, 0, -10, 100),
            "EPSG:32643",
            unit="m2",
        )

    with pytest.raises(InvalidRasterArrayError):
        measure_mask_area(
            np.array([[np.nan, 0.0], [0.0, 1.0]]),
            Affine(10, 0, 0, 0, -10, 100),
            "EPSG:32643",
            unit="m2",
        )


def test_affine_transform_model_is_accepted() -> None:
    transform_model = AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=100.0)

    result = measure_mask_area(
        _three_positive_mask(), transform_model, "EPSG:32643", unit="m2"
    )

    assert result.value == pytest.approx(300.0)


def test_unsupported_unit_rejects() -> None:
    with pytest.raises(ValueError):
        measure_mask_area(
            _three_positive_mask(), Affine(10, 0, 0, 0, -10, 100), "EPSG:32643", unit="acres"
        )


def test_valid_mask_shape_mismatch_rejects() -> None:
    with pytest.raises(InvalidRasterArrayError):
        measure_mask_area(
            _three_positive_mask(),
            Affine(10, 0, 0, 0, -10, 100),
            "EPSG:32643",
            unit="m2",
            valid=np.ones((3, 3), dtype=bool),
        )
