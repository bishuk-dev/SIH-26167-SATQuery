"""Tests for deterministic GIS measurements (R-GEO-004, R-GEO-005)."""

from __future__ import annotations

import math

from affine import Affine
import numpy as np
import pytest

from satquery.geo.exceptions import (
    CrsMeasurementError,
    MeasurementError,
)
from satquery.geo.measurements import (
    calculate_area,
    calculate_pixel_area_m2,
    convert_area,
    count_pixels,
)
from satquery.geo.models import MeasurementUnit


def test_convert_area_exact_factors() -> None:
    # 10,000 m2 = 1 hectare
    assert convert_area(10000.0, MeasurementUnit.HA) == 1.0
    assert convert_area(10000.0, "ha") == 1.0

    # 1,000,000 m2 = 1 square kilometer
    assert convert_area(1_000_000.0, MeasurementUnit.KM2) == 1.0
    assert convert_area(1_000_000.0, "km2") == 1.0

    # Identity for m2
    assert convert_area(543.21, MeasurementUnit.M2) == 543.21

    # Invalid area values
    with pytest.raises(MeasurementError, match="finite and non-negative"):
        convert_area(-10.0, "ha")
    with pytest.raises(MeasurementError, match="finite and non-negative"):
        convert_area(float("nan"), "ha")
    with pytest.raises(MeasurementError, match="Unsupported measurement unit"):
        convert_area(100.0, "acres")
    with pytest.raises(MeasurementError, match="Cannot convert square meters to pixel count"):
        convert_area(100.0, "count")


def test_count_pixels_boolean_and_threshold() -> None:
    mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
    assert count_pixels(mask) == 3

    # Numeric array with threshold
    values = np.array([[0.1, 0.4, 0.8], [0.2, 0.5, np.nan]], dtype=np.float32)
    # values > 0.3 are: 0.4, 0.8, 0.5 (3 pixels). NaN must not be counted.
    assert count_pixels(values, threshold=0.3) == 3

    # Numeric array without threshold: count non-zeros (excluding NaN)
    non_zeros = np.array([[0, 5, 0], [1, 0, np.nan]])
    assert count_pixels(non_zeros) == 2

    # Empty array
    assert count_pixels(np.array([], dtype=bool)) == 0


def test_calculate_pixel_area_m2_projected() -> None:
    # 10m x 10m pixel grid in UTM 32N (EPSG:32632)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)
    area_m2, path = calculate_pixel_area_m2(transform, "EPSG:32632")

    assert math.isclose(area_m2, 100.0, rel_tol=1e-6)
    assert path == "projected_planar"


def test_calculate_pixel_area_m2_geographic_geodesic() -> None:
    # 0.001 degree pixels in EPSG:4326
    # 1 deg latitude is approx 111,320m
    d_deg = 0.001
    transform = Affine(d_deg, 0.0, 10.0, 0.0, -d_deg, 0.0)

    # At the equator (lat = 0)
    area_equator, path = calculate_pixel_area_m2(
        transform, "EPSG:4326", center_coord=(10.0, 0.0)
    )
    assert path == "geodesic_wgs84"
    # ~111.32m x 110.57m ≈ 12300 m2
    assert 12000.0 < area_equator < 12500.0

    # At latitude 60 degrees: longitude convergence halves the width
    area_60, path = calculate_pixel_area_m2(
        transform, "EPSG:4326", center_coord=(10.0, 60.0)
    )
    assert math.isclose(area_60 / area_equator, 0.5, rel_tol=0.02)


def test_calculate_pixel_area_m2_crs_safety() -> None:
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)

    # Missing CRS must raise CrsMeasurementError
    with pytest.raises(CrsMeasurementError, match="valid CRS"):
        calculate_pixel_area_m2(transform, None)

    # Invalid CRS must raise CrsMeasurementError
    with pytest.raises(CrsMeasurementError, match="Invalid or unparseable CRS"):
        calculate_pixel_area_m2(transform, "NOT_A_VALID_CRS_STRING_12345")


def test_calculate_area_complete_workflow() -> None:
    # 10x10 raster where 50 pixels are True
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True  # 50 pixels True

    # 10m x 10m pixel grid = 100 m2 per pixel
    transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    crs = "EPSG:32632"

    # Total area: 50 * 100 m2 = 5000 m2 = 0.5 ha = 0.005 km2
    res_m2 = calculate_area(mask, transform, crs, target_unit="m2")
    assert res_m2.pixel_count == 50
    assert math.isclose(res_m2.area, 5000.0, rel_tol=1e-6)
    assert res_m2.unit == MeasurementUnit.M2
    assert res_m2.calculation_path == "projected_planar"
    assert math.isclose(res_m2.pixel_area_m2, 100.0, rel_tol=1e-6)

    res_ha = calculate_area(mask, transform, crs, target_unit="ha")
    assert math.isclose(res_ha.area, 0.5, rel_tol=1e-6)
    assert res_ha.unit == MeasurementUnit.HA

    res_km2 = calculate_area(mask, transform, crs, target_unit="km2")
    assert math.isclose(res_km2.area, 0.005, rel_tol=1e-6)
    assert res_km2.unit == MeasurementUnit.KM2

    res_count = calculate_area(mask, transform, crs, target_unit="count")
    assert res_count.pixel_count == 50
    assert res_count.area == 50.0
    assert res_count.unit == MeasurementUnit.COUNT
