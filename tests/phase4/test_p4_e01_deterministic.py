"""Validation experiment P4-E01: Deterministic GIS, spectral indices, and area measurement."""

from __future__ import annotations

import math
import numpy as np
import pytest

from affine import Affine

from satquery.analytics.measurement import MeasurementEngine, measure_mask_area
from satquery.analytics.spectral import SpectralAnalytics, compute_index
from satquery.analytics.temporal import TemporalAnalytics, compute_bitemporal_change
from satquery.core.contracts.evidence import (
    ChangeMaskEvidence,
    IndexRasterEvidence,
    MaskEvidence,
    MeasurementEvidence,
    SpectralAnalysisResult,
)
from satquery.core.contracts.temporal import (
    AnalysisROI,
    TemporalChangeResult,
    TemporalObservationPair,
)
from satquery.geo.exceptions import CrsMeasurementError, MissingBandError, SpectralIndexError
from satquery.geo.models import MeasurementUnit
from satquery.sensors.semantics import SemanticBandRole, get_semantic_band, resolve_band_for_role


def test_semantic_band_resolution():
    """Verify semantic role resolution across different band alias naming conventions."""
    # Sentinel-2 style
    s2_bands = {
        "B02": np.ones((10, 10), dtype=np.float32) * 0.1,
        "B03": np.ones((10, 10), dtype=np.float32) * 0.2,
        "B04": np.ones((10, 10), dtype=np.float32) * 0.15,
        "B08": np.ones((10, 10), dtype=np.float32) * 0.6,
        "B11": np.ones((10, 10), dtype=np.float32) * 0.3,
    }
    red_key, _ = get_semantic_band(s2_bands, SemanticBandRole.RED, sensor="sentinel2")
    nir_key, _ = get_semantic_band(s2_bands, SemanticBandRole.NIR, sensor="sentinel2")
    swir_key, _ = get_semantic_band(s2_bands, SemanticBandRole.SWIR, sensor="sentinel2")

    assert red_key == "B04"
    assert nir_key == "B08"
    assert swir_key == "B11"

    # Landsat style
    landsat_bands = {
        "sr_b2": np.ones((10, 10), dtype=np.float32),
        "sr_b3": np.ones((10, 10), dtype=np.float32),
        "sr_b4": np.ones((10, 10), dtype=np.float32),
        "sr_b5": np.ones((10, 10), dtype=np.float32),
        "sr_b6": np.ones((10, 10), dtype=np.float32),
    }
    ls_red, _ = get_semantic_band(landsat_bands, SemanticBandRole.RED, sensor="landsat")
    ls_nir, _ = get_semantic_band(landsat_bands, SemanticBandRole.NIR, sensor="landsat")
    assert ls_red == "sr_b4"
    assert ls_nir == "sr_b5"


def test_missing_semantic_band_raises_error():
    """Verify that attempting to resolve an absent semantic band raises MissingBandError."""
    incomplete_bands = {
        "B02": np.ones((10, 10), dtype=np.float32),
        "B03": np.ones((10, 10), dtype=np.float32),
    }
    with pytest.raises(MissingBandError):
        get_semantic_band(incomplete_bands, SemanticBandRole.NIR)


def test_spectral_index_math_and_clamping():
    """Verify spectral index math, zero-division masking, and range clamping [-1, 1]."""
    # Create 4x4 test bands
    # NIR: high values, RED: low values
    nir = np.array([
        [0.8, 0.6, 0.5, 0.0],
        [0.9, 0.7, 0.4, np.nan],
        [0.0, 0.5, 0.8, 0.1],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float32)

    red = np.array([
        [0.2, 0.2, 0.5, 0.0],  # Cell (0,3): 0/0 division -> invalid
        [0.1, 0.1, 0.4, 0.2],
        [0.0, 0.5, 0.2, 0.9],  # Cell (2,0): 0/0 division -> invalid
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float32)

    bands = {"nir": nir, "red": red}
    ndvi, stats = compute_index(bands, "ndvi")

    # (0,0): (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    assert math.isclose(ndvi[0, 0], 0.6, abs_tol=1e-5)
    # (0,1): (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    assert math.isclose(ndvi[0, 1], 0.5, abs_tol=1e-5)
    # (0,2): (0.5 - 0.5) / (0.5 + 0.5) = 0.0
    assert math.isclose(ndvi[0, 2], 0.0, abs_tol=1e-5)

    # Division by zero pixels (0,3) and (2,0) should be NaN
    assert np.isnan(ndvi[0, 3])
    assert np.isnan(ndvi[2, 0])
    assert np.isnan(ndvi[1, 3])  # NaN input

    assert stats.valid_pixels == 13
    assert stats.nodata_pixels == 3
    assert stats.min_value >= -1.0
    assert stats.max_value <= 1.0


def test_spectral_index_roi_clipping():
    """Verify that AnalysisROI pixel bounds restrict spectral index computation."""
    nir = np.ones((20, 20), dtype=np.float32) * 0.8
    red = np.ones((20, 20), dtype=np.float32) * 0.2
    bands = {"nir": nir, "red": red}

    # ROI covering rows 5..15, cols 5..15
    roi = AnalysisROI(
        roi_id="test_roi",
        crs="EPSG:32633",
        geometry_type="bbox",
        coordinates=((300000.0, 5000000.0), (310000.0, 5000000.0), (310000.0, 5010000.0), (300000.0, 5010000.0)),
        pixel_bounds=(5, 5, 15, 15),
    )

    ndvi, stats = SpectralAnalytics.compute_index(bands, "ndvi", roi=roi)

    # Outside ROI should be NaN
    assert np.isnan(ndvi[0, 0])
    assert np.isnan(ndvi[4, 4])
    # Inside ROI (5..14, 5..14) should be valid 0.6
    assert math.isclose(ndvi[10, 10], 0.6, abs_tol=1e-5)
    assert stats.valid_pixels == 100  # 10x10 ROI


def test_deterministic_area_measurement():
    """Verify planar and geodesic area calculation engine."""
    # 100x100 mask, affine transform with 10m x 10m pixels (100 m2 per pixel)
    mask = np.ones((100, 100), dtype=bool)
    # Affine: dx=10, dy=-10, origin=(500000, 4000000)
    affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)

    # Projected CRS: EPSG:32633 (UTM 33N)
    meas = MeasurementEngine.measure_mask(mask, affine, "EPSG:32633", target_unit=MeasurementUnit.M2)

    assert meas.pixel_count == 10000
    assert math.isclose(meas.pixel_area_m2, 100.0, abs_tol=1e-5)
    assert math.isclose(meas.area_value, 1000000.0, abs_tol=1e-5)  # 1 km2 = 1,000,000 m2
    assert meas.calculation_path == "projected_planar"

    # Convert to hectares
    meas_ha = MeasurementEngine.measure_mask(mask, affine, "EPSG:32633", target_unit=MeasurementUnit.HA)
    assert math.isclose(meas_ha.area_value, 100.0, abs_tol=1e-5)  # 100 hectares

    # Convert to km2
    meas_km2 = MeasurementEngine.measure_mask(mask, affine, "EPSG:32633", target_unit=MeasurementUnit.KM2)
    assert math.isclose(meas_km2.area_value, 1.0, abs_tol=1e-5)  # 1 km2


def test_geodesic_area_measurement_wgs84():
    """Verify ellipsoidal geodesic area calculation for EPSG:4326 geographic coordinates."""
    mask = np.ones((10, 10), dtype=bool)
    # 0.001 deg x 0.001 deg pixels (~111m x ~111m at equator)
    affine = Affine(0.001, 0.0, 0.0, 0.0, -0.001, 0.0)

    meas = MeasurementEngine.measure_mask(mask, affine, "EPSG:4326", target_unit=MeasurementUnit.M2)

    assert meas.calculation_path == "geodesic_wgs84"
    assert meas.pixel_count == 100
    assert meas.pixel_area_m2 > 10000.0  # Approx 12,300 m2 per 0.001 deg cell near equator
    assert meas.area_value > 1000000.0


def test_bitemporal_vegetation_change():
    """Verify bi-temporal vegetation loss and gain computation."""
    # T1: High vegetation
    nir_t1 = np.ones((50, 50), dtype=np.float32) * 0.8
    red_t1 = np.ones((50, 50), dtype=np.float32) * 0.2  # NDVI T1 = 0.6

    # T2: Cleared vegetation in left half, unchanged in right half
    nir_t2 = np.ones((50, 50), dtype=np.float32) * 0.8
    red_t2 = np.ones((50, 50), dtype=np.float32) * 0.2
    # Left half cleared: high red, low NIR -> NDVI T2 = (0.2 - 0.7)/(0.2 + 0.7) = -0.55
    nir_t2[:, :25] = 0.2
    red_t2[:, :25] = 0.7

    bands_t1 = {"nir": nir_t1, "red": red_t1}
    bands_t2 = {"nir": nir_t2, "red": red_t2}
    affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)

    delta, change_res, mask_ev = compute_bitemporal_change(
        bands_t1,
        bands_t2,
        affine,
        "EPSG:32633",
        change_category="vegetation",
        threshold=0.20,
        detect_mode="loss",
    )

    assert change_res.change_type == "vegetation_loss"
    assert change_res.change_pixel_count == 1250  # 50 rows x 25 cols = 1250 pixels
    assert math.isclose(change_res.change_area_m2, 125000.0, abs_tol=1e-5)  # 1250 * 100 m2
    assert math.isclose(change_res.percent_change, 50.0, abs_tol=1e-5)  # 50% of baseline lost
    assert mask_ev.changed_pixels == 1250


def test_change_description_contract():
    """Verify ChangeDescriptionResult domain contract and LEVIR-CC metadata."""
    from satquery.core.contracts.temporal import ChangeDescriptionResult

    desc_result = ChangeDescriptionResult(
        pair_id="pair_levir_001",
        description="New residential buildings constructed on cleared land between T1 and T2.",
        captions=("New residential buildings constructed on cleared land between T1 and T2.",),
        dataset_source="LEVIR-CC",
        evaluation_split="val",
        limitations=(
            "Change description generated by learned bi-temporal specialist.",
            "Imagery provenance: LEVIR-CC (academic/non-commercial research only).",
            "Evaluation restricted to validation split; test set sealed.",
        ),
    )

    assert isinstance(desc_result, ChangeDescriptionResult)
    assert desc_result.pair_id == "pair_levir_001"
    assert desc_result.dataset_source == "LEVIR-CC"
    assert desc_result.evaluation_split == "val"
    assert len(desc_result.description) > 0
    assert any("LEVIR-CC" in lim for lim in desc_result.limitations)


