"""Validation experiment P4-E04: SAR temporal backscatter, flood detection, and temporal pair verification."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import numpy as np
import pytest

from satquery.analytics.sar import SarTemporalAnalytics, detect_sar_flood
from satquery.analytics.temporal import TemporalAnalytics
from satquery.core.contracts.evidence import ChangeMaskEvidence, SarChangeResult
from satquery.core.contracts.temporal import TemporalObservationPair
from satquery.geo.exceptions import MeasurementError, MissingBandError
from affine import Affine
from satquery.ingestion.models import (
    AffineTransform,
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)
from satquery.sensors.semantics import SemanticBandRole
from satquery.verification.models import VerificationStatus


def test_sar_backscatter_unit_conversion():
    """Verify linear amplitude to decibel (dB) conversion and back."""
    linear = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    db = SarTemporalAnalytics.linear_to_db(linear)

    # 10*log10(0.01) = -20, 10*log10(0.1) = -10, 10*log10(1) = 0, 10*log10(10) = 10, 10*log10(100) = 20
    assert math.isclose(db[0], -20.0, abs_tol=1e-4)
    assert math.isclose(db[1], -10.0, abs_tol=1e-4)
    assert math.isclose(db[2], 0.0, abs_tol=1e-4)
    assert math.isclose(db[3], 10.0, abs_tol=1e-4)
    assert math.isclose(db[4], 20.0, abs_tol=1e-4)

    # Roundtrip conversion
    reconstructed_linear = SarTemporalAnalytics.db_to_linear(db)
    np.testing.assert_allclose(linear, reconstructed_linear, rtol=1e-4)


def test_sar_backscatter_delta_computation():
    """Verify SAR decibel differencing: Delta = Post - Pre."""
    pre_db = np.array([[-10.0, -12.0], [-8.0, -15.0]], dtype=np.float32)
    post_db = np.array([[-20.0, -12.0], [-5.0, -18.0]], dtype=np.float32)

    delta, stats = SarTemporalAnalytics.compute_backscatter_delta(pre_db, post_db)

    # (-20 - (-10)) = -10.0 (decrease)
    assert math.isclose(delta[0, 0], -10.0, abs_tol=1e-4)
    # (-12 - (-12)) = 0.0 (no change)
    assert math.isclose(delta[0, 1], 0.0, abs_tol=1e-4)
    # (-5 - (-8)) = +3.0 (increase)
    assert math.isclose(delta[1, 0], 3.0, abs_tol=1e-4)
    # (-18 - (-15)) = -3.0 (decrease)
    assert math.isclose(delta[1, 1], -3.0, abs_tol=1e-4)

    assert stats["valid_pixels"] == 4
    assert math.isclose(stats["min_delta_db"], -10.0, abs_tol=1e-4)
    assert math.isclose(stats["max_delta_db"], 3.0, abs_tol=1e-4)


def test_sar_flood_detection_workflow():
    """Verify deterministic SAR flood inundation mapping and area calculation."""
    # 50x50 SAR scene
    # Pre-event dry land: backscatter = -10.0 dB
    pre_vv = np.ones((50, 50), dtype=np.float32) * -10.0

    # Post-event: Flooded area in top-left 25x25 quadrant (backscatter drops to -22.0 dB)
    post_vv = np.ones((50, 50), dtype=np.float32) * -10.0
    post_vv[:25, :25] = -22.0

    bands_pre = {"vv": pre_vv}
    bands_post = {"vv": post_vv}
    affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)  # 100 m2 / pixel

    mask, sar_res, mask_ev = detect_sar_flood(
        bands_pre,
        bands_post,
        affine,
        "EPSG:32633",
        polarization_role=SemanticBandRole.SAR_CO_POL,
        decrease_threshold_db=3.0,
        water_max_threshold_db=-16.0,
    )

    assert sar_res.flood_detected is True
    assert sar_res.flood_pixel_count == 625  # 25x25 = 625 pixels
    assert math.isclose(sar_res.flood_area_m2, 62500.0, abs_tol=1e-5)  # 625 * 100 m2
    assert mask_ev.changed_pixels == 625
    assert mask_ev.change_type == "sar_flood_inundation"


def test_temporal_pair_verification_and_refusal():
    """Verify TemporalAnalytics.build_temporal_pair chronological ordering and verification checks."""
    t1 = datetime(2025, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 5, 12, 10, 0, 0, tzinfo=timezone.utc)

    asset1 = SourceAsset(
        asset_id="asset1",
        original_name="scene1.tif",
        path="/data/scene1.tif",
        sha256="1111111111111111111111111111111111111111111111111111111111111111",
    )
    asset2 = SourceAsset(
        asset_id="asset2",
        original_name="scene2.tif",
        path="/data/scene2.tif",
        sha256="2222222222222222222222222222222222222222222222222222222222222222",
    )

    geo1 = GeoMetadata(
        crs="EPSG:32633",
        bounds=GeoBounds(left=500000, bottom=4000000, right=510000, top=4010000),
        transform=AffineTransform(a=10, b=0, c=500000, d=0, e=-10, f=4010000),
        native_gsd_x=10.0,
        native_gsd_y=10.0,
    )

    raster1 = RasterMetadata(
        driver="GTiff",
        width=1000,
        height=1000,
        band_count=1,
        dtypes=("float32",),
        nodata=(None,),
    )

    sensor1 = SensorMetadata(
        sensor_name="Sentinel-1",
        modality=Modality.SAR,
        bands=(BandMetadata(index=1, dtype="float32"),),
        polarizations=("VV",),
    )

    validity1 = ValidityMetadata(
        has_crs=True,
        has_transform=True,
        has_nodata=False,
    )

    prov = ObservationProvenance(
        created_at=datetime(2025, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        ingestion_version="1.0.0",
    )

    obs1 = ObservationState(
        observation_id="obs_20250501",
        source_asset=asset1,
        raster=raster1,
        sensor=sensor1,
        geo=geo1,
        temporal=TemporalMetadata(acquisition_time=t1),
        validity=validity1,
        provenance=prov,
    )

    obs2 = ObservationState(
        observation_id="obs_20250512",
        source_asset=asset2,
        raster=raster1,
        sensor=sensor1,
        geo=geo1,
        temporal=TemporalMetadata(acquisition_time=t2),
        validity=validity1,
        provenance=prov,
    )

    # 1. Valid temporal pair
    pair, report = TemporalAnalytics.build_temporal_pair(obs1, obs2)
    assert pair.is_valid_temporal_order is True
    assert pair.delta_seconds == 11 * 86400
    assert report.is_valid is True
    assert report.overall_status == VerificationStatus.PASS

    # 2. Refusal on identical observation IDs (duplicate input)
    with pytest.raises(ValueError, match="Temporal pair must contain distinct observations"):
        TemporalAnalytics.build_temporal_pair(obs1, obs1)

    # 3. Inverted temporal order (T1 > T2) produces invalid verification status
    obs_inverted_1 = ObservationState(
        observation_id="obs_20250512_inv",
        source_asset=asset1,
        raster=raster1,
        sensor=sensor1,
        geo=geo1,
        temporal=TemporalMetadata(acquisition_time=t2),  # Later time
        validity=validity1,
        provenance=prov,
    )
    obs_inverted_2 = ObservationState(
        observation_id="obs_20250501_inv",
        source_asset=asset2,
        raster=raster1,
        sensor=sensor1,
        geo=geo1,
        temporal=TemporalMetadata(acquisition_time=t1),  # Earlier time
        validity=validity1,
        provenance=prov,
    )

    pair_inv, report_inv = TemporalAnalytics.build_temporal_pair(obs_inverted_1, obs_inverted_2)
    assert pair_inv.is_valid_temporal_order is False
    assert report_inv.is_valid is False
    assert report_inv.overall_status == VerificationStatus.FAIL
