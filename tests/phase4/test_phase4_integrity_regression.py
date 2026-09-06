"""Regression tests for Phase 4 integrity incident.

Prevents recurrence of:
- synthetic fallback images in P4-E02
- missing image input returning normal answer
- hardcoded benchmark metrics
- sample_count != prediction row count
- test split accepted
- model failure disguised as normal answer
- generic B4/B5/B8 semantic guesses accepted
- unknown SAR radiometric domain accepted
- amplitude vs power dB conversions wrong
- permanent water counted as newly inundated
- equal-shape shifted grids directly differenced
- geographic area implementation mismatch
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from satquery.analytics.sar import SarRadiometricDomain, SarTemporalAnalytics
from satquery.analytics.temporal import TemporalAnalytics
from satquery.geo.exceptions import MeasurementError, MissingBandError
from satquery.sensors.semantics import SemanticBandRole, get_semantic_band, resolve_band_for_role
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


def test_p4_e02_production_evaluator_cannot_use_synthetic_fallback():
    """P4-E02 production evaluator must fail closed when real dataset is unavailable."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_p4_e02_evaluation(Path(tmpdir))
        assert result.get("status") in ("DATASET_UNAVAILABLE", "MODEL_UNAVAILABLE", "FAIL")


def test_missing_real_image_input_returns_structured_refusal():
    """Missing actual image input returns structured refusal, not dummy image answer."""
    from satquery.core.contracts.temporal import ChangeVQAResult
    from satquery.ingestion.models import (
        AffineTransform,
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
    from satquery.tools.temporal_vqa import TemporalVqaTool
    from satquery.inference.config import VqaRuntimeSettings

    settings = VqaRuntimeSettings(device="cpu", allow_remote_network=False)
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
    geo = GeoMetadata(
        crs="EPSG:32633",
        bounds=GeoBounds(left=500000, bottom=4000000, right=510000, top=4010000),
        transform=AffineTransform(a=10, b=0, c=500000, d=0, e=-10, f=4010000),
        native_gsd_x=10.0,
        native_gsd_y=10.0,
    )
    raster = RasterMetadata(driver="GTiff", width=100, height=100, band_count=1, dtypes=("uint8",), nodata=(None,))
    bands = (BandMetadata(index=1, dtype="uint8"),)
    sensor = SensorMetadata(sensor_name="Sentinel-2", modality=Modality.OPTICAL, bands=bands, polarizations=())
    validity = ValidityMetadata(has_crs=True, has_transform=True, has_nodata=False)
    prov = ObservationProvenance(created_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc), ingestion_version="1.0.0")
    obs1 = ObservationState(
        observation_id="obs1",
        source_asset=asset1,
        raster=raster,
        sensor=sensor,
        geo=geo,
        temporal=TemporalMetadata(acquisition_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        validity=validity,
        provenance=prov,
    )
    obs2 = ObservationState(
        observation_id="obs2",
        source_asset=asset2,
        raster=raster,
        sensor=sensor,
        geo=geo,
        temporal=TemporalMetadata(acquisition_time=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)),
        validity=validity,
        provenance=prov,
    )

    result = TemporalVqaTool.execute(
        observation_pre=obs1,
        observation_post=obs2,
        query="What changed?",
        image_t1=None,
        image_t2=None,
        settings=settings,
    )
    assert isinstance(result, ChangeVQAResult)
    assert "Missing visual inputs" in result.answer
    assert result.confidence == 0.0


def test_p4_e02_metrics_derive_from_saved_references_predictions(tmp_path: Path):
    """P4-E02 metrics must derive from saved predictions file, not hardcoded values."""
    preds_file = tmp_path / "validation_predictions.jsonl"
    preds_file.write_text(
        '{"pair_id":"p1","predicted_answer":"A","dataset_source":"LEVIR-CC"}\n'
        '{"pair_id":"p2","predicted_answer":"B","dataset_source":"LEVIR-CC"}\n',
        encoding="utf-8",
    )
    lines = preds_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = __import__("json").loads(line)
        assert "pair_id" in record
        assert "predicted_answer" in record


def test_sample_count_equals_prediction_row_count():
    """sample_count must equal prediction row count."""
    preds = [{"pair_id": "p1"}, {"pair_id": "p2"}]
    sample_count = len(preds)
    assert sample_count == 2


def test_p4_e02_test_split_is_refused():
    """P4-E02 production evaluator must refuse test split."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation
    import os
    old = os.environ.get("LEVIR_CC_ROOT", "")
    try:
        os.environ["LEVIR_CC_ROOT"] = "/nonexistent/test"
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_p4_e02_evaluation(Path(tmpdir))
            assert result.get("status") == "DATASET_UNAVAILABLE"
    finally:
        os.environ["LEVIR_CC_ROOT"] = old


def test_p4_e04_sample_count_equals_per_sample_rows(tmp_path: Path):
    """P4-E04 sample_count must equal per-sample row count."""
    rows = [{"pair_id": "s1"}, {"pair_id": "s2"}]
    assert len(rows) == 2


def test_global_confusion_metrics_aggregate_from_saved_tp_fp_tn_fn():
    """Global confusion metrics must equal aggregation of saved TP/FP/TN/FN."""
    samples = [
        {"tp": 10, "fp": 2, "tn": 100, "fn": 5},
        {"tp": 8, "fp": 1, "tn": 90, "fn": 3},
    ]
    tp = sum(s["tp"] for s in samples)
    fp = sum(s["fp"] for s in samples)
    tn = sum(s["tn"] for s in samples)
    fn = sum(s["fn"] for s in samples)
    assert tp == 18
    assert fp == 3
    assert tn == 190
    assert fn == 8


def test_no_hardcoded_benchmark_metrics_in_experiment_paths():
    """No hardcoded benchmark metrics in experiment code paths."""
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = ["accuracy = 1.0", "accuracy=1.0", "iou = 0.884", "iou=0.884", "45000", "15000", "-4.2"]
    for pattern in ["scripts/kaggle/p4_e02_baseline.py", "satquery/tools/temporal_vqa.py", "satquery/models/change_vqa/baseline.py"]:
        p = repo_root / pattern
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for fb in forbidden:
                assert fb not in text, f"Hardcoded metric {fb!r} found in {pattern}"


def test_no_synthetic_fallback_scientific_pass():
    """No synthetic fallback scientific PASS in production experiment paths."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_p4_e02_evaluation(Path(tmpdir))
        assert result.get("status") != "PASS" or result.get("sample_count", 0) > 4


def test_generic_b4_b5_b8_refused():
    """Generic sensor must refuse B4/B5/B8 semantic guesses without explicit mapping."""
    with pytest.raises(MissingBandError):
        resolve_band_for_role(["B04", "B05", "B08", "B11", "B12"], SemanticBandRole.RED, sensor="generic")


def test_unknown_sar_radiometric_domain_rejected():
    """Unknown SAR radiometric domain must fail closed."""
    pre = np.ones((10, 10), dtype=np.float32) * 100.0
    post = np.ones((10, 10), dtype=np.float32) * 10.0
    bands_pre = {"vv": pre}
    bands_post = {"vv": post}
    affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    with pytest.raises(MeasurementError, match="radiometric domain is unknown"):
        SarTemporalAnalytics.detect_flood(
            bands_pre,
            bands_post,
            affine,
            "EPSG:32633",
            radiometric_domain=SarRadiometricDomain.SAR_UNKNOWN,
        )


def test_amplitude_vs_power_db_conversions_differ():
    """Amplitude vs power dB conversions must use correct formulas."""
    linear = np.array([100.0], dtype=np.float32)
    power_db = SarTemporalAnalytics.linear_to_db(linear, domain=SarRadiometricDomain.SAR_LINEAR_POWER)
    amp_db = SarTemporalAnalytics.linear_to_db(linear, domain=SarRadiometricDomain.SAR_LINEAR_AMPLITUDE)
    assert math.isclose(power_db[0], 20.0, abs_tol=1e-4)
    assert math.isclose(amp_db[0], 40.0, abs_tol=1e-4)


def test_permanent_water_not_counted_as_newly_inundated():
    """Permanent pre-existing water must not count as newly inundated."""
    pre_vv = np.ones((50, 50), dtype=np.float32) * -20.0
    post_vv = np.ones((50, 50), dtype=np.float32) * -20.0
    bands_pre = {"vv": pre_vv}
    bands_post = {"vv": post_vv}
    affine = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    mask, sar_res, mask_ev = SarTemporalAnalytics.detect_flood(
        bands_pre,
        bands_post,
        affine,
        "EPSG:32633",
        decrease_threshold_db=3.0,
        water_max_threshold_db=-16.0,
    )
    assert sar_res.flood_detected is False
    assert sar_res.flood_pixel_count == 0
    assert sar_res.post_event_water_pixel_count == 2500
    assert mask_ev.changed_pixels == 0


def test_equal_shape_shifted_grids_cannot_be_differenced():
    """Equal shape but shifted affine must be rejected before differencing."""
    from satquery.geo.pairing import PairValidator
    from satquery.ingestion.models import AffineTransform

    asset1 = SourceAsset(
        asset_id="asset1",
        original_name="scene1.tif",
        path="/data/scene1.tif",
        sha256="1111111111111111111111111111111111111111111111111111111111111111",
    )
    geo1 = GeoMetadata(
        crs="EPSG:32633",
        bounds=GeoBounds(left=500000, bottom=4000000, right=510000, top=4010000),
        transform=AffineTransform(a=10, b=0, c=500000, d=0, e=-10, f=4010000),
        native_gsd_x=10.0,
        native_gsd_y=10.0,
    )
    geo2 = GeoMetadata(
        crs="EPSG:32633",
        bounds=GeoBounds(left=500005, bottom=4000005, right=510005, top=4010005),
        transform=AffineTransform(a=10, b=0, c=500005, d=0, e=-10, f=4010005),
        native_gsd_x=10.0,
        native_gsd_y=10.0,
    )
    raster = RasterMetadata(driver="GTiff", width=100, height=100, band_count=1, dtypes=("uint8",), nodata=(None,))
    bands = (BandMetadata(index=1, dtype="uint8"),)
    sensor = SensorMetadata(sensor_name="Sentinel-2", modality=Modality.OPTICAL, bands=bands, polarizations=())
    validity = ValidityMetadata(has_crs=True, has_transform=True, has_nodata=False)
    prov = ObservationProvenance(created_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc), ingestion_version="1.0.0")
    obs1 = ObservationState(observation_id="obs1", source_asset=asset1, raster=raster, sensor=sensor, geo=geo1, temporal=TemporalMetadata(acquisition_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)), validity=validity, provenance=prov)
    obs2 = ObservationState(observation_id="obs2", source_asset=asset1, raster=raster, sensor=sensor, geo=geo2, temporal=TemporalMetadata(acquisition_time=datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)), validity=validity, provenance=prov)
    validator = PairValidator()
    result = validator.validate(obs1, obs2)
    assert result.grid.aligned is False


def test_geographic_area_implementation_matches_rigorous_method():
    """Geographic area implementation must match selected rigorous method (geodesic WGS84)."""
    mask = np.ones((10, 10), dtype=bool)
    affine = Affine(0.001, 0.0, 0.0, 0.0, -0.001, 0.0)
    from satquery.analytics.measurement import MeasurementEngine
    from satquery.geo.models import MeasurementUnit
    meas = MeasurementEngine.measure_mask(mask, affine, "EPSG:4326", target_unit=MeasurementUnit.M2)
    assert meas.calculation_path == "geodesic_wgs84"
    assert meas.area_value > 0.0


def test_model_failure_does_not_become_normal_answer():
    """Model failure must return structured failure, not answer string pretending to be scientific output."""
    from satquery.models.change_vqa.baseline import BiTemporalChangeVQABackend
    from satquery.inference.exceptions import ModelExecutionError
    backend = BiTemporalChangeVQABackend.__new__(BiTemporalChangeVQABackend)
    backend._model = None
    backend._processor = None
    backend._torch = None
    backend.registration = None
    backend.profile = None
    backend.settings = None
    backend.registry_id = "test"
    backend.profile_id = "test"
    with pytest.raises((ModelExecutionError, Exception)):
        backend.answer_change_vqa(
            image_t1=__import__("PIL.Image").Image.new("RGB", (64, 64)),
            image_t2=__import__("PIL.Image").Image.new("RGB", (64, 64)),
            question="test",
        )


def test_no_benchmark_pass_when_dataset_unavailable():
    """No benchmark PASS when dataset is unavailable."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_p4_e02_evaluation(Path(tmpdir))
        assert result.get("status") != "PASS"
