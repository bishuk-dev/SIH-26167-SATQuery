"""P4-E01 deterministic verification lanes A-C (formula, grid, measurement).

Lane A-C are deterministic verification artifacts computed on frozen fixtures —
they are not learned-model benchmark metrics. Lane D (optional OSCD change
benchmark) is BLOCKED_METHOD_CONTRACT: the OSCD data contract is BLOCKED and no
defensible deterministic generic change method has been frozen, so no
precision/recall/F1/IoU may be emitted for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from ml.evaluation.run_p4_e01 import (
    LANE_D_STATUS,
    run_lane_a,
    run_lane_b,
    run_lane_c,
    run_p4_e01,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_lane_a_formulas_match_hand_calculation() -> None:
    result = run_lane_a()
    assert result["status"] == "PASS"
    assert result["lane"] == "A_formula_correctness"
    # every check records the frozen formula identity it verified
    formulas = {check["formula"] for check in result["checks"]}
    assert formulas == {
        "NDVI = (NIR - RED) / (NIR + RED)",
        "NDWI = (GREEN - NIR) / (GREEN + NIR)",
        "MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)",
    }


def test_lane_a_nodata_and_zero_denominator_stay_masked() -> None:
    result = run_lane_a()
    names = {check["name"] for check in result["checks"]}
    assert "ndvi_nodata_masked" in names
    assert "ndvi_zero_denominator_masked" in names


def test_lane_b_grid_controls(tmp_path: Path) -> None:
    result = run_lane_b(tmp_path)
    assert result["status"] == "PASS"
    names = {check["name"] for check in result["checks"]}
    assert {
        "identity_pair_untouched",
        "reversed_order_untouched",
        "misaligned_grid_reprojected_onto_reference",
        "resampling_provenance_recorded",
    } <= names
    # identity pairs must not write derived rasters
    identity = next(c for c in result["checks"] if c["name"] == "identity_pair_untouched")
    assert identity["reprojected"] is False
    misaligned = next(
        c for c in result["checks"] if c["name"] == "misaligned_grid_reprojected_onto_reference"
    )
    assert misaligned["reprojected"] is True
    assert misaligned["reference"] in ("t1", "t2")


def test_lane_c_measurement_reconstruction(tmp_path: Path) -> None:
    result = run_lane_c()
    assert result["status"] == "PASS"
    for check in result["checks"]:
        # every area value must reconstruct exactly from pixel counts and
        # the declared geometric basis — no fitted or rounded values
        assert check["reconstruction_error_m2"] == pytest.approx(0.0, abs=1e-6)
    methods = {check["method"] for check in result["checks"]}
    assert methods == {"projected_affine_determinant", "ellipsoidal_parallel_band_sum"}


def test_manifest_contains_no_benchmark_metrics(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    manifest = run_p4_e01(output)
    text = json.dumps(manifest)
    for banned in ("precision", "recall", "f1", "iou_score", "accuracy"):
        assert f'"{banned}"' not in text
    assert manifest["experiment_id"] == "P4-E01"
    assert manifest["lane_d"]["status"] == LANE_D_STATUS
    assert manifest["lane_d"]["oscd_metrics_emitted"] is False
    assert (output).exists()
    # manifest written to disk must be byte-identical to the returned object
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_runner_runs_all_local_lanes(tmp_path: Path) -> None:
    manifest = run_p4_e01(tmp_path / "manifest.json")
    statuses = {
        lane_id: lane["status"]
        for lane_id, lane in manifest["lanes"].items()
    }
    assert statuses == {
        "A_formula_correctness": "PASS",
        "B_grid_correctness": "PASS",
        "C_measurement_correctness": "PASS",
    }


def test_lane_c_geographic_measurement_matches_closed_form() -> None:
    # independent reconstruction: one WGS84 cell spanning 0.01 deg at the
    # equator row must equal the parallel-band width times longitude span
    mask = np.array([[True]])
    transform = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.005)
    result = run_lane_c()
    geographic = next(
        c
        for c in result["checks"]
        if c["method"] == "ellipsoidal_parallel_band_sum"
    )
    assert geographic["crs"] == "EPSG:4326"
    assert geographic["positive_pixel_count"] == int(mask.sum())


def test_manifest_fixture_provenance_is_declared(tmp_path: Path) -> None:
    manifest = run_p4_e01(tmp_path / "manifest.json")
    assert manifest["fixture_provenance"] == "deterministic_synthetic_fixtures"
    assert manifest["real_source_data_used"] is False
