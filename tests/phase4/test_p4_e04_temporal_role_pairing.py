"""Regression tests for P4-E04 temporal role classification, scene-key normalization,
independent pairing, and strict geospatial grid compatibility verification.

Covers:
1. PRE and POST with IDENTICAL basenames (e.g. PRE/..._S1Hand.tif and POST/..._S1Hand.tif)
   correctly classified by archive/directory provenance and paired (pre=1, post=1, label=1, paired=1).
2. Optional _S1Hand_post normalization succeeds when present.
3. Uppercase .TIF and .TIFF extensions enumerated and paired.
4. Duplicate PRE key detected and rejected.
5. Duplicate POST key detected and rejected.
6. Missing label results in unmatched key and pairing failure when paired_count == 0.
7. CRS mismatch detected -> GRID_MISMATCH.
8. Affine shift with identical array shape -> GRID_MISMATCH.
9. Fully aligned triplet passes grid verification and evaluates correctly.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = REPO_ROOT / "scripts" / "kaggle"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.kaggle.p4_e04_baseline import (
    DatasetRoles,
    _derive_scene_key,
    _enumerate_rasters,
    _build_role_map,
    _check_grid_alignment,
    _load_raster_data,
    run_p4_e04_evaluation,
)


def _create_mock_raster(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    transform: Affine = Affine(0.0001, 0, 10.0, 0, -0.0001, 20.0),
    nodata: float | None = -1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


class TestP4E04TemporalRolePairingAndGrid:
    def test_1_pre_and_post_identical_basenames_pair_by_provenance(self, tmp_path: Path) -> None:
        """PRE and POST with IDENTICAL basenames pair cleanly via directory provenance."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)
        arr_lbl = np.ones((10, 10), dtype=np.float32)

        # IDENTICAL filename in PRE and POST: Bolivia_103757_S1Hand.tif
        _create_mock_raster(pre_dir / "Bolivia_103757_S1Hand.tif", arr_sar)
        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand.tif", arr_sar)
        _create_mock_raster(label_dir / "Bolivia_103757_LabelHand.tif", arr_lbl)

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        metrics = run_p4_e04_evaluation(output_dir, roles=roles)

        audit = metrics["scene_pairing_audit"]
        assert audit["pre_count"] == 1
        assert audit["post_count"] == 1
        assert audit["label_count"] == 1
        assert audit["paired_count"] == 1
        assert metrics["sample_count"] == 1
        assert metrics["status"] == "PASS"

    def test_2_optional_s1hand_post_normalization(self, tmp_path: Path) -> None:
        """_S1Hand_post normalizes correctly to the base scene key."""
        key = _derive_scene_key("Bolivia_103757_S1Hand_post.tif")
        assert key == "Bolivia_103757"

        # Also test with .tiff
        key_tiff = _derive_scene_key("Bolivia_103757_S1Hand_post.tiff")
        assert key_tiff == "Bolivia_103757"

    def test_3_uppercase_tif_and_tiff_enumerated_and_paired(self, tmp_path: Path) -> None:
        """Uppercase .TIF and .TIFF extensions are discovered and paired."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)
        arr_lbl = np.ones((10, 10), dtype=np.float32)

        _create_mock_raster(pre_dir / "SceneA_100_S1Hand.TIF", arr_sar)
        _create_mock_raster(post_dir / "SceneA_100_S1Hand_post.TIFF", arr_sar)
        _create_mock_raster(label_dir / "SceneA_100_LabelHand.tif", arr_lbl)

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        metrics = run_p4_e04_evaluation(output_dir, roles=roles)
        audit = metrics["scene_pairing_audit"]
        assert audit["paired_count"] == 1
        assert metrics["sample_count"] == 1

    def test_4_duplicate_pre_key_detected_and_rejected(self, tmp_path: Path) -> None:
        """Duplicate scene key in PRE is rejected and tracked."""
        pre_dir = tmp_path / "PRE"
        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)

        _create_mock_raster(pre_dir / "dir1" / "Bolivia_103757_S1Hand.tif", arr_sar)
        _create_mock_raster(pre_dir / "dir2" / "Bolivia_103757_S1Hand.tif", arr_sar)

        files = _enumerate_rasters(pre_dir)
        mapping, dupes, unrec = _build_role_map(files)

        assert "Bolivia_103757" in dupes
        assert "Bolivia_103757" not in mapping, "Duplicate key must be rejected from map"

    def test_5_duplicate_post_key_detected_and_rejected(self, tmp_path: Path) -> None:
        """Duplicate scene key in POST is rejected and tracked."""
        post_dir = tmp_path / "POST"
        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)

        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand_post.tif", arr_sar)
        _create_mock_raster(post_dir / "sub" / "Bolivia_103757_S1Hand.tif", arr_sar)

        files = _enumerate_rasters(post_dir)
        mapping, dupes, unrec = _build_role_map(files)

        assert "Bolivia_103757" in dupes
        assert "Bolivia_103757" not in mapping

    def test_6_missing_label_results_in_unmatched_and_pairing_failed(self, tmp_path: Path) -> None:
        """Missing labels cause paired_count == 0 -> PAIRING_FAILED."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)
        _create_mock_raster(pre_dir / "Bolivia_103757_S1Hand.tif", arr_sar)
        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand.tif", arr_sar)
        # label_dir is empty!

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        result = run_p4_e04_evaluation(output_dir, roles=roles)
        assert result["status"] == "PAIRING_FAILED"
        assert (output_dir / "evaluation_failure.json").exists()
        fail_data = json.loads((output_dir / "evaluation_failure.json").read_text(encoding="utf-8"))
        assert fail_data["scene_pairing_audit"]["unmatched_pre"] == 1
        assert fail_data["scene_pairing_audit"]["paired_count"] == 0

    def test_7_crs_mismatch_detected_as_grid_mismatch(self, tmp_path: Path) -> None:
        """CRS mismatch between POST and LABEL triggers GRID_MISMATCH."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)
        arr_lbl = np.ones((10, 10), dtype=np.float32)

        _create_mock_raster(pre_dir / "Bolivia_103757_S1Hand.tif", arr_sar, crs="EPSG:4326")
        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand.tif", arr_sar, crs="EPSG:4326")
        # Label has different CRS
        _create_mock_raster(label_dir / "Bolivia_103757_LabelHand.tif", arr_lbl, crs="EPSG:32619")

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        metrics = run_p4_e04_evaluation(output_dir, roles=roles)
        assert metrics["grid_mismatch_count"] == 1
        assert metrics["grid_valid_count"] == 0
        assert metrics["sample_count"] == 0

        # Check prediction line has GRID_MISMATCH
        preds = [json.loads(line) for line in (output_dir / "sar_validation_predictions.jsonl").read_text().splitlines()]
        assert len(preds) == 1
        assert preds[0]["status"] == "GRID_MISMATCH"
        assert "CRS mismatch" in preds[0]["failure_reason"]

    def test_8_affine_shift_with_identical_shape_detected_as_grid_mismatch(self, tmp_path: Path) -> None:
        """Spatial affine shift with identical array shape triggers GRID_MISMATCH."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        arr_sar = np.full((10, 10), -18.0, dtype=np.float32)
        arr_lbl = np.ones((10, 10), dtype=np.float32)

        tf1 = Affine(0.0001, 0, 10.0, 0, -0.0001, 20.0)
        # Shifted origin
        tf_shifted = Affine(0.0001, 0, 10.5, 0, -0.0001, 20.0)

        _create_mock_raster(pre_dir / "Bolivia_103757_S1Hand.tif", arr_sar, transform=tf1)
        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand.tif", arr_sar, transform=tf1)
        _create_mock_raster(label_dir / "Bolivia_103757_LabelHand.tif", arr_lbl, transform=tf_shifted)

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        metrics = run_p4_e04_evaluation(output_dir, roles=roles)
        assert metrics["grid_mismatch_count"] == 1
        assert metrics["grid_valid_count"] == 0
        assert metrics["sample_count"] == 0

        preds = [json.loads(line) for line in (output_dir / "sar_validation_predictions.jsonl").read_text().splitlines()]
        assert preds[0]["status"] == "GRID_MISMATCH"
        assert "affine transform coeff" in preds[0]["failure_reason"]

    def test_9_fully_aligned_triplet_passes_grid_verification_and_evaluates(self, tmp_path: Path) -> None:
        """Fully aligned triplet passes grid verification and computes pixel metrics accurately."""
        pre_dir = tmp_path / "PRE"
        post_dir = tmp_path / "POST"
        label_dir = tmp_path / "LABEL"

        # SAR post: -18 dB (water <= -16 dB)
        arr_pre = np.full((10, 10), -12.0, dtype=np.float32)
        arr_post = np.full((10, 10), -18.0, dtype=np.float32)
        # Label: 1 (water)
        arr_lbl = np.ones((10, 10), dtype=np.float32)

        tf = Affine(0.0001, 0, 10.0, 0, -0.0001, 20.0)
        _create_mock_raster(pre_dir / "Bolivia_103757_S1Hand.tif", arr_pre, transform=tf)
        _create_mock_raster(post_dir / "Bolivia_103757_S1Hand.tif", arr_post, transform=tf)
        _create_mock_raster(label_dir / "Bolivia_103757_LabelHand.tif", arr_lbl, transform=tf)

        roles = DatasetRoles(pre_dir=pre_dir, post_dir=post_dir, label_dir=label_dir)
        output_dir = tmp_path / "out"

        metrics = run_p4_e04_evaluation(output_dir, roles=roles)
        assert metrics["status"] == "PASS"
        assert metrics["grid_valid_count"] == 1
        assert metrics["grid_mismatch_count"] == 0
        assert metrics["sample_count"] == 1
        assert metrics["total_tp"] == 100
        assert metrics["total_fp"] == 0
        assert metrics["total_fn"] == 0
        assert metrics["post_event_water_iou"] == 1.0
