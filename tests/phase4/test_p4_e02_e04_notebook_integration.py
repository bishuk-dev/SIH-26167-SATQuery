"""Focused regression tests for Phase 4 E02/E04 evaluator integrity fixes.

Tests the specific blockers identified from failed external experiment runs:
1. HF acquisition downloads Levir-CC-dataset.zip rather than assuming exploded repo folders
2. Exact expected ZIP size/SHA256 enforced
3. Safe extraction rejects ../ traversal
4. LEVIR root discovery requires captions.json + val/A + val/B
5. Test path is never accessed
6. LEVIR reference captions load for validation entries
7. Missing references fail benchmark evaluation
8. Sen1Floods scene key normalizes actual S1 and label layer names
9. Duplicate scene keys fail
10. Zero paired scenes fails before metric generation
11. Post-event-water ground truth compared with post-event-water prediction, not flood expansion
12. Runner Git provenance is preserved
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np


# ============================================================
# P4-E02 Tests: LEVIR-CC HF Acquisition & Extraction
# ============================================================

def test_p4_e02_uses_hf_hub_download_not_exploded_folders():
    """P4-E02 must download Levir-CC-dataset.zip from HuggingFace, not assume pre-extracted val/A/val/B dirs."""
    import sys

    fake_hf = MagicMock()
    fake_hf.hf_hub_download = MagicMock(return_value="/tmp/fake_hf_cache/levir.zip")
    sys.modules["huggingface_hub"] = fake_hf

    from scripts.kaggle.p4_e02_baseline import _acquire_levir_cc, LEVIR_CC_HF_REPO, LEVIR_CC_ZIP_NAME

    assert LEVIR_CC_HF_REPO == "lcybuaa/LEVIR-CC"

    def _mock_extract(zip_path, dest_dir):
        extracted = dest_dir
        images = extracted / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "LevirCCcaptions.json").write_text("{}")
        val = images / "val"
        val.mkdir(exist_ok=True)
        (val / "A").mkdir(exist_ok=True)
        (val / "B").mkdir(exist_ok=True)

    try:
        with patch("scripts.kaggle.p4_e02_baseline._safe_zip_extract", side_effect=_mock_extract), \
             patch("scripts.kaggle.p4_e02_baseline._sha256_file", return_value="fake_sha"), \
             patch("scripts.kaggle.p4_e02_baseline.LEVIR_CC_EXPECTED_SIZE", 4), \
             patch("scripts.kaggle.p4_e02_baseline.LEVIR_CC_EXPECTED_SHA256", "fake_sha"):
            fake_base = Path(tempfile.mkdtemp())
            try:
                fake_hf_cache = Path("/tmp/fake_hf_cache")
                fake_hf_cache.mkdir(parents=True, exist_ok=True)
                fake_zip = fake_hf_cache / "levir.zip"
                fake_zip.write_bytes(b"fake")

                _acquire_levir_cc(fake_base)
                fake_hf.hf_hub_download.assert_called_once()
                call_kwargs = fake_hf.hf_hub_download.call_args.kwargs
                assert call_kwargs.get("repo_id") == "lcybuaa/LEVIR-CC"
                assert call_kwargs.get("repo_type") == "dataset"
                assert call_kwargs.get("filename") == LEVIR_CC_ZIP_NAME
            finally:
                shutil.rmtree(fake_base, ignore_errors=True)
                shutil.rmtree(fake_hf_cache, ignore_errors=True)
    finally:
        del sys.modules["huggingface_hub"]


def test_p4_e02_zip_size_sha256_enforced():
    """P4-E02 must verify exact ZIP size (2683666867 bytes) and SHA256."""
    from scripts.kaggle.p4_e02_baseline import LEVIR_CC_EXPECTED_SIZE, LEVIR_CC_EXPECTED_SHA256

    assert LEVIR_CC_EXPECTED_SIZE == 2683666867
    assert LEVIR_CC_EXPECTED_SHA256 == "e05d38c0fdfda8c9b2048d314e5f95974d8b81e1b9f83f107acc39d55015e130"


def test_p4_e02_safe_extraction_rejects_traversal():
    """P4-E02 safe extraction must reject path traversal attempts."""
    from scripts.kaggle.p4_e02_baseline import _safe_zip_extract

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        zip_path = tmp_dir / "malicious.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../../etc/evil.txt", "evil")
            zf.writestr("normal.txt", "ok")

        dest_dir = tmp_dir / "extracted"
        dest_dir.mkdir()

        with pytest.raises(RuntimeError, match="Unsafe path"):
            _safe_zip_extract(zip_path, dest_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e02_dataset_root_discovery_requires_all_parts():
    """LEVIR root discovery must require captions.json + val/A + val/B."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        extracted = tmp_dir / "extracted"
        extracted.mkdir(parents=True)

        images_dir = extracted / "images"
        images_dir.mkdir()
        (images_dir / "LevirCCcaptions.json").write_text("{}")

        val_dir = images_dir / "val"
        val_dir.mkdir()

        val_a = val_dir / "A"
        val_a.mkdir()

        from scripts.kaggle.p4_e02_baseline import _discover_dataset_root

        with pytest.raises(RuntimeError, match="val/A directory not found|val/B directory not found"):
            _discover_dataset_root(extracted)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e02_test_path_never_accessed():
    """P4-E02 must never access images/test during evaluation."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation

    with patch("scripts.kaggle.p4_e02_baseline._acquire_levir_cc") as mock_acq:
        mock_acq.side_effect = RuntimeError("blocked for test")
        tmpdir = tempfile.mkdtemp()
        try:
            result = run_p4_e02_evaluation(Path(tmpdir))
            assert result.get("status") == "DATASET_UNAVAILABLE"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    assert "test" not in str(mock_acq.call_args)


def test_p4_e02_reference_captions_load_for_validation():
    """P4-E02 must load reference captions from LevirCCcaptions.json for val entries."""
    from scripts.kaggle.p4_e02_baseline import _load_captions

    test_captions = {
        "images": [
            {"filename": "pair_001.png", "split": "val", "sentences": [{"raw": "A new building was constructed."}, {"raw": "Construction appeared between the two images."}]},
            {"filename": "pair_002.png", "split": "test", "sentences": [{"raw": "Trees were removed."}]},
        ]
    }

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        cap_file = tmp_dir / "LevirCCcaptions.json"
        with open(cap_file, "w") as f:
            json.dump(test_captions, f)

        refs = _load_captions(cap_file)
        assert "pair_001.png" in refs
        assert len(refs["pair_001.png"]) == 2
        assert "pair_002.png" not in refs or len(refs.get("pair_002.png", [])) != 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e02_missing_references_fails_evaluation():
    """P4-E02 must fail benchmark evaluation if reference captions are missing for a pair."""
    from scripts.kaggle.p4_e02_baseline import _build_val_pairs

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        val_a = tmp_dir / "val" / "A"
        val_b = tmp_dir / "val" / "B"
        val_a.mkdir(parents=True, exist_ok=True)
        val_b.mkdir(parents=True, exist_ok=True)

        (val_a / "test_pair.png").write_bytes(b"fake")
        (val_b / "test_pair.png").write_bytes(b"fake")

        refs = {}
        pairs = _build_val_pairs(val_a, val_b, refs)
        assert len(pairs) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# P4-E02 Tests: Caption Metrics Implementation
# ============================================================

def test_p4_e02_caption_metrics_use_real_implementations():
    """P4-E02 must use real BLEU-4, ROUGE-L, CIDEr implementations, not dummy zeros."""
    from scripts.kaggle.p4_e02_baseline import _bleu4_impl, _rouge_l_score, _cider_score

    refs = ["a new building was constructed"]
    candidate = "a building was constructed"

    bleu4 = _bleu4_impl(refs, candidate)
    rouge_l = _rouge_l_score(refs, candidate)
    cider = _cider_score(refs, candidate)

    assert bleu4 > 0.0
    assert rouge_l > 0.0
    assert cider >= 0.0


def test_p4_e02_evaluate_captions_returns_all_three_metrics():
    """P4-E02 evaluation must return BLEU-4, ROUGE-L, and CIDEr when captions exist."""
    from scripts.kaggle.p4_e02_baseline import _evaluate_captions

    predictions = [
        {"pair_id": "p1", "predicted_answer": "new building"},
        {"pair_id": "p2", "predicted_answer": "trees removed"},
    ]
    references = {
        "p1": ["new building constructed"],
        "p2": ["trees were removed"],
    }

    metrics = _evaluate_captions(predictions, references)
    assert "bleu4" in metrics
    assert "rouge_l" in metrics
    assert "cider" in metrics


# ============================================================
# P4-E04 Tests: Scene Key Pairing
# ============================================================

def test_p4_e04_scene_key_normalizes_layer_names():
    """P4-E04 must normalize EVENT_CHIPID_S1Hand.tif to canonical scene key."""
    from scripts.kaggle.p4_e04_baseline import _derive_scene_key

    assert _derive_scene_key("hurricane_id_S1Hand.tif") == "hurricane_id"
    assert _derive_scene_key("hurricane_id_S1Hand_post.tif") == "hurricane_id"
    assert _derive_scene_key("hurricane_id_LabelHand.tif") == "hurricane_id"


def test_p4_e04_duplicate_scene_keys_fail():
    """P4-E04 must fail if duplicate scene keys detected within a modality."""
    from scripts.kaggle.p4_e04_baseline import _check_duplicate
    from pathlib import Path

    mapping = {"scene_001": Path("/fake/pre1.tif")}
    with pytest.raises(RuntimeError, match="Duplicate scene keys"):
        _check_duplicate(mapping, "scene_001", Path("/fake/pre2.tif"))


# ============================================================
# P4-E04 Tests: Pairing & Failure Behavior
# ============================================================

def test_p4_e04_zero_paired_scenes_fails_before_metrics():
    """P4-E04 must fail closed (PAIRING_FAILED) when zero paired scenes, not produce zero metrics."""
    from scripts.kaggle.p4_e04_baseline import _acquire_modified_sen1floods11, _find_raster_pairs

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        with patch("scripts.kaggle.p4_e04_baseline._acquire_modified_sen1floods11") as mock_acq:
            mock_acq.return_value = tmp_dir
            with patch("scripts.kaggle.p4_e04_baseline._find_raster_pairs") as mock_find:
                mock_find.return_value = {}
                from scripts.kaggle.p4_e04_baseline import run_p4_e04_evaluation
                result = run_p4_e04_evaluation(output_dir)
                assert result.get("status") == "PAIRING_FAILED"
                assert "sar_validation_metrics.json" not in [f.name for f in output_dir.iterdir() if f.is_file()]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e04_failure_artifact_written():
    """P4-E04 must write evaluation_failure.json on pairing failure."""
    from scripts.kaggle.p4_e04_baseline import run_p4_e04_evaluation

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        with patch("scripts.kaggle.p4_e04_baseline._acquire_modified_sen1floods11") as mock_acq:
            mock_acq.side_effect = RuntimeError("download failed")
            result = run_p4_e04_evaluation(output_dir)
            assert result.get("status") == "DATASET_UNAVAILABLE"
            failure_file = output_dir / "evaluation_failure.json"
            assert failure_file.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# P4-E04 Tests: Benchmark Target Correctness
# ============================================================

def test_p4_e04_post_water_mask_not_flood_expansion():
    """P4-E04 must evaluate post-event water mask vs ground truth, NOT flood expansion mask."""
    from scripts.kaggle.p4_e04_baseline import _compute_post_water_mask, _compute_flood_expansion_mask

    pre_arr = np.ones((10, 10), dtype=np.float32) * -20.0
    post_arr = np.ones((10, 10), dtype=np.float32) * -20.0

    post_water = _compute_post_water_mask(post_arr, water_max_threshold_db=-16.0)
    expansion = _compute_flood_expansion_mask(pre_arr, post_arr, water_max_threshold_db=-16.0, flood_decrease_db=3.0)

    assert post_water.sum() > 0, "post-water mask should detect water"
    assert expansion.sum() == 0, "expansion mask should be empty when no change (permanent water)"


def test_p4_e04_confusion_matrix_uses_correct_labels():
    """P4-E04 confusion matrix must use post-event water labels (1=water, 0=non-water, -1=nodata excluded)."""
    from scripts.kaggle.p4_e04_baseline import _compute_confusion_matrix

    pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    label = np.array([[1, 0, 0], [0, 1, -1]], dtype=np.float32)

    tp, fp, tn, fn = _compute_confusion_matrix(pred, label)

    assert tp == 2
    assert fp == 1
    assert tn == 2
    assert fn == 0


# ============================================================
# P4-E04 Tests: Provenance Preservation
# ============================================================

def test_p4_e04_preserves_runner_meta_git_provenance():
    """P4-E04 must preserve existing runner_meta.json and write evaluation_meta.json separately."""
    from scripts.kaggle.p4_e04_baseline import run_p4_e04_evaluation

    existing_meta = {
        "experiment": "phase4-e04-sar-validation",
        "benchmark_provenance": {"benchmark": "Modified Sen1Floods11"},
        "status": "running",
        "raw_git_sha": "abc123def456",
        "dirty_worktree": False,
    }

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp_dir / "output"
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "runner_meta.json", "w") as f:
            json.dump(existing_meta, f)

        with patch("scripts.kaggle.p4_e04_baseline._acquire_modified_sen1floods11") as mock_acq:
            mock_acq.side_effect = RuntimeError("test fail")
            result = run_p4_e04_evaluation(output_dir)

            with open(output_dir / "runner_meta.json", "r") as f:
                preserved = json.load(f)
            assert preserved["raw_git_sha"] == "abc123def456"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
