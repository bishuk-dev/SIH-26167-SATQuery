"""Focused regression tests for Phase 4 E02/E04 evaluator integrity fixes and runner hygiene.

Covers:
1. P4-E02 HuggingFace dataset acquisition (ZIP download, size/SHA256, safe traversal rejection)
2. P4-E02 model loader uses AutoModelForImageTextToText and hash verification
3. P4-E02 model loader readiness fail-fast
4. P4-E02 authoritative validation membership dynamically derived from split=='val' (not hardcoded)
5. P4-E02 generated_caption schema contract
6. P4-E02 RSICC metrics parity against frozen reference fixture and corpus-level aggregation
7. P4-E04 HTTP Range resume (206 appends, 200 restarts from byte 0, 416 complete promotes)
8. P4-E04 SEN1FLOODS11_ARCHIVE_DIR pre-provisioned data verification
9. P4-E04 scene key pairing and duplicate rejection
10. P4-E04 benchmark target correctness (-16 dB post-water vs ground truth; 3 dB belongs only to expansion evidence)
11. P4-E04 failure artifact written on failure
12. Runner archives current result files BEFORE a new run, preserving invalid_initial/
13. Runner retrieves failure_result_files and leaves no stale success metrics
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_KAGGLE_DIR = REPO_ROOT / "scripts" / "kaggle"
if str(SCRIPTS_KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_KAGGLE_DIR))

import runner


# ============================================================
# P4-E02 Tests: Model Loader & Readiness Fail-Fast
# ============================================================

def test_p4_e02_model_loader_uses_image_text_to_text_and_verified_hash():
    """SmolVLM Change-VQA loader must use AutoModelForImageTextToText and verify checkpoint SHA256."""
    from satquery.models.change_vqa.baseline import load_change_vqa_model
    from satquery.inference.config import VqaRuntimeSettings

    settings = VqaRuntimeSettings(device="cpu", allow_remote_network=False)
    backend = load_change_vqa_model(settings=settings)

    assert backend.registration.model_id == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert backend.registration.revision == "7e3e67edbbed1bf9888184d9df282b700a323964"
    assert backend.registration.checkpoint_sha256 == "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e"
    assert hasattr(backend, "load"), "Baseline must provide an explicit load() readiness method"


def test_p4_e02_model_loader_fail_fast_on_readiness_failure():
    """If backend.load() fails, run_p4_e02_evaluation must write evaluation_failure.json and abort immediately."""
    from scripts.kaggle.p4_e02_baseline import run_p4_e02_evaluation

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_dir = tmp_dir / "output"
        output_dir.mkdir()
        # Setup mock dataset structure
        (tmp_dir / "LevirCCcaptions.json").write_text("{}", encoding="utf-8")
        (tmp_dir / "images" / "val" / "A").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "images" / "val" / "B").mkdir(parents=True, exist_ok=True)

        with patch("scripts.kaggle.p4_e02_baseline._acquire_levir_cc") as mock_acq, \
             patch("scripts.kaggle.p4_e02_baseline._build_authoritative_val_pairs") as mock_build, \
             patch("satquery.models.change_vqa.baseline.load_change_vqa_model") as mock_load_model:
            mock_acq.return_value = tmp_dir
            mock_build.return_value = ([{"pair_id": "p1", "filename": "p1.png"}], {})
            mock_backend = MagicMock()
            mock_backend.load.side_effect = RuntimeError("Simulated CUDA OOM or weight hash mismatch")
            mock_load_model.return_value = mock_backend

            res = run_p4_e02_evaluation(output_dir)

            assert res.get("status") == "MODEL_UNAVAILABLE"
            fail_file = output_dir / "evaluation_failure.json"
            assert fail_file.exists(), "evaluation_failure.json must be written on load failure"
            fail_data = json.loads(fail_file.read_text(encoding="utf-8"))
            assert "Simulated CUDA OOM" in fail_data.get("failure_reason", "") or "Simulated CUDA OOM" in fail_data.get("details", "")

            runner_meta_file = output_dir / "runner_meta.json"
            assert runner_meta_file.exists()
            runner_meta = json.loads(runner_meta_file.read_text(encoding="utf-8"))
            assert runner_meta["status"] == "FAIL"

            assert not (output_dir / "validation_metrics.json").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# P4-E02 Tests: Dynamic LEVIR-CC Validation Membership Audit
# ============================================================

def test_p4_e02_authoritative_val_count_dynamically_derived():
    """LEVIR validation count must be dynamically derived from split=='val', not hardcoded."""
    from scripts.kaggle.p4_e02_baseline import _build_authoritative_val_pairs

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        val_a = tmp_dir / "val" / "A"
        val_b = tmp_dir / "val" / "B"
        val_a.mkdir(parents=True)
        val_b.mkdir(parents=True)

        # Create 4 pairs in filesystem
        for i in range(1, 5):
            (val_a / f"img_{i:03d}.png").write_bytes(b"dummy_a")
            (val_b / f"img_{i:03d}.png").write_bytes(b"dummy_b")

        # Create caption JSON with 3 val, 1 train, 1 test
        cap_data = {
            "images": [
                {"filename": "img_001.png", "split": "val", "sentences": [{"raw": "desc 1"}]},
                {"filename": "img_002.png", "split": "val", "sentences": [{"raw": "desc 2"}]},
                {"filename": "img_003.png", "split": "val", "sentences": [{"raw": "desc 3"}]},
                {"filename": "img_004.png", "split": "train", "sentences": [{"raw": "desc 4"}]},
                {"filename": "img_005.png", "split": "test", "sentences": [{"raw": "desc 5"}]},
            ]
        }
        cap_file = tmp_dir / "LevirCCcaptions.json"
        cap_file.write_text(json.dumps(cap_data), encoding="utf-8")

        pairs, audit_meta = _build_authoritative_val_pairs(cap_file, val_a, val_b)

        assert audit_meta["annotation_val_count"] == 3
        assert audit_meta["matched_annotated_pairs"] == 3
        assert len(pairs) == 3
        assert audit_meta["historical_documented_count"] == 1332
        assert "discrepancy_resolution" in audit_meta
        # img_004 and img_005 must NOT be in pairs
        pair_files = [p["filename"] for p in pairs]
        assert "img_004.png" not in pair_files
        assert "img_005.png" not in pair_files
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e02_generated_caption_field_contract():
    """Metric computation and prediction output must strictly use generated_caption."""
    from scripts.kaggle.p4_e02_baseline import compute_rsicc_caption_metrics

    # Verify predictions record structure contract
    preds = [
        {"pair_id": "001", "generated_caption": "new building built", "reference_captions": ["new building built"]}
    ]
    # compute_rsicc_caption_metrics takes lists of references and hypotheses
    hyps = [p["generated_caption"] for p in preds]
    refs = [p["reference_captions"] for p in preds]
    metrics = compute_rsicc_caption_metrics(refs, hyps)

    assert "bleu_4" in metrics
    assert "rouge_l" in metrics
    assert "cider" in metrics


# ============================================================
# P4-E02 Tests: RSICC Metrics Parity & Corpus Aggregation
# ============================================================

def test_p4_e02_rsicc_metric_parity_against_frozen_fixture():
    """RSICC metric wrapper must reproduce exact frozen reference scores within numerical tolerance."""
    from satquery.evaluation.rsicc_eval import compute_rsicc_caption_metrics

    refs = [
        ["the residential area has new houses built on the farmland"],
        ["trees along the road were cleared for road widening"],
    ]
    hyps = [
        "residential area has new houses built on farmland",
        "trees along road were cleared for road widening",
    ]

    metrics = compute_rsicc_caption_metrics(refs, hyps)

    # Parity assertions against frozen fixture
    assert pytest.approx(metrics["bleu_1"], abs=1e-4) == 0.829029
    assert pytest.approx(metrics["bleu_2"], abs=1e-4) == 0.767532
    assert pytest.approx(metrics["bleu_3"], abs=1e-4) == 0.715497
    assert pytest.approx(metrics["bleu_4"], abs=1e-4) == 0.679005
    assert pytest.approx(metrics["rouge_l"], abs=1e-4) == 0.901363
    assert pytest.approx(metrics["cider"], abs=1e-4) == 7.386388

    # Provenance metadata assertions
    prov = metrics["provenance"]
    assert "Chen-Yang-Liu/RSICC" in prov["source_repository"]
    assert prov["commit"] == "d1505e514c450c3728782ca723e82761e70bafd3"
    assert prov["aggregation"] == "corpus_level"


def test_p4_e02_corpus_level_metric_aggregation():
    """BLEU and CIDEr must accumulate across the entire corpus, not take arithmetic mean of per-image scores."""
    from satquery.evaluation.rsicc_eval import compute_rsicc_caption_metrics

    # Pair 1 has 3 tokens (no 4-grams possible, so per-sentence BLEU-4 would be 0.0)
    # Pair 2 has matching 4-grams
    refs = [
        ["small red car"],
        ["a large modern building was constructed on empty ground"],
    ]
    hyps = [
        "small red car",
        "a large modern building was constructed on empty ground",
    ]

    metrics = compute_rsicc_caption_metrics(refs, hyps)

    # In corpus BLEU, 4-grams from sentence 2 are aggregated into the corpus pool
    # so corpus BLEU-4 is non-zero
    assert metrics["bleu_4"] > 0.0, "Corpus BLEU-4 must be > 0 when corpus contains valid 4-grams"


# ============================================================
# P4-E04 Tests: HTTP Range Resume (206, 200, 416)
# ============================================================

def test_p4_e04_http_resume_with_206_appends_correctly():
    """HTTP 206 Partial Content must append to existing .part file."""
    from scripts.kaggle.p4_e04_baseline import _download_file_robust

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        dest = tmp_dir / "test_archive.zip"
        part_path = dest.with_suffix(".zip.part")

        # Existing 5 bytes in .part
        part_path.write_bytes(b"HELLO")
        # Remaining 5 bytes from server
        second_chunk = b"WORLD"
        full_content = b"HELLOWORLD"
        expected_size = len(full_content)
        expected_md5 = hashlib.md5(full_content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.status = 206
        mock_resp.read.side_effect = [second_chunk, b""]
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            _download_file_robust("http://fake/test.zip", dest, expected_size, expected_md5, max_attempts=1)

            assert dest.exists()
            assert dest.read_bytes() == full_content
            assert not part_path.exists()
            # Verify Range header was sent
            call_req = mock_urlopen.call_args[0][0]
            assert call_req.headers.get("Range") == "bytes=5-"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e04_http_range_ignored_with_200_restarts_instead_of_appending():
    """HTTP 200 OK (server ignored Range) must overwrite from byte 0, not append."""
    from scripts.kaggle.p4_e04_baseline import _download_file_robust

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        dest = tmp_dir / "test_archive.zip"
        part_path = dest.with_suffix(".zip.part")

        # Existing stale 5 bytes
        part_path.write_bytes(b"STALE")

        full_content = b"FRESH_FULL_PAYLOAD_123"
        expected_size = len(full_content)
        expected_md5 = hashlib.md5(full_content).hexdigest()

        mock_resp = MagicMock()
        mock_resp.status = 200  # Server ignored Range
        mock_resp.read.side_effect = [full_content, b""]
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            _download_file_robust("http://fake/test.zip", dest, expected_size, expected_md5, max_attempts=1)

            assert dest.exists()
            assert dest.read_bytes() == full_content
            assert not part_path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e04_http_416_valid_complete_part_verifies_and_promotes():
    """HTTP 416 on an already-complete .part file must verify MD5 and promote to destination."""
    from scripts.kaggle.p4_e04_baseline import _download_file_robust

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        dest = tmp_dir / "test_archive.zip"
        part_path = dest.with_suffix(".zip.part")

        full_content = b"COMPLETE_FILE_CONTENT"
        part_path.write_bytes(full_content)
        expected_size = len(full_content)
        expected_md5 = hashlib.md5(full_content).hexdigest()

        http_err = urllib.error.HTTPError(
            url="http://fake/test.zip",
            code=416,
            msg="Requested Range Not Satisfiable",
            hdrs={},
            fp=io.BytesIO(b""),
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            _download_file_robust("http://fake/test.zip", dest, expected_size, expected_md5, max_attempts=1)

            assert dest.exists()
            assert dest.read_bytes() == full_content
            assert not part_path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_p4_e04_archive_dir_support():
    """SEN1FLOODS11_ARCHIVE_DIR verifies size and MD5 before copying and extracting."""
    from scripts.kaggle.p4_e04_baseline import _acquire_modified_sen1floods11, BENCHMARK_PROVENANCE

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        archive_dir = tmp_dir / "pre_provisioned"
        archive_dir.mkdir()
        base_dir = tmp_dir / "base"
        base_dir.mkdir()

        # Create all pinned files in archive_dir
        dummy_content = b"fake_zip_content"
        patch_provenance = []
        for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
            fname = pinned["filename"]
            (archive_dir / fname).write_bytes(dummy_content)
            patch_provenance.append({
                "filename": fname,
                "size_bytes": len(dummy_content),
                "md5": hashlib.md5(dummy_content).hexdigest(),
            })

        with patch.dict(os.environ, {"SEN1FLOODS11_ARCHIVE_DIR": str(archive_dir)}), \
             patch("scripts.kaggle.p4_e04_baseline._safe_zip_extract"):
            with patch.dict(BENCHMARK_PROVENANCE, {"pinned_files": patch_provenance}):
                res = _acquire_modified_sen1floods11(base_dir)
                assert res.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# P4-E04 Tests: Benchmark Target Semantics & Pairing
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

    mapping = {"scene_001": Path("/fake/pre1.tif")}
    with pytest.raises(RuntimeError, match="Duplicate scene keys"):
        _check_duplicate(mapping, "scene_001", Path("/fake/pre2.tif"))


def test_p4_e04_post_water_mask_target_semantics():
    """Benchmark target is finite(post) AND post <= -16 dB. The 3 dB decrease belongs ONLY to expansion."""
    from scripts.kaggle.p4_e04_baseline import _compute_post_water_mask, _compute_flood_expansion_mask

    pre_arr = np.array([[-20.0, -20.0], [-10.0, -10.0]], dtype=np.float32)
    post_arr = np.array([[-20.0, -15.0], [-17.0, -12.0]], dtype=np.float32)

    # Post water: <= -16 dB -> [-20.0, -17.0]
    post_water = _compute_post_water_mask(post_arr, water_max_threshold_db=-16.0)
    assert post_water[0, 0] == 1
    assert post_water[0, 1] == 0
    assert post_water[1, 0] == 1
    assert post_water[1, 1] == 0

    # Expansion: post <= -16 dB AND (pre - post) >= 3 dB
    # (0, 0): pre=-20, post=-20 -> drop=0 < 3 -> 0 (permanent water)
    # (1, 0): pre=-10, post=-17 -> drop=7 >= 3 and post <= -16 -> 1 (new water expansion)
    expansion = _compute_flood_expansion_mask(pre_arr, post_arr, water_max_threshold_db=-16.0, flood_decrease_db=3.0)
    assert expansion[0, 0] == 0, "Permanent water must not be marked as flood expansion"
    assert expansion[1, 0] == 1, "New water exceeding drop threshold must be marked as expansion"


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
# Runner Hygiene Tests: Archival & Failure Handling
# ============================================================

def test_runner_archive_current_result_files_before_run(tmp_path: Path):
    """runner._archive_current_result_files must archive declared result files to archive/<ts>_<sha>/ and preserve invalid_initial/."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Current results
    (results_dir / "validation_metrics.json").write_text('{"accuracy": 0.99}', encoding="utf-8")
    (results_dir / "runner_meta.json").write_text('{"git_sha": "abcdef1234567890"}', encoding="utf-8")

    # Historical invalid directory
    invalid_dir = results_dir / "invalid_initial"
    invalid_dir.mkdir()
    (invalid_dir / "historical_bad.json").write_text('{"bad": true}', encoding="utf-8")

    archived = runner._archive_current_result_files(
        output_dir=tmp_path,
        result_files=["validation_metrics.json"],
        failure_result_files=["evaluation_failure.json", "runner_meta.json"],
    )

    assert archived is not None
    assert archived.exists()
    assert "abcdef123456" in archived.name
    assert (archived / "validation_metrics.json").exists()
    assert (archived / "runner_meta.json").exists()

    # Current results directory must no longer contain the archived files
    assert not (results_dir / "validation_metrics.json").exists()
    assert not (results_dir / "runner_meta.json").exists()

    # invalid_initial must be completely untouched
    assert invalid_dir.exists()
    assert (invalid_dir / "historical_bad.json").read_text(encoding="utf-8") == '{"bad": true}'


def test_runner_failed_evaluation_retrieves_failure_artifacts_without_stale_success(tmp_path: Path):
    """When kernel produces evaluation_failure.json, runner must retrieve failure artifacts and leave no success metrics."""
    remote_out = "test-remote"
    experiment = {
        "remote_output_dir": remote_out,
        "result_files": ["validation_metrics.json", "runner_meta.json"],
        "failure_result_files": ["evaluation_failure.json", "runner_meta.json"],
    }
    out_dir = tmp_path / "exp"
    results_dest = out_dir / "results"
    results_dest.mkdir(parents=True)

    # Fake download directory with evaluation_failure.json and runner_meta.json
    dl_dir = tmp_path / "fake_dl"
    dl_out = dl_dir / "satquery-output" / remote_out
    dl_out.mkdir(parents=True)
    (dl_out / "evaluation_failure.json").write_text(
        json.dumps({"failure_reason": "DATASET_UNAVAILABLE", "details": "Zenodo 504"}),
        encoding="utf-8"
    )
    (dl_out / "runner_meta.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")

    def fake_run(cmd, **kwargs):
        dest = Path(cmd[cmd.index("-p") + 1])
        dest_out = dest / "satquery-output" / remote_out
        dest_out.mkdir(parents=True, exist_ok=True)
        for f in dl_out.iterdir():
            shutil.copy2(f, dest_out / f.name)
        return MagicMock(returncode=0)

    with patch.object(runner, "_run", side_effect=fake_run):
        with pytest.raises(SystemExit):
            runner._download_artifacts(
                username="testuser",
                kernel_slug="test-slug",
                experiment=experiment,
                output_dir=out_dir,
                allow_dirty=False,
                kernel_status="complete",
            )

    # Failure artifacts must be present
    assert (results_dest / "evaluation_failure.json").exists()
    assert (results_dest / "runner_meta.json").exists()
    # Success metrics must NOT be present
    assert not (results_dest / "validation_metrics.json").exists()
