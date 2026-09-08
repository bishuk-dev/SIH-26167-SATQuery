"""Regression tests for Kaggle runner path-qualified artifact retrieval and identity validation.

Verifies:
1. Cloned-repo historical validation_metrics.json cannot be selected for E02 when a correct remote-output artifact exists.
2. invalid_initial files can never be selected as current remote results.
3. Basename collision across two directories under remote_output causes failure (ARTIFACT_AMBIGUOUS), never arbitrary hits[0].
4. Path-qualified result retrieval succeeds.
5. E04 runner_meta is rejected for E02.
6. Wrong Git SHA is rejected.
7. Prediction row count must equal metrics sample_count.
8. Artifact publication is atomic: validation failure leaves no partially published current result set.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "kaggle"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import runner


@pytest.fixture
def fake_e02_experiment() -> dict:
    return {
        "_name": "phase4-e02-levircc-change-description",
        "remote_output_dir": "phase4-e02-levircc-change-description",
        "result_files": [
            "validation_metrics.json",
            "validation_predictions.jsonl",
            "runner_meta.json",
        ],
        "failure_result_files": [
            "evaluation_failure.json",
            "runner_meta.json",
        ],
    }


def _make_e02_valid_payload(sample_count: int = 2) -> tuple[dict, list[dict], dict]:
    metrics = {
        "experiment": "P4-E02",
        "task": "bitemporal_change_description",
        "sample_count": sample_count,
        "status": "PASS",
        "caption_metrics": {"bleu1": 0.1747},
    }
    predictions = [
        {
            "pair_id": f"pair_{i}",
            "generated_caption": f"Change detected {i}",
            "reference_captions": [f"Ground truth {i}"],
            "evaluation_split": "val",
        }
        for i in range(sample_count)
    ]
    meta = {
        "experiment": "phase4-e02-levircc-change-description",
        "task": "bitemporal_change_description",
        "git_sha": "c73861537c2e2a724378e0d9d3ab7d9844216712",
        "reproducible": True,
        "sample_count": sample_count,
        "status": "PASS",
    }
    return metrics, predictions, meta


class TestRunnerArtifactSelectionRegression:
    def test_1_cloned_repo_historical_metrics_cannot_be_selected_for_e02(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """When Kaggle output contains cloned SATQuery repo and correct remote output, only remote output is selected."""
        remote_out = fake_e02_experiment["remote_output_dir"]
        dest_dir = tmp_path / "exp"
        results_dest = dest_dir / "results"

        # Setup mock dl_dir
        dl_dir = tmp_path / "dl_tree"

        # Historical repo file with old/invalid metrics
        repo_bad = dl_dir / "SATQuery" / "experiments" / "phase4_bitemporal_vqa" / "results" / "invalid_initial"
        repo_bad.mkdir(parents=True)
        (repo_bad / "validation_metrics.json").write_text(
            json.dumps({"status": "INVALID_PLACEHOLDER_RESULT", "sample_count": 4}), encoding="utf-8"
        )
        (repo_bad / "runner_meta.json").write_text(
            json.dumps({"experiment": "phase4-e04-sar-validation"}), encoding="utf-8"
        )

        # Real experiment output
        real_out = dl_dir / "satquery-output" / remote_out
        real_out.mkdir(parents=True)
        metrics, preds, meta = _make_e02_valid_payload(sample_count=2)
        (real_out / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (real_out / "validation_predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
        )
        (real_out / "runner_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        def fake_run(cmd, **_kwargs):
            dest = Path(cmd[cmd.index("-p") + 1])
            # Mirror entire dl_dir into CLI temp dir
            shutil.copytree(dl_dir, dest, dirs_exist_ok=True)
            return mock.MagicMock(returncode=0)

        with mock.patch.object(runner, "_run", side_effect=fake_run):
            runner._download_artifacts(
                username="testuser",
                kernel_slug="test-slug",
                experiment=fake_e02_experiment,
                output_dir=dest_dir,
                allow_dirty=False,
            )

        published_metrics = json.loads((results_dest / "validation_metrics.json").read_text(encoding="utf-8"))
        assert published_metrics["status"] == "PASS"
        assert published_metrics["sample_count"] == 2
        published_meta = json.loads((results_dest / "runner_meta.json").read_text(encoding="utf-8"))
        assert published_meta["experiment"] == "phase4-e02-levircc-change-description"

    def test_2_invalid_initial_files_never_selected_when_remote_missing(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """If remote output is missing, runner must fail closed rather than falling back to invalid_initial."""
        dest_dir = tmp_path / "exp"
        dl_dir = tmp_path / "dl_tree"

        # Only invalid_initial files exist in cloned SATQuery repo
        repo_bad = dl_dir / "SATQuery" / "experiments" / "phase4_bitemporal_vqa" / "results" / "invalid_initial"
        repo_bad.mkdir(parents=True)
        (repo_bad / "validation_metrics.json").write_text(
            json.dumps({"status": "INVALID_PLACEHOLDER_RESULT"}), encoding="utf-8"
        )
        (repo_bad / "validation_predictions.jsonl").write_text("{}\n", encoding="utf-8")
        (repo_bad / "runner_meta.json").write_text(json.dumps({"experiment": "P4-E02"}), encoding="utf-8")

        def fake_run(cmd, **_kwargs):
            dest = Path(cmd[cmd.index("-p") + 1])
            shutil.copytree(dl_dir, dest, dirs_exist_ok=True)
            return mock.MagicMock(returncode=0)

        with mock.patch.object(runner, "_run", side_effect=fake_run):
            with pytest.raises(SystemExit):
                runner._download_artifacts(
                    username="testuser",
                    kernel_slug="test-slug",
                    experiment=fake_e02_experiment,
                    output_dir=dest_dir,
                    allow_dirty=False,
                )

        assert not (dest_dir / "results" / "validation_metrics.json").exists()

    def test_3_basename_collision_across_two_directories_causes_failure(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """If multiple matches exist within the qualified remote path hierarchy, raise ARTIFACT_AMBIGUOUS."""
        remote_out = fake_e02_experiment["remote_output_dir"]
        dest_dir = tmp_path / "exp"
        dl_dir = tmp_path / "dl_tree"

        # Create two subdirectories under remote_output_dir with the same basename
        dir1 = dl_dir / "satquery-output" / remote_out / "sub1"
        dir2 = dl_dir / "satquery-output" / remote_out / "sub2"
        dir1.mkdir(parents=True)
        dir2.mkdir(parents=True)
        (dir1 / "validation_metrics.json").write_text("{}", encoding="utf-8")
        (dir2 / "validation_metrics.json").write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="ARTIFACT_AMBIGUOUS"):
            runner._resolve_artifact_path(dl_dir, remote_out, "validation_metrics.json")

    def test_4_path_qualified_result_succeeds(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """Correct path satquery-output/<remote_output_dir>/<file> succeeds."""
        remote_out = fake_e02_experiment["remote_output_dir"]
        dl_dir = tmp_path / "dl_tree"
        target_dir = dl_dir / "satquery-output" / remote_out
        target_dir.mkdir(parents=True)
        (target_dir / "validation_metrics.json").write_text('{"status": "PASS"}', encoding="utf-8")

        resolved = runner._resolve_artifact_path(dl_dir, remote_out, "validation_metrics.json")
        assert resolved is not None
        assert resolved == target_dir / "validation_metrics.json"

    def test_5_e04_runner_meta_is_rejected_for_e02(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """If an E04 runner_meta.json is presented for E02 download, validation rejects it."""
        staged = tmp_path / "staged"
        staged.mkdir()
        metrics, preds, _ = _make_e02_valid_payload(sample_count=2)
        (staged / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (staged / "validation_predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
        )
        # E04 metadata
        (staged / "runner_meta.json").write_text(
            json.dumps({"experiment": "phase4-e04-sar-validation"}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="Identity mismatch.*E04 metadata"):
            runner._validate_downloaded_artifacts(
                staged,
                experiment=fake_e02_experiment,
                allow_dirty=False,
            )

    def test_6_wrong_git_sha_is_rejected(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """When expected_git_sha is provided, a mismatched git_sha in runner_meta.json is rejected."""
        staged = tmp_path / "staged"
        staged.mkdir()
        metrics, preds, meta = _make_e02_valid_payload(sample_count=2)
        meta["git_sha"] = "wrong_sha_1234567890abcdef"
        (staged / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (staged / "validation_predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
        )
        (staged / "runner_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="Git SHA mismatch"):
            runner._validate_downloaded_artifacts(
                staged,
                experiment=fake_e02_experiment,
                allow_dirty=False,
                expected_git_sha="c73861537c2e2a724378e0d9d3ab7d9844216712",
            )

    def test_7_prediction_row_count_must_equal_metrics_sample_count(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """Prediction row count != sample_count must fail validation."""
        staged = tmp_path / "staged"
        staged.mkdir()
        metrics, preds, meta = _make_e02_valid_payload(sample_count=5)
        # Metrics says 5, but predictions has only 2 rows
        metrics["sample_count"] = 5
        preds = preds[:2]
        (staged / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (staged / "validation_predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
        )
        (staged / "runner_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match prediction rows"):
            runner._validate_downloaded_artifacts(
                staged,
                experiment=fake_e02_experiment,
                allow_dirty=False,
            )

    def test_8_artifact_publication_is_atomic(
        self, tmp_path: Path, fake_e02_experiment: dict
    ) -> None:
        """Validation failure in staging leaves results_dest completely untouched."""
        remote_out = fake_e02_experiment["remote_output_dir"]
        dest_dir = tmp_path / "exp"
        results_dest = dest_dir / "results"
        results_dest.mkdir(parents=True)

        # Place a sentinel file in results_dest to verify nothing is modified/published
        (results_dest / "pre_existing_sentinel.txt").write_text("untouched", encoding="utf-8")

        dl_dir = tmp_path / "dl_tree"
        real_out = dl_dir / "satquery-output" / remote_out
        real_out.mkdir(parents=True)

        # Prepare invalid payload: sample_count mismatch
        metrics, preds, meta = _make_e02_valid_payload(sample_count=5)
        preds = preds[:1]  # 1 row vs sample_count 5
        (real_out / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (real_out / "validation_predictions.jsonl").write_text(
            "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
        )
        (real_out / "runner_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        def fake_run(cmd, **_kwargs):
            dest = Path(cmd[cmd.index("-p") + 1])
            shutil.copytree(dl_dir, dest, dirs_exist_ok=True)
            return mock.MagicMock(returncode=0)

        with mock.patch.object(runner, "_run", side_effect=fake_run):
            with pytest.raises(ValueError, match="does not match prediction rows"):
                runner._download_artifacts(
                    username="testuser",
                    kernel_slug="test-slug",
                    experiment=fake_e02_experiment,
                    output_dir=dest_dir,
                    allow_dirty=False,
                )

        # Ensure results_dest was NOT populated with partial download files
        assert not (results_dest / "validation_metrics.json").exists()
        assert not (results_dest / "validation_predictions.jsonl").exists()
        assert not (results_dest / "runner_meta.json").exists()
        # Sentinel remains untouched
        assert (results_dest / "pre_existing_sentinel.txt").read_text(encoding="utf-8") == "untouched"
