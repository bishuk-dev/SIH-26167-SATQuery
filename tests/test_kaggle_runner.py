"""
tests/test_kaggle_runner.py
===========================
Unit tests for the Kaggle experiment runner.
No network calls, no CLI invocations, no Kaggle credentials required.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Make the scripts directory importable without installing
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "kaggle"
sys.path.insert(0, str(SCRIPTS_DIR))

import runner  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Test notebook\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, subprocess, sys\n",
                "from pathlib import Path\n",
                "REPO_URL = os.environ.get('SATQUERY_REPO_URL', 'https://github.com/example/repo.git')\n",
                "REPO_DIR = Path('/kaggle/working/repo')\n",
                "if (REPO_DIR / '.git').is_dir():\n",
                "    subprocess.run(['git', 'pull', '--ff-only'], cwd=REPO_DIR, check=True)\n",
                "else:\n",
                "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)], check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["print('hello')\n"],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


@pytest.fixture()
def source_nb(tmp_path: Path) -> Path:
    """Write a minimal notebook and return its path."""
    nb_file = tmp_path / "kaggle_test.ipynb"
    nb_file.write_text(json.dumps(MINIMAL_NOTEBOOK, indent=1))
    return nb_file


@pytest.fixture()
def dest_dir(tmp_path: Path) -> Path:
    d = tmp_path / "push"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Tests — notebook patching
# ---------------------------------------------------------------------------

class TestPatchNotebook:
    def test_injected_cell_is_first(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc123def456",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        first = nb["cells"][0]
        assert first["cell_type"] == "code"
        src = "".join(first["source"])
        assert "injected-by-runner" in str(first.get("metadata", {}).get("tags", []))
        assert "abc123def456" in src
        assert "SATQUERY_GIT_REF" in src
        assert "SATQUERY_REPO_URL" in src

    def test_original_cells_preserved(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc123",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        assert len(nb["cells"]) == len(MINIMAL_NOTEBOOK["cells"]) + 1

    def test_clone_cell_gets_checkout_code(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="deadbeef",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        clone_cell = nb["cells"][2]  # injected(0) + markdown(1) + clone(2)
        src = "".join(clone_cell["source"])
        assert "'checkout'" in src or '"checkout"' in src, (
            f"Expected git checkout call in clone cell source, got: {src!r}"
        )
        assert "SATQUERY_GIT_REF" in src

    def test_dirty_worktree_flag_in_meta(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=True,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        first_src = "".join(nb["cells"][0]["source"])
        assert "_RUNNER_REPRODUCIBLE = False" in first_src
        assert "'dirty_worktree': True" in first_src

    def test_clean_run_is_reproducible(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        first_src = "".join(nb["cells"][0]["source"])
        assert "_RUNNER_REPRODUCIBLE = True" in first_src

    def test_patched_file_is_valid_json(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        nb = json.loads(patched.read_text())
        assert nb["nbformat"] == 4

    def test_patched_filename_matches_source(self, source_nb: Path, dest_dir: Path) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="test-exp",
            remote_output_dir="test-output",
        )
        assert patched.name == source_nb.name
        assert patched.parent == dest_dir

    def test_experiment_name_and_output_in_injected_cell(
        self, source_nb: Path, dest_dir: Path
    ) -> None:
        patched = runner._patch_notebook(
            source_nb=source_nb,
            dest_dir=dest_dir,
            git_sha="abc",
            repo_url="https://github.com/example/repo.git",
            allow_dirty=False,
            experiment_name="phase3a-grounding-baseline",
            remote_output_dir="phase3a-grounding-dino",
        )
        nb = json.loads(patched.read_text())
        first_src = "".join(nb["cells"][0]["source"])
        assert "phase3a-grounding-baseline" in first_src
        assert "phase3a-grounding-dino" in first_src


# ---------------------------------------------------------------------------
# Tests — kernel metadata
# ---------------------------------------------------------------------------

class TestWriteKernelMetadata:
    def test_metadata_structure(self, dest_dir: Path) -> None:
        runner._write_kernel_metadata(
            dest_dir=dest_dir,
            username="testuser",
            kernel_slug="satquery-test",
            notebook_name="test.ipynb",
            gpu=True,
            internet=True,
        )
        meta = json.loads((dest_dir / "kernel-metadata.json").read_text())
        assert meta["id"] == "testuser/satquery-test"
        assert meta["code_file"] == "test.ipynb"
        assert meta["enable_gpu"] is True
        assert meta["enable_internet"] is True
        assert meta["kernel_type"] == "notebook"
        assert meta["language"] == "python"

    def test_metadata_gpu_false(self, dest_dir: Path) -> None:
        runner._write_kernel_metadata(
            dest_dir=dest_dir,
            username="testuser",
            kernel_slug="satquery-cpu",
            notebook_name="test.ipynb",
            gpu=False,
            internet=False,
        )
        meta = json.loads((dest_dir / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is False
        assert meta["enable_internet"] is False

    def test_metadata_resolves_kernel_source_slugs_for_same_user(
        self, dest_dir: Path
    ) -> None:
        runner._write_kernel_metadata(
            dest_dir=dest_dir,
            username="testuser",
            kernel_slug="satquery-s2",
            notebook_name="test.ipynb",
            gpu=False,
            internet=True,
            kernel_sources=["satquery-s1", "other/source"],
        )
        meta = json.loads((dest_dir / "kernel-metadata.json").read_text())
        assert meta["kernel_sources"] == ["testuser/satquery-s1", "other/source"]


# ---------------------------------------------------------------------------
# Tests — experiment registry
# ---------------------------------------------------------------------------

class TestExperimentRegistry:
    def test_load_registry_returns_dict(self) -> None:
        registry = runner._load_registry()
        assert isinstance(registry, dict)

    @pytest.mark.parametrize("modality", ["s1", "s2"])
    def test_phase4e_unimodal_baselines_use_kaggle_materialization_outputs(
        self, modality: str
    ) -> None:
        registry = runner._load_registry()
        entry = registry[f"phase4e-bifold-{modality}-validation"]

        assert entry["notebook"] == f"notebooks/kaggle_phase4e_bifold_{modality}.ipynb"
        assert entry["kernel_sources"] == [
            "satquery-phase4-materialize-s1",
            "satquery-phase4-materialize-s2",
        ]
        assert entry["result_files"] == [
            "validation_result.json",
            "validation_predictions.jsonl",
            "runner_meta.json",
        ]
        assert entry["gpu"] is True
        assert entry["internet"] is True

    @pytest.mark.parametrize("modality", ["s1", "s2"])
    def test_phase4e_notebook_installs_pinned_configilm_on_python_312(
        self, modality: str
    ) -> None:
        notebook_path = (
            Path(__file__).parent.parent
            / "notebooks"
            / f"kaggle_phase4e_bifold_{modality}.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join(
            line
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )

        assert "'--no-deps', '--ignore-requires-python', 'configilm==0.7.0'" in source
        assert "'timm==0.9.16'" in source
        assert "'lmdb==1.6.2'" in source

    def test_phase4d_native_audit_uses_exact_private_kernel_outputs(self) -> None:
        registry = runner._load_registry()
        entry = registry["phase4d-native-raster-audit"]

        assert entry["notebook"] == "notebooks/kaggle_phase4d_native_raster_audit.ipynb"
        assert entry["kernel_sources"] == [
            "technobishu/satquery-phase4-materialize-s1",
            "technobishu/satquery-phase4-materialize-s2",
        ]
        assert entry["result_files"] == [
            "representative_raster_audit.json",
            "native_audit_runner_meta.json",
        ]
        assert entry["gpu"] is False
        assert entry["internet"] is True
        assert len(registry) > 0

    def test_known_experiments_have_required_fields(self) -> None:
        registry = runner._load_registry()
        required = {"notebook", "kernel_slug", "experiment_dir", "remote_output_dir", "result_files"}
        for name, entry in registry.items():
            missing = required - entry.keys()
            assert not missing, f"Experiment {name!r} missing fields: {missing}"

    def test_result_files_are_lists(self) -> None:
        registry = runner._load_registry()
        for name, entry in registry.items():
            assert isinstance(entry["result_files"], list), (
                f"Experiment {name!r}: result_files must be a list"
            )

    def test_get_unknown_experiment_raises(self) -> None:
        with pytest.raises(SystemExit):
            runner._get_experiment("this-experiment-does-not-exist-xyz")

    def test_get_known_experiment(self) -> None:
        registry = runner._load_registry()
        if not registry:
            pytest.skip("No experiments registered")
        name = next(iter(registry))
        entry = runner._get_experiment(name)
        assert entry["_name"] == name

    def test_phase4_modalities_are_independent_cpu_experiments(self) -> None:
        registry = runner._load_registry()
        for modality in ("s1", "s2"):
            entry = registry[f"phase4-materialize-{modality}"]
            assert entry["gpu"] is False
            assert entry["internet"] is True
            assert entry["large_result_files"] == [
                f"phase4_{modality}_selected.tar.zst"
            ]
            assert entry["download_policy"] == "metadata_only"
            assert not set(entry["large_result_files"]) & set(entry["result_files"])
        assert registry["phase4-materialize-s2"]["kernel_sources"] == [
            "satquery-phase4-materialize-s1"
        ]


# ---------------------------------------------------------------------------
# Tests — status normalizer  (covers exact Kaggle CLI 2.2.4 observed values)
# ---------------------------------------------------------------------------

class TestNormalizeStatus:
    # --- Kaggle CLI 2.2.4 observed enum-style values ---
    def test_enum_running(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.running") == "running"

    def test_enum_complete(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.complete") == "complete"

    def test_enum_error(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.error") == "error"

    def test_enum_queued(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.queued") == "queued"

    def test_enum_cancelled(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.cancelled") == "cancelled"

    def test_enum_cancelacknowledged(self) -> None:
        assert runner._normalize_status("kernelworkerstatus.cancelAcknowledged") == "cancelacknowledged"

    # --- Backward-compat: older bare statuses still work ---
    def test_bare_running(self) -> None:
        assert runner._normalize_status("running") == "running"

    def test_bare_complete(self) -> None:
        assert runner._normalize_status("complete") == "complete"

    def test_bare_error(self) -> None:
        assert runner._normalize_status("error") == "error"

    def test_bare_queued(self) -> None:
        assert runner._normalize_status("queued") == "queued"

    # --- Whitespace and quote stripping ---
    def test_leading_trailing_whitespace(self) -> None:
        assert runner._normalize_status("  running  ") == "running"

    def test_double_quoted(self) -> None:
        assert runner._normalize_status('"kernelworkerstatus.complete"') == "complete"

    def test_single_quoted(self) -> None:
        assert runner._normalize_status("'kernelworkerstatus.running'") == "running"

    def test_whitespace_and_quotes(self) -> None:
        assert runner._normalize_status('  "kernelworkerstatus.error"  ') == "error"

    # --- Terminal-set membership after normalisation ---
    def test_enum_complete_is_terminal(self) -> None:
        status = runner._normalize_status("kernelworkerstatus.complete")
        assert status in runner._STATUS_TERMINAL

    def test_enum_error_is_terminal(self) -> None:
        status = runner._normalize_status("kernelworkerstatus.error")
        assert status in runner._STATUS_TERMINAL

    def test_enum_running_is_not_terminal(self) -> None:
        status = runner._normalize_status("kernelworkerstatus.running")
        assert status not in runner._STATUS_TERMINAL
        assert status in runner._STATUS_RUNNING

    def test_enum_queued_is_not_terminal(self) -> None:
        status = runner._normalize_status("kernelworkerstatus.queued")
        assert status not in runner._STATUS_TERMINAL
        assert status in runner._STATUS_RUNNING


# ---------------------------------------------------------------------------
# Tests — _build_file_pattern
# ---------------------------------------------------------------------------

class TestBuildFilePattern:
    def test_single_file(self) -> None:
        pattern = runner._build_file_pattern(["calibration.json"])
        assert re.match(pattern, "satquery-output/phase3b/calibration.json")
        assert not re.match(pattern, "satquery-output/phase3b/old_calibration.json")

    def test_two_files(self) -> None:
        pattern = runner._build_file_pattern(["calibration.json", "validation_candidates.jsonl"])
        assert re.match(pattern, "satquery-output/phase3b/calibration.json")
        assert re.match(pattern, "satquery-output/phase3b/validation_candidates.jsonl")
        assert not re.match(pattern, "satquery-output/phase3b/other.txt")

    def test_phase3b_exact_result_list(self) -> None:
        """Verify the pattern built from Phase 3B's actual configured result_files."""
        registry = runner._load_registry()
        entry = registry.get("phase3b-grounding-thresholds")
        if entry is None:
            pytest.skip("phase3b-grounding-thresholds not in registry")
        result_files = entry["result_files"]
        remote_out = entry["remote_output_dir"]
        assert result_files == ["calibration.json", "validation_candidates.jsonl"], (
            f"Unexpected phase3b result_files: {result_files}"
        )
        pattern = runner._build_file_pattern(result_files)
        for rf in result_files:
            remote_path = f"satquery-output/{remote_out}/{rf}"
            assert re.match(pattern, remote_path), (
                f"Pattern {pattern!r} did not match {remote_path!r}"
            )

    def test_phase3_final_test_exact_result_list(self) -> None:
        entry = runner._load_registry()["phase3-final-grounding-test"]
        assert entry["result_files"] == [
            "final_test_metrics.json",
            "final_test_predictions.jsonl",
        ]
        assert entry["gpu"] is True
        assert entry["internet"] is True

    def test_special_chars_escaped(self) -> None:
        """Dots in filenames must be treated as literals, not regex wildcards."""
        pattern = runner._build_file_pattern(["metrics.json"])
        # 'metricsXjson' has a char in place of '.'; must NOT match
        assert not re.match(pattern, "some/path/metricsXjson")
        # The real name must match
        assert re.match(pattern, "some/path/metrics.json")

    def test_pattern_is_valid_regex(self) -> None:
        """Constructed pattern must compile without error."""
        files = ["calibration.json", "validation_candidates.jsonl", "run.json"]
        pattern = runner._build_file_pattern(files)
        compiled = re.compile(pattern)
        assert compiled is not None

    def test_pattern_ends_with_dollar(self) -> None:
        """Pattern must end with $ to avoid partial-suffix matches."""
        pattern = runner._build_file_pattern(["metrics.json"])
        assert pattern.endswith("$"), f"Pattern should end with $, got: {pattern!r}"

    def test_subdirectory_path(self) -> None:
        """Files in nested directories should also match."""
        pattern = runner._build_file_pattern(["metrics/train_metrics.json"])
        assert re.match(pattern, "satquery-output/phase2b/metrics/train_metrics.json")


# ---------------------------------------------------------------------------
# Tests — _download_artifacts  (filesystem only, no network calls)
# ---------------------------------------------------------------------------

class TestDownloadArtifacts:
    """
    Exercises the file-copy logic inside _download_artifacts by pre-populating
    a fake download directory that mirrors what `kaggle kernels output` produces.
    """

    @staticmethod
    def _make_fake_download(base: Path, remote_output_dir: str, files: list[str]) -> Path:
        """Populate a dir as if kaggle kernels output extracted into it."""
        out_base = base / "satquery-output" / remote_output_dir
        out_base.mkdir(parents=True)
        for f in files:
            dest = out_base / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f"fake content for {f}")
        return base

    def _fake_run_factory(self, dl_dir: Path, remote_output_dir: str):
        """Return a _run replacement that mirrors dl_dir into the CLI dest arg."""
        def fake_run(cmd, **_kwargs):
            dest = Path(cmd[cmd.index("-p") + 1])
            src = dl_dir / "satquery-output" / remote_output_dir
            dest_sub = dest / "satquery-output" / remote_output_dir
            dest_sub.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.copy2(f, dest_sub / f.name)
            return mock.MagicMock(returncode=0)
        return fake_run

    def test_copies_configured_result_files(self, tmp_path: Path) -> None:
        remote_out = "phase3a-grounding-dino"
        dl_dir = self._make_fake_download(
            tmp_path / "dl", remote_out,
            ["validation_metrics.json", "validation_predictions.jsonl"],
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        experiment = {
            "remote_output_dir": remote_out,
            "result_files": ["validation_metrics.json", "validation_predictions.jsonl"],
        }
        with mock.patch.object(runner, "_run", side_effect=self._fake_run_factory(dl_dir, remote_out)):
            runner._download_artifacts("u", "s", experiment, out_dir, allow_dirty=False)

        results = out_dir / "results"
        assert (results / "validation_metrics.json").exists()
        assert (results / "validation_predictions.jsonl").exists()
        assert not (results / ".dirty_worktree").exists()

    def test_file_pattern_passed_to_cli(self, tmp_path: Path) -> None:
        """Verify --file-pattern is included in the kaggle kernels output call."""
        remote_out = "phase3b-grounding-calibration"
        dl_dir = self._make_fake_download(
            tmp_path / "dl", remote_out,
            ["calibration.json", "validation_candidates.jsonl"],
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        experiment = {
            "remote_output_dir": remote_out,
            "result_files": ["calibration.json", "validation_candidates.jsonl"],
        }
        captured_cmds: list[list[str]] = []

        def capturing_run(cmd, **_kwargs):
            captured_cmds.append(cmd[:])
            # Also copy files so the rest of _download_artifacts succeeds
            dest = Path(cmd[cmd.index("-p") + 1])
            src = dl_dir / "satquery-output" / remote_out
            dest_sub = dest / "satquery-output" / remote_out
            dest_sub.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.copy2(f, dest_sub / f.name)
            return mock.MagicMock(returncode=0)

        with mock.patch.object(runner, "_run", side_effect=capturing_run):
            runner._download_artifacts("u", "s", experiment, out_dir, allow_dirty=False)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "--file-pattern" in cmd
        pattern_idx = cmd.index("--file-pattern") + 1
        pattern = cmd[pattern_idx]
        # Pattern must match the configured basenames
        assert re.match(pattern, "satquery-output/phase3b/calibration.json")
        assert re.match(pattern, "satquery-output/phase3b/validation_candidates.jsonl")
        # Must not match unrelated files
        assert not re.match(pattern, "satquery-output/phase3b/README.md")
        assert not re.match(pattern, "SIH-26167-SATQuery/satquery/main.py")

    def test_missing_result_file_raises_system_exit(self, tmp_path: Path) -> None:
        """Any configured file absent from the download must cause SystemExit."""
        remote_out = "test-output"
        # Provide only one of the two configured files
        dl_dir = self._make_fake_download(tmp_path / "dl", remote_out, ["metrics.json"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        experiment = {
            "remote_output_dir": remote_out,
            "result_files": ["metrics.json", "missing_file.json"],
        }
        with mock.patch.object(runner, "_run", side_effect=self._fake_run_factory(dl_dir, remote_out)):
            with pytest.raises(SystemExit):
                runner._download_artifacts("u", "s", experiment, out_dir, allow_dirty=False)

    def test_dirty_flag_written_when_allow_dirty(self, tmp_path: Path) -> None:
        remote_out = "test-output"
        dl_dir = self._make_fake_download(tmp_path / "dl", remote_out, ["metrics.json"])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        experiment = {
            "remote_output_dir": remote_out,
            "result_files": ["metrics.json"],
        }
        with mock.patch.object(runner, "_run", side_effect=self._fake_run_factory(dl_dir, remote_out)):
            runner._download_artifacts("u", "s", experiment, out_dir, allow_dirty=True)

        assert (out_dir / "results" / ".dirty_worktree").exists()

    def test_metadata_only_policy_never_requests_large_packages(
        self, tmp_path: Path
    ) -> None:
        remote_out = "phase4-bigearthnet-materialize-s1"
        small_files = ["materialization_report.json", "package_manifest.json"]
        dl_dir = self._make_fake_download(tmp_path / "dl", remote_out, small_files)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        experiment = {
            "remote_output_dir": remote_out,
            "result_files": small_files,
            "large_result_files": ["phase4_s1_selected.tar.zst"],
            "download_policy": "metadata_only",
        }
        captured: list[list[str]] = []

        def capturing_run(cmd, **kwargs):
            captured.append(cmd[:])
            return self._fake_run_factory(dl_dir, remote_out)(cmd, **kwargs)

        with mock.patch.object(runner, "_run", side_effect=capturing_run):
            runner._download_artifacts("u", "s", experiment, out_dir, allow_dirty=False)

        pattern = captured[0][captured[0].index("--file-pattern") + 1]
        assert re.match(pattern, f"satquery-output/{remote_out}/package_manifest.json")
        assert not re.match(
            pattern, f"satquery-output/{remote_out}/phase4_s1_selected.tar.zst"
        )

    def test_metadata_only_policy_rejects_large_package_in_result_files(
        self, tmp_path: Path
    ) -> None:
        experiment = {
            "remote_output_dir": "phase4",
            "result_files": ["phase4_s1_selected.tar.zst"],
            "large_result_files": ["phase4_s1_selected.tar.zst"],
            "download_policy": "metadata_only",
        }

        with pytest.raises(ValueError, match="large_result_files"):
            runner._download_artifacts(
                "u", "s", experiment, tmp_path / "out", allow_dirty=False
            )
