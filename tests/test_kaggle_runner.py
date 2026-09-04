"""
tests/test_kaggle_runner.py
===========================
Unit tests for the Kaggle experiment runner.
These tests cover notebook patching logic only — no network calls, no CLI
invocations, no Kaggle credentials required.
"""

from __future__ import annotations

import json
import sys
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
        # Original had 3 cells; patched adds 1 injected cell → 4 total
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
        # Cell at index 1 should be the markdown cell (unchanged)
        # Cell at index 2 should be the clone cell with appended checkout
        clone_cell = nb["cells"][2]  # injected(0) + markdown(1) + clone(2)
        src = "".join(clone_cell["source"])
        # The runner appends subprocess.run(['git', 'checkout', _ref], ...)
        # so we check for the actual generated fragments, not a shell string.
        assert "'checkout'" in src or "\"checkout\"" in src, (
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
        # Must not raise
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


# ---------------------------------------------------------------------------
# Tests — experiment registry
# ---------------------------------------------------------------------------

class TestExperimentRegistry:
    def test_load_registry_returns_dict(self) -> None:
        registry = runner._load_registry()
        assert isinstance(registry, dict)
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
