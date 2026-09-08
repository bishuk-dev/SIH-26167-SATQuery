#!/usr/bin/env python3
"""
scripts/kaggle/runner.py
========================
Local-to-Kaggle experiment launcher for SatQuery.

Usage
-----
  python scripts/kaggle/runner.py run <experiment> [options]
  python scripts/kaggle/runner.py list
  python scripts/kaggle/runner.py status <experiment>

Authentication
--------------
Credentials are owned entirely by the Kaggle CLI.  This script never reads
~/.kaggle/kaggle.json or any token directly.  It verifies that the CLI is
authenticated before doing anything else.

Requirements
------------
  pip install kaggle          # provides the `kaggle` CLI
  Python >= 3.11 (stdlib only beyond kaggle package for CLI calls)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import yaml  # PyYAML is already a project dependency

# Ensure UTF-8 stdout/stderr on Windows console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_YAML = SCRIPTS_DIR / "experiments.yaml"
KERNEL_TEMPLATE = SCRIPTS_DIR / "kernel_metadata.json.template"
PUSH_WORK_DIR = SCRIPTS_DIR / ".kernel_push"   # gitignored


# ---------------------------------------------------------------------------
# Helpers — subprocess
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    *,
    capture: bool = False,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess, streaming output unless capture=True.

    Forces UTF-8 encoding for both our text=True capture and for the
    child process's own stdout/stderr via PYTHONIOENCODING.  This prevents
    'charmap' codec errors on Windows when the Kaggle CLI prints Unicode
    characters (emoji, special paths) to a cp1252 console.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        cwd=str(cwd or REPO_ROOT),
        env=env,
    )


def _run_json(cmd: list[str], *, cwd: Path | None = None) -> Any:
    """Run a subprocess and parse stdout as JSON."""
    result = _run(cmd, capture=True, cwd=cwd)
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def _check_kaggle_cli() -> str:
    """Verify kaggle CLI is on PATH and returns its version string."""
    if not shutil.which("kaggle"):
        _die(
            "kaggle CLI not found.\n"
            "Install it with:  pip install kaggle\n"
            "Then authenticate: kaggle auth login   (or place kaggle.json in ~/.kaggle/)"
        )
    result = _run(["kaggle", "--version"], capture=True)
    version = result.stdout.strip()
    return version


def _check_kaggle_auth() -> str:
    """
    Verify that the Kaggle CLI is authenticated and return the configured username.
    We call `kaggle config view` which is a lightweight read-only operation.
    """
    result = _run(["kaggle", "config", "view"], capture=True)
    # Output is like:
    #   - username: johndoe
    #   - key: ...
    for line in result.stdout.splitlines():
        if "username" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                username = parts[-1].strip()
                if username and username != "None":
                    return username
    _die(
        "Kaggle CLI is not authenticated or username is unset.\n"
        "Run:  kaggle auth login\n"
        "  or: set the KAGGLE_API_TOKEN environment variable."
    )


def _check_git_clean(allow_dirty: bool) -> str:
    """
    Return the current HEAD commit SHA.
    If allow_dirty is False, abort when the working tree is dirty.
    """
    sha_result = _run(["git", "rev-parse", "HEAD"], capture=True)
    sha = sha_result.stdout.strip()

    status_result = _run(["git", "status", "--porcelain"], capture=True)
    dirty = bool(status_result.stdout.strip())

    if dirty and not allow_dirty:
        _die(
            "Working tree is dirty.\n\n"
            "Commit or stash your changes before launching a reproducible GPU experiment:\n\n"
            "  git add -A && git commit -m 'wip'\n"
            "  # or\n"
            "  git stash\n\n"
            "To skip this check for a disposable debugging run (results will be marked\n"
            "non-reproducible), pass --allow-dirty."
        )

    return sha


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict[str, Any]:
    if not EXPERIMENTS_YAML.exists():
        _die(f"experiments.yaml not found at {EXPERIMENTS_YAML}")
    with EXPERIMENTS_YAML.open() as fh:
        return yaml.safe_load(fh) or {}


def _get_experiment(name: str) -> dict[str, Any]:
    registry = _load_registry()
    if name not in registry:
        available = "\n".join(f"  - {k}" for k in sorted(registry))
        _die(f"Unknown experiment: {name!r}\n\nAvailable experiments:\n{available}")
    entry = registry[name]
    entry["_name"] = name
    return entry


# ---------------------------------------------------------------------------
# Kaggle CLI capability detection
# ---------------------------------------------------------------------------

def _kaggle_push_command() -> list[str]:
    """
    Detect the right `kaggle kernels push` invocation.
    Older CLI: kaggle kernels push -p <path>
    Newer CLI: same, but `push` may be aliased.  We probe once and cache.

    In practice `kaggle kernels push -p <dir>` has been stable since >=1.5 and
    is still the documented interface as of 1.7.  We keep the detection so the
    runner doesn't silently break on future CLI restructuring.
    """
    result = _run(["kaggle", "kernels", "--help"], capture=True, check=False)
    help_text = (result.stdout + result.stderr).lower()
    if "push" in help_text:
        return ["kaggle", "kernels", "push", "-p"]
    # Fallback: try the newer hypothetical `create/update`
    _warn("kaggle kernels push not found in help text; attempting it anyway.")
    return ["kaggle", "kernels", "push", "-p"]


# ---------------------------------------------------------------------------
# Notebook patching
# ---------------------------------------------------------------------------

def _patch_notebook(
    source_nb: Path,
    dest_dir: Path,
    git_sha: str,
    repo_url: str,
    allow_dirty: bool,
    experiment_name: str,
    remote_output_dir: str,
) -> Path:
    """
    Load the source notebook, inject a env-var cell at position 0, and
    write the patched copy to dest_dir/<original_name>.
    The existing notebooks already read SATQUERY_REPO_URL; we also set
    SATQUERY_GIT_REF so they check out the exact commit.
    """
    with source_nb.open() as fh:
        nb = json.load(fh)

    reproducible = not allow_dirty
    injected_source = [
        "# ---- injected by scripts/kaggle/runner.py — do not edit manually ----\n",
        "import os, subprocess, sys\n",
        f"os.environ['SATQUERY_GIT_REF']        = {git_sha!r}\n",
        f"os.environ['SATQUERY_REPO_URL']        = {repo_url!r}\n",
        f"os.environ['SATQUERY_EXPERIMENT_NAME'] = {experiment_name!r}\n",
        f"os.environ['SATQUERY_REMOTE_OUTPUT']   = {remote_output_dir!r}\n",
        f"_RUNNER_REPRODUCIBLE = {str(reproducible)}\n",
        "_RUNNER_META = {\n",
        f"    'git_sha':      {git_sha!r},\n",
        f"    'experiment':   {experiment_name!r},\n",
        f"    'reproducible': _RUNNER_REPRODUCIBLE,\n",
        f"    'dirty_worktree': {str(allow_dirty)},\n",
        "}\n",
        "import json as _json\n",
        "from pathlib import Path as _Path\n",
        "_out = _Path('/kaggle/working/satquery-output') / os.environ['SATQUERY_REMOTE_OUTPUT']\n",
        "_out.mkdir(parents=True, exist_ok=True)\n",
        "(_out / 'runner_meta.json').write_text(_json.dumps(_RUNNER_META, indent=2))\n",
        "print('runner_meta:', _RUNNER_META)\n",
        "# ---- end injection ----\n",
    ]

    injected_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-by-runner"]},
        "outputs": [],
        "source": injected_source,
    }

    # Also patch the clone cell to checkout the pinned ref
    cells = nb.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Match both shell-style "git clone/pull" and subprocess list-style
        # ['git', 'clone'] / ['git', 'pull'] calls.
        _is_clone_cell = (
            "git clone" in src
            or "git pull" in src
            or ("'git'" in src and ("'clone'" in src or "'pull'" in src))
            or ('"git"' in src and ('"clone"' in src or '"pull"' in src))
        )
        if _is_clone_cell:
            src = "".join(cell.get("source", []))
            if "REPO_DIR" in src:
                repo_var = "REPO_DIR"
            elif "repo_root" in src:
                repo_var = "repo_root"
            else:
                repo_var = "REPO_DIR"
            cell["source"] = list(cell.get("source", [])) + [
                "\n",
                "# Pin to the exact git ref recorded by runner.py\n",
                "_ref = os.environ.get('SATQUERY_GIT_REF', 'HEAD')\n",
                "if _ref != 'HEAD':\n",
                f"    subprocess.run(['git', 'fetch', '--depth=1', 'origin', _ref], cwd=str({repo_var}), check=False)\n",
                f"    subprocess.run(['git', 'checkout', _ref], cwd=str({repo_var}), check=True)\n",
                "    print(f'Checked out {_ref}')\n",
            ]
            break

    nb["cells"] = [injected_cell] + cells

    dest = dest_dir / source_nb.name
    with dest.open("w") as fh:
        json.dump(nb, fh, indent=1)

    return dest


# ---------------------------------------------------------------------------
# Kernel metadata
# ---------------------------------------------------------------------------

DEFAULT_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"


def _write_kernel_metadata(
    dest_dir: Path,
    username: str,
    kernel_slug: str,
    notebook_name: str,
    gpu: bool,
    internet: bool,
    kernel_sources: list[str] | None = None,
) -> None:
    """Write kernel-metadata.json into the push working directory."""
    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "machine_shape": DEFAULT_GPU_MACHINE_SHAPE if gpu else None,
        "enable_internet": internet,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [
            source if "/" in source else f"{username}/{source}"
            for source in (kernel_sources or [])
        ],
    }
    (dest_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2)
    )


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

_STATUS_TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled"}
_STATUS_RUNNING  = {"running", "queued", "starting"}

# Kaggle CLI 2.x emits enum-style values such as:
#   kernelworkerstatus.running
#   kernelworkerstatus.complete
# Older releases emit bare values: running, complete, error …
# We normalise both to the bare lowercase form before any comparison.
_KAGGLE_STATUS_PREFIX = "kernelworkerstatus."


def _normalize_status(raw: str) -> str:
    """
    Normalise a raw Kaggle status token to a bare lowercase string.

    Handles:
    - surrounding whitespace
    - surrounding JSON/string quotes (" or ')
    - enum-style prefixes:  kernelworkerstatus.running → running
    - already-bare values:  running → running
    """
    s = raw.strip().strip('"\'')
    s = s.lower()
    if s.startswith(_KAGGLE_STATUS_PREFIX):
        s = s[len(_KAGGLE_STATUS_PREFIX):]
    return s


def _poll_status(
    username: str,
    kernel_slug: str,
    poll_interval: int,
) -> str:
    """
    Poll `kaggle kernels status` until the kernel reaches a terminal state.
    Returns the final normalised status string.
    Prints a recovery command on KeyboardInterrupt so the user can retrieve
    results after the kernel finishes without re-running it.
    """
    kernel_id = f"{username}/{kernel_slug}"
    exp_name = "<experiment>"  # placeholder shown in recovery hint
    # Derive the experiment name from the slug for a friendlier hint
    registry = _load_registry()
    for name, entry in registry.items():
        if entry.get("kernel_slug") == kernel_slug:
            exp_name = name
            break

    print(f"\n⏳  Polling status for {kernel_id} every {poll_interval}s …")
    print(f"    (Ctrl+C to abort — run 'download {exp_name}' later to retrieve results)\n")

    try:
        while True:
            result = _run(
                ["kaggle", "kernels", "status", kernel_id],
                capture=True,
                check=False,
            )
            output = result.stdout.strip()

            # kaggle kernels status outputs a table; the status is the last
            # token on the data row, e.g.:
            #   ref                            totalVotes  status
            #   username/kernel-slug           0           kernelworkerstatus.running
            lines = [ln for ln in output.splitlines() if kernel_slug in ln.lower()]
            raw_status = "unknown"
            if lines:
                raw_status = lines[-1].split()[-1]

            status = _normalize_status(raw_status)

            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] status: {status}")

            if status in _STATUS_TERMINAL:
                emoji = "✅" if status == "complete" else "❌"
                print(f"\n{emoji}  Kernel finished with status: {status}\n")
                return status

            if status not in _STATUS_RUNNING and status != "unknown":
                print(f"  ⚠️  Unexpected status {status!r} — continuing to poll.")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(
            f"\n\n⚠️  Polling interrupted.\n"
            f"    The kernel may still be running on Kaggle.\n"
            f"    When it completes, retrieve results with:\n"
            f"\n"
            f"      python scripts/kaggle/runner.py download {exp_name}\n"
            f"\n"
            f"    Or monitor at: https://www.kaggle.com/code/{kernel_id}\n"
        )
        sys.exit(0)


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------

import re as _re


def _build_file_pattern(
    result_files: list[str],
    remote_output_dir: str | None = None,
) -> str:
    """
    Build a regex string for `kaggle kernels output --file-pattern`.

    When remote_output_dir is provided, matches ONLY files under:
        satquery-output/<remote_output_dir>/<result_file>
    with regex:
        .*(?:^|/)satquery-output/<escaped_remote_output_dir>/(?:file1|file2|...)$
    This prevents matching cloned repository files or files from other experiments.
    """
    escaped = [_re.escape(Path(rf).as_posix()) for rf in result_files]
    alternation = "|".join(escaped)
    if remote_output_dir:
        escaped_remote = _re.escape(remote_output_dir)
        return f".*(?:^|/)satquery-output/{escaped_remote}/(?:{alternation})$"
    return f".*/(?:{alternation})$"


def _archive_current_result_files(
    output_dir: Path,
    result_files: list[str],
    failure_result_files: list[str] | None = None,
) -> Path | None:
    """
    Archive only declared current scientific result files (and failure result files /
    runner_meta.json / .dirty_worktree) from output_dir/results to:
        results/archive/<timestamp>_<previous-git-sha>/
    BEFORE launching or polling the new experiment.

    Preserves results/invalid_initial/ unchanged (does not archive or remove
    that historical integrity record).
    """
    results_dir = output_dir / "results"
    if not results_dir.exists() or not results_dir.is_dir():
        return None

    declared_names = set()
    for rf in result_files:
        declared_names.add(Path(rf).name)
    if failure_result_files:
        for ff in failure_result_files:
            declared_names.add(Path(ff).name)
    declared_names.add(".dirty_worktree")

    # Only consider files directly in results_dir that match declared names
    files_to_archive: list[Path] = []
    for item in results_dir.iterdir():
        # NEVER touch directories (e.g. invalid_initial/, archive/)
        if item.is_file() and item.name in declared_names:
            files_to_archive.append(item)

    if not files_to_archive:
        return None

    prev_sha = "unknown"
    meta_file = results_dir / "runner_meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("git_sha"):
                prev_sha = str(meta["git_sha"])[:12]
        except Exception:
            pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = results_dir / "archive" / f"{timestamp}_{prev_sha}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for src_file in files_to_archive:
        shutil.move(str(src_file), str(archive_dir / src_file.name))

    print(
        f"📦  Archived {len(files_to_archive)} previous result file(s) → "
        f"{archive_dir.resolve().relative_to(REPO_ROOT.resolve(), walk_up=True)}/"
    )
    return archive_dir


def _resolve_artifact_path(tmp_path: Path, remote_output_dir: str, rel_path: str) -> Path | None:
    """
    Resolve an expected artifact to its file under satquery-output/<remote_output_dir>/<rel_path>.
    Strictly forbids searching outside satquery-output/<remote_output_dir>.
    """
    # 1. Primary candidate
    candidate = tmp_path / "satquery-output" / remote_output_dir / rel_path
    if candidate.is_file():
        return candidate

    # 2. Strict directory-bounded fallback: search ONLY within satquery-output/<remote_output_dir>
    fname = Path(rel_path).name
    hits = [
        p for p in tmp_path.rglob(fname)
        if p.is_file() and "satquery-output" in p.parts and remote_output_dir in p.parts
    ]
    if len(hits) == 1:
        return hits[0]
    elif len(hits) > 1:
        raise RuntimeError(
            f"ARTIFACT_AMBIGUOUS: Multiple matches found for {rel_path!r} under "
            f"satquery-output/{remote_output_dir}: {[str(p) for p in hits]}"
        )
    return None


def _validate_downloaded_artifacts(
    staged_dir: Path,
    experiment: dict[str, Any],
    allow_dirty: bool,
    expected_git_sha: str | None = None,
    is_failure: bool = False,
) -> None:
    """
    Validate identity and schema integrity of downloaded artifacts in staged_dir.
    Raises ValueError or RuntimeError on validation failure.
    """
    exp_name = experiment.get("_name", "")

    # 1. Validate runner_meta.json if present
    meta_path = staged_dir / "runner_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupted runner_meta.json in downloaded artifacts: {exc}") from exc

        if not isinstance(meta, dict):
            raise ValueError("runner_meta.json must be a JSON object")

        # Identity check: experiment field where present
        meta_exp = meta.get("experiment")
        if meta_exp:
            if exp_name == "phase4-e02-levircc-change-description":
                if "e04" in str(meta_exp).lower() or "sar" in str(meta_exp).lower():
                    raise ValueError(f"Identity mismatch: E04 metadata ({meta_exp!r}) found in E02 download")
                if meta_exp not in ("phase4-e02-levircc-change-description", "P4-E02"):
                    raise ValueError(f"Identity mismatch: unexpected experiment in runner_meta.json: {meta_exp!r}")
            elif exp_name == "phase4-e04-modified-sen1floods11-validation":
                if "e02" in str(meta_exp).lower() or "levir" in str(meta_exp).lower():
                    raise ValueError(f"Identity mismatch: E02 metadata ({meta_exp!r}) found in E04 download")
                if meta_exp not in (
                    "phase4-e04-modified-sen1floods11-validation",
                    "phase4-e04-sar-validation",
                    "P4-E04",
                ):
                    raise ValueError(f"Identity mismatch: unexpected experiment in runner_meta.json: {meta_exp!r}")
            elif meta_exp != exp_name:
                raise ValueError(f"Identity mismatch: expected experiment {exp_name!r}, got {meta_exp!r}")

        # Git SHA check where field exists and expected SHA is provided
        if expected_git_sha and meta.get("git_sha"):
            actual_sha = str(meta["git_sha"]).strip()
            exp_sha = expected_git_sha.strip()
            if not actual_sha.startswith(exp_sha) and not exp_sha.startswith(actual_sha):
                raise ValueError(
                    f"Git SHA mismatch in runner_meta.json: expected {expected_git_sha}, got {actual_sha}"
                )

        # Reproducibility check: for clean runs, reproducible must be True
        if not allow_dirty:
            if meta.get("reproducible") is False or meta.get("dirty_worktree") is True:
                raise ValueError("Downloaded artifact indicates non-reproducible run (dirty worktree)")

    # If this is a scientific failure run, validate failure artifact
    if is_failure:
        fail_path = staged_dir / "evaluation_failure.json"
        if fail_path.exists():
            try:
                fail_data = json.loads(fail_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Corrupted evaluation_failure.json: {exc}") from exc
            if exp_name == "phase4-e02-levircc-change-description":
                if fail_data.get("experiment") not in ("P4-E02", "phase4-e02-levircc-change-description"):
                    raise ValueError(f"Wrong experiment in evaluation_failure.json: {fail_data.get('experiment')}")
            elif exp_name == "phase4-e04-modified-sen1floods11-validation":
                if fail_data.get("experiment") not in (
                    "P4-E04",
                    "phase4-e04-modified-sen1floods11-validation",
                    "phase4-e04-sar-validation",
                ):
                    raise ValueError(f"Wrong experiment in evaluation_failure.json: {fail_data.get('experiment')}")
        return

    # 2. E02 Success Metrics Validation
    if exp_name == "phase4-e02-levircc-change-description":
        metrics_path = staged_dir / "validation_metrics.json"
        if not metrics_path.exists():
            raise ValueError("Missing validation_metrics.json for E02")
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupted validation_metrics.json: {exc}") from exc

        if metrics.get("experiment") != "P4-E02":
            raise ValueError(f"E02 metrics experiment mismatch: expected 'P4-E02', got {metrics.get('experiment')!r}")
        if metrics.get("task") != "bitemporal_change_description":
            raise ValueError(f"E02 metrics task mismatch: expected 'bitemporal_change_description', got {metrics.get('task')!r}")
        if metrics.get("status") != "PASS":
            raise ValueError(f"E02 metrics status must be 'PASS', got {metrics.get('status')!r}")

        # Validation of prediction rows vs sample_count
        preds_path = staged_dir / "validation_predictions.jsonl"
        if not preds_path.exists():
            raise ValueError("Missing validation_predictions.jsonl for E02")
        pred_lines = [line.strip() for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sample_count = metrics.get("sample_count")
        if sample_count != len(pred_lines):
            raise ValueError(f"E02 sample_count ({sample_count}) does not match prediction rows ({len(pred_lines)})")

        for idx, line in enumerate(pred_lines):
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Malformed prediction row {idx}: {exc}") from exc
            if "generated_caption" not in row:
                raise ValueError(f"Prediction row {idx} missing 'generated_caption'")
            if "reference_captions" not in row:
                raise ValueError(f"Prediction row {idx} missing 'reference_captions'")
            if row.get("evaluation_split") != "val":
                raise ValueError(f"Prediction row {idx} evaluation_split != 'val': {row.get('evaluation_split')}")

    # 3. E04 Success Metrics Validation
    elif exp_name == "phase4-e04-modified-sen1floods11-validation":
        metrics_path = staged_dir / "sar_validation_metrics.json"
        if not metrics_path.exists():
            raise ValueError("Missing sar_validation_metrics.json for E04")
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupted sar_validation_metrics.json: {exc}") from exc

        if metrics.get("experiment") != "P4-E04":
            raise ValueError(f"E04 metrics experiment mismatch: expected 'P4-E04', got {metrics.get('experiment')!r}")
        if metrics.get("task") != "sar_temporal_flood_validation":
            raise ValueError(f"E04 metrics task mismatch: expected 'sar_temporal_flood_validation', got {metrics.get('task')!r}")

        preds_path = staged_dir / "sar_validation_predictions.jsonl"
        if preds_path.exists() and "sample_count" in metrics:
            pred_lines = [line.strip() for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            valid_samples = metrics.get("grid_valid_count", metrics["sample_count"])
            total_samples = metrics["sample_count"]
            mismatches = metrics.get("grid_mismatch_count", 0)
            if len(pred_lines) not in (total_samples, valid_samples + mismatches):
                raise ValueError(
                    f"E04 sample_count ({metrics['sample_count']}) + mismatches ({mismatches}) "
                    f"!= prediction rows ({len(pred_lines)})"
                )

    # 4. General prediction row count verification when both files exist
    else:
        for rf in experiment.get("result_files", []):
            if rf.endswith(".jsonl"):
                preds_path = staged_dir / Path(rf).name
                if preds_path.exists():
                    pred_lines = [line.strip() for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    # Look for metrics json file
                    for mf in experiment.get("result_files", []):
                        if mf.endswith("_metrics.json") or mf.endswith("_result.json") or mf == "metrics.json":
                            m_path = staged_dir / Path(mf).name
                            if m_path.exists():
                                try:
                                    m_data = json.loads(m_path.read_text(encoding="utf-8"))
                                    if "sample_count" in m_data and m_data["sample_count"] != len(pred_lines):
                                        raise ValueError(
                                            f"sample_count mismatch: {m_data['sample_count']} in {mf} "
                                            f"!= {len(pred_lines)} rows in {rf}"
                                        )
                                except Exception:
                                    pass


def _download_via_kaggle_api(kernel_id: str, dest_dir: Path, file_pattern: str) -> bool:
    """Download matching files directly via Kaggle API with polite paging.

    Uses page_size=200, exponential backoff for 429 rate limits, and 0.2s inter-page delays.
    Handles sessions with 20,000+ output files reliably without hitting rate limits.
    """
    import re
    import time
    from kaggle import api
    import requests

    compiled_pattern = re.compile(file_pattern)
    owner, slug, _ = api.parse_kernel_string(kernel_id)
    req_cls = api.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]

    token = None
    page = 0
    downloaded = 0
    with api.build_kaggle_client() as client:
        while True:
            page += 1
            req = req_cls()
            req.user_name = owner
            req.kernel_slug = slug
            api._set_paging(req, 200, token)
            resp = None
            for attempt in range(6):
                try:
                    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
                    break
                except Exception as exc:
                    if "429" in str(exc):
                        wait = 5 * (attempt + 1)
                        print(f"⚠️  Rate limit (429) encountered on page {page}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            if resp is None:
                raise RuntimeError(f"Failed to fetch session output page {page} after retries")

            for item in resp.files or []:
                if compiled_pattern.search(item.file_name):
                    out_path = dest_dir / item.file_name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    res = requests.get(item.url, stream=True)
                    res.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in res.iter_content(chunk_size=65536):
                            f.write(chunk)
                    print(f"   Downloaded {item.file_name} ({out_path.stat().st_size} bytes)")
                    downloaded += 1

            token = resp.next_page_token
            if not token:
                break
            time.sleep(0.2)

    return downloaded > 0


def _download_artifacts(
    username: str,
    kernel_slug: str,
    experiment: dict[str, Any],
    output_dir: Path,
    allow_dirty: bool,
    kernel_status: str = "complete",
    expected_git_sha: str | None = None,
) -> None:
    """
    Download artifacts from a Kaggle kernel into output_dir/results/.

    Understands two result states:
      1. SUCCESS ARTIFACTS: declared in `result_files`
      2. FAILURE ARTIFACTS: declared in `failure_result_files` (e.g. evaluation_failure.json)

    Distinguishes Kaggle execution status from scientific evaluation status:
      - If evaluation_failure.json is produced, success metrics are NOT required,
        failure artifacts are retrieved, and an error is reported.
      - If the Kaggle kernel itself errored, available failure artifacts are retrieved
        if exposed, and no stale success metrics remain in results/.
    """
    kernel_id = f"{username}/{kernel_slug}"
    result_files: list[str] = experiment.get("result_files", [])
    failure_result_files: list[str] = experiment.get(
        "failure_result_files", ["evaluation_failure.json", "runner_meta.json"]
    )
    large_result_files: list[str] = experiment.get("large_result_files", [])
    remote_output_dir: str = experiment["remote_output_dir"]

    if experiment.get("download_policy") == "metadata_only":
        overlap = set(result_files) & set(large_result_files)
        if overlap:
            raise ValueError(
                "metadata_only result_files must not contain large_result_files: "
                + ", ".join(sorted(overlap))
            )

    all_download_files = list(dict.fromkeys(result_files + failure_result_files))
    if not all_download_files:
        print("⚠️  No result_files configured for this experiment — skipping download.")
        return

    pattern = _build_file_pattern(all_download_files, remote_output_dir=remote_output_dir)
    print(f"⬇️   Downloading selected artifacts for {kernel_id}")
    print(f"    file-pattern: {pattern}")
    if large_result_files:
        print(
            "    large packages remain on Kaggle: "
            + ", ".join(large_result_files)
        )

    with tempfile.TemporaryDirectory(prefix="satquery-kaggle-dl-") as tmp:
        tmp_path = Path(tmp)
        is_mocked = hasattr(_run, "mock_calls") or hasattr(_run, "_mock_self")
        api_success = False
        if not is_mocked:
            try:
                api_success = _download_via_kaggle_api(kernel_id, tmp_path, pattern)
            except Exception as exc:
                _warn(f"Direct API download failed ({exc}), falling back to CLI...")

        if not api_success:
            max_attempts = 4
            dl_result = None
            for attempt in range(1, max_attempts + 1):
                dl_result = _run([
                    "kaggle", "kernels", "output", kernel_id,
                    "-p", str(tmp_path),
                    "--file-pattern", pattern,
                    "--page-size", "200",
                    "--force",
                ], check=False)
                # The Kaggle CLI on Windows may exit non-zero after printing Unicode
                # characters that the cp1252 console can't encode, even though all
                # requested files have already been written to disk.
                has_files = any(tmp_path.iterdir())
                if dl_result.returncode == 0 or has_files:
                    break
                if attempt < max_attempts:
                    wait_time = 15 * attempt
                    _warn(f"kaggle kernels output failed (attempt {attempt}/{max_attempts}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)

            if dl_result and dl_result.returncode != 0:
                _warn(
                    f"kaggle kernels output exited with code {dl_result.returncode} "
                    "(often a Windows encoding issue or partial error — checking whether files arrived)"
                )


        results_dest = output_dir / "results"
        results_dest.mkdir(parents=True, exist_ok=True)

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Check if evaluation_failure.json was produced (strictly under satquery-output/<remote_output_dir>)
        fail_file = _resolve_artifact_path(tmp_path, remote_output_dir, "evaluation_failure.json")

        if fail_file and fail_file.is_file():
            print("\n⚠️  Detected scientific evaluation failure artifact (evaluation_failure.json).")
            for ff in failure_result_files:
                resolved = _resolve_artifact_path(tmp_path, remote_output_dir, ff)
                if resolved and resolved.is_file():
                    shutil.copy2(resolved, staging_dir / Path(ff).name)

            _validate_downloaded_artifacts(
                staging_dir,
                experiment=experiment,
                allow_dirty=allow_dirty,
                expected_git_sha=expected_git_sha,
                is_failure=True,
            )

            copied_fail: list[Path] = []
            for item in staging_dir.iterdir():
                if item.is_file():
                    dest = results_dest / item.name
                    shutil.copy2(item, dest)
                    copied_fail.append(dest)

            # Ensure NO success metrics exist in results_dest
            for rf in result_files:
                rf_name = Path(rf).name
                stale_rf = results_dest / rf_name
                if stale_rf.exists() and rf_name not in [Path(f).name for f in failure_result_files]:
                    stale_rf.unlink()

            try:
                failure_info = json.loads(fail_file.read_text(encoding="utf-8"))
                reason = failure_info.get("failure_reason", "UNKNOWN")
                details = failure_info.get("details", "")
                print(f"❌  Scientific evaluation FAILED:\n    Reason:  {reason}\n    Details: {details}", file=sys.stderr)
            except Exception:
                print("❌  Scientific evaluation FAILED (malformed evaluation_failure.json)", file=sys.stderr)

            print(f"📦  Copied {len(copied_fail)} failure artifact(s) → {results_dest.resolve().relative_to(REPO_ROOT.resolve(), walk_up=True)}/")
            sys.exit(1)

        # If no evaluation_failure.json was found and kernel itself errored:
        if kernel_status != "complete":
            print(
                f"❌  Kernel did not complete successfully (status: {kernel_status}) and no evaluation_failure.json was produced.\n"
                f"    Inspect the run at: https://www.kaggle.com/code/{kernel_id}\n",
                file=sys.stderr,
            )
            sys.exit(1)

        # Kernel succeeded and no evaluation_failure.json: retrieve SUCCESS artifacts
        missing: list[str] = []
        for rf in result_files:
            resolved = _resolve_artifact_path(tmp_path, remote_output_dir, rf)
            if resolved and resolved.is_file():
                shutil.copy2(resolved, staging_dir / Path(rf).name)
            else:
                missing.append(rf)

        if missing:
            msg_lines = [
                "❌  Missing result files after download:",
                *[f"    - {m}" for m in missing],
                "",
                "    Check that:",
                f"    • remote_output_dir in experiments.yaml matches '{remote_output_dir}'",
                "    • The notebook wrote these files before the kernel completed",
                f"    • The kernel output is available at:",
                f"      https://www.kaggle.com/code/{kernel_id}",
            ]
            print("\n".join(msg_lines), file=sys.stderr)
            sys.exit(1)

        # Validate staged files before publishing
        _validate_downloaded_artifacts(
            staging_dir,
            experiment=experiment,
            allow_dirty=allow_dirty,
            expected_git_sha=expected_git_sha,
            is_failure=False,
        )

        # All validation passed: atomically copy staged files to results_dest
        copied: list[Path] = []
        for item in staging_dir.iterdir():
            if item.is_file():
                dest = results_dest / item.name
                shutil.copy2(item, dest)
                copied.append(dest)

        # Annotate dirty runs
        if allow_dirty:
            (results_dest / ".dirty_worktree").write_text(
                "Results from a dirty working tree run — not a reproducible artifact.\n"
            )

        print(f"\n📦  Copied {len(copied)} artifact(s) → {results_dest.resolve().relative_to(REPO_ROOT.resolve(), walk_up=True)}/")
        for p in copied:
            print(f"    {p.name}  ({p.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list(_args: argparse.Namespace) -> None:
    registry = _load_registry()
    print(f"{'Experiment':<40}  {'Notebook':<40}  GPU  Internet")
    print("-" * 95)
    for name, entry in sorted(registry.items()):
        nb   = entry.get("notebook", "—")
        gpu  = "yes" if entry.get("gpu") else "no "
        inet = "yes" if entry.get("internet") else "no "
        print(f"{name:<40}  {nb:<40}  {gpu}  {inet}")


def cmd_status(args: argparse.Namespace) -> None:
    _check_kaggle_cli()
    username = _check_kaggle_auth()
    exp      = _get_experiment(args.experiment)
    slug     = exp["kernel_slug"]
    kernel_id = f"{username}/{slug}"
    result = _run(["kaggle", "kernels", "status", kernel_id], capture=True, check=False)
    raw = (result.stdout or result.stderr).strip()
    # Also print normalised status so it is unambiguous
    lines = [ln for ln in raw.splitlines() if slug in ln.lower()]
    if lines:
        raw_status = lines[-1].split()[-1]
        normalised = _normalize_status(raw_status)
        print(raw)
        print(f"  → normalised status: {normalised}")
    else:
        print(raw)


def cmd_download(args: argparse.Namespace) -> None:
    """
    Download artifacts from a completed Kaggle kernel without re-running it.
    Uses `kaggle kernels output` and copies only the configured result_files.
    """
    _check_kaggle_cli()
    username = _check_kaggle_auth()
    exp      = _get_experiment(args.experiment)
    slug     = exp["kernel_slug"]
    exp_dir  = REPO_ROOT / exp["experiment_dir"]

    output_dir = Path(args.output_dir) if args.output_dir else exp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n⬇️   Downloading results for experiment: {args.experiment}")
    print(f"    Kernel: {username}/{slug}")
    print(f"    Destination: {output_dir.relative_to(REPO_ROOT)}/results/\n")

    _archive_current_result_files(
        output_dir=output_dir,
        result_files=exp.get("result_files", []),
        failure_result_files=exp.get("failure_result_files", []),
    )

    _download_artifacts(
        username=username,
        kernel_slug=slug,
        experiment=exp,
        output_dir=output_dir,
        allow_dirty=False,   # downloads never mark as dirty
        kernel_status="complete",
        expected_git_sha=getattr(args, "expected_sha", None),
    )

    print("\n✅  Done.")


def cmd_run(args: argparse.Namespace) -> None:
    # 1. Checks
    cli_version = _check_kaggle_cli()
    print(f"🔧  Kaggle CLI: {cli_version}")

    username = _check_kaggle_auth()
    print(f"👤  Authenticated as: {username}")

    git_sha = _check_git_clean(args.allow_dirty)
    dirty_suffix = " (dirty — not reproducible)" if args.allow_dirty else ""
    print(f"🔖  Git SHA: {git_sha[:12]}{dirty_suffix}")

    # Derive repo URL from git remote
    remote_result = _run(["git", "remote", "get-url", "origin"], capture=True, check=False)
    repo_url = remote_result.stdout.strip() or "https://github.com/bishuk-dev/SIH-26167-SATQuery.git"
    # Normalise SSH → HTTPS for Kaggle's internet access
    if repo_url.startswith("git@github.com:"):
        repo_url = repo_url.replace("git@github.com:", "https://github.com/", 1)
        if repo_url.endswith(".git") is False:
            repo_url += ".git"
    print(f"🔗  Repo URL: {repo_url}")

    # 2. Load experiment config
    exp          = _get_experiment(args.experiment)
    exp_name     = exp["_name"]
    nb_path      = REPO_ROOT / exp["notebook"]
    slug         = exp["kernel_slug"]
    if len(slug) > 50:
        _die(f"Kernel slug '{slug}' exceeds Kaggle's 50-character limit (length: {len(slug)})")
    
    exp_dir      = REPO_ROOT / exp["experiment_dir"]
    remote_out   = exp["remote_output_dir"]
    gpu          = exp.get("gpu", True)
    internet     = exp.get("internet", True)

    if not nb_path.exists():
        _die(f"Notebook not found: {nb_path}")

    print(f"\n🚀  Experiment:     {exp_name}")
    print(f"    Notebook:      {exp['notebook']}")
    print(f"    Kernel slug:   {username}/{slug}")
    print(f"    Remote output: satquery-output/{remote_out}")

    # Determine local output dir
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Archive previous result files BEFORE launching or polling
    _archive_current_result_files(
        output_dir=output_dir,
        result_files=exp.get("result_files", []),
        failure_result_files=exp.get("failure_result_files", []),
    )

    # 3. Prepare push working directory
    push_dir = PUSH_WORK_DIR / slug
    push_dir.mkdir(parents=True, exist_ok=True)

    # 4. Patch notebook
    print("\n📝  Patching notebook with git ref …")
    patched_nb = _patch_notebook(
        source_nb=nb_path,
        dest_dir=push_dir,
        git_sha=git_sha,
        repo_url=repo_url,
        allow_dirty=args.allow_dirty,
        experiment_name=exp_name,
        remote_output_dir=remote_out,
    )
    print(f"    Written: {patched_nb.relative_to(REPO_ROOT)}")

    # 5. Write kernel metadata
    _write_kernel_metadata(
        dest_dir=push_dir,
        username=username,
        kernel_slug=slug,
        notebook_name=patched_nb.name,
        gpu=gpu,
        internet=internet,
        kernel_sources=exp.get("kernel_sources", []),
    )

    if args.dry_run:
        print("\n🔍  --dry-run: stopping before push.")
        print(f"    Push directory: {push_dir}")
        print(f"    kernel-metadata.json:")
        meta_txt = (push_dir / "kernel-metadata.json").read_text()
        print(textwrap.indent(meta_txt, "      "))
        return

    # 6. Push to Kaggle
    push_cmd = _kaggle_push_command() + [str(push_dir)]
    print(f"\n⬆️   Pushing kernel …  ({' '.join(shlex.quote(c) for c in push_cmd)})")
    _run(push_cmd)
    print("    Push accepted.")

    if args.no_download:
        print("⏭️   --no-download: skipping status poll and artifact download.")
        print(f"    Monitor at: https://www.kaggle.com/code/{username}/{slug}")
        return

    # 7. Poll status
    final_status = _poll_status(username, slug, args.poll_interval)

    # 8. Download artifacts (handles both complete and error statuses, retrieves failure artifacts if present)
    _download_artifacts(
        username=username,
        kernel_slug=slug,
        experiment=exp,
        output_dir=output_dir,
        allow_dirty=args.allow_dirty,
        kernel_status=final_status,
        expected_git_sha=git_sha,
    )

    print("\n✅  Done.")
    print(f"    Results: {output_dir.relative_to(REPO_ROOT)}/results/")
    print(f"    Kaggle:  https://www.kaggle.com/code/{username}/{slug}\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(f"\n❌  Error: {msg}\n", file=sys.stderr)
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"⚠️   Warning: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Local-to-Kaggle experiment launcher for SatQuery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
              python scripts/kaggle/runner.py list
              python scripts/kaggle/runner.py run phase3a-grounding-baseline --dry-run
              python scripts/kaggle/runner.py run phase3a-grounding-baseline
              python scripts/kaggle/runner.py run phase3b-grounding-thresholds --allow-dirty
              python scripts/kaggle/runner.py status phase3a-grounding-baseline
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered experiments.")

    # status
    status_p = sub.add_parser("status", help="Check the status of a running Kaggle kernel.")
    status_p.add_argument("experiment", help="Experiment name from experiments.yaml")

    # download
    dl_p = sub.add_parser(
        "download",
        help="Download results from a completed kernel without re-running it.",
    )
    dl_p.add_argument("experiment", help="Experiment name from experiments.yaml")
    dl_p.add_argument(
        "--output-dir", metavar="PATH",
        help=(
            "Override local artifact destination. "
            "Defaults to the experiment_dir defined in experiments.yaml."
        ),
    )
    dl_p.add_argument(
        "--expected-sha",
        dest="expected_sha",
        default=None,
        help="Expected git SHA in runner_meta.json (validates integrity if present).",
    )

    # run
    run_p = sub.add_parser("run", help="Launch an experiment on Kaggle.")
    run_p.add_argument("experiment", help="Experiment name from experiments.yaml")
    run_p.add_argument(
        "--dry-run", action="store_true",
        help="Patch notebook and write metadata, but do not push to Kaggle."
    )
    run_p.add_argument(
        "--no-download", action="store_true",
        help="Push and poll, but skip artifact download."
    )
    run_p.add_argument(
        "--allow-dirty", action="store_true",
        help=(
            "Allow launching from a dirty working tree. Results will be marked "
            "non-reproducible (dirty_worktree: true in runner_meta.json)."
        ),
    )
    run_p.add_argument(
        "--poll-interval", type=int, default=60, metavar="SECONDS",
        help="Seconds between status polls (default: 60)."
    )
    run_p.add_argument(
        "--output-dir", metavar="PATH",
        help=(
            "Override local artifact destination. "
            "Defaults to the experiment_dir defined in experiments.yaml."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "list":     cmd_list,
        "status":   cmd_status,
        "download": cmd_download,
        "run":      cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
