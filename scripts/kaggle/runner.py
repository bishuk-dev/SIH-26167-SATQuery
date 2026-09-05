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
            # Append a checkout step after clone/pull
            cell["source"] = list(cell.get("source", [])) + [
                "\n",
                "# Pin to the exact git ref recorded by runner.py\n",
                "_ref = os.environ.get('SATQUERY_GIT_REF', 'HEAD')\n",
                "if _ref != 'HEAD':\n",
                "    subprocess.run(['git', 'fetch', '--depth=1', 'origin', _ref], cwd=str(REPO_DIR), check=False)\n",
                "    subprocess.run(['git', 'checkout', _ref], cwd=str(REPO_DIR), check=True)\n",
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


def _build_file_pattern(result_files: list[str]) -> str:
    """
    Build a regex string for `kaggle kernels output --file-pattern` that
    matches ONLY the basenames listed in result_files.

    Kaggle matches the pattern against the full remote path of each output
    file.  We require the filename to appear immediately after a '/' (or at
    the start of the string) so that e.g. "calibration.json" does not
    accidentally match "old_calibration.json".

    The produced pattern is:
        .*/(?:calibration[.]json|validation_candidates[.]jsonl)$

    All special regex characters in the filenames are escaped before use.
    """
    escaped = [_re.escape(Path(rf).name) for rf in result_files]
    alternation = "|".join(escaped)
    return f".*/(?:{alternation})$"


def _download_artifacts(
    username: str,
    kernel_slug: str,
    experiment: dict[str, Any],
    output_dir: Path,
    allow_dirty: bool,
) -> None:
    """
    Download ONLY the configured result_files from a completed Kaggle kernel.

    Uses `kaggle kernels output --file-pattern <regex>` so only matching files
    are transferred.  The full /kaggle/working tree (cloned repo, .git packs,
    hundreds of MB of source) is never downloaded.

    Raises SystemExit if any configured result file is absent after download.
    """
    kernel_id = f"{username}/{kernel_slug}"
    result_files: list[str] = experiment.get("result_files", [])
    large_result_files: list[str] = experiment.get("large_result_files", [])
    remote_output_dir: str  = experiment["remote_output_dir"]

    if experiment.get("download_policy") == "metadata_only":
        overlap = set(result_files) & set(large_result_files)
        if overlap:
            raise ValueError(
                "metadata_only result_files must not contain large_result_files: "
                + ", ".join(sorted(overlap))
            )

    if not result_files:
        print("⚠️  No result_files configured for this experiment — skipping download.")
        return

    pattern = _build_file_pattern(result_files)
    print(f"⬇️   Downloading selected artifacts for {kernel_id}")
    print(f"    file-pattern: {pattern}")
    if large_result_files:
        print(
            "    large packages remain on Kaggle: "
            + ", ".join(large_result_files)
        )

    with tempfile.TemporaryDirectory(prefix="satquery-kaggle-dl-") as tmp:
        tmp_path = Path(tmp)
        dl_result = _run([
            "kaggle", "kernels", "output", kernel_id,
            "-p", str(tmp_path),
            "--file-pattern", pattern,
        ], check=False)
        # The Kaggle CLI on Windows may exit non-zero after printing Unicode
        # characters that the cp1252 console can't encode, even though all
        # requested files have already been written to disk.  We detect this
        # by checking file presence below and only fail if files are absent.
        if dl_result.returncode != 0:
            _warn(
                f"kaggle kernels output exited with code {dl_result.returncode} "
                "(often a Windows encoding issue — checking whether files arrived)"
            )

        # Kaggle CLI mirrors the remote path structure under tmp_path.
        # Notebook artifacts live at:
        #   /kaggle/working/satquery-output/<remote_output_dir>/<file>
        # Downloaded mirror:
        #   <tmp>/satquery-output/<remote_output_dir>/<file>
        # We also accept a flat layout in case the CLI version differs.
        results_dest = output_dir / "results"
        results_dest.mkdir(parents=True, exist_ok=True)

        missing: list[str] = []
        copied: list[Path] = []

        for rf in result_files:
            fname = Path(rf).name

            # 1. Expected structured path
            candidate = tmp_path / "satquery-output" / remote_output_dir / rf
            if candidate.exists():
                dest = results_dest / fname
                shutil.copy2(candidate, dest)
                copied.append(dest)
                continue

            # 2. Flat fallback — search whole download tree by basename
            hits = [p for p in tmp_path.rglob(fname) if p.is_file()]
            if hits:
                dest = results_dest / fname
                shutil.copy2(hits[0], dest)
                copied.append(dest)
                continue

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

        # Annotate dirty runs
        if allow_dirty:
            (results_dest / ".dirty_worktree").write_text(
                "Results from a dirty working tree run — not a reproducible artifact.\n"
            )

        print(f"\n📦  Copied {len(copied)} artifact(s) → {results_dest.relative_to(REPO_ROOT, walk_up=True)}/")
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

    _download_artifacts(
        username=username,
        kernel_slug=slug,
        experiment=exp,
        output_dir=output_dir,
        allow_dirty=False,   # downloads never mark as dirty
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

    if final_status != "complete":
        print(
            f"❌  Kernel did not complete successfully (status: {final_status}).\n"
            f"    Inspect the run at: https://www.kaggle.com/code/{username}/{slug}\n"
        )
        sys.exit(1)

    # 8. Download artifacts
    _download_artifacts(
        username=username,
        kernel_slug=slug,
        experiment=exp,
        output_dir=output_dir,
        allow_dirty=args.allow_dirty,
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
