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
    """Run a subprocess, streaming output unless capture=True."""
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        cwd=str(cwd or REPO_ROOT),
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

def _write_kernel_metadata(
    dest_dir: Path,
    username: str,
    kernel_slug: str,
    notebook_name: str,
    gpu: bool,
    internet: bool,
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
        "enable_internet": internet,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (dest_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2)
    )


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

_STATUS_TERMINAL = {"complete", "error", "cancelAcknowledged", "cancelled"}
_STATUS_RUNNING  = {"running", "queued", "starting"}


def _poll_status(
    username: str,
    kernel_slug: str,
    poll_interval: int,
) -> str:
    """
    Poll `kaggle kernels status` until the kernel reaches a terminal state.
    Returns the final status string.
    """
    kernel_id = f"{username}/{kernel_slug}"
    print(f"\n⏳  Polling status for {kernel_id} every {poll_interval}s …\n")

    while True:
        result = _run(
            ["kaggle", "kernels", "status", kernel_id],
            capture=True,
            check=False,
        )
        output = result.stdout.strip()

        # kaggle kernels status outputs a table; the status is the last token
        # on the data line, e.g.:
        #   ref                            totalVotes  status
        #   username/kernel-slug           0           running
        lines = [l for l in output.splitlines() if kernel_slug in l.lower()]
        status = "unknown"
        if lines:
            status = lines[-1].split()[-1].lower()

        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] status: {status}")

        if status in _STATUS_TERMINAL:
            print(f"\n{'✅' if status == 'complete' else '❌'}  Kernel finished with status: {status}\n")
            return status

        if status not in _STATUS_RUNNING and status != "unknown":
            print(f"  ⚠️  Unexpected status {status!r} — continuing to poll.")

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------

def _download_artifacts(
    username: str,
    kernel_slug: str,
    experiment: dict[str, Any],
    output_dir: Path,
    allow_dirty: bool,
) -> None:
    """
    Download kernel output using `kaggle kernels output` into a temp directory,
    then selectively copy result_files into the local experiment results dir.
    """
    kernel_id = f"{username}/{kernel_slug}"
    result_files: list[str] = experiment.get("result_files", [])
    remote_output_dir: str  = experiment["remote_output_dir"]

    if not result_files:
        print("⚠️  No result_files configured for this experiment — skipping download.")
        return

    with tempfile.TemporaryDirectory(prefix="satquery-kaggle-dl-") as tmp:
        tmp_path = Path(tmp)
        print(f"⬇️   Downloading kernel output for {kernel_id} …")
        _run(["kaggle", "kernels", "output", kernel_id, "-p", str(tmp_path)])

        # The output zip is extracted by the CLI into tmp_path.
        # The notebook writes artifacts under:
        #   /kaggle/working/satquery-output/<remote_output_dir>/
        # After download the mirror layout is typically:
        #   <tmp>/satquery-output/<remote_output_dir>/<file>
        # but `kaggle kernels output` may also flatten the directory. We search.
        found: dict[str, Path] = {}
        for rf in result_files:
            # Try direct path match first
            candidate = tmp_path / "satquery-output" / remote_output_dir / rf
            if candidate.exists():
                found[rf] = candidate
                continue
            # Fallback: recursive search by filename
            fname = Path(rf).name
            hits = list(tmp_path.rglob(fname))
            if hits:
                found[rf] = hits[0]
            else:
                print(f"  ⚠️  {rf} not found in downloaded output — skipping.")

        if not found:
            print("❌  No result files found in download. Check the experiment's remote_output_dir.")
            return

        # Write to experiment results dir
        results_dest = output_dir / "results"
        results_dest.mkdir(parents=True, exist_ok=True)

        copied = []
        for rf, src in found.items():
            dest = results_dest / Path(rf).name
            shutil.copy2(src, dest)
            copied.append(dest)

        # Annotate dirty runs
        if allow_dirty:
            dirty_flag = results_dest / ".dirty_worktree"
            dirty_flag.write_text(
                "Results from a dirty working tree run — not a reproducible artifact.\n"
            )

        print(f"\n📦  Copied {len(copied)} artifact(s) → {results_dest.relative_to(REPO_ROOT)}/")
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
    print(result.stdout or result.stderr)


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
        "list":   cmd_list,
        "status": cmd_status,
        "run":    cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
