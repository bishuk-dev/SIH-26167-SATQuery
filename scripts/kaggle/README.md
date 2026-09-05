# Kaggle Experiment Runner

Automates the full local-to-Kaggle workflow: patch → push → poll → download.

## Prerequisites

1. **Python ≥ 3.11** — no extra packages beyond the project's existing `PyYAML` dependency.
2. **Kaggle CLI** — install once:
   ```bash
   pip install kaggle
   ```
3. **Authenticate** the CLI (pick one method):
   ```bash
   kaggle auth login             # interactive (recommended)
   # or
   export KAGGLE_API_TOKEN=<token>
   # or
   # place ~/.kaggle/kaggle.json with {"username":"…","key":"…"}
   ```
   The runner never reads credentials itself — it delegates entirely to the CLI.

4. **Kaggle Internet** — your kernel must have Internet enabled (runner sets this automatically via `kernel-metadata.json`).  The first run creates the kernel in your account automatically.

---

## Usage

```
python scripts/kaggle/runner.py <command> [options]
```

### List registered experiments

```bash
python scripts/kaggle/runner.py list
```

### Dry-run (patch notebook, write metadata, do not push)

```bash
python scripts/kaggle/runner.py run phase3a-grounding-baseline --dry-run
```

Useful to inspect the patched notebook and `kernel-metadata.json` before committing to a GPU session.

### Launch an experiment

```bash
python scripts/kaggle/runner.py run phase3a-grounding-baseline
```

Phase 4D uses two independent CPU notebooks. Run S1 first, verify its small
downloaded manifests, then run S2:

```bash
python scripts/kaggle/runner.py run phase4-materialize-s1
python scripts/kaggle/runner.py run phase4-materialize-s2
```

The S2 kernel automatically uses the S1 kernel output as an input and verifies
its package SHA-256 before starting the S2 transfer.

Each run keeps its multi-GiB `phase4_<modality>_selected.tar.zst` package in
Kaggle output. It downloads only `materialization_report.json`,
`package_manifest.json`, and `runner_meta.json`. The runner rejects registry
configurations that place a declared large package in `result_files` while
`download_policy: metadata_only` is active.

The one-time frozen Phase 3 grounding test uses:

```bash
python scripts/kaggle/runner.py run phase3-final-grounding-test
```

Run it only after the final grounding implementation is committed. Its evaluator
refuses to overwrite either final-test artifact.

The runner will:
1. Verify the Kaggle CLI is authenticated.
2. Refuse to proceed if the working tree is dirty (see below).
3. Get the current `HEAD` SHA.
4. Patch the notebook to clone/checkout that exact commit.
5. Push the kernel to Kaggle (creates it on first use, updates on subsequent runs).
6. Poll status every 60 seconds.
7. Download result artifacts into `experiments/<experiment_dir>/results/`.

### Check kernel status without blocking

```bash
python scripts/kaggle/runner.py status phase3a-grounding-baseline
```

### Skip artifact download after polling

```bash
python scripts/kaggle/runner.py run phase3a-grounding-baseline --no-download
```

### Custom poll interval

```bash
python scripts/kaggle/runner.py run phase2b-smolvlm-lora --poll-interval 120
```

---

## Reproducibility

By default the runner **refuses** to push from a dirty working tree:

```
❌  Error: Working tree is dirty.

Commit or stash your changes before launching a reproducible GPU experiment:

  git add -A && git commit -m 'wip'
  # or
  git stash
```

For a disposable debugging run only:

```bash
python scripts/kaggle/runner.py run phase3a-grounding-baseline --allow-dirty
```

Dirty-run results are marked in `runner_meta.json` inside the experiment output:
```json
{
  "git_sha": "abc123…",
  "experiment": "phase3a-grounding-baseline",
  "reproducible": false,
  "dirty_worktree": true
}
```
A `.dirty_worktree` sentinel file is also written next to the downloaded results so they are never accidentally treated as accepted benchmark artifacts.

---

## Adding a new experiment

1. Add a notebook under `notebooks/kaggle_<experiment>.ipynb`.  Structure it exactly like `kaggle_phase3a.ipynb`: env-vars are injected by the runner, no hardcoded git refs.
2. Add an entry to `scripts/kaggle/experiments.yaml`:

```yaml
my-new-experiment:
  notebook: notebooks/kaggle_my_experiment.ipynb
  kernel_slug: satquery-my-experiment
  experiment_dir: experiments/my_new_experiment
  remote_output_dir: my-new-experiment
  result_files:
    - metrics.json
    - predictions.jsonl
  gpu: true
  internet: true
```

For large outputs intended as downstream Kaggle inputs, declare them separately:

```yaml
  result_files:
    - materialization_report.json
    - package_manifest.json
  large_result_files:
    - phase4_s1_selected.tar.zst
  download_policy: metadata_only
```

3. Run:
```bash
python scripts/kaggle/runner.py run my-new-experiment --dry-run
```

---

## Files

| File | Purpose |
|------|---------|
| `runner.py` | CLI entry-point — push, poll, download |
| `experiments.yaml` | Registry of all named experiments |
| `kernel_metadata.json.template` | Template for Kaggle's push metadata |
| `.kernel_push/` | Gitignored scratch directory for push artifacts |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `kaggle CLI not found` | `pip install kaggle` |
| `username is unset` | `kaggle auth login` or set `KAGGLE_API_TOKEN` |
| Dirty working tree error | Commit/stash, or use `--allow-dirty` |
| Kernel stuck in `queued` | GPU quota exhausted — wait for a free slot |
| `result_files` not found | Check `remote_output_dir` matches the notebook's output path |
| SSH remote URL rejected | Runner auto-converts `git@github.com:` → `https://github.com/` |
