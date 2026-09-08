# AGENTS.md

> Repository-wide instructions for AI coding agents working on SatQuery AI.
> Keep this file stable and task-general. Put volatile phase status, experiment details,
> and one-off implementation plans in the existing docs/experiment artifacts.

## Mission

SatQuery AI (SIH 26167) is an **evidence-grounded, sensor-aware remote-sensing analysis system with a natural-language interface**.

Core rule:

> **The language model may explain evidence; it must not create or manufacture evidence.**

The intended flow is:

```text
understand sensor → understand query → validate feasibility → choose specialist
→ process correctly → generate evidence → verify → explain
```

Numeric/spatial claims must come from reproducible model or GIS outputs, never visual guessing.

## Scope and Precedence

- This root file applies to the whole repository.
- Direct user/task instructions take precedence.
- A nested `AGENTS.md` may add more specific rules for its subtree; the nearest applicable file wins on conflicts.
- Avoid duplicating or contradicting instructions across `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or tool-specific files.
- Current phase status belongs in `docs/DEVELOPMENT_PLAN.md`, not here.

## Start Here

Before editing:

1. Run:
   ```bash
   git status
   git log --oneline -10
   ```
2. Read only the task-relevant sources:
   - `README.md` — product philosophy.
   - `docs/REQUIREMENTS.md` — required capabilities.
   - `docs/ARCHITECTURE.md` — architecture contracts.
   - `docs/EVALUATION.md` — scientific evaluation policy.
   - `docs/FAILURE_POLICY.md` — failure/abstention semantics.
   - `docs/DEVELOPMENT_PLAN.md` — current phase/state.
   - `docs/KAGGLE.md` — external-run workflow when relevant.
3. Inspect the specific code, tests, registries, and experiment artifacts for the task.
4. Do not scan the whole repository when targeted reads are sufficient.
5. Do not load `docs/Beginner-Learning-Guide.md` unless explicitly relevant; it is intentionally large.

If code/config and prose disagree, report the discrepancy instead of silently choosing one.

## Repository Map

- `satquery/` — reusable scientific/domain package.
- `apps/api/` — FastAPI transport boundary.
- `apps/web/` — frontend workspace; leave alone unless explicitly tasked.
- `ml/` — offline training/adaptation/evaluation.
- `models/registry.yaml` — registered model IDs, revisions, hashes, preprocessing references.
- `scripts/kaggle/` — external GPU/data experiment tooling.
- `tests/` — pytest suite.
- `experiments/` — versioned scientific evidence and closeouts.
- `data/` — local data root; large datasets do not belong in Git.
- `docs/` — requirements, architecture, evaluation, failure policy, development plan.

Keep transport, scientific logic, benchmark logic, and orchestration separated.

## Environment and Commands

`pyproject.toml` is the dependency source of truth.

```bash
pip install -e ".[dev]"
pip install -e ".[training]"      # only when needed
pip install -e ".[grounding]"     # only when needed
pip install -e ".[multisensor]"   # only when needed

uvicorn apps.api.app.main:app --reload

python -m pytest
python -m pytest tests/path/to/test_file.py
python -m pytest tests/path/to/test_file.py::test_name

git diff --check
```

- Do not casually change dependency versions.
- Do not replace/reinstall PyTorch in GPU notebooks unless the experiment contract requires it.
- Prefer targeted tests while iterating; run the broader affected suite before completion.
- Detect hardware capabilities; do not assume CUDA.
- Do not launch expensive downloads/training unless required by the task.

## Scientific Invariants

### Evidence first
Every scientific answer must be traceable to model output, spatial evidence, raster statistics, metadata, or deterministic measurement.

### Metadata is first-class
Preserve sensor/platform, modality, acquisition time, CRS, affine, bounds, resolution/GSD, band roles, polarization, radiometric domain, and NoData/masks when available.

### Optical and SAR are not interchangeable
- Never treat SAR as grayscale RGB.
- Never assume SAR means Sentinel-1 or VV/VH.
- Unknown/generic sensors require explicit semantic mappings or must fail closed.
- Never infer SAR radiometric units from brightness.

### Temporal claims require temporal evidence
- Change requires ordered T1/T2 observations.
- Equal array shape is not proof of alignment.
- Pixelwise comparison requires verified grids or explicit documented reprojection/resampling.

### Quantitative claims are deterministic
Area, distance, coverage, and counts must be derived from masks/pixels plus geospatial metadata.
A caption/VQA model must not invent them.

### Fail closed
Missing/unknown semantics, incompatible observations, unsupported model contracts, or invalid metadata must produce structured request-input/abstain/reject behavior rather than plausible output.

Use repository failure-policy outcomes where applicable:
`ALLOW`, `ALLOW_WITH_WARNING`, `REQUEST_INPUT`, `ABSTAIN`, `REJECT`.

## Dataset and Model Provenance

Before adopting a dataset/model:

- Prefer original paper, official repo, DOI/Zenodo, or official model/dataset card.
- Treat Kaggle/Hugging Face mirrors as transport unless they are the authoritative publisher.
- Record source, version/revision, license/upstream restrictions, split semantics, modality/band order, preprocessing, and label/output semantics.
- Record file size/hash when practical.
- Do not let a mirror license override upstream imagery terms.
- Do not commit restricted imagery or large checkpoints.
- Pin model revisions and verify registered checkpoint hashes.
- Never guess preprocessing, normalization, channel order, polarization, or radiometric domain.
- If provenance/licensing remains unresolved, record it as unresolved and choose another source when practical.

## Experiment Discipline

Before a scientific run, freeze:

- experiment ID and hypothesis,
- dataset/version and split,
- sample-selection rule,
- model/checkpoint revision,
- preprocessing,
- metrics,
- seeds when relevant,
- output schema,
- test-sealing policy.

Rules:

- Validation may guide development; sealed test data must not guide tuning.
- Never tune after seeing sealed-test results.
- Preserve failed and negative experiments.
- Never manufacture a `PASS`, metric, prediction, confidence value, or benchmark row.
- A zero-sample run is not a measured zero score.
- Prediction/stat row counts must match declared sample counts.
- Aggregate metrics should be reconstructible from saved evidence where feasible.
- Synthetic fixtures belong in tests, never as benchmark fallbacks.
- Prefer the smallest justified adaptation: frozen baseline → head/projector → LoRA/PEFT → deeper tuning only with evidence.

## Kaggle / External Runs

For canonical external results:

- begin from a clean Git worktree;
- record and checkout the exact Git SHA;
- clone the repository exactly once;
- use a unique experiment/kernel identity;
- write outputs only to the declared experiment output directory;
- retrieve artifacts by exact path, never by first matching basename;
- keep runner/provenance metadata separate from scientific metrics;
- preserve logs/failure artifacts;
- do not use `--allow-dirty` for canonical runs;
- audit downloaded artifacts locally before claiming a result.

Notebooks should remain thin:

```text
environment check → exact repo/ref bootstrap → acquire data/model
→ call tested Python evaluator → verify artifacts → exit
```

No hand-authored benchmark predictions or hardcoded scientific metrics in notebooks.

## Testing and Verification

Tests should prove behavior, not merely search source strings.

Prioritize:

1. deterministic scientific/geospatial tests;
2. sensor semantic and failure-policy tests;
3. model contract/provenance tests;
4. API/integration tests;
5. benchmark reconstruction checks.

For changed code:

- add/update the smallest meaningful regression test;
- run targeted tests during iteration;
- run the broader affected suite before completion;
- run `git diff --check`;
- report skipped/unrunnable checks explicitly.

Do not inflate test counts with trivial assertions.
Do not label failures as environment-only without evidence.

## Coding and Architecture Style

- Reuse existing patterns before inventing abstractions.
- Prefer simple, typed, testable modules and explicit error paths.
- Avoid unrelated refactors.
- Keep benchmark adapters separate from production scientific APIs.
- Do not add speculative infrastructure (Kubernetes, Kafka, vector DBs, service mesh, multi-agent frameworks) without a demonstrated requirement.
- Reuse current FastAPI/storage abstractions.
- Comment non-obvious remote-sensing/geospatial assumptions, not obvious syntax.
- If adding a dependency, state why existing dependencies are insufficient.

## Security and Data Safety

- Never commit credentials, tokens, `.env` contents, private URLs, datasets, or large model weights.
- Treat rasters/archives as untrusted input.
- Validate paths, archive extraction, file sizes, dimensions, band counts, and formats.
- Prevent path traversal and unsafe extraction.
- Keep network access opt-in where the existing system expects it.
- Do not execute arbitrary code from user queries.
- Agent/orchestration code must use bounded registered tools and parameters.

## Git and Artifact Safety

- Inspect `git status` before and after work.
- Do not force-push, rewrite shared history, reset unrelated work, or delete evidence without explicit authorization.
- Do not amend existing commits unless requested.
- Prefer focused, coherent commits; conventional commit style is preferred.
- Historical experiment artifacts are evidence, not scratch space.
- Do not modify frozen artifacts merely to make docs or metrics agree.
- Keep caches/temp/generated staging files out of Git.

## Documentation Ownership

Update docs when a public contract, scientific assumption, dataset/model decision, or workflow changes.

Use:
- `docs/DEVELOPMENT_PLAN.md` for phase/status;
- `docs/EVALUATION.md` for evaluation policy;
- `docs/FAILURE_POLICY.md` for failure semantics;
- experiment manifests/closeouts for exact scientific results;
- `models/registry.yaml` for model/checkpoint provenance.

Do not turn `AGENTS.md` into a running project diary.

## Completion Standard

Before reporting completion:

1. inspect the final diff;
2. run applicable tests/checks;
3. verify generated scientific artifacts, if any;
4. confirm no secrets/large files/temp artifacts were added;
5. state what changed and what was actually verified;
6. state unresolved risks/external dependencies;
7. never claim a test, benchmark, download, or deployment ran unless it did.

For scientific work, distinguish clearly between:

**implemented → locally verified → externally evaluated → scientifically supported → not yet validated**

Prefer concise, evidence-backed reporting over optimistic completion claims.
