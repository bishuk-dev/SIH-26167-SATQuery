# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SatQuery AI** — Evidence-grounded, sensor-aware remote-sensing analysis via natural-language queries (Problem Statement ID: 26167). The system combines PyTorch-based vision-language models, deterministic GIS, and a constrained orchestration layer so that every answer is tied to sensor data and spatial evidence rather than LLM hallucination.

## Tech Stack & Key Commands

- **Python 3.11+**, managed with `pip`/`uv` (see `pyproject.toml`)
- **Backend:** FastAPI + Uvicorn; app entry at `apps/api/app/main.py`
- **ML:** PyTorch, HuggingFace Transformers, PEFT/LoRA; models registered in `models/registry.yaml`
- **Geospatial:** Rasterio, Affine, Pillow, GDAL ecosystem; `rio-tiler`/TiTiler optional
- **Frontend:** React + TypeScript + Vite + OpenLayers + Tailwind CSS (`apps/web/`) — scaffolded, not built
- **Storage:** Filesystem-first; SQLite for metadata; Postgres optional at scale
- **Orchestration:** Docker + Docker Compose (when topology is runnable)
- **Training infra:** Kaggle GPU runner (`scripts/kaggle/runner.py`), `accelerate`

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Optional extras
pip install -e ".[training]"      # PEFT/accelerate/tensorboard
pip install -e ".[multisensor]"   # Parquet/zstd for BigEarthNet
pip install -e ".[grounding]"     # fsspec[http]

# Run the FastAPI server
uvicorn apps.api.app.main:app --reload

# Run the full test suite
python -m pytest

# Run a single test file
python -m pytest tests/geo/test_coordinates.py

# Phase 2B LoRA adaptation
python -m ml.training.phase2b --config ml/configs/phase2b_smolvlm_lora.yaml --output-dir outputs/phase2b_smolvlm_lora

# Phase 3A grounding
python -m ml.evaluation.run_phase3a_grounding --allow-download

# Phase 4D materialization (requires --confirm-full-stream-transfer)
python -m ml.evaluation.materialize_phase4_bigearthnet --plan
python -m ml.evaluation.materialize_phase4_bigearthnet --confirm-full-stream-transfer

# Kaggle GPU training
python scripts/kaggle/runner.py
```

## Architecture Principles (Non-Negotiable)

1. **Sensor metadata is first-class.** Every observation carries CRS, GSD, bands, modality, polarity, acquisition time. Never discard it in preprocessing.
2. **Language models explain evidence; they never manufacture it.** Numeric claims (area, counts) come from deterministic GIS operations.
3. **Change requires two temporally ordered observations.** One image + "What changed?" must request a second observation — never infer change from a single image.
4. **Optical and SAR are different physical measurements.** Never treat SAR as grayscale RGB. Preserve modality identity in evidence (`OPTICAL_SUPPORTED`, `SAR_SUPPORTED`, `BOTH_SUPPORTED`, `CONFLICTING`).
5. **Validation gates before processing.** `PASS` / `WARN` / `FAIL` statuses govern every operation. Missing CRS + area request = `FAIL`. Missing NIR + NDVI = `FAIL`.
6. **Evidence exists independently of language.** Every result includes structured evidence (bbox/mask/polygon), measurements, model provenance, verification results, and execution trace.
7. **Frozen baselines first.** Adaptation hierarchy: frozen pretrained → linear/projector → LoRA/PEFT → sensor adapter → partial unfreezing → full fine-tuning only if justified.
8. **Abstention over unsupported certainty.** Low confidence, out-of-domain sensors, or invalid inputs produce qualified results or explicit refusal — never silent hallucination.

## Module Boundaries

- `satquery/` — reusable scientific/domain package: `ingestion`, `geo`, `inference`, `orchestration`, `registry`, `verification`, `evidence`, `reporting`, `visualization`
- `apps/api/` — FastAPI transport boundary (uploads, observations, VQA, grounding, tiles)
- `apps/web/` — React frontend (imagery-first, not chat-first)
- `ml/` — offline model work: `adapters`, `preprocessing`, `training`, `inference`, `evaluation`, `configs`
- `models/registry.yaml` — single source of truth for model checkpoint IDs, revisions, preprocessing profiles
- `scripts/kaggle/` — Kaggle GPU runner
- `tests/` — pytest test suite
- `experiments/` — versioned experiment records (results, manifests, configs)
- `data/` — immutable dataset storage; never commit large artifacts to Git

## Development Workflow

**Git:** Feature branches (`feat/...`, `fix/...`) + PR reviews. Conventional commits preferred.

**Code style:** Ruff (linting) + Black (formatting). Configure in `pyproject.toml` when adding.

**No CI workflows yet** — `.github/workflows/` does not exist.

## Testing Strategy

Three layers (per README):
1. **Deterministic GIS tests** — strict unit tests for pixel↔world coordinate transforms, CRS conversion, area calculation, overlap detection (`tests/geo/`, `tests/ingestion/`)
2. **Model evaluation** — fixed benchmark/golden datasets; evaluate every approved checkpoint against VQA, grounding, SAR, optical-SAR, change, cross-sensor (`tests/ml/`)
3. **Orchestration tests** — validate routing rules (e.g., 1 image + change query → `MISSING_TEMPORAL_PAIR`; RGB + NDVI → `MISSING_NIR_BAND`) (`tests/routing/`)

```bash
python -m pytest                          # all
python -m pytest tests/geo/               # layer
python -m pytest tests/geo/test_coordinates.py::test_name  # single
```

Integration tests in `tests/integration/` use `httpx` async client against the live FastAPI app.

## Key Gotchas

**Raster upload limits** (enforced by the inspector):
- `MAX_UPLOAD_SIZE_MB=512`
- `MAX_RASTER_WIDTH=50000`, `MAX_RASTER_HEIGHT=50000`
- `MAX_RASTER_PIXELS=150000000`
- `MAX_RASTER_BANDS=32`

**VQA inference defaults:**
- `GPU_DEVICE=cpu` — frozen VQA runs on CPU by default
- `ENABLE_REMOTE_NETWORK=false` — network access is opt-in
- `VQA_CPU_THREADS=2`, `GROUNDING_CPU_THREADS=2`

**Model cache:** HuggingFace Hub caches weights at `~/.cache/huggingface/`. No custom cache path configured.

**Model registry checksum verification:** Every registered model in `models/registry.yaml` has a `checkpoint_sha256`. Downloads are verified before use.

**No Git LFS:** Large model weights not in Git. Fetched from HuggingFace or Kaggle at runtime. `.gitignore` excludes `models/checkpoints/`, `models/cache/`, `*.safetensors`, `*.pt`, `*.ckpt`.

**Data root:** `DATA_ROOT=./data` stores observations, analyses, SQLite database (`DATABASE_URL=sqlite:///./data/satquery.db`).

**BigEarthNet materialization:** Phase 4D requires ~109 GiB HTTP transfer from Zenodo. CROMA model explicitly blocked (no positional band semantics published).

**Kaggle GPU requirements:** Notebook bootstrap uninstalls `torchao`, pins `torch==2.8.0+cu126`, BF16 only on compute capability >= 8.0 (Ampere+). P100 falls back to FP16 after stability smoke pass.

**Phase 3 grounding policy:** Threshold 0.30 + normalized-area cap 0.80 is the frozen production policy. Test run once on untouched data; will not be rerun.

**Frontend is not yet built:** `apps/web/` contains only a `README.md` placeholder.

## Environment Setup

```bash
cp .env.example .env
# Edit .env as needed
```

Key env vars: `ENABLE_REMOTE_NETWORK=false` (default), upload limits, GPU device, CPU threads, data root, database URL.