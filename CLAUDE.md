# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in code in this repository.

## Project Overview

**SatQuery AI** — Evidence-grounded, sensor-aware remote-sensing analysis via natural-language queries (Problem Statement ID: 26167). The system combines PyTorch-based vision-language models, deterministic GIS, and a constrained orchestration layer so that every answer is tied to sensor data and spatial evidence rather than LLM hallucination.

## Tech Stack

- **Python 3.11+**, managed with `pip`/`uv` (see `pyproject.toml`)
- **Backend:** FastAPI + Uvicorn; app entry at `apps/api/app/main.py`
- **ML:** PyTorch, HuggingFace Transformers, PEFT/LoRA; models registered in `models/registry.yaml`
- **Geospatial:** Rasterio, Affine, Pillow, GDAL ecosystem; `rio-tiler`/TiTiler optional
- **Frontend:** React + TypeScript + Vite + OpenLayers + Tailwind CSS (`apps/web/`)
- **Storage:** Filesystem-first; SQLite for metadata; Postgres optional at scale
- **Orchestration:** Docker + Docker Compose (when topology is runnable)
- **Training infra:** Kaggle GPU runner (`scripts/kaggle/runner.py`), `accelerate`

## Project Layout

- `satquery/` — reusable scientific/domain package: `ingestion`, `geo`, `inference`, `orchestration`, `registry`, `verification`, `evidence`, `reporting`, `visualization`
- `apps/api/` — FastAPI transport boundary (uploads, observations, VQA, grounding, tiles)
- `apps/web/` — React frontend (imagery-first, not chat-first)
- `ml/` — offline model work: `adapters`, `preprocessing`, `training`, `inference`, `evaluation`, `configs`
- `models/registry.yaml` — single source of truth for model checkpoint IDs, revisions, preprocessing profiles
- `scripts/kaggle/` — Kaggle GPU runner
- `tests/` — pytest test suite
- `experiments/` — versioned experiment records (results, manifests, configs)
- `data/` — immutable dataset storage; never commit large artifacts to Git

## Key Commands

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install with optional extras
pip install -e ".[multisensor]"   # Parquet/zstd for BigEarthNet
pip install -e ".[training]"      # PEFT/accelerate/tensorboard

# Run the FastAPI server
uvicorn apps.api.app.main:app --reload

# Run the full test suite
python -m pytest

# Run a single test file
python -m pytest tests/geo/test_coordinate_mapping.py

# Phase 2A frozen VQA baseline
python -m ml.evaluation.run_phase2a_baseline --allow-download

# Phase 2B LoRA adaptation
python -m ml.training.phase2b --config ml/configs/phase2b_smolvlm_lora.yaml --output-dir outputs/phase2b_smolvlm_lora
python -m ml.training.phase2b --config ml/configs/phase2b_smolvlm_lora.yaml --stability-smoke

# Phase 3A grounding
python -m ml.evaluation.run_phase3a_grounding --allow-download

# Phase 4B BigEarthNet metadata manifest
python -m ml.evaluation.prepare_phase4_bigearthnet

# Phase 4D materialization (requires --confirm-full-stream-transfer)
python -m ml.evaluation.materialize_phase4_bigearthnet --plan
python -m ml.evaluation.materialize_phase4_bigearthnet --confirm-full-stream-transfer

# Kaggle GPU training
python scripts/kaggle/runner.py
```

## Architecture Principles

The following rules are non-negotiable and must hold in all code:

1. **Sensor metadata is first-class.** Every observation carries CRS, GSD, bands, modality, polarity, acquisition time. Never discard it in preprocessing.
2. **Language models explain evidence; they never manufacture it.** Numeric claims (area, counts) come from deterministic GIS operations.
3. **Change requires two temporally ordered observations.** One image + "What changed?" must request a second observation — never infer change from a single image.
4. **Optical and SAR are different physical measurements.** Never treat SAR as grayscale RGB. Preserve modality identity in evidence (`OPTICAL_SUPPORTED`, `SAR_SUPPORTED`, `BOTH_SUPPORTED`, `CONFLICTING`).
5. **Validation gates before processing.** `PASS` / `WARN` / `FAIL` statuses govern every operation. Missing CRS + area request = `FAIL`. Missing NIR + NDVI = `FAIL`.
6. **Evidence exists independently of language.** Every result includes structured evidence (bbox/mask/polygon), measurements, model provenance, verification results, and execution trace.
7. **Frozen baselines first.** Adaptation hierarchy: frozen pretrained → linear/projector → LoRA/PEFT → sensor adapter → partial unfreezing → full fine-tuning only if justified.
8. **Abstention over unsupported certainty.** Low confidence, out-of-domain sensors, or invalid inputs produce qualified results or explicit refusal — never silent hallucination.

## Module Boundaries

- `satquery/` holds reusable scientific logic (domain-agnostic, no framework coupling)
- `apps/api/` holds FastAPI-specific transport code (schemas, routes, upload handling)
- `ml/` holds offline model code; training code must NOT be required by the API runtime
- `models/registry.yaml` is the canonical model catalog; add entries only when a real implementation satisfies its contract
- `apps/web/` owns all UI code; `satquery/` must remain frontend-agnostic

## Testing Strategy

Three layers (per README):
1. **Deterministic GIS tests** — strict unit tests for pixel↔world coordinate transforms, CRS conversion, area calculation, overlap detection
2. **Model evaluation** — fixed benchmark/golden datasets; evaluate every approved checkpoint against VQA, grounding, SAR, optical-SAR, change, cross-sensor
3. **Orchestration tests** — validate routing rules (e.g., 1 image + change query → `MISSING_TEMPORAL_PAIR`; RGB + NDVI → `MISSING_NIR_BAND`)

## Environment

- Copy `.env.example` → `.env` before running locally
- `ENABLE_REMOTE_NETWORK=false` by default (models only load from local cache after first verified download)
- Upload limits enforced in env: `MAX_UPLOAD_SIZE_MB`, `MAX_RASTER_WIDTH/HEIGHT/PIXELS/BANDS`
- Model cache lives under `models/cache/` (gitignored); checkpoints verified by SHA-256 before loading

## Development Notes

- Phase status is tracked in `docs/DEVELOPMENT_PLAN.md` (Phase 1A–4D are complete; Phases 4E+ pending)
- The frozen Phase 3 grounding policy (threshold 0.30, 80% area cap) must not be changed without a documented rationale
- Experiment results go under `experiments/<phase_name>/` with versioned manifests and result summaries
- Never download datasets or checkpoints into Git; use `data/` (gitignored) and `models/cache/` (gitignored)
- Kaggle-specific setup, resume, and artifact paths documented in `docs/KAGGLE.md`
