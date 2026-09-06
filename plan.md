# Phase 4 Implementation Plan: Temporal + Deterministic Remote-Sensing Analytics

This plan outlines the end-to-end execution strategy for Phase 4, focusing on temporal analysis, deterministic spectral tools, and evidence generation. 

## User Review Required

> [!WARNING]
> **Dataset for Change-VQA:** The plan proposes using CDVQA (based on SECOND) for P4-E02. We need to verify the exact OSS dataset license and provenance before proceeding. Is CDVQA fully authorized for this use case, or should we use another dataset?

> [!IMPORTANT]
> **SAR Dataset for P4-E04:** We need an authoritative OSS dataset with real SAR flood/change labels or before/after scenes. Are there specific datasets (e.g., Sen1Floods11, although we must ensure it's temporal if possible) that are preferred for the SAR flood validation?

## Proposed Changes

---

### Part 1 & 2: Core Contracts & Architecture Setup
We will establish typed domain contracts and audit the existing `PairValidator`, avoiding unrelated refactors.

#### [NEW] `satquery/core/contracts/temporal.py`
- `TemporalObservationPair`
- `AnalysisROI`
- `TemporalChangeResult`
- `ChangeVQAResult`

#### [NEW] `satquery/core/contracts/evidence.py`
- `RasterStatistic`
- `IndexRasterEvidence`
- `MaskEvidence`
- `ChangeMaskEvidence`
- `MeasurementEvidence`
- `SpectralAnalysisResult`
- `SarChangeResult`

---

### Part 3: Sensor-Semantic Band Resolution
Implement a semantic band mapping layer rather than hardcoding band indices.

#### [NEW] `satquery/sensors/semantics.py`
- Map known sensors (Sentinel-1/2) to semantic roles: `RED`, `GREEN`, `NIR`, `SWIR`, `SAR_CO_POL`, `SAR_CROSS_POL`.
- Add generic adapters for unknown data that rely on metadata constraints.

---

### Part 4, 5, 6, 7 & 13: Deterministic Spectral & Temporal Tools
Implement deterministic spectral indices, the geospatial measurement engine, the temporal alignment engine, and temporal analytics.

#### [NEW] `satquery/analytics/spectral.py`
- Implementation of `NDVI`, `NDWI`, `MNDWI`, and `NDBI` with strict provenance, zero-division handling, and ROI checking.
- Explicit threshold policies for masks.

#### [NEW] `satquery/analytics/measurement.py`
- Geospatial measurement engine converting masks to physical area, percentage, and bounding geometry utilizing projected CRSs safely.

#### [NEW] `satquery/analytics/temporal.py`
- Common before/after pair engine (input validation, CRS, footprint overlap).
- Deterministic change analytics: vegetation change (NDVI delta), water change (NDWI delta), built-up change, and generic raster change.

#### [NEW] `satquery/analytics/sar.py`
- SAR Temporal Analysis (calibrated backscatter, co-pol/cross-pol).
- Probable SAR flood/water-change workflow relying on pre/post-event SAR backscatter change and temporal alignment.

---

### Part 8, 16 & 19: Outputs, Tool Registration & Contract
Format outputs for evidence generation and register them for Phase 5 orchestration.

#### [MODIFY] `satquery/registry/preprocessing.yaml` (or tools registry)
- Register new temporal/spectral specialists (`NDVI`, `VegetationChange`, `SARTemporalChange`, `ChangeVQA`, etc.)

#### [NEW] `satquery/tools/temporal_vqa.py`
- Defines the Temporal Change-VQA tool contract. 

---

### Part 9, 11, 12: Learned Bi-Temporal Change Intelligence (P4-E02, P4-E03)
Implement the learned specialist for bi-temporal Change-VQA.

#### [NEW] `satquery/models/change_vqa/baseline.py`
- Load the chosen OSS baseline model for CDVQA/SECOND.
- Implement reproducible checkpoint loading and preprocessing logic.

#### [NEW] `scripts/kaggle/p4_e02_baseline.py`
- Kaggle runner script for P4-E02 baseline evaluation (and P4-E03 if adaptation is justified).

---

### Part 10 & 15: Validation Experiments (P4-E01, P4-E04)
Create synthetic and real-data validation for deterministic and SAR capabilities.

#### [NEW] `tests/phase4/test_p4_e01_deterministic.py`
- Small synthetic/real geospatial fixtures to test spectral math, ROI, CRS handling, and mask-to-area.

#### [NEW] `tests/phase4/test_p4_e04_sar.py`
- Fixtures verifying temporal alignment, SAR polarization semantics, and deterministic change generation for SAR data.

---

### Part 17, 18, 20-24: Policies, Tests, and Closeout
- **Confidence Policy:** No fabricated confidences; rely on physical measurement stats. Learned specialists only return confidence if calibrated.
- **Failure Policy:** Map missing pairs to `REQUEST_INPUT`, incompatible pairs to `REJECT`, reprojections to `ALLOW_WITH_WARNING`.
- **Docs:** Update `ARCHITECTURE.md`, `REQUIREMENTS.md`, and `DEVELOPMENT_PLAN.md`.
- **Closeout:** Produce `experiments/phase4_temporal_closeout.json` upon completion of the phase.

## Verification Plan

### Automated Tests
- `pytest tests/phase4/test_p4_e01_deterministic.py -v`
- `pytest tests/phase4/test_p4_e04_sar.py -v`
- `pytest tests/` (for regression on Phase 1-3)

### Manual / External Verification
- Run P4-E02 Kaggle external dependency to evaluate the Change-VQA learned baseline.
- Provide the exact Kaggle command for execution, and pause work while awaiting results.
