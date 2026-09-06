# Phase 4 Implementation Plan: Temporal + Deterministic Remote-Sensing Analytics

This plan outlines the end-to-end execution strategy for Phase 4, focusing on temporal analysis, deterministic spectral tools, and evidence generation. 

## Dataset Blocker Resolutions (RESOLVED)

### P4-E02 — Learned Bi-Temporal Language Specialist (CDVQA Blocked / LEVIR-CC Adopted)
- **CDVQA / SECOND License Audit & Gate Status**:
  - `cdvqa_annotation_license`: Apache-2.0
  - `second_dataset_access`: public
  - `second_image_license_status`: UNRESOLVED
  - `cdvqa_full_dataset_license_gate`: BLOCKED
  - *Policy*: Do not infer that CDVQA's Apache-2.0 license relicenses upstream SECOND imagery.
- **Adopted SIH MVP Solution**: Switched P4-E02 to **CHANGE DESCRIPTION** using **LEVIR-CC**.
  - Permitted by mandatory requirement (allows bi-temporal change description OR change-VQA).
  - Upstream imagery: Derived mainly from LEVIR-CD (Hao Chen & Zhenwei Shi, Beihang University / LEVIR Lab).
  - Image license/terms: Restricted strictly to academic / non-commercial research use.
  - Repositories & Provenance:
    - Dataset: `https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset` / `lcybuaa/LEVIR-CC`
    - Paper: Chenyang Liu et al., "Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset", IEEE TGRS 2022.
    - Code: `https://github.com/Chen-Yang-Liu/RSICC`
  - Invariants:
    - Acceptable for SIH academic prototype.
    - Attribution preserved.
    - No redistribution of imagery in SatQuery repository.
    - Do not describe source imagery as unrestricted commercial OSS.
    - HuggingFace mirror Apache-2.0 metadata does NOT override upstream terms.
  - Baseline: RSICCformer / SmolVLM multi-image change captioner (<2.5 GB VRAM on T4).
  - Split: 6,815 train, 1,332 val, 1,930 test (test set strictly sealed; validation split only for evaluation).

### P4-E04 — SAR Temporal Flood Inundation Validation (Modified Sen1Floods11 Pinned)
- **Primary Benchmark**: Modified Sen1Floods11 Dataset for Change Detection
  - DOI: [10.5281/zenodo.7946594](https://doi.org/10.5281/zenodo.7946594) (Concept DOI: 10.5281/zenodo.7946593)
  - Version: v1 (Publication Date: 2023-05-17)
  - Creator: Ritu Yadav / KTH Royal Institute of Technology
  - License: CC BY 4.0
  - Associated Paper: "Attentive Dual Stream Siamese U-Net for Flood Detection on Multi-Temporal Sentinel-1 Data", IGARSS 2022 (DOI: [10.1109/IGARSS46834.2022.9883132](https://doi.org/10.1109/IGARSS46834.2022.9883132))
  - Authoritative Pinned Files & Published Hashes:
    1. `PRE_S1-20230517T191707Z-001.zip` (1,520,699,949 bytes, MD5: `4a32637c56ea519bd3c4baca208b289d`)
    2. `POST_S1-20230517T191716Z-001.zip` (729,719,788 bytes, MD5: `40a505cfbd5318d94a9d5bd7aef88561`)
    3. `Labels-20230517T191741Z-001.zip` (2,462,219 bytes, MD5: `069b4c05eefb7a6e72c1adb34aaf1a24`)
- **Label Raster Semantic Audit & Target**:
  - *Audit Finding*: Ground truth labels in Modified Sen1Floods11 delineate **post-event water/flood extent** (1 = water/inundated, 0 = non-water, -1 = NoData). They do NOT represent newly inundated change minus permanent baseline water.
  - *Evaluation Target*: Evaluates post-event water extent against authoritative labels.
  - *SatQuery Evidence*: Bi-temporal flood expansion (post-water minus pre-water, backscatter drop >= 3 dB) is computed as a separate deterministic GIS evidence computation (`flood_expansion_m2`).
  - *Policy*: Single-date Sen1Floods11 is NOT used alone as proof of temporal change. No random mirrors are permitted.

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
