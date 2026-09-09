# Phase 4 Clean Rebuild Implementation Plan

> **For agentic workers:** Execute this plan strictly task-by-task. If your environment provides a plan-execution/subagent skill, use it; otherwise follow the checklist directly. Never advance past a human-review or scientific gate automatically. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible temporal remote-sensing specialists for deterministic multispectral analysis, high-resolution optical change masks, change captions, and SAR flood/change evidence without allowing prose to manufacture spatial or numeric claims.

**Architecture:** Keep four independent scientific paths: deterministic multispectral processing, ChangerEx structural-change segmentation, Chg2Cap change captioning, and STURM Sentinel-1 flood segmentation plus deterministic SAR change. Each path emits typed evidence. Shared verification checks temporal order, grids, sensor contracts, provenance, masks, and measurements before any evidence is reconciled or explained.

**Tech Stack:** Python 3.11+, NumPy, Rasterio, Affine, Pydantic 2, PyTorch for existing/optical specialists, the existing Kaggle runner, and only audit-approved model-specific dependencies. STURM remains an isolated TensorFlow/Keras specialist if the authoritative runtime contract requires it; do not convert it to PyTorch merely for stack uniformity.

**Spec:** User-approved Phase 4 clean-rebuild brief, captured in “Approved Phase 4 design” below.

**Current canonical state (2026-09-09):**
- Phase 3 is frozen.
- Task 1 audit commit: `a7b9317`.
- Corrected Task 1 re-audit commit: `339be0b`.
- `339be0b` changes only the four audit files under `experiments/phase4_temporal_analytics/`.
- The previous premature scaffold is preserved on `archive/phase4-scaffold-with-reaudit`.
- Task 2 is the next authorized task. No Task 3+ work may be started automatically.

## Global Constraints

- Preserve all frozen historical artifacts under `experiments/phase2*`, `experiments/phase3*`, and `experiments/phase4_bigearthnet_multisensor/`; never rewrite them to fit this rebuild.
- Treat this work as canonical Phase 4. Historical Phase 4A–4F and Phase 5A remain canonical Phase 3 evidence.
- Work only from a clean `phase4-clean-rebuild` worktree and record the exact starting Git SHA before each canonical task or external run.
- First implementation-series commit after this planning document contains audit artifacts only. No runtime code, registry entry, dependency, notebook, or benchmark result belongs in that commit.
- Preferred datasets/models remain candidates until audit status is `PASS`. A failed license, provenance, checkpoint, preprocessing, or dependency audit changes status to `BLOCKED`; it never gets guessed around.
- Task 2 may validate `BLOCKED` contracts, but specialist runtime tasks may not execute until their required primary dataset/model contracts are `PASS`.
- Optional robustness/holdout blockers must not block an otherwise valid primary lane: S2Looking is non-gating for P4-E02; Sen1Floods11 is conditional/non-gating for the STURM primary reproduction.
- `PASS` means the contract is structurally ready for reproduction, not that the model is already promoted to production.
- Use original papers, official repositories, DOI/Zenodo records, or publisher dataset cards as provenance authority. Kaggle and Hugging Face mirrors are transport only unless maintained by the publisher.
- Do not accept DeepWiki, ResearchGate, Kaggle descriptions, or an ephemeral Hugging Face pull-request ref as sole authority for metrics, licenses, revisions, or preprocessing.
- Keep source-code license, checkpoint provenance/license, dataset-package license, and upstream imagery terms as separate fields; one never implies another.
- Compute SHA-256 locally for every downloaded checkpoint and stable source archive. Record publisher MD5/SHA only as an additional transport check when provided; a publisher hash is not mandatory when the publisher does not provide one, but a locally verified SHA-256 is mandatory before canonical scientific use.
- Never commit source imagery, model weights, unpacked external repositories, credentials, caches, or large generated artifacts.
- Never train before reproducing the official pretrained baseline on validation data.
- Never tune preprocessing, thresholds, prompts, tiling, or postprocessing using sealed test or robustness results.
- Keep each temporal pair indivisible across splits. Group related crops, captions, and masks with their source pair.
- Equal array shape is not proof of alignment. Pixelwise operations require verified common grids or explicit reprojection/resampling recorded as a derived operation.
- Never infer band identity, polarization, radiometric domain, T1/T2 order, CRS, or GSD from visual appearance.
- Never run the STURM learned model unless sensor, polarization order, radiometric domain, scaling, normalization, resolution, and dimensions satisfy its audited contract.
- RISAT or generic SAR with incompatible/unknown semantics must bypass learned STURM inference. Deterministic SAR processing may run only with explicit compatible polarization and radiometric mappings.
- Caption output never supplies masks, area, counts, or scientific confidence. Numeric claims come only from deterministic GIS evidence.
- Model scores remain `raw_model_score` until calibration is measured. No aggregate confidence weighting.
- Use repository outcomes exactly: `ALLOW`, `ALLOW_WITH_WARNING`, `REQUEST_INPUT`, `ABSTAIN`, `REJECT`.
- Keep API routing and natural-language orchestration outside Phase 4. Phase 5 consumes these registered specialists and evidence contracts.
- Add no dependency until existing dependencies and official source installation options are evaluated in Task 1.
- Run targeted tests after each task, broader affected tests before each commit, and `python -m pytest` plus `git diff --check` before closeout.
- A zero-sample or blocked run is not a zero metric. Preserve failure artifacts and report `not_measured`.

---

## Approved Phase 4 Design

| Capability | Dataset | Specialist | Role |
|---|---|---|---|
| Multispectral temporal analytics | OSCD plus explicitly selected Sentinel-2 L2A pairs | Deterministic Rasterio/NumPy operations | Primary |
| High-resolution structural change mask | LEVIR-CD | Open-CD ChangerEx-R18 | Primary |
| High-resolution robustness | S2Looking | Frozen ChangerEx-R18, unchanged | Optional non-gating robustness source |
| Learned change description | LEVIR-CC | Chg2Cap | Primary |
| SAR flood/water extent | STURM-Flood | Official Sentinel-1 U-Net | Primary |
| SAR independent validation | Sen1Floods11 | Frozen STURM model plus deterministic SAR evidence | Conditional external holdout |
| Modified Sen1Floods11 | None | None | Removed from core Phase 4 |
| RSCaMa | LEVIR-CC | None in core plan | Optional challenger after closeout only |

### Scientific lanes

```text
Temporal pair
├── multispectral
│   ├── deterministic indices/differences
│   ├── ChangerEx structural-change mask when RGB/high-resolution contract matches
│   └── Chg2Cap caption when LEVIR-CC-like contract matches
└── SAR
    ├── STURM flood mask when exact Sentinel-1 contract matches
    └── deterministic temporal change when explicit SAR semantics match

mask evidence
├── geometry verification
├── deterministic measurement
└── optional agreement/conflict assessment

caption evidence
└── textual description only; never measurement source
```

### Experiment identities

- `P4-E01`: deterministic multispectral and temporal engine on OSCD; operational demonstration on real Sentinel-2 L2A.
- `P4-E02`: ChangerEx-R18 structural-change validation on LEVIR-CD; unchanged S2Looking robustness evaluation only if the S2Looking license/input contract reaches `PASS`.
- `P4-E03`: Chg2Cap validation on LEVIR-CC.
- `P4-E04`: STURM Sentinel-1 U-Net validation on STURM-Flood; unchanged Sen1Floods11 external evaluation only if transfer preprocessing/input semantics are proven compatible; deterministic SAR agreement is diagnostic only.

### Promotion rule

A specialist enters `models/registry.yaml` only after all are true:

1. official source and license are verified;
2. exact source revision is immutable;
3. checkpoint bytes, nonzero size, and locally computed SHA-256 are verified; publisher hash is also verified when one exists;
4. preprocessing and output semantics are verified from authoritative source/code;
5. local contract/smoke tests pass;
6. official pretrained validation is reproduced or discrepancy is documented and accepted;
7. domain limits and failure outcomes are explicit;
8. source-code license, checkpoint provenance/license, dataset-package license, and imagery terms are independently acceptable for the intended use.

---

## Planned File Map

### Audit and experiment evidence

- Create `experiments/phase4_temporal_analytics/README.md`: scope, status, sealing policy, artifact index.
- Create `experiments/phase4_temporal_analytics/dataset_contracts.yaml`: authoritative dataset contracts and audit status.
- Create `experiments/phase4_temporal_analytics/model_contracts.yaml`: authoritative model/checkpoint contracts and audit status.
- Create `experiments/phase4_temporal_analytics/experiment_plan.yaml`: frozen IDs, splits, metrics, selection rules, outputs, and stop conditions.
- Create `experiments/phase4_temporal_analytics/p4_e01/` through `p4_e04/`: reviewed manifests, plans, and later measured results.
- Create `experiments/phase4_temporal_analytics/PHASE_4_CLOSEOUT.json` only at final closeout.

### Reusable scientific package

- Modify `satquery/evidence/models.py`: temporal mask, caption, measurement, and agreement evidence contracts.
- Create `satquery/analytics/__init__.py`: public deterministic temporal exports only after implementations exist.
- Create `satquery/analytics/temporal.py`: common-grid preparation and deterministic temporal differencing.
- Create `satquery/analytics/spectral.py`: NDVI, NDWI, MNDWI, and semantic-band validation.
- Create `satquery/analytics/measurement.py`: CRS-safe mask area and units.
- Create `satquery/analytics/sar.py`: radiometric-domain-aware SAR differencing and mask agreement.
- Create `satquery/inference/change_detection.py`: ChangerEx adapter, tiling, stitching, and domain checks.
- Create `satquery/inference/change_captioning.py`: Chg2Cap adapter and caption-only evidence.
- Create `satquery/inference/flood.py`: STURM adapter and strict Sentinel-1 contract checks.
- Modify `satquery/inference/config.py`: bounded runtime settings for temporal specialists.
- Modify `satquery/inference/exceptions.py`: explicit temporal model contract and execution failures.
- Modify `satquery/registry/models.py`: typed temporal model and preprocessing registrations.
- Modify `satquery/registry/preprocessing.yaml`: only promoted, audit-backed profiles.
- Modify `models/registry.yaml`: only promoted, reproduced checkpoints.
- Modify `satquery/registry/tools.yaml`: deterministic index, temporal difference, agreement, and mask-area tools after implementation.

### Offline preparation and evaluation

- Create `ml/evaluation/phase4_contracts.py`: validate audit files and frozen manifests.
- Create `ml/evaluation/prepare_p4_e01.py`: OSCD and selected Sentinel-2 L2A manifest validation/materialization.
- Create `ml/evaluation/run_p4_e01.py`: deterministic multispectral evaluation.
- Create `ml/evaluation/prepare_p4_e02.py`: LEVIR-CD and S2Looking pair-safe manifests.
- Create `ml/evaluation/run_p4_e02.py`: ChangerEx validation and robustness evaluation.
- Create `ml/evaluation/prepare_p4_e03.py`: LEVIR-CC pair/caption manifest.
- Create `ml/evaluation/run_p4_e03.py`: Chg2Cap caption evaluation.
- Create `ml/evaluation/prepare_p4_e04.py`: STURM-Flood and Sen1Floods11 manifests.
- Create `ml/evaluation/run_p4_e04.py`: flood segmentation and cross-dataset audit.

### External execution

- Modify `scripts/kaggle/experiments.yaml`: four exact Phase 4 experiment entries.
- Create `notebooks/kaggle_p4_e01_oscd.ipynb`.
- Create `notebooks/kaggle_p4_e02_changer.ipynb`.
- Create `notebooks/kaggle_p4_e03_chg2cap.ipynb`.
- Create `notebooks/kaggle_p4_e04_sturm.ipynb`.
- Modify `scripts/kaggle/README.md` and `docs/KAGGLE.md`: exact dry-run/run/retrieval commands.

### Tests

- Create `tests/ml/test_phase4_contracts.py`.
- Create `tests/analytics/test_temporal.py`.
- Create `tests/analytics/test_spectral.py`.
- Create `tests/analytics/test_measurement.py`.
- Create `tests/analytics/test_sar.py`.
- Create `tests/inference/test_change_detection.py`.
- Create `tests/inference/test_change_captioning.py`.
- Create `tests/inference/test_flood.py`.
- Create `tests/ml/test_p4_e01.py` through `test_p4_e04.py`.
- Modify `tests/test_kaggle_runner.py`: exact Phase 4 output-path and metadata-only retrieval tests.

---

### Task 0: Baseline Isolation — COMPLETE

**Status:** Complete before the current canonical history. Do not rerun unless the owner explicitly starts another clean rebuild.

Current canonical branch is clean and the previous scaffold is preserved separately.

---

### Task 1: Authoritative Dataset and Model Audit — COMPLETE / HUMAN-APPROVED

**Canonical commits:**
- `a7b9317` — initial Task 1 source-contract audit.
- `339be0b` — authoritative re-audit; exactly four audit files changed.

**Authoritative Task 1 files:**
- `experiments/phase4_temporal_analytics/README.md`
- `experiments/phase4_temporal_analytics/dataset_contracts.yaml`
- `experiments/phase4_temporal_analytics/model_contracts.yaml`
- `experiments/phase4_temporal_analytics/experiment_plan.yaml`

These files are now the source of truth. Do not reconstruct their schema from this plan.

Important frozen outcomes:
- ChangerEx/LEVIR-CD is the strongest primary promotion path, but remains `BLOCKED` until local archive/checkpoint byte verification and remaining dependency/source details are closed.
- Chg2Cap/LEVIR-CC remains `BLOCKED` until checkpoint byte identity/provenance/license and imagery-use constraints are acceptable.
- STURM/STURM-Flood remains `BLOCKED` until local bytes, exact Sentinel-1 radiometric/scaling behavior, and TensorFlow/Keras runtime details are verified.
- Sen1Floods11 is a conditional external lane; incompatible/unknown transfer preprocessing produces `BLOCKED_INPUT_CONTRACT`.
- S2Looking is optional/non-gating; unresolved licensing must never block the LEVIR-CD primary P4-E02 reproduction.
- OSCD native layout must not be assumed; its deterministic benchmark lane remains conditional on authorized source access/materialization.
- Agreement IoU is diagnostic only in Phase 4. Numeric ALLOW/WARN/ABSTAIN thresholds are forbidden.
- No Phase 4 closeout artifact exists before Task 13.

**Do not rerun Task 1 automatically.** Any later fact resolution is a small, reviewable audit amendment and must not silently turn guessed fields into `PASS`.

---

### Task 2: Validate Audit and Registry Contracts

**Authorized scope only:**
- Create `ml/evaluation/phase4_contracts.py`
- Create `tests/ml/test_phase4_contracts.py`
- Modify `satquery/registry/models.py`

**Do not create:** runtime specialists, preprocessing profiles, model registry entries, notebooks, Kaggle entries, analytics, reconciliation, closeout models, or `PHASE_4_CLOSEOUT.json`.

**Interfaces:**
- `load_phase4_contracts(root: Path) -> Phase4ContractSet`
- typed `ChangeDetectionRegistration`, `ChangeCaptionRegistration`, `FloodSegmentationRegistration`

- [ ] **Step 1: Write RED tests against the real `339be0b` audit schema**

Required behavioral coverage:
1. current audit loads successfully;
2. `BLOCKED` + non-empty blockers is valid and non-runnable;
3. `BLOCKED` without blockers fails;
4. `PASS` model with missing/malformed SHA-256 fails;
5. `PASS` model with missing authoritative source/license/checkpoint identity fails;
6. source-code license does not substitute for unresolved checkpoint provenance/license;
7. dataset-package license does not erase upstream imagery restrictions;
8. unknown fields/statuses fail;
9. robustness source cannot become development/tuning source;
10. external holdout cannot become development/tuning source;
11. `excluded_from_core` cannot become runnable;
12. sealed test cannot become development split;
13. missing referenced dataset/model IDs fail;
14. optional challenger remains non-core by default;
15. blocked S2Looking does not invalidate the LEVIR-CD primary P4-E02 structure;
16. blocked/conditional Sen1Floods11 does not invalidate the STURM primary P4-E04 structure;
17. Phase 4 agreement policy containing numeric decision thresholds fails;
18. P4-E03 requires BLEU-4, METEOR, ROUGE-L, CIDEr declarations;
19. loading contracts performs no registry mutation;
20. Task-2 loader expects only the three audit YAMLs, not a closeout file.

- [ ] **Step 2: Implement strict Pydantic audit loader**

Rules:
- `extra="forbid"` (or equivalent) throughout scientific records.
- Preserve `BLOCKED` as a valid scientific state.
- Derive `runnable`/`promotable`; never trust YAML-supplied booleans for them.
- `BLOCKED` → non-runnable/non-promotable.
- `PASS` means structurally eligible for later reproduction, not already production-promoted.
- Validate cross-references between experiments, datasets, models/tools where IDs exist.
- Do not redesign free-text relationships merely to satisfy type checking.
- Preserve license separation exactly as audited.
- Preserve non-gating optional/conditional lanes.
- No closeout classes/functions in Task 2.

- [ ] **Step 3: Add temporal registry schemas without entries**

Add explicit task discriminators:
```python
Literal["structural_change_segmentation"]
Literal["change_captioning"]
Literal["flood_segmentation"]
```

Add typed registrations compatible with existing registry architecture:
- `ChangeDetectionRegistration`
- `ChangeCaptionRegistration`
- `FloodSegmentationRegistration`

Temporal-specific records must make domain/preprocessing/output contracts explicit while preserving all Phase 1–3 registry compatibility.

Do not hardcode STURM/Sentinel-1 values in generic base classes. Those belong to a future audited registry entry after reproduction.

- [ ] **Step 4: Verify**

```bash
python -m pytest tests/ml/test_phase4_contracts.py -v
# discover and run the existing registry/model tests affected by satquery/registry/models.py
python -m compileall -q ml/evaluation/phase4_contracts.py satquery/registry/models.py
git diff --check
git status --short
```

- [ ] **Step 5: Commit exactly three files**

Expected diff:
```text
ml/evaluation/phase4_contracts.py
tests/ml/test_phase4_contracts.py
satquery/registry/models.py
```

```bash
git add ml/evaluation/phase4_contracts.py tests/ml/test_phase4_contracts.py satquery/registry/models.py
git commit -m "feat: validate phase 4 scientific contracts"
```

**STOP for human review. Do not begin Task 3 automatically.**

---

### Task 2A: Close Lane-Specific Source Contracts Before Learned Specialist Execution

This is a **gate**, not permission to implement learned specialists immediately. It may be resolved incrementally after Task 2 and may run in parallel with deterministic Tasks 3–5.

A learned specialist task may start only when all of its required primary contracts are `PASS`.

#### P4-E01 data closure
- Authorized OSCD download: inspect actual publisher archive layout, compute local SHA-256, freeze materialization to a declared common grid.
- If OSCD access remains unavailable, deterministic formula/grid/area lanes may still be implemented and verified with fixtures, but OSCD benchmark results remain `not_measured`.
- Do not invent a validation split from the sealed publisher test set. If no publisher validation exists, freeze a pair-level development split from the training partition before looking at test.

#### P4-E02 closure
Before Task 6:
- locally acquire/verify LEVIR-CD bytes/layout under accepted terms;
- locally verify ChangerEx checkpoint bytes/size/SHA-256 against the official model-zoo object;
- freeze exact Open-CD source revision, config chain, dependencies, preprocessing, channel convention, output semantics and official metric protocol.
S2Looking may remain `BLOCKED`; that does not block primary LEVIR-CD reproduction.

#### P4-E03 closure
Before Task 7:
- locally verify the LEVIR-CC package SHA and actual `LevirCCcaptions.json` schema/split membership;
- confirm acceptable upstream imagery-use terms for this project;
- acquire the official Chg2Cap checkpoint once, compute local SHA-256, and resolve checkpoint provenance/license;
- freeze exact source revision, vocabulary, feature extractor, decoding and official metric implementation;
- if official Torch requirements conflict with the project runtime, use an isolated external environment rather than downgrading the whole project.

#### P4-E04 closure
Before Task 9:
- locally verify STURM dataset/model archives and compute SHA-256;
- safely inspect the model archive and exact weight member;
- freeze actual Sentinel-1 band order, radiometric/scaling/normalization path from authoritative code plus real dataset bytes;
- freeze exact TensorFlow/Keras runtime versions;
- because STURM has no publisher split, derive and freeze an event-grouped development/validation policy before evaluation and keep any final holdout sealed.
Sen1Floods11 remains conditional until license and exact no-tuning transfer preprocessing are justified.

All blocker-resolution edits must be limited, evidence-backed amendments to the Task 1 audit files and reviewed before a contract changes from `BLOCKED` to `PASS`.

---

### Task 3: Add Temporal Evidence and Derived-Mask Contracts

**Files:**
- Modify: `satquery/evidence/models.py`
- Modify: `satquery/evidence/__init__.py`
- Create: `tests/evidence/test_temporal_models.py`

**Interfaces:**
- Produces: `MaskAsset`, `TemporalPairEvidence`, `ChangeMaskEvidence`, `FloodMaskEvidence`, `ChangeCaptionEvidence`, `MeasurementEvidence`, `AgreementEvidence`.
- Consumes: `ObservationState`, `AffineTransform`, `GeoBounds`, `EvidenceModelProvenance`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_change_mask_requires_ordered_distinct_observations():
    with pytest.raises(ValidationError):
        TemporalPairEvidence(t1_observation_id="obs_1", t2_observation_id="obs_1")


def test_caption_cannot_carry_measurement_fields():
    with pytest.raises(ValidationError):
        ChangeCaptionEvidence.model_validate({**valid_caption(), "area_ha": 4.2})


def test_mask_requires_binary_semantics_and_source_grid():
    evidence = ChangeMaskEvidence.model_validate(valid_change_mask())
    assert evidence.mask.value_semantics == "binary_0_1"
    assert evidence.mask.source_grid_observation_id == evidence.temporal.t1_observation_id
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/evidence/test_temporal_models.py -v
```

Expected: missing contract imports.

- [ ] **Step 3: Implement minimal contracts**

Required fields:

```python
class MaskAsset(ContractModel):
    asset_id: str
    path: str
    sha256: str
    width: int
    height: int
    crs: str | None
    transform: AffineTransform | None
    source_grid_observation_id: str
    value_semantics: Literal["binary_0_1"]
    immutable: Literal[True] = True

class TemporalPairEvidence(ContractModel):
    t1_observation_id: str
    t2_observation_id: str
    order_source: Literal["metadata", "explicit_user_mapping", "frozen_dataset_contract"]

class ChangeMaskEvidence(ContractModel):
    evidence_id: str
    task: Literal["change_localize"]
    target_class: str
    change_kind: Literal["gain", "loss", "symmetric_change"]
    temporal: TemporalPairEvidence
    mask: MaskAsset
    raw_model_score: float | None
    model: EvidenceModelProvenance | None
    tool_id: str | None
    domain: DomainAssessment
    warnings: tuple[str, ...]
    provenance: EvidenceProvenance
```

`FloodMaskEvidence` contains one source observation, water/flood label semantics, mask, model, domain, warnings, and provenance; it does not require a temporal pair. `ChangeCaptionEvidence` contains caption/model/provenance but no mask or measurement. `MeasurementEvidence` references one source evidence ID and records value, unit, method, calculation CRS, pixel count, and valid area method. `AgreementEvidence` references two mask evidence IDs and labels agreement as diagnostic, not confidence.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/evidence/test_temporal_models.py tests/inference -v
git diff --check
git add satquery/evidence tests/evidence/test_temporal_models.py
git commit -m "feat: add temporal evidence contracts"
```

---

### Task 4: Build Deterministic Temporal, Spectral, and Area Core

**Files:**
- Create: `satquery/analytics/__init__.py`
- Create: `satquery/analytics/temporal.py`
- Create: `satquery/analytics/spectral.py`
- Create: `satquery/analytics/measurement.py`
- Create: `tests/analytics/test_temporal.py`
- Create: `tests/analytics/test_spectral.py`
- Create: `tests/analytics/test_measurement.py`
- Modify: `satquery/registry/tools.yaml`

**Interfaces:**
- Produces: `prepare_common_grid(t1_path, t2_path, output_dir, *, resampling) -> AlignedPair` with two explicit derived raster paths.
- Produces: `normalized_difference(first, second, valid) -> np.ma.MaskedArray`.
- Produces: `compute_index(name, bands, valid) -> np.ma.MaskedArray`.
- Produces: `threshold_temporal_difference(t1, t2, *, threshold, valid) -> np.ndarray` as an explicitly parameterized diagnostic operator; it is **not** a generic semantic change detector and must not be benchmarked against OSCD labels unless a separately audited method defines that use.
- Produces: `measure_mask_area(mask, transform, crs, *, unit) -> MeasurementResult`.

- [ ] **Step 1: Write failing spectral tests**

```python
def test_indices_match_hand_calculation_and_preserve_nodata():
    red = np.array([[1.0, 2.0], [0.0, 4.0]])
    nir = np.array([[3.0, 2.0], [0.0, 0.0]])
    valid = np.array([[True, True], [False, True]])
    ndvi = compute_index("ndvi", {"RED": red, "NIR": nir}, valid)
    np.testing.assert_allclose(ndvi.compressed(), [0.5, 0.0, -1.0])
    assert ndvi.mask[1, 0]


def test_ndvi_rejects_unknown_or_missing_nir():
    with pytest.raises(MissingRequiredBandError):
        compute_index("ndvi", {"RED": np.ones((2, 2))}, np.ones((2, 2), bool))
```

Formula freeze:

```text
NDVI  = (NIR - RED)   / (NIR + RED)
NDWI  = (GREEN - NIR) / (GREEN + NIR)
MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
```

Denominator zero and NoData remain masked, never coerced to valid zero.

- [ ] **Step 2: Write failing grid and temporal tests**

```python
def test_equal_shape_misaligned_grids_require_reprojection(tmp_path):
    result = prepare_common_grid(t1_path, shifted_t2_path, tmp_path / "aligned", resampling="bilinear")
    assert result.reprojected
    assert result.t1_transform == result.t2_transform


def test_identity_pair_has_zero_deterministic_change():
    mask = threshold_temporal_difference(image, image, threshold=0.2, valid=valid)
    assert not mask.any()
```

Use `PairValidator` before pixelwise operations. Select the destination grid from an explicit dataset/analysis contract. Reproject continuous bands with explicit bilinear resampling and masks/labels with nearest-neighbor. Record parent hashes, source/destination grids, CRS, resampling, transform, valid-pixel handling, and whether the operation produced derived rasters. Equal shape alone never passes the grid gate.

- [ ] **Step 3: Write failing area tests**

```python
def test_projected_rotated_pixel_area_uses_affine_determinant():
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    transform = Affine(10, 2, 0, 1, -10, 0)
    result = measure_mask_area(mask, transform, "EPSG:32643", unit="m2")
    assert result.value == pytest.approx(306.0)
    assert result.positive_pixel_count == 3


def test_geographic_crs_never_squares_degrees():
    result = measure_mask_area(mask, transform, "EPSG:4326", unit="ha")
    assert result.method == "equal_area_reprojection_epsg_6933"
```

Projected metric area uses `abs(a*e - b*d)` per pixel. Geographic/non-metric CRS uses nearest-neighbor reprojection to EPSG:6933 before counting. Missing CRS raises `CrsRequiredForMeasurementError`.

- [ ] **Step 4: Verify RED, implement minimum, verify GREEN**

```bash
python -m pytest tests/analytics -v
```

Expected first run: missing modules. Implement only tested operations; no generic raster algebra framework.

- [ ] **Step 5: Register deterministic tools and commit**

Register exact IDs: `compute_ndvi_v1`, `compute_ndwi_v1`, `compute_mndwi_v1`, `temporal_difference_v1`, `compute_mask_area_v1`. Parameters use enums/ranges, not arbitrary expressions or GDAL arguments.

```bash
python -m pytest tests/analytics tests/geo -v
git diff --check
git add satquery/analytics satquery/registry/tools.yaml tests/analytics
git commit -m "feat: add deterministic temporal analytics"
```

---

### Task 5: Prepare and Evaluate P4-E01

**Purpose:** validate deterministic multispectral/temporal science without pretending that NDVI difference is generic change detection.

**Files:**
- Create `ml/evaluation/prepare_p4_e01.py`
- Create `ml/evaluation/run_p4_e01.py`
- Create `tests/ml/test_p4_e01.py`
- Add reviewed manifests/results under `experiments/phase4_temporal_analytics/p4_e01/` only when real source data are available.

**P4-E01 lanes:**
- **Lane A — formula correctness (mandatory):** NDVI, NDWI, MNDWI against hand-calculated fixtures and NoData/zero-denominator cases.
- **Lane B — grid correctness (mandatory):** common-grid preparation, identity pair, reversed-order audit, deliberate equal-shape misalignment, resampling provenance.
- **Lane C — measurement correctness (mandatory):** projected and geographic CRS area fixtures, pixel-count reconstruction.
- **Lane D — optional OSCD deterministic change benchmark:** only if the audit freezes a scientifically defensible method such as CVA/normalized spectral change and the OSCD source contract is `PASS`.

**Forbidden:** `abs(NDVI_T2 - NDVI_T1) > arbitrary_threshold` labeled as generic OSCD change.

- [ ] **Step 1: Write RED tests for real-source preparation**

The OSCD preparer must consume the **actual audited publisher layout**, not assume `T1.tif`/`T2.tif` 13-band stacks. It must preserve semantic band identity and native-resolution provenance through an explicit materialization step.

If the publisher exposes only train/test, derive any development validation split from the training partition at the temporal-pair level and freeze it before test access.

- [ ] **Step 2: Implement lanes A–C independently of OSCD benchmark availability**

Their results are deterministic verification artifacts, not learned-model benchmark metrics.

- [ ] **Step 3: Implement Lane D only if authorized by the audit**

If no defensible deterministic generic change method has been frozen:
- set Lane D to `BLOCKED_METHOD_CONTRACT` or `NOT_EVALUATED`;
- do not emit precision/recall/F1/IoU for OSCD.

If Lane D is enabled:
- save one row per temporal pair;
- save confusion counts and method/profile identity;
- reconstruct aggregate metrics from rows;
- never tune from sealed test data.

Sentinel-2 L2A examples remain `demonstration_only` and are never mixed into benchmark metrics.

- [ ] **Step 4: Verify and commit**

Run targeted analytics/P4-E01 tests and `git diff --check`. If real OSCD data are unavailable, commit implementation plus a structured blocked/not-measured artifact only if the repository's experiment policy allows such an artifact; never synthesize metrics.

---

### Task 6: Implement and Reproduce ChangerEx P4-E02

**Precondition:** LEVIR-CD and ChangerEx primary contracts are `PASS`. If either is still `BLOCKED`, stop this task.

**Files:** same planned adapter/evaluator/test paths as above.

**Principle:** reproduce the official Open-CD pipeline first; the SatQuery service must delegate to the same pinned model/config behavior rather than inventing an alternate benchmark pipeline.

- [ ] **Step 1: RED tests for domain and evidence boundaries**
Test:
- optical RGB/high-resolution contract only;
- known T1/T2 order;
- verified/common pixel grid;
- exact checkpoint hash rejection;
- source-grid mask provenance;
- model exceptions fail closed;
- no semantic claim broader than `HIGH_RES_STRUCTURAL_CHANGE`.

- [ ] **Step 2: Implement thin Open-CD adapter**
Requirements:
- pinned Open-CD revision/config chain from Task 2A;
- verify checkpoint SHA-256 before handing path to Open-CD;
- no arbitrary remote code or untracked package patching;
- official normalization/data preprocessor and two-image ordering;
- official probability/logit/decoder semantics;
- benchmark path does not introduce custom tiling/thresholding absent from the official protocol.

Production tiling for arbitrary large user rasters may be added only as a separately tested inference adaptation. It must preserve source-grid mapping and must not be described as the reproduced benchmark path.

- [ ] **Step 3: Reproduce LEVIR-CD validation**
Use the exact audited split/protocol and metric definitions. Save per-sample confusion counts/predictions sufficient to reconstruct precision/recall/F1/IoU.

If reproduction exceeds the predeclared tolerance:
- write `reproduction_failure.json`;
- do not register the model;
- do not run optional robustness;
- stop for review.

- [ ] **Step 4: Optional S2Looking robustness**
Run only if the S2Looking contract is `PASS`.
If S2Looking remains license/contract blocked:
- record robustness status as `BLOCKED`;
- do **not** fail the LEVIR-CD primary lane;
- do not replace the missing result with another unreviewed dataset.

If run, use the identical frozen checkpoint/preprocessing/decision rule; no tuning after robustness exposure.

- [ ] **Step 5: Promote only after primary reproduction**
Add registry/preprocessing entries only after LEVIR-CD reproduction passes. Promotion must state the model is a high-resolution structural/building-change specialist, not a universal change detector.

Run targeted tests, existing registry tests, `git diff --check`, commit the coherent implementation/result set, then stop for review.

---

### Task 7: Implement and Reproduce Chg2Cap P4-E03

**Precondition:** LEVIR-CC and Chg2Cap primary contracts are `PASS`, including acceptable imagery-use terms and locally verified checkpoint identity/provenance. Otherwise stop.

**Files:** same planned adapter/evaluator/test paths as above.

- [ ] **Step 1: RED tests for lane boundaries**
Require:
- ordered optical pair;
- correct LEVIR-CC-like input contract;
- empty/model-error paths fail closed;
- exact checkpoint hash;
- caption evidence contains text/provenance only, never mask/area/count/confidence;
- unknown temporal order rejects.

- [ ] **Step 2: Reproduce official Chg2Cap runtime in an isolated environment if needed**
Use the exact audited Chg2Cap source revision/checkpoint/vocabulary/feature extractor and decoding path.

Do **not** downgrade or replace the project-wide PyTorch stack merely because the official model targets an older Torch version. If required, run Chg2Cap in a pinned Kaggle/container/isolated environment and keep the integration boundary explicit.

Do not patch third-party installed packages in place without an audited, versioned source change.

- [ ] **Step 3: Prepare the real LEVIR-CC layout**
Consume:
- `LevirCCcaptions.json`
- `images/<split>/A`
- `images/<split>/B`

Preserve every reference caption for each pair and authoritative split membership. Do not invent per-pair `references.json`.

- [ ] **Step 4: Evaluate with pinned standard metrics**
Required:
- BLEU-4
- METEOR
- ROUGE-L
- CIDEr

Use the audited official/standard implementation and corpus semantics. Save generated caption + all references per pair. Prediction/reference counts must match declared sample counts.

Run predeclared T1+T1 and reversed-pair diagnostics without altering decoding.

- [ ] **Step 5: Promote only after reproduction**
If official validation cannot be reproduced within the predeclared tolerance, preserve failure evidence and stop. Do not substitute a generic VLM.

After pass, register the exact change-caption specialist/profile, run focused tests and `git diff --check`, commit, and stop for review.

RSCaMa remains absent from runtime/dependencies during core Phase 4.

---

### Task 8: Build Deterministic SAR Change and Agreement

**Files:**
- Create: `satquery/analytics/sar.py`
- Create: `tests/analytics/test_sar.py`
- Modify: `satquery/registry/tools.yaml`

**Interfaces:**
- Produces: `SarInputContract(polarizations, radiometric_domain, sensor, calibration)`.
- Produces: `sar_temporal_change(t1, t2, contract, *, threshold) -> SarChangeResult`.
- Produces: `mask_agreement(first, second, valid) -> AgreementResult`.

- [ ] **Step 1: Write failing physical-contract tests**

```python
def test_db_contract_uses_difference_not_log_ratio():
    result = sar_temporal_change(t1_db, t2_db, db_contract, threshold=3.0)
    np.testing.assert_array_equal(result.score, np.abs(t2_db - t1_db))


def test_linear_contract_uses_safe_log_ratio():
    result = sar_temporal_change(t1_linear, t2_linear, linear_contract, threshold=0.5)
    assert np.isfinite(result.score[result.valid]).all()


def test_unknown_radiometric_domain_or_polarization_rejects():
    with pytest.raises(UnknownSarSemanticsError):
        sar_temporal_change(t1, t2, unknown_contract, threshold=1.0)
```

- [ ] **Step 2: Verify RED and implement only two explicit domains**

```bash
python -m pytest tests/analytics/test_sar.py -v
```

Supported domains:

- `backscatter_db`: absolute dB difference;
- `backscatter_linear`: absolute log-ratio with audited epsilon.

Unknown/amplitude/complex/raw-DN inputs reject unless a future explicit conversion contract is added. Match polarization by semantic name; never assume channel position.

- [ ] **Step 3: Add agreement tests and implementation**

```python
def test_agreement_is_iou_not_confidence():
    result = mask_agreement(first, second, valid)
    assert result.metric == "mask_iou"
    assert result.interpretation == "agreement_not_accuracy"
```

Return intersection, union, IoU or `None` for empty union, and valid-pixel count. Agreement is diagnostic only. Do not label low agreement as either model being wrong and do not map IoU to ALLOW/WARN/ABSTAIN thresholds in Phase 4.

- [ ] **Step 4: Register tools and commit**

Add `sar_temporal_change_v1` and `mask_agreement_v1` with enum-constrained domains and polarization lists.

```bash
python -m pytest tests/analytics/test_sar.py tests/analytics/test_temporal.py -v
git diff --check
git add satquery/analytics/sar.py satquery/analytics/__init__.py \
  satquery/registry/tools.yaml tests/analytics/test_sar.py
git commit -m "feat: add deterministic SAR change evidence"
```

---

### Task 9: Implement and Reproduce STURM P4-E04

**Precondition:** STURM model + STURM-Flood primary contracts are `PASS`, including verified archive/weight SHA-256, exact Sentinel-1 radiometric/scaling/normalization behavior, and pinned TensorFlow/Keras runtime. Otherwise stop.

**Files:** same planned adapter/evaluator/test paths as above.

**Runtime rule:** STURM remains an official TensorFlow/Keras specialist if that is the audited implementation. Do not reimplement it in PyTorch for convenience.

- [ ] **Step 1: RED strict sensor/model tests**
Test exact audited constraints:
- Sentinel-1/platform requirements;
- required polarization names and order;
- calibrated/radiometric domain;
- scaling/normalization;
- GSD and input dimensions;
- NoData handling;
- exact archive/weight hashes;
- finite output probabilities/logits;
- output threshold/decoder semantics;
- RISAT and incompatible polarizations reject before model execution.

Generic `FloodBackend` types must remain sensor-neutral; Sentinel-1 restrictions belong to the STURM registration/service contract.

- [ ] **Step 2: Securely acquire/load official model**
Safely inspect/extract only allowed regular files; reject traversal, links/devices, duplicates, oversized/unexpected members. Verify outer archive and selected weight member identities before TensorFlow/Keras loading.

- [ ] **Step 3: Prepare real STURM-Flood data**
Use the actual audited dataset layout/metadata and event grouping. Because no publisher split exists, use the frozen event-level split policy from Task 2A. Never split related tiles from one event across tuning/validation boundaries if the frozen contract forbids it.

- [ ] **Step 4: Reproduce primary STURM validation**
Require nonzero evaluated samples. Report at least precision, recall, F1, IoU, event breakdown, latency, and memory/runtime information. Accuracy alone cannot pass.

Metrics must reconstruct from saved valid-pixel confusion counts.

- [ ] **Step 5: Conditional Sen1Floods11 external evaluation**
Run only when Sen1Floods11 license and exact no-tuning STURM transfer preprocessing/input semantics are proven.

Freeze actual Sen1Floods11 source behavior:
- 512×512 Sentinel-1 chips;
- VV/VH semantics as audited;
- labels: `-1` invalid, `0` non-water, `1` water;
- **exclude `-1` pixels from all confusion metrics**.

Do not reshape the dataset contract into fake `128×128 image.tif/label.tif` fixtures. Any resizing/preparation must be an explicit STURM preprocessing operation with provenance.

If transfer preprocessing cannot be justified, record:
`BLOCKED_INPUT_CONTRACT`
and keep the primary STURM result valid.

No threshold/preprocessing tuning on the external holdout.

- [ ] **Step 6: Promote only after primary reproduction**
Register STURM only after the primary STURM-Flood gate passes. Registry entry must state the exact Sentinel-1 domain and must not imply RISAT compatibility.

Run focused tests, `git diff --check`, commit, and stop for review.

---

### Task 10: Add Semantic Compatibility and Diagnostic Mask Agreement

**Phase 4 scope change:** do **not** implement an evidence-to-ALLOW/WARN/ABSTAIN policy from uncalibrated IoU bands. Phase 4 may verify whether two masks are semantically/spatially comparable and report diagnostic agreement only. Decision policy belongs to Phase 5 unless later calibrated evidence explicitly authorizes it.

**Files:**
- Create `satquery/analytics/reconciliation.py` (or a more precise compatibility module name if consistent with repository style)
- Create `tests/analytics/test_reconciliation.py`

**Interfaces:**
- `assess_mask_compatibility(first, second) -> CompatibilityResult`
- optional `diagnostic_mask_agreement(first, second, valid) -> AgreementEvidence`

- [ ] **Step 1: RED compatibility tests**
Reject or mark non-comparable when:
- grids differ and have not been explicitly aligned;
- target phenomena differ (for example post-event water extent vs generic temporal change);
- time semantics differ;
- mask value semantics differ;
- source observations cannot be related.

A `FloodMaskEvidence` and generic `ChangeMaskEvidence` are **not** automatically comparable merely because both are binary rasters.

- [ ] **Step 2: Diagnostic agreement only**
For compatible masks report:
- intersection;
- union;
- IoU (or `None` for empty union);
- valid-pixel count;
- interpretation = `agreement_not_accuracy`.

Do not expose:
- aggregate confidence;
- `IoU < X -> ABSTAIN`;
- `IoU < Y -> WARNING`;
- claims that one model is correct.

Missing optional evidence should simply yield `not_available/not_comparable`, not a fabricated policy decision.

- [ ] **Step 3: Verify and commit**
Run analytics tests and `git diff --check`, then commit the diagnostic/compatibility implementation. Stop for review.

---

### Task 11: Harden and Extend the Existing Kaggle Runner

**Do not build another execution framework.** First audit the runner that exists at the current branch; preserve working Phase 1–3 behavior.

**Files:**
- modify `scripts/kaggle/experiments.yaml`
- create the four Phase 4 notebooks
- modify `tests/test_kaggle_runner.py`
- update `scripts/kaggle/README.md` / `docs/KAGGLE.md` only as needed
- modify `scripts/kaggle/runner.py` only if required to satisfy the exact-path/stale-artifact contract below.

- [ ] **Step 1: RED runner tests**
Require:
- unique new kernel slug and `remote_output_dir` per Phase 4 experiment;
- no reuse of historical Phase 4 slugs;
- exact Git SHA injection;
- clean-run provenance;
- repository clone/checkout exactly once per notebook run;
- exact remote result path retrieval, never first matching basename or unrestricted recursive search;
- ambiguous/missing result path fails closed;
- current results are staged/validated atomically before publication;
- a failed new run cannot leave an older success metric looking current;
- success and failure artifact sets are distinguishable;
- dry-run launches nothing.

- [ ] **Step 2: Register only experiments whose execution path exists**
Use the exact audited dependency/runtime needs:
- P4-E01: CPU unless a concrete reason needs GPU;
- P4-E02: Open-CD/PyTorch environment;
- P4-E03: isolated audited Chg2Cap environment;
- P4-E04: isolated TensorFlow/Keras STURM environment.

A `BLOCKED` model contract may have a dry-run preparation entry, but no canonical execution should be launched until the contract is `PASS`.

- [ ] **Step 3: Keep notebooks thin**
Each notebook may only:
```text
report environment
→ clone exact repository once
→ checkout injected SHA
→ assert clean checkout
→ install/use audited environment
→ verify exact data/model identities
→ call one tested Python evaluator
→ verify exact declared output files
→ fail nonzero on scientific/runtime failure
```

No manually authored predictions, metrics, thresholds, scientific transformations, fallback datasets, or placeholder PASS values in cells.

- [ ] **Step 4: Dry-run verification**
Run runner tests plus all four `--dry-run` commands. Confirm exact output directories and no push/run.

- [ ] **Step 5: Commit**
Commit only the runner/notebook/registry/doc changes actually needed. Stop before any external run.

---

### Task 12: Execute, Retrieve, and Audit External Results

Run only lanes whose required contracts and implementation gates are `PASS`.

**Canonical preconditions:**
```bash
git status --short
git rev-parse HEAD
python -m pytest
git diff --check
```
Worktree must be clean. Never use `--allow-dirty` for canonical evidence.

**Execution independence:**
- P4-E01 deterministic lanes may proceed independently of learned specialists.
- P4-E02 primary requires LEVIR-CD + ChangerEx PASS.
- P4-E03 requires LEVIR-CC + Chg2Cap PASS.
- P4-E04 primary requires STURM-Flood + STURM PASS.
- S2Looking and Sen1Floods11 optional/conditional lanes are skipped or explicitly blocked without invalidating a successful primary lane.

- [ ] **Run one experiment at a time and audit before the next promotion decision**

For each canonical run:
1. record launched Git SHA/kernel identity;
2. retrieve only exact declared output paths;
3. verify runner metadata and dirty flag;
4. verify dataset/checkpoint/profile hashes against frozen contracts;
5. verify split is allowed and sealed test was not accessed;
6. verify sample count > 0 for measured results;
7. verify prediction row count == sample count;
8. reconstruct aggregate metrics from saved rows/counts;
9. inspect representative predictions/evidence, not only the aggregate score;
10. verify output does not contain source datasets, checkpoints, cloned repository files or caches;
11. preserve structured failure artifacts as failures.

Do not run robustness/holdout lanes after a failed primary reproduction.

Commit each independently audited small result in a separate experiment commit. Never create a success artifact manually.

---

### Task 13: Freeze Phase 4 Decisions and Close Out

`PHASE_4_CLOSEOUT.json` is created **only here**.

**Mandatory core closeout gates:**
- P4-E01 lanes A–C have measured deterministic verification evidence. Lane D may be `NOT_EVALUATED/BLOCKED` if no defensible OSCD benchmark method/data contract is available.
- P4-E02 primary LEVIR-CD/ChangerEx reproduction passes.
- P4-E03 primary LEVIR-CC/Chg2Cap reproduction passes.
- P4-E04 primary STURM-Flood/STURM reproduction passes.
- Optional S2Looking/Sen1Floods11 absence does not block `COMPLETE`, but their limitations/status must be explicit.
- No model is promoted beyond its audited sensor/domain limits.
- No Cartosat/RISAT transfer claim is made without measured evidence.

**Files:** create closeout; update development/evaluation/architecture/failure docs only where actual public contracts changed; update registries only for specialists already promoted by measured evidence.

- [ ] **Step 1: Add closeout reconstruction tests**
Every referenced artifact must exist and hash-match. Every `SUPPORTED*` capability must have nonzero measured evidence and reconstructible metrics appropriate to that capability.

- [ ] **Step 2: Build closeout from locally audited artifacts only**
Record:
- exact Git SHA(s);
- audit-contract hashes;
- dataset/model/profile identities;
- measured sample counts and metrics;
- sealed-test status;
- domain limitations;
- optional/blocked lanes;
- accepted/rejected promotions;
- explicit Cartosat/RISAT non-generalization;
- RSCaMa and Modified Sen1Floods11 exclusions;
- any no-training decision.

- [ ] **Step 3: Set phase state truthfully**
`COMPLETE` only when all mandatory core gates above pass.
Otherwise `IN_PROGRESS` or `BLOCKED` with exact reasons.

Never rewrite frozen historical artifacts to make results agree.

- [ ] **Step 4: Full release verification**
```bash
python -m pytest
python -m compileall -q satquery ml apps scripts
git diff --check
git status --short
```

Inspect tracked file sizes and staged diff; verify no credentials, source datasets, checkpoints, caches, or accidental large artifacts.

- [ ] **Step 5: Commit closeout**
Stage only files actually changed and commit:
```text
docs: close phase 4 temporal analytics
```

---

## Scientific Acceptance Gates

### P4-E01 — deterministic multispectral/temporal core

Pass core lanes only when:
- semantic bands are explicit; no positional guessing;
- native resolution, NoData and derived-grid provenance are explicit;
- hand-calculated NDVI/NDWI/MNDWI fixtures pass;
- identity/reversal/misalignment grid controls pass;
- area fixtures reconstruct exactly within declared tolerance;
- Sentinel-2 L2A examples remain demonstrations.

OSCD segmentation precision/recall/F1/IoU are reported **only** if optional Lane D has a frozen, defensible deterministic change method and a `PASS` OSCD data contract. NDVI-difference thresholding is not generic change detection.

### P4-E02 — ChangerEx

Primary pass only when:
- LEVIR-CD data contract and ChangerEx model contract are `PASS`;
- exact official source/config/checkpoint bytes are verified;
- official LEVIR-CD validation is reproduced within predeclared tolerance;
- saved rows reconstruct metrics;
- masks map to the declared source/common grid;
- identity/misalignment/domain controls pass;
- capability is labeled high-resolution structural/building change, not universal change.

S2Looking is optional/non-gating. If unavailable or license-blocked, report robustness `BLOCKED/NOT_EVALUATED`; do not fail an otherwise valid primary reproduction.

### P4-E03 — Chg2Cap

Pass only when:
- LEVIR-CC imagery/package use and Chg2Cap checkpoint provenance/license are acceptable;
- exact checkpoint/source/vocabulary/feature-extractor/decoder are pinned;
- official validation protocol is reproduced or discrepancy is explicitly rejected/accepted under the frozen tolerance rule;
- every generated caption and full reference set is preserved;
- BLEU-4/METEOR/ROUGE-L/CIDEr use pinned standard implementations;
- identity/reversal diagnostics are reported;
- caption evidence contains no mask, area, count or manufactured confidence.

### P4-E04 — STURM

Primary pass only when:
- exact STURM Sentinel-1 input/radiometric/scaling contract and TensorFlow/Keras runtime are verified;
- exact archive/weight SHA-256 is verified;
- event-grouped STURM validation has nonzero samples and reconstructible precision/recall/F1/IoU;
- deterministic SAR processing refuses unknown radiometric domains/polarizations;
- learned/deterministic agreement is diagnostic only;
- RISAT incompatibility routes away from STURM rather than forcing Sentinel-1 assumptions.

Sen1Floods11 is conditional/non-gating. If run, `-1` pixels are excluded from metrics and the STURM preprocessing/threshold remains unchanged. Otherwise report `BLOCKED_INPUT_CONTRACT`/license blocker explicitly.

## Stop Conditions

Stop the affected lane and preserve evidence when any occurs:

- required dataset/model contract is still `BLOCKED`;
- upstream license/imagery rights are unacceptable or unresolved for the intended execution;
- checkpoint authority, bytes, local SHA-256, revision, or provenance cannot be established;
- preprocessing, band order, radiometric/scaling domain, T1/T2 semantics, or label semantics are unknown;
- official source requires unsafe arbitrary remote code or untracked installed-package patching;
- validation cannot be isolated from sealed test data;
- prediction/stat counts do not match sample count;
- metrics cannot be reconstructed;
- sealed test/robustness/holdout data influence tuning;
- official reproduction exceeds the predeclared tolerance;
- output retrieval is ambiguous, basename-based, stale, or from the wrong experiment path;
- canonical run uses dirty/unrecorded source state;
- zero samples are evaluated;
- optional robustness/holdout failure is incorrectly being used to rewrite the primary result.

A stop in one independent lane does not automatically block unrelated Phase 4 lanes.

## Deferred Work

- RSCaMa integration: only after Phase 4 closeout if Chg2Cap is scientifically insufficient and its Linux/Mamba/runtime risk is separately accepted.
- ChangerEx adaptation/retraining on S2Looking: only after S2Looking rights are resolved and a new untouched evaluation source is frozen first.
- Any STURM adaptation to Sen1Floods11 or RISAT: only through an explicit sensor/radiometric transfer experiment with untouched holdout.
- Numeric agreement-to-decision thresholds or aggregate confidence: Phase 5+ only after calibration evidence; Phase 4 agreement remains diagnostic.
- Temporal API routes, natural-language routing, answer composition, UI overlays and downloadable reports: canonical Phases 5–6.
- Generic framework for arbitrary spectral indices/models: only when a second concrete implementation proves the explicit registered tools insufficient.
