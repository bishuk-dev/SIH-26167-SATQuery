# Phase 4 Clean Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible temporal remote-sensing specialists for deterministic multispectral analysis, high-resolution optical change masks, change captions, and SAR flood/change evidence without allowing prose to manufacture spatial or numeric claims.

**Architecture:** Keep four independent scientific paths: deterministic multispectral processing, ChangerEx structural-change segmentation, Chg2Cap change captioning, and STURM Sentinel-1 flood segmentation plus deterministic SAR change. Each path emits typed evidence. Shared verification checks temporal order, grids, sensor contracts, provenance, masks, and measurements before any evidence is reconciled or explained.

**Tech Stack:** Python 3.11+, NumPy, Rasterio, Affine, Pydantic 2, PyTorch, existing Kaggle runner, and only audit-approved model-specific dependencies.

**Spec:** User-approved Phase 4 clean-rebuild brief, captured in “Approved Phase 4 design” below.

## Global Constraints

- Preserve all frozen historical artifacts under `experiments/phase2*`, `experiments/phase3*`, and `experiments/phase4_bigearthnet_multisensor/`; never rewrite them to fit this rebuild.
- Treat this work as canonical Phase 4. Historical Phase 4A–4F and Phase 5A remain canonical Phase 3 evidence.
- Current branch `phase4-clean-rebuild` already has unrelated staged SHA-256 refactors. Isolate or commit those separately before Task 1. Never include them in the audit-only commit.
- Begin implementation from a clean worktree created with `superpowers:using-git-worktrees`. Record exact starting Git SHA.
- First implementation-series commit after this planning document contains audit artifacts only. No runtime code, registry entry, dependency, notebook, or benchmark result belongs in that commit.
- Preferred datasets/models remain candidates until audit status is `PASS`. A failed license, provenance, checkpoint, preprocessing, or dependency audit changes status to `BLOCKED`; it never gets guessed around.
- Use original papers, official repositories, DOI/Zenodo records, or publisher dataset cards as provenance authority. Kaggle and Hugging Face mirrors are transport only unless maintained by the publisher.
- Do not accept DeepWiki, ResearchGate, Kaggle descriptions, or an ephemeral Hugging Face pull-request ref as sole authority for metrics, licenses, revisions, or preprocessing.
- Compute SHA-256 locally for every downloaded checkpoint and source archive. Record publisher MD5 only as an additional transport check, never as the sole SatQuery identity.
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
| High-resolution robustness | S2Looking | Frozen ChangerEx-R18, unchanged | Robustness gate |
| Learned change description | LEVIR-CC | Chg2Cap | Primary |
| SAR flood/water extent | STURM-Flood | Official Sentinel-1 U-Net | Primary |
| SAR independent validation | Sen1Floods11 | Frozen STURM model plus deterministic SAR evidence | Secondary holdout |
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
- `P4-E02`: ChangerEx-R18 structural-change validation on LEVIR-CD; unchanged robustness evaluation on S2Looking.
- `P4-E03`: Chg2Cap validation on LEVIR-CC.
- `P4-E04`: STURM Sentinel-1 U-Net validation on STURM-Flood; unchanged external evaluation on Sen1Floods11; deterministic SAR agreement audit.

### Promotion rule

A specialist enters `models/registry.yaml` only after all are true:

1. official source and license are verified;
2. exact source revision is immutable;
3. checkpoint bytes, size, SHA-256, and publisher hash are verified;
4. preprocessing and output semantics are verified from authoritative source/code;
5. local contract/smoke tests pass;
6. official pretrained validation is reproduced or discrepancy is documented and accepted;
7. domain limits and failure outcomes are explicit.

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

### Task 0: Isolate Existing Work and Freeze Baseline

**Files:** None.

**Interfaces:**
- Consumes: current `phase4-clean-rebuild` worktree.
- Produces: clean isolated worktree and baseline verification record in shell history.

- [ ] **Step 1: Inspect existing staged work**

```bash
git status --short
git diff --cached --stat
git diff --cached
git log --oneline -10
```

Expected: staged SHA-256 refactors are visible. Do not reset, amend, or mix them into Phase 4 audit.

- [ ] **Step 2: Ask owner to commit or stash unrelated work if still present**

Do not choose on the owner's behalf. After owner action, require:

```bash
git status --short
```

Expected: clean output.

- [ ] **Step 3: Create isolated worktree**

Use `superpowers:using-git-worktrees`, then record:

```bash
git rev-parse HEAD
git branch --show-current
git status --short
```

Expected: exact start SHA, `phase4-clean-rebuild` or owner-approved child branch, clean status.

- [ ] **Step 4: Run baseline checks**

```bash
python -m pytest
git diff --check
```

Expected: all current tests pass; any pre-existing warning is recorded without reclassifying failure.

---

### Task 1: Audit Dataset and Model Contracts — First Commit Only

**Files:**
- Create: `experiments/phase4_temporal_analytics/README.md`
- Create: `experiments/phase4_temporal_analytics/dataset_contracts.yaml`
- Create: `experiments/phase4_temporal_analytics/model_contracts.yaml`
- Create: `experiments/phase4_temporal_analytics/experiment_plan.yaml`

**Interfaces:**
- Consumes: authoritative sources listed by the approved design.
- Produces: human-readable, machine-parseable audit records with `PASS` or `BLOCKED` status. Later tasks consume only `PASS` records.

- [ ] **Step 1: Audit each dataset from authoritative sources**

For OSCD, Sentinel-2 L2A, LEVIR-CD, S2Looking, LEVIR-CC, STURM-Flood, and Sen1Floods11, record this exact schema:

```yaml
id: oscd
status: PASS|BLOCKED
authority:
  title: "..."
  landing_url: "https://..."
  repository_url: "https://..."
  doi: "..."                 # null only when authority publishes no DOI
  version: "..."
license:
  dataset_license: "..."
  imagery_terms: "..."
  redistribution_allowed: true|false
transport:
  url: "https://..."
  mirror_role: official|transport_only
  expected_files: []
  expected_size_bytes: 0
  publisher_hashes: {}
  locally_verified_sha256: null
contract:
  modalities: []
  sensors: []
  bands_or_polarizations: []
  radiometric_domain: "..."
  spatial_resolution_m: "..."
  pair_order: T1_then_T2
  registration_claim: "..."
  labels: "..."
  label_limitations: []
splits:
  authority_defined: true|false
  grouping_unit: temporal_pair
  train: "..."
  validation: "..."
  test: "..."
  test_sealed: true
references:
  - url: "https://..."
    supports: "..."
blockers: []
```

Rules:

- OSCD labels validate generic/urban structural change, not water or vegetation semantics.
- Sentinel-2 L2A pairs are operational demonstrations, not benchmark substitutes.
- S2Looking is a frozen robustness source. Never tune from it.
- Sen1Floods11 is an unchanged external source. Never tune thresholds from it.
- Modified Sen1Floods11 is recorded as `excluded_from_core`; do not download it.

- [ ] **Step 2: Audit each model and checkpoint**

For ChangerEx-R18, Chg2Cap, STURM Sentinel-1 U-Net, and optional RSCaMa, record:

```yaml
id: changerex_r18_levircd
status: PASS|BLOCKED
role: primary|optional_challenger
source:
  repository_url: "https://..."
  revision: "40_hex_git_commit"
  license: "..."
checkpoint:
  authority_url: "https://..."
  filename: "..."
  size_bytes: 0
  publisher_hashes: {}
  locally_verified_sha256: "64_hex_or_null"
contract:
  task: "..."
  training_domain: "..."
  sensor_domain: []
  band_order: []
  radiometric_domain: "..."
  input_shape: []
  normalization: "..."
  temporal_order: T1_then_T2
  output_semantics: "..."
  threshold_or_decoder: "..."
dependencies:
  python: "..."
  pytorch: "..."
  packages: []
  operating_system: []
official_evaluation:
  dataset: "..."
  split: "..."
  command: "..."
  metrics: {}
blockers: []
```

Verify rather than assume:

- ChangerEx candidate filename, `136880383` bytes, and SHA-256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`.
- Whether `refs/pr/1` resolves to immutable official checkpoint provenance. If not, block promotion.
- Chg2Cap exact official checkpoint, source revision, feature extractor weights, vocabulary, decoder, and metric implementation.
- STURM archive publisher MD5 `14a046d9d7965f2a3c511acb1bbca57b`; download bytes once, verify MD5, compute SHA-256, inspect archive safely, and identify actual weight file.
- RSCaMa remains `optional_challenger` and `not_authorized_for_core_execution` even if its audit passes.

- [ ] **Step 3: Freeze experiment contracts**

Each `P4-E0x` record must define:

```yaml
experiment_id: P4-E01
hypothesis: "..."
datasets: []
model_or_tools: []
sample_selection:
  seed: 0
  grouping_unit: temporal_pair
  development_split: validation
  robustness_split: null
  sealed_splits: [test]
preprocessing_profile_ids: []
metrics: []
sanities: []
output_files: []
pass_condition: "..."
stop_conditions: []
```

Required metrics/sanities:

- `P4-E01`: valid-pixel index error, change precision/recall/F1/IoU, T1+T1, T2+T1, deliberate misalignment, area fixture error.
- `P4-E02`: precision/recall/F1/IoU, official split identity, T1+T1 false-positive rate, reversed-pair behavior, LEVIR-to-S2Looking absolute/relative degradation, latency, peak VRAM.
- `P4-E03`: BLEU-4, METEOR, ROUGE-L, CIDEr, caption row count, T1+T1 sanity, reversed-pair directional audit, latency, peak VRAM.
- `P4-E04`: flood precision/recall/F1/IoU, STURM-to-Sen1Floods11 degradation, T1+T1 false-positive rate for deterministic change, learned/deterministic agreement IoU, abstention count, latency, peak VRAM.

- [ ] **Step 4: Review audit status before code**

```bash
python - <<'PY'
from pathlib import Path
import yaml
root = Path("experiments/phase4_temporal_analytics")
for name in ("dataset_contracts.yaml", "model_contracts.yaml", "experiment_plan.yaml"):
    value = yaml.safe_load((root / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict), name
print("phase4 audit YAML valid")
PY
rg -n "unknown_to_fill|replace_me|FIXME" experiments/phase4_temporal_analytics
```

Expected: YAML valid; `rg` finds nothing. Legitimate unresolved facts use `status: BLOCKED` plus explicit blockers, not placeholders.

- [ ] **Step 5: Commit audit artifacts only**

```bash
git status --short
git add experiments/phase4_temporal_analytics/README.md \
  experiments/phase4_temporal_analytics/dataset_contracts.yaml \
  experiments/phase4_temporal_analytics/model_contracts.yaml \
  experiments/phase4_temporal_analytics/experiment_plan.yaml
git diff --cached --name-only
git commit -m "docs: freeze phase 4 source contracts"
```

Expected staged names: exactly four audit files. Stop after this commit for human review. Any primary model marked `BLOCKED` prevents its implementation task.

---

### Task 2: Validate Audit and Registry Contracts

**Files:**
- Create: `ml/evaluation/phase4_contracts.py`
- Create: `tests/ml/test_phase4_contracts.py`
- Modify: `satquery/registry/models.py`

**Interfaces:**
- Produces: `load_phase4_contracts(root: Path) -> Phase4ContractSet`.
- Produces: typed `ChangeDetectionRegistration`, `ChangeCaptionRegistration`, and `FloodSegmentationRegistration`.
- Consumes: exact revisions/hashes/profile fields from Task 1.

- [ ] **Step 1: Write failing audit-contract tests**

```python
def test_primary_contracts_cannot_pass_without_sha256_and_license(tmp_path):
    audit = write_contract(tmp_path, status="PASS", license=None, sha256=None)
    with pytest.raises(ValueError):
        load_phase4_contracts(audit.parent)


def test_blocked_contract_is_valid_but_not_runnable(tmp_path):
    contracts = load_phase4_contracts(write_blocked_contract_set(tmp_path))
    assert not contracts.models["chg2cap_levircc"].runnable
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/ml/test_phase4_contracts.py -v
```

Expected: import/function failure because module does not exist.

- [ ] **Step 3: Implement minimal strict loader**

Implement frozen Pydantic records. Enforce:

- no unknown fields;
- `PASS` primary records require authority, license, revision, size, SHA-256, semantics, preprocessing, and split policy;
- `BLOCKED` records require at least one blocker;
- test split is sealed;
- robustness sources cannot be development sources;
- model records reference existing dataset/profile IDs.

- [ ] **Step 4: Add temporal registry union tests, then schemas**

```python
def test_temporal_registry_rejects_missing_domain_contract(tmp_path):
    path = write_temporal_registry(tmp_path, training_domain=None)
    with pytest.raises(ValidationError):
        load_model_registry(path)
```

Add only fields proven by Task 1. Keep task discriminators explicit:

```python
Literal["structural_change_segmentation"]
Literal["change_captioning"]
Literal["flood_segmentation"]
```

Do not add model entries yet. Registry promotion occurs after Tasks 6, 7, and 9 reproduce baselines.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/ml/test_phase4_contracts.py tests/ingestion/test_models.py -v
git diff --check
git add ml/evaluation/phase4_contracts.py tests/ml/test_phase4_contracts.py satquery/registry/models.py
git commit -m "feat: validate phase 4 scientific contracts"
```

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
- Produces: `threshold_temporal_difference(t1, t2, *, threshold, valid) -> np.ndarray`.
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

Use `PairValidator` before raster reads. Reproject continuous bands with explicit bilinear resampling and masks/labels with nearest-neighbor. Record parent hashes, source/destination grids, CRS, resampling, and transform.

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

**Files:**
- Create: `ml/evaluation/prepare_p4_e01.py`
- Create: `ml/evaluation/run_p4_e01.py`
- Create: `tests/ml/test_p4_e01.py`
- Create after preparation: `experiments/phase4_temporal_analytics/p4_e01/manifest.json`
- Create after validation: `experiments/phase4_temporal_analytics/p4_e01/validation_result.json`
- Create after validation: `experiments/phase4_temporal_analytics/p4_e01/validation_predictions.jsonl`

**Interfaces:**
- Produces: `prepare_oscd_manifest(source_root: Path, contract: DatasetContract) -> dict`.
- Produces: `run_p4_e01(manifest: Path, data_root: Path, output_dir: Path, split: Literal["validation"]) -> dict`.
- Consumes: deterministic functions from Task 4.

- [ ] **Step 1: Write failing manifest tests**

```python
def test_oscd_manifest_preserves_all_13_semantic_bands_and_pairs():
    manifest = prepare_oscd_manifest(fixture_root, contract)
    assert manifest["band_order"] == contract.contract.bands_or_polarizations
    assert all(row["pair_id"] and row["t1"] != row["t2"] for row in manifest["samples"])


def test_runner_refuses_test_split():
    with pytest.raises(SealedTestAccessError):
        run_p4_e01(manifest, data_root, output_dir, split="test")
```

- [ ] **Step 2: Verify RED and implement manifest validation**

```bash
python -m pytest tests/ml/test_p4_e01.py -v
```

Materialization must validate expected paths/hashes, 13-band identity, pair order, label semantics, dimensions, CRS/transforms, and pair grouping. No full download occurs without explicit `--allow-download` and accepted source terms.

- [ ] **Step 3: Implement deterministic evaluator**

Save one JSONL row per pair with source hashes, selected bands, grid operation, threshold source, pixel counts, confusion counts, and artifact paths. Reconstruct aggregate precision/recall/F1/IoU from rows in test.

Do not claim NDVI/NDWI/MNDWI benchmark accuracy from OSCD labels. Evaluate deterministic formula/grid correctness separately. Mark selected Sentinel-2 L2A examples `demonstration_only` and do not mix them into OSCD metrics.

- [ ] **Step 4: Run local fixture and approved validation**

```bash
python -m pytest tests/ml/test_p4_e01.py tests/analytics -v
python -m ml.evaluation.prepare_p4_e01 --source-root "$OSCD_ROOT" --output experiments/phase4_temporal_analytics/p4_e01/manifest.json
python -m ml.evaluation.run_p4_e01 --manifest experiments/phase4_temporal_analytics/p4_e01/manifest.json --data-root "$OSCD_ROOT" --output-dir experiments/phase4_temporal_analytics/p4_e01/results --split validation
```

If data are unavailable, commit implementation and a structured `blocked.json`; do not synthesize metrics.

- [ ] **Step 5: Audit and commit**

```bash
python -m pytest tests/ml/test_p4_e01.py -v
git diff --check
git add ml/evaluation/prepare_p4_e01.py ml/evaluation/run_p4_e01.py tests/ml/test_p4_e01.py experiments/phase4_temporal_analytics/p4_e01
git commit -m "feat: evaluate deterministic temporal analytics"
```

---

### Task 6: Implement and Reproduce ChangerEx P4-E02

**Files:**
- Create: `satquery/inference/change_detection.py`
- Modify: `satquery/inference/config.py`
- Modify: `satquery/inference/exceptions.py`
- Create: `ml/evaluation/prepare_p4_e02.py`
- Create: `ml/evaluation/run_p4_e02.py`
- Create: `tests/inference/test_change_detection.py`
- Create: `tests/ml/test_p4_e02.py`

**Interfaces:**
- Produces: `StructuralChangeBackend.predict(t1_rgb, t2_rgb) -> np.ndarray` protocol.
- Produces: `ChangerExBackend` using exact audited Open-CD revision/checkpoint.
- Produces: `StructuralChangeService.detect(t1, t2) -> ChangeMaskEvidence`.
- Produces: `run_p4_e02(..., dataset: Literal["levir_cd", "s2looking"], split: Literal["validation", "robustness"])`.

- [ ] **Step 1: Write failing service tests with a deterministic fake backend**

```python
def test_structural_change_service_emits_source_grid_mask(fake_pair, backend):
    evidence = service(backend).detect(*fake_pair)
    assert evidence.target_class == "high_res_structural_change"
    assert evidence.mask.source_grid_observation_id == fake_pair[0].observation_id


def test_structural_model_rejects_sar_and_unverified_alignment(fake_sar_pair):
    with pytest.raises(ModelInputUnsupportedError):
        service(backend).detect(*fake_sar_pair)
```

Also test 512×512 tiling, edge padding, deterministic overlap stitching, nearest-neighbor mask restoration, T1/T2 order, and exact checkpoint hash rejection.

- [ ] **Step 2: Verify RED and implement thin adapter**

```bash
python -m pytest tests/inference/test_change_detection.py -v
```

Adapter requirements:

- lazy import official Open-CD code from audit-approved installation;
- no dynamic remote-code execution;
- exact checkpoint hash before `torch.load`;
- load weights only with safe supported mode where checkpoint format permits;
- model input normalization exactly matches audited config;
- output only binary structural-change probability/mask;
- no semantic claims beyond `HIGH_RES_STRUCTURAL_CHANGE`;
- domain warning outside audited RGB/GSD/sensor range.

- [ ] **Step 3: Write evaluator reconstruction tests and implement preparation**

```python
def test_p4_e02_metrics_reconstruct_from_prediction_rows(tmp_path):
    result = run_fixture_evaluation(tmp_path)
    rows = read_jsonl(result.prediction_path)
    assert reconstruct_binary_metrics(rows) == result.metrics


def test_s2looking_cannot_change_threshold_or_profile():
    with pytest.raises(ValueError):
        run_p4_e02(..., dataset="s2looking", threshold=0.4)
```

LEVIR-CD validation chooses no new model weights. Any allowed binary decision threshold must come from official config or be frozen on LEVIR validation before S2Looking. S2Looking consumes the exact frozen checkpoint/profile/threshold.

- [ ] **Step 4: Reproduce official validation before robustness**

```bash
python -m ml.evaluation.prepare_p4_e02 --dataset levir_cd --source-root "$LEVIR_CD_ROOT" --output experiments/phase4_temporal_analytics/p4_e02/levir_manifest.json
python -m ml.evaluation.run_p4_e02 --dataset levir_cd --manifest experiments/phase4_temporal_analytics/p4_e02/levir_manifest.json --data-root "$LEVIR_CD_ROOT" --output-dir outputs/p4_e02_levir --split validation
```

Compare only matching metric definitions/splits. If discrepancy exceeds the predeclared tolerance in `experiment_plan.yaml`, stop and write `reproduction_failure.json`; do not run S2Looking or register model.

- [ ] **Step 5: Run unchanged robustness gate**

```bash
python -m ml.evaluation.prepare_p4_e02 --dataset s2looking --source-root "$S2LOOKING_ROOT" --output experiments/phase4_temporal_analytics/p4_e02/s2looking_manifest.json
python -m ml.evaluation.run_p4_e02 --dataset s2looking --manifest experiments/phase4_temporal_analytics/p4_e02/s2looking_manifest.json --data-root "$S2LOOKING_ROOT" --output-dir outputs/p4_e02_s2looking --split robustness
```

Report raw score, absolute drop, relative drop, sensor/GSD shift, latency, and peak VRAM. No post-robustness tuning.

- [ ] **Step 6: Promote only after reproduction, then commit**

Add exact `structural_change_segmentation` model/profile registry entries only if gate passes. Copy reviewed small results into `experiments/phase4_temporal_analytics/p4_e02/results/`.

```bash
python -m pytest tests/inference/test_change_detection.py tests/ml/test_p4_e02.py tests/ingestion/test_models.py -v
git diff --check
git add satquery/inference/change_detection.py satquery/inference/config.py satquery/inference/exceptions.py \
  ml/evaluation/prepare_p4_e02.py ml/evaluation/run_p4_e02.py \
  tests/inference/test_change_detection.py tests/ml/test_p4_e02.py \
  models/registry.yaml satquery/registry/models.py satquery/registry/preprocessing.yaml \
  experiments/phase4_temporal_analytics/p4_e02
git commit -m "feat: add validated structural change specialist"
```

---

### Task 7: Implement and Reproduce Chg2Cap P4-E03

**Files:**
- Create: `satquery/inference/change_captioning.py`
- Create: `ml/evaluation/prepare_p4_e03.py`
- Create: `ml/evaluation/run_p4_e03.py`
- Create: `tests/inference/test_change_captioning.py`
- Create: `tests/ml/test_p4_e03.py`

**Interfaces:**
- Produces: `ChangeCaptionBackend.caption(t1_rgb, t2_rgb) -> str` protocol.
- Produces: `Chg2CapBackend` using exact audited source/checkpoint/vocabulary.
- Produces: `ChangeCaptionService.describe(t1, t2) -> ChangeCaptionEvidence`.
- Produces: `run_p4_e03(..., split: Literal["validation"]) -> CaptionEvaluationResult`.

- [ ] **Step 1: Write failing lane-boundary tests**

```python
def test_caption_service_preserves_pair_order_and_returns_text_only(pair, backend):
    evidence = service(backend).describe(*pair)
    assert evidence.temporal.t1_observation_id == pair[0].observation_id
    assert not hasattr(evidence, "measurement")


def test_caption_service_rejects_unknown_temporal_order(pair_without_dates):
    with pytest.raises(TemporalOrderUnknownError):
        service(backend).describe(*pair_without_dates)
```

Also test empty caption, model exception, wrong modality, bad alignment, and exact checkpoint hash.

- [ ] **Step 2: Verify RED and implement minimal adapter**

```bash
python -m pytest tests/inference/test_change_captioning.py -v
```

Use official Chg2Cap code/config at audited revision. Do not modify third-party CLIP/model internals in-place. If official source cannot load without patching installed packages, mark model `BLOCKED` and stop P4-E03 rather than creating an untracked patch.

- [ ] **Step 3: Write manifest and metric integrity tests**

```python
def test_levir_cc_pair_and_all_captions_stay_in_one_split():
    manifest = prepare_levir_cc(fixture_root, contract)
    assert no_pair_crosses_splits(manifest)


def test_caption_metric_input_counts_match_declared_samples(tmp_path):
    result = evaluate_fixture_captions(tmp_path)
    assert result.prediction_count == count_jsonl(result.prediction_path)
    assert result.reference_count == sum(len(row["references"]) for row in read_jsonl(result.prediction_path))
```

Pin the exact official/standard BLEU-4, METEOR, ROUGE-L, and CIDEr evaluator revision. Save generated caption and all references per pair. Do not substitute custom approximate metrics under official names.

- [ ] **Step 4: Reproduce validation**

```bash
python -m ml.evaluation.prepare_p4_e03 --source-root "$LEVIR_CC_ROOT" --output experiments/phase4_temporal_analytics/p4_e03/manifest.json
python -m ml.evaluation.run_p4_e03 --manifest experiments/phase4_temporal_analytics/p4_e03/manifest.json --data-root "$LEVIR_CC_ROOT" --output-dir outputs/p4_e03_chg2cap --split validation
```

Run T1+T1 and reversed-pair controls on predeclared validation samples. Controls diagnose temporal sensitivity; they do not alter decoding.

- [ ] **Step 5: Promote only after reproduction, then commit**

Add exact `change_captioning` model/profile registry entries only after pass. Keep RSCaMa absent from runtime and dependency files.

```bash
python -m pytest tests/inference/test_change_captioning.py tests/ml/test_p4_e03.py tests/ingestion/test_models.py -v
git diff --check
git add satquery/inference/change_captioning.py ml/evaluation/prepare_p4_e03.py \
  ml/evaluation/run_p4_e03.py tests/inference/test_change_captioning.py tests/ml/test_p4_e03.py \
  models/registry.yaml satquery/registry/preprocessing.yaml \
  experiments/phase4_temporal_analytics/p4_e03
git commit -m "feat: add validated change caption specialist"
```

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

Return intersection, union, IoU or `None` for empty union, and valid-pixel count. Do not label low agreement as either model being wrong.

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

**Files:**
- Create: `satquery/inference/flood.py`
- Create: `ml/evaluation/prepare_p4_e04.py`
- Create: `ml/evaluation/run_p4_e04.py`
- Create: `tests/inference/test_flood.py`
- Create: `tests/ml/test_p4_e04.py`

**Interfaces:**
- Produces: `FloodBackend.segment(image) -> FloodBackendResult` protocol.
- Produces: `SturmS1Backend` using exact audited model source/checkpoint.
- Produces: `FloodSegmentationService.segment(observation) -> FloodMaskEvidence` with `target_class="water_or_flood_extent"` only when label semantics support it.

- [ ] **Step 1: Write failing strict sensor tests**

```python
def test_sturm_accepts_only_exact_audited_sentinel1_contract(valid_s1_observation):
    evidence = service(fake_backend).segment(valid_s1_observation)
    assert evidence.domain.status is DomainStatus.IN_DOMAIN

@pytest.mark.parametrize("polarizations", [("HH", "HV"), ("VV",), ()])
def test_sturm_rejects_incompatible_polarization(polarizations):
    with pytest.raises(ModelInputUnsupportedError):
        service(fake_backend).segment(observation_with(polarizations=polarizations))


def test_risat_never_falls_through_to_sturm(risat_observation):
    with pytest.raises(ModelInputUnsupportedError):
        service(fake_backend).segment(risat_observation)
```

Also test radiometric domain, calibrated product requirement, 10 m GSD tolerance, 128×128 preparation, VV/VH order, NoData, archive/checkpoint SHA-256, finite logits, binary output, and raw-score labeling.

- [ ] **Step 2: Verify RED and implement model adapter**

```bash
python -m pytest tests/inference/test_flood.py -v
```

Extract official `.tar.gz` only through validated regular-file allowlist into quarantine. Reject absolute paths, `..`, links, devices, duplicate members, oversized members, and unexpected weights. Verify outer archive and selected checkpoint hashes before loading.

- [ ] **Step 3: Write evaluator and split-sealing tests**

```python
def test_sen1floods11_cannot_tune_sturm_threshold():
    with pytest.raises(ValueError):
        run_p4_e04(dataset="sen1floods11", threshold=0.6)


def test_flood_metrics_reconstruct_from_saved_confusion_counts(tmp_path):
    result = run_fixture_evaluation(tmp_path)
    assert reconstruct_binary_metrics(read_jsonl(result.predictions)) == result.metrics
```

Preparation records VV/VH order, dB/linear domain, CRS, transform, GSD, acquisition metadata, mask label semantics, source event/location, pair grouping, and hashes. If STURM and Sen1Floods11 radiometric/preprocessing contracts cannot be reconciled from authoritative sources, record external evaluation as `BLOCKED_INPUT_CONTRACT`; never invent conversion.

- [ ] **Step 4: Reproduce STURM validation**

```bash
python -m ml.evaluation.prepare_p4_e04 --dataset sturm_flood --source-root "$STURM_ROOT" --output experiments/phase4_temporal_analytics/p4_e04/sturm_manifest.json
python -m ml.evaluation.run_p4_e04 --dataset sturm_flood --manifest experiments/phase4_temporal_analytics/p4_e04/sturm_manifest.json --data-root "$STURM_ROOT" --output-dir outputs/p4_e04_sturm --split validation
```

Require nonzero evaluated samples. Report precision, recall, F1, IoU, event-level breakdown, latency, and peak VRAM. Accuracy alone cannot pass segmentation gate.

- [ ] **Step 5: Run unchanged independent evaluation**

```bash
python -m ml.evaluation.prepare_p4_e04 --dataset sen1floods11 --source-root "$SEN1FLOODS11_ROOT" --output experiments/phase4_temporal_analytics/p4_e04/sen1floods11_manifest.json
python -m ml.evaluation.run_p4_e04 --dataset sen1floods11 --manifest experiments/phase4_temporal_analytics/p4_e04/sen1floods11_manifest.json --data-root "$SEN1FLOODS11_ROOT" --output-dir outputs/p4_e04_sen1floods11 --split external_holdout
```

No threshold/preprocessing changes after STURM validation. Report cross-dataset degradation and domain caveats.

- [ ] **Step 6: Promote only after reproduction, then commit**

Add exact `flood_segmentation` model/profile registry entries only after pass. Save reviewed metadata/results, never the 1.8 GB archive/checkpoint.

```bash
python -m pytest tests/inference/test_flood.py tests/ml/test_p4_e04.py tests/ingestion/test_models.py -v
git diff --check
git add satquery/inference/flood.py ml/evaluation/prepare_p4_e04.py ml/evaluation/run_p4_e04.py \
  tests/inference/test_flood.py tests/ml/test_p4_e04.py models/registry.yaml \
  satquery/registry/preprocessing.yaml experiments/phase4_temporal_analytics/p4_e04
git commit -m "feat: add validated SAR flood specialist"
```

---

### Task 10: Add Fail-Closed Evidence Reconciliation

**Files:**
- Create: `satquery/analytics/reconciliation.py`
- Create: `tests/analytics/test_reconciliation.py`

**Interfaces:**
- Produces: `ReconciliationResult` with outcome, source evidence IDs, optional agreement evidence, warnings, and no aggregate confidence.
- Produces: `reconcile_masks(learned: FloodMaskEvidence | ChangeMaskEvidence | None, deterministic: ChangeMaskEvidence | None) -> ReconciliationResult`.
- Consumes: agreement calculation from Task 8 and failure outcomes.

- [ ] **Step 1: Write failing outcome tests**

```python
def test_compatible_agreeing_masks_allow_with_separate_scores():
    result = reconcile_masks(learned, deterministic)
    assert result.outcome == "ALLOW"
    assert result.agreement.interpretation == "agreement_not_accuracy"
    assert result.aggregate_confidence is None


def test_strong_conflict_abstains_without_destroying_independent_evidence():
    result = reconcile_masks(learned, conflicting_deterministic)
    assert result.outcome == "ABSTAIN"
    assert result.evidence_ids == (learned.evidence_id, conflicting_deterministic.evidence_id)


def test_missing_learned_model_degrades_to_deterministic_with_warning():
    result = reconcile_masks(None, deterministic)
    assert result.outcome == "ALLOW_WITH_WARNING"
    assert "LEARNED_MODEL_NOT_APPLICABLE" in result.warnings
```

- [ ] **Step 2: Verify RED and implement fixed policy**

```bash
python -m pytest tests/analytics/test_reconciliation.py -v
```

Policy uses predeclared agreement bands from `experiment_plan.yaml`. Agreement changes support status only; it never becomes ground-truth correctness or calibrated confidence. Invalid geometry blocks reconciliation. Missing optional evidence degrades locally. Unknown sensor semantics reject model execution before reconciliation.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest tests/analytics -v
git diff --check
git add satquery/analytics/reconciliation.py satquery/analytics/__init__.py \
  tests/analytics/test_reconciliation.py
git commit -m "feat: reconcile temporal evidence safely"
```

---

### Task 11: Extend Existing Kaggle Runner, Not Build Another Framework

**Files:**
- Modify: `scripts/kaggle/experiments.yaml`
- Create: `notebooks/kaggle_p4_e01_oscd.ipynb`
- Create: `notebooks/kaggle_p4_e02_changer.ipynb`
- Create: `notebooks/kaggle_p4_e03_chg2cap.ipynb`
- Create: `notebooks/kaggle_p4_e04_sturm.ipynb`
- Modify: `tests/test_kaggle_runner.py`
- Modify: `scripts/kaggle/README.md`
- Modify: `docs/KAGGLE.md`

**Interfaces:**
- Consumes: existing `scripts/kaggle/runner.py`; no second runner.
- Produces: exact experiment names and exact result-file retrieval.

- [ ] **Step 1: Write failing registry tests**

```python
@pytest.mark.parametrize("name", ["p4-e01-oscd", "p4-e02-changer", "p4-e03-chg2cap", "p4-e04-sturm"])
def test_phase4_experiment_has_unique_identity_and_output(name):
    experiment = get_experiment(name)
    assert experiment["remote_output_dir"].startswith(name)
    assert experiment["download_policy"] == "metadata_only"


def test_phase4_download_uses_exact_relative_paths_not_basename_search(tmp_path):
    command = build_download_command("p4-e02-changer", tmp_path)
    assert "**/validation_result.json" not in command
```

- [ ] **Step 2: Verify RED and add exact entries**

Each entry has one unique notebook, kernel slug, experiment directory, remote output directory, declared small `result_files`, declared `large_result_files`, `download_policy: metadata_only`, GPU flag, internet flag, and exact kernel sources. Never reuse a historical Phase 4 slug.

- [ ] **Step 3: Keep notebooks thin**

Each notebook contains only:

```text
environment report
exact Git clone once
checkout injected SHA
assert clean checkout
install audited dependencies
verify dataset/model inputs
call one tested Python evaluator
verify exact declared outputs
exit nonzero on failure
```

No metrics, predictions, thresholds, scientific transforms, or fallback data are authored in notebook cells.

- [ ] **Step 4: Test dry runs without launching jobs**

```bash
python -m pytest tests/test_kaggle_runner.py -v
python scripts/kaggle/runner.py run p4-e01-oscd --dry-run
python scripts/kaggle/runner.py run p4-e02-changer --dry-run
python scripts/kaggle/runner.py run p4-e03-chg2cap --dry-run
python scripts/kaggle/runner.py run p4-e04-sturm --dry-run
```

Expected: exact commit, unique identity, exact output path, no upload/run during dry run.

- [ ] **Step 5: Commit framework extension**

```bash
git diff --check
git add scripts/kaggle/experiments.yaml scripts/kaggle/README.md docs/KAGGLE.md \
  notebooks/kaggle_p4_e01_oscd.ipynb notebooks/kaggle_p4_e02_changer.ipynb \
  notebooks/kaggle_p4_e03_chg2cap.ipynb notebooks/kaggle_p4_e04_sturm.ipynb \
  tests/test_kaggle_runner.py
git commit -m "feat: register phase 4 external experiments"
```

---

### Task 12: Execute, Retrieve, and Audit External Results

**Files:**
- Add reviewed small outputs under `experiments/phase4_temporal_analytics/p4_e0x/results/`.
- Do not add model weights, data, caches, or large archives.

**Interfaces:**
- Consumes: clean committed Git SHA and Task 11 experiment identities.
- Produces: reproducible measured evidence or preserved failure artifacts.

- [ ] **Step 1: Confirm canonical-run preconditions**

```bash
git status --short
git rev-parse HEAD
python -m pytest
```

Expected: clean worktree and passing suite. Do not use `--allow-dirty`.

- [ ] **Step 2: Run experiments in order**

```bash
python scripts/kaggle/runner.py run p4-e01-oscd
python scripts/kaggle/runner.py run p4-e02-changer
python scripts/kaggle/runner.py run p4-e03-chg2cap
python scripts/kaggle/runner.py run p4-e04-sturm
```

Run only experiments whose contracts are `PASS`. A failed earlier reproduction blocks its robustness/promotion step but does not block independent experiments.

- [ ] **Step 3: Retrieve exact declared metadata/results**

```bash
python scripts/kaggle/runner.py download p4-e01-oscd
python scripts/kaggle/runner.py download p4-e02-changer
python scripts/kaggle/runner.py download p4-e03-chg2cap
python scripts/kaggle/runner.py download p4-e04-sturm
```

Never select first matching basename. Preserve runner metadata separately from scientific metrics.

- [ ] **Step 4: Audit every result locally**

For each experiment verify:

- returned Git SHA equals launched SHA;
- dirty-worktree flag is false;
- dataset manifest and checkpoint/profile hashes match frozen contracts;
- split is validation/robustness/external holdout, never sealed test;
- prediction row count equals declared sample count and is greater than zero;
- aggregate metrics reconstruct from prediction/confusion rows;
- no source image, model weight, repository file, or cache appears in output;
- latency/VRAM records identify hardware;
- failures remain failures, not zero scores.

- [ ] **Step 5: Commit each independently audited result**

```bash
git add experiments/phase4_temporal_analytics/p4_e01/results
git commit -m "exp: record P4-E01 validation"
```

Repeat with separate commits for `P4-E02`, `P4-E03`, and `P4-E04`. Skip absent/blocked experiments; never create a success artifact manually.

---

### Task 13: Freeze Phase 4 Decisions and Close Out

**Files:**
- Create: `experiments/phase4_temporal_analytics/PHASE_4_CLOSEOUT.json`
- Modify: `experiments/phase4_temporal_analytics/README.md`
- Modify: `docs/DEVELOPMENT_PLAN.md`
- Modify if public scientific contracts changed: `docs/ARCHITECTURE.md`
- Modify if evaluation policy changed: `docs/EVALUATION.md`
- Modify if new failure behavior exists: `docs/FAILURE_POLICY.md`
- Modify: `models/registry.yaml`
- Modify: `satquery/registry/preprocessing.yaml`
- Modify: `tests/ml/test_phase4_contracts.py`

**Interfaces:**
- Consumes: locally audited experiment artifacts only.
- Produces: immutable Phase 4 status with per-capability outcomes: `SUPPORTED`, `SUPPORTED_WITH_LIMITS`, `BLOCKED`, or `NOT_EVALUATED`.

- [ ] **Step 1: Write closeout reconstruction test**

Add to `tests/ml/test_phase4_contracts.py`:

```python
def test_phase4_closeout_references_existing_hash_matching_artifacts():
    closeout = load_closeout(CLOSEOUT_PATH)
    for artifact in closeout.artifacts:
        assert artifact.path.is_file()
        assert sha256_file(artifact.path) == artifact.sha256


def test_supported_capability_has_nonzero_measured_evidence():
    closeout = load_closeout(CLOSEOUT_PATH)
    for capability in closeout.capabilities:
        if capability.status.startswith("SUPPORTED"):
            assert capability.sample_count > 0
            assert capability.metrics
```

- [ ] **Step 2: Verify RED and write closeout from evidence**

```bash
python -m pytest tests/ml/test_phase4_contracts.py -v
```

Closeout records:

- exact Git SHA;
- contract hashes;
- dataset/model/profile identities;
- validation/robustness sample counts;
- reconstructible metrics;
- test-sealing status;
- domain limitations;
- external dependencies;
- accepted/rejected model promotions;
- explicit RISAT/Cartosat non-generalization statement;
- RSCaMa and Modified Sen1Floods11 exclusions;
- no-training decision unless later separately authorized.

- [ ] **Step 3: Reconcile docs with code and evidence**

Set canonical Phase 4 to `COMPLETE` only if mandatory accepted capabilities have measured evidence and all closeout checks pass. Otherwise use `IN PROGRESS` or `BLOCKED` with exact reason. Never edit frozen historical artifacts to make metrics agree.

- [ ] **Step 4: Run full release verification**

```bash
python -m pytest
python -m compileall -q satquery ml apps scripts
git diff --check
git status --short
```

Also inspect tracked file sizes and secrets:

```bash
git ls-files -z | xargs -0 -n1 sh -c 'n=$(wc -c < "$0"); [ "$n" -gt 20000000 ] && echo "$n $0"'
git diff --cached --check
```

Expected: all tests pass; no unexpected large files, credentials, caches, datasets, or checkpoints.

- [ ] **Step 5: Commit closeout**

```bash
git add experiments/phase4_temporal_analytics/PHASE_4_CLOSEOUT.json \
  experiments/phase4_temporal_analytics/README.md \
  docs/DEVELOPMENT_PLAN.md docs/ARCHITECTURE.md docs/EVALUATION.md docs/FAILURE_POLICY.md \
  models/registry.yaml satquery/registry/preprocessing.yaml tests/ml/test_phase4_contracts.py
git commit -m "docs: close phase 4 temporal analytics"
```

Stage only files actually changed. Omit unchanged docs/registries from `git add`.

---

## Scientific Acceptance Gates

### P4-E01 — deterministic multispectral

Pass only when:

- all 13 OSCD bands are identified from contract, not position guesses;
- native resolutions and NoData are handled explicitly;
- output grids are verified;
- hand-calculated NDVI/NDWI/MNDWI and area fixtures pass;
- identity/misalignment/reversal controls execute;
- prediction rows reconstruct all reported mask metrics;
- Sentinel-2 L2A examples remain demonstrations, not benchmark evidence.

### P4-E02 — ChangerEx

Pass only when:

- official checkpoint/source/profile hashes are verified;
- official LEVIR-CD validation is reproduced within predeclared tolerance;
- masks map back to source grid;
- identity/misalignment controls pass;
- unchanged S2Looking degradation is reported;
- capability is labeled high-resolution structural change, not generic change.

### P4-E03 — Chg2Cap

Pass only when:

- official checkpoint/source/vocabulary/decoder are pinned;
- official validation protocol is reproduced or discrepancy is accepted explicitly;
- every generated caption and reference set is preserved;
- BLEU-4/METEOR/ROUGE-L/CIDEr use pinned standard implementations;
- identity/reversal diagnostics are reported;
- caption evidence contains no manufactured mask or measurement.

### P4-E04 — STURM/Sen1Floods11

Pass only when:

- exact Sentinel-1 input contract and checkpoint are verified;
- STURM validation has nonzero samples and segmentation F1/IoU;
- Sen1Floods11 uses unchanged preprocessing/threshold or is blocked for contract incompatibility;
- deterministic SAR processing refuses unknown radiometric domains/polarizations;
- learned/deterministic agreement is labeled diagnostic only;
- RISAT incompatibility routes away from STURM rather than forcing VV/VH assumptions.

## Stop Conditions

Stop affected path and preserve evidence when any occurs:

- upstream license or imagery terms are unresolved;
- checkpoint authority, revision, bytes, or hash cannot be established;
- preprocessing, band order, radiometric domain, or T1/T2 semantics are unknown;
- official source requires unsafe runtime patching or arbitrary remote code;
- validation split cannot be separated from test;
- prediction/stat counts do not match declared sample count;
- metrics cannot be reconstructed;
- sealed test is accessed before freeze;
- official reproduction exceeds predeclared discrepancy tolerance;
- robustness/holdout results accidentally influence tuning;
- output path is ambiguous or retrieved by basename search;
- canonical run uses dirty/unrecorded source state.

## Deferred Work

- RSCaMa integration: add only after Phase 4 closeout if Chg2Cap is scientifically insufficient and Linux/Mamba/CLIP-patch risk is accepted.
- ChangerEx retraining on S2Looking: add only if robustness degradation is unacceptable and a new sealed evaluation source is frozen first.
- Any STURM adaptation to Sen1Floods11 or RISAT: add only with an explicit radiometric/sensor transfer experiment and untouched holdout.
- Temporal API routes, natural-language query routing, answer composition, UI overlays, and downloadable reports: canonical Phases 5–6.
- Generic framework for arbitrary spectral indices/models: add only when a second concrete implementation cannot use the explicit registered tools.

---

# Post-Implementation Code-Quality Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:safe-refactor` with `superpowers:test-driven-development`, then `superpowers:verification-before-completion`. Execute one task per commit.

**Reviewed against:** `4aa8a786213fce61cfe0ce8c7d3486135b051f59`

**Goal:** Remove proven duplication in the Phase 4 implementation without changing scientific behavior, failure outcomes, evidence schemas, experiment outputs, or the intentionally separate production and benchmark boundaries.

**Architecture:** Keep model-specific Protocols and adapters because their signatures are typed dependency-injection seams used by tests and future audited runtimes. Share only stable mechanics: checkpoint integrity checks, aligned RGB-pair preparation, and Phase 4 evaluator utilities. Keep specialist-specific domain constraints in their specialists and keep offline evaluator contracts independent from production inference contracts.

**Tech Stack:** Python 3.11+, NumPy, Rasterio, Pydantic 2, pytest.

**Spec:** Thermo-nuclear review supplied after Phase 4 closeout, vetted against `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/FAILURE_POLICY.md`, existing tests, and the Phase 4 scope in this plan.

## Review Disposition

| Review item | Decision | Reason |
|---|---|---|
| Replace all specialist adapters with one generic `CheckpointedBackend` | **Reject** | `StructuralChangeBackend`, `ChangeCaptionBackend`, and `FloodBackend` intentionally expose different typed method contracts and support fake injection. A generic callable wrapper would save little while hiding model-specific signatures and provenance. |
| Share checkpoint hashing/verification | **Accept, narrowed** | The three new specialist modules duplicate the same integrity mechanism. Share that mechanism only; retain each named adapter and its fail-closed runtime message. Do not sweep historical evaluators or unrelated ingestion/visualization modules merely to remove a one-line hash helper. |
| Remove checks duplicated after `require_domain` | **Accept, narrowed** | Modality checks in both optical specialists and modality/sensor/polarization checks in `flood.py` exactly repeat the canonical gate. Remove only exact duplicates. Keep STURM radiometric-domain set, calibration, GSD, dimensions, NoData, and georeferencing checks because they are model-specific and are not fully represented by `require_domain`. |
| Add GSD, calibration, and band-description options to `require_domain` | **Reject** | Those constraints differ by specialist and would turn the small universal gate into a model-policy parameter bag. Add a shared optical-pair input helper for the two real callers instead. |
| Share optical aligned-pair validation and RGB reads | **Accept, relocated** | The duplication is real, but `satquery/analytics/temporal.py` is the deterministic GIS layer. Model tensor preparation belongs in `satquery/inference/temporal_inputs.py`. Temporal-order policy remains in captioning because structural change accepts explicit user ordering. |
| Centralize API exception mapping | **Defer** | No `apps/api` file changed on this branch, and this plan explicitly keeps temporal API routing in Phases 5–6. Refactoring existing Phase 2/3 routes here would expand scope and risk public response-contract drift without enabling Phase 4. Revisit when the first temporal route is added. |
| Share all evaluation orchestration and CLI code | **Reject** | P4-E01 through P4-E04 have different manifests, splits, outputs, result types, and scientific controls. A common runner/template would obscure those contracts. Share only byte-identical path, hash, and binary-metric functions. |
| Reuse production backend Protocols in evaluators | **Reject** | `AGENTS.md` requires benchmark adapters to remain separate from production scientific APIs. The structurally typed evaluator Protocols are cheap and intentionally decouple offline evaluation. Remove the unused concrete `ChangerExBackend` import instead. |
| Refactor `_parse_audit_file` into one generic Pydantic parse | **Reject for this pass** | The loader intentionally accepts heterogeneous audit metadata, then applies dataset-specific typing and explicit PASS/BLOCKED policy checks. A generic model would either admit the same loose shape behind more machinery or require a separate scientific-contract migration. |
| Remove unused `ValidationError` import | **Accept** | It has no special purpose and no callers. |
| Split files due to the 1,000-line rule | **No action** | No changed source file crossed 1,000 lines. `scripts/kaggle/runner.py` remains below the threshold and is outside this cleanup. |

## Global Constraints for Tasks 14–17

- Preserve exception classes, fail-closed ordering, accepted/rejected input domains, evidence fields, mask bytes, hashes, metrics, and JSON/JSONL schemas.
- Do not modify frozen experiment artifacts, registries, model contracts, thresholds, preprocessing profiles, notebooks, API routes, or documentation outside `plan.md`.
- Do not add a generic backend factory, generic model runner, evaluator framework, or new dependency.
- Characterize behavior before moving logic. Existing fake backend seams must continue to work without inheritance or casts.
- Do not claim Phase 4 scientific support: the recorded closeout remains `BLOCKED`, and these are maintainability changes only.
- If a proposed extraction changes an error type, output path, serialized field, numerical result, or hash, stop and report instead of updating expectations.

### Task 14: Centralize New-Specialist Checkpoint Integrity

**Files:**
- Create: `satquery/inference/checkpoints.py`
- Modify: `satquery/inference/change_detection.py`
- Modify: `satquery/inference/change_captioning.py`
- Modify: `satquery/inference/flood.py`
- Modify: `tests/inference/test_change_detection.py`
- Modify: `tests/inference/test_change_captioning.py`
- Modify: `tests/inference/test_flood.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `require_checkpoint(path: Path, expected_sha256: str, *, model_name: str) -> None` raising `ModelUnavailableError` with the existing `<model> checkpoint is unavailable|invalid` messages.
- Preserves: all three named backend Protocols and adapter classes unchanged at their public boundaries.

- [x] **Step 1: Add characterization tests before extraction**

For each concrete adapter, invoke its public method and assert that a missing checkpoint and a wrong digest raise `ModelUnavailableError`. Keep the existing model label in the message:

```python
with pytest.raises(ModelUnavailableError, match="ChangerEx checkpoint is unavailable"):
    ChangerExBackend(missing, "0" * 64, predictor=fake_predictor).predict(t1, t2)

checkpoint.write_bytes(b"not-the-registered-checkpoint")
with pytest.raises(ModelUnavailableError, match="checkpoint hash is invalid"):
    Chg2CapBackend(checkpoint, "0" * 64, captioner=fake_captioner).caption(t1, t2)
```

Add the equivalent STURM assertion in `tests/inference/test_flood.py`.

- [x] **Step 2: Run tests to establish the behavior baseline**

```bash
python -m pytest tests/inference/test_change_detection.py tests/inference/test_change_captioning.py tests/inference/test_flood.py -v
```

Expected: PASS before refactoring.

- [x] **Step 3: Extract only integrity mechanics**

Implement:

```python
def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_checkpoint(path: Path, expected_sha256: str, *, model_name: str) -> None:
    if not path.is_file():
        raise ModelUnavailableError(f"{model_name} checkpoint is unavailable")
    if sha256_file(path) != expected_sha256:
        raise ModelUnavailableError(f"{model_name} checkpoint hash is invalid")
```

Replace the three private checkpoint-verification implementations and use `sha256_file` for derived mask hashes in change detection and flood. Remove only now-unused `hashlib` imports and the unused `ValidationError` import. Do not alter backend Protocols, constructor parameters, callable injection, or runtime-unavailable behavior.

- [x] **Step 4: Verify and commit**

```bash
python -m pytest tests/inference/test_change_detection.py tests/inference/test_change_captioning.py tests/inference/test_flood.py -v
python -m compileall -q satquery/inference

git diff --check
git add satquery/inference/checkpoints.py satquery/inference/change_detection.py \
  satquery/inference/change_captioning.py satquery/inference/flood.py tests/inference
git commit -m "refactor: share specialist checkpoint verification"
```

### Task 15: Canonicalize Temporal RGB Input Preparation

**Files:**
- Create: `satquery/inference/temporal_inputs.py`
- Modify: `satquery/inference/change_detection.py`
- Modify: `satquery/inference/change_captioning.py`
- Modify: `satquery/inference/flood.py`
- Create: `tests/inference/test_temporal_inputs.py`
- Modify: `tests/inference/test_change_detection.py`
- Modify: `tests/inference/test_change_captioning.py`
- Modify: `tests/inference/test_flood.py`

**Interfaces:**
- Produces: `read_aligned_rgb_pair(t1: ObservationState, t2: ObservationState) -> tuple[np.ndarray, np.ndarray]`.
- Preserves: captioning's metadata-based `T1 < T2` rule and structural change's explicit-user-mapping order source.
- Preserves: STURM's accepted radiometric domains (`backscatter_db` and `backscatter_linear`) and all specialist-specific checks.

- [x] **Step 1: Characterize shared and distinct policies**

Add focused tests proving that `read_aligned_rgb_pair` rejects missing georeferencing, shifted transforms, and non-`("R", "G", "B")` semantics, and returns two `(3, H, W)` float32 arrays for a valid pair. Keep service tests proving captioning rejects unknown/reversed time while structural change does not begin requiring metadata order.

Add flood tests for wrong modality, sensor, polarization, radiometric domain, calibration, GSD, and georeferencing. These tests distinguish checks owned by `require_domain` from checks that must remain local.

- [x] **Step 2: Run the characterization suite**

```bash
python -m pytest tests/inference/test_temporal_inputs.py \
  tests/inference/test_change_detection.py \
  tests/inference/test_change_captioning.py \
  tests/inference/test_flood.py \
  tests/verification/test_domain.py -v
```

Expected first run: FAIL only because `temporal_inputs.py` does not exist; existing service tests remain PASS.

- [x] **Step 3: Implement the shared inference-layer helper**

`read_aligned_rgb_pair` must:

1. call `require_domain` for each observation with optical/multispectral modalities;
2. require CRS and affine transforms on both observations;
3. require exact equality of `(width, height, CRS, transform)` because these model services do not perform reprojection;
4. require exact semantic band descriptions `("R", "G", "B")` on both observations;
5. read bands `(1, 2, 3)` as float32 and divide by `255.0` only after every metadata check passes.

Use a generic temporal-RGB error message; do not pass model-name strings or boolean modes into this helper. Keep temporal ordering in `ChangeCaptionService.describe` before calling the helper. `StructuralChangeService.detect` calls the helper directly.

- [x] **Step 4: Remove only exact domain-gate duplicates in flood**

After `require_domain(... supported_modalities=(SAR,), required_sensor_names=(...), required_polarizations=("VV", "VH"))`, delete the repeated modality, sensor-name, and polarization branches. Keep local checks for:

- radiometric domain membership in `{backscatter_db, backscatter_linear}`;
- `CALIBRATION == calibrated`;
- approximately 10 m X/Y GSD;
- CRS and transform;
- 128×128 dimensions, two channels, and NoData behavior.

Do not extend `require_domain` in this task.

- [x] **Step 5: Verify and commit**

```bash
python -m pytest tests/inference tests/verification/test_domain.py -v
python -m compileall -q satquery/inference satquery/verification

git diff --check
git add satquery/inference/temporal_inputs.py satquery/inference/change_detection.py \
  satquery/inference/change_captioning.py satquery/inference/flood.py \
  tests/inference/test_temporal_inputs.py tests/inference/test_change_detection.py \
  tests/inference/test_change_captioning.py tests/inference/test_flood.py
git commit -m "refactor: share temporal RGB input checks"
```

### Task 16: Share Only Stable Phase 4 Evaluation Utilities

**Files:**
- Create: `ml/evaluation/common.py`
- Modify: `ml/evaluation/run_p4_e01.py`
- Modify: `ml/evaluation/run_p4_e02.py`
- Modify: `ml/evaluation/run_p4_e03.py`
- Modify: `ml/evaluation/run_p4_e04.py`
- Create: `tests/ml/test_phase4_evaluation_common.py`
- Modify: `tests/ml/test_p4_e01.py`
- Modify: `tests/ml/test_p4_e02.py`
- Modify: `tests/ml/test_p4_e03.py`
- Modify: `tests/ml/test_p4_e04.py`

**Interfaces:**
- Produces: `resolve_under_root(root: Path, relative: str) -> Path`.
- Produces: `verify_sha256(path: Path, expected: str) -> None` with the existing `ValueError("manifest hash mismatch: ...")` behavior.
- Produces: `binary_confusion(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]`.
- Produces: `binary_metrics(counts: Mapping[str, int]) -> dict[str, float | None]`.
- Preserves: each evaluator's Protocol, manifest checks, split restrictions, threshold freeze, output rows, result type, and CLI.

- [x] **Step 1: Add direct utility tests and strengthen output characterization**

Test path traversal rejection, correct SHA-256 acceptance, digest mismatch rejection, valid-mask exclusion, empty-positive metric `None` values, and exact TP/FP/FN/TN calculations. In each existing P4 evaluator test, retain the current fixture and assert the complete `confusion` and `metrics` dictionaries (or caption counts for P4-E03), not only IoU.

- [x] **Step 2: Run the baseline tests**

```bash
python -m pytest tests/ml/test_phase4_evaluation_common.py tests/ml/test_p4_e01.py \
  tests/ml/test_p4_e02.py tests/ml/test_p4_e03.py tests/ml/test_p4_e04.py -v
```

Expected first run: FAIL only because `ml.evaluation.common` does not exist.

- [x] **Step 3: Extract the four stable helpers**

Move only the byte-identical path containment, digest verification, confusion-count, and binary-metric logic. `binary_confusion` treats `valid=None` as an all-valid mask; P4-E01 passes its real valid mask, while P4-E02 and P4-E04 omit it.

Do not extract manifest parsing, sample loops, output serialization, argparse setup, thresholds, split logic, or caption evaluation. Keep `BinaryChangeBackend`, `CaptionBackend`, and `FloodEvaluationBackend` local to the benchmark modules. Remove the unused `ChangerExBackend` import from `run_p4_e02.py`.

- [x] **Step 4: Prove serialized behavior remains unchanged**

Run all four fixture evaluators and compare their generated JSON/JSONL fields through the strengthened tests. No golden scientific metrics or experiment artifacts may be regenerated.

```bash
python -m pytest tests/ml/test_phase4_evaluation_common.py tests/ml/test_p4_e01.py \
  tests/ml/test_p4_e02.py tests/ml/test_p4_e03.py tests/ml/test_p4_e04.py -v
python -m compileall -q ml/evaluation

git diff --check
git add ml/evaluation/common.py ml/evaluation/run_p4_e01.py ml/evaluation/run_p4_e02.py \
  ml/evaluation/run_p4_e03.py ml/evaluation/run_p4_e04.py \
  tests/ml/test_phase4_evaluation_common.py tests/ml/test_p4_e01.py \
  tests/ml/test_p4_e02.py tests/ml/test_p4_e03.py tests/ml/test_p4_e04.py
git commit -m "refactor: share phase 4 evaluation utilities"
```

### Task 17: Final Regression Gate

**Files:** No source changes expected.

- [x] **Step 1: Inspect the complete cleanup diff**

```bash
git diff 4aa8a786213fce61cfe0ce8c7d3486135b051f59...HEAD --stat
git diff 4aa8a786213fce61cfe0ce8c7d3486135b051f59...HEAD -- satquery/inference ml/evaluation tests
```

Confirm there are no changes under `apps/api`, `models`, `satquery/registry`, `experiments`, or `notebooks`.

- [x] **Step 2: Run affected and full verification**

```bash
python -m pytest tests/inference tests/verification tests/ml/test_p4_e01.py \
  tests/ml/test_p4_e02.py tests/ml/test_p4_e03.py tests/ml/test_p4_e04.py -v
python -m pytest
python -m compileall -q satquery ml apps scripts
git diff --check
git status --short
```

Expected: all tests pass; compileall and diff checks are clean; only intentional source/test changes plus this planning document are present.

- [x] **Step 3: Confirm scientific non-impact**

Verify explicitly:

- no experiment artifact or registry hash changed;
- no accepted sensor, modality, polarization, radiometric domain, calibration, GSD, dimensions, grid, bands, or temporal-order rule changed;
- no exception class or evidence schema changed;
- no evaluator row key, metric formula, threshold, split, output path, or result type changed;
- fake backends still satisfy Protocols structurally without inheritance.

If any item differs, stop and revert the responsible task rather than redefining expected behavior.
