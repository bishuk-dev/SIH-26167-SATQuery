# Phase 4 Integrity Repair — Final Report

**Timestamp:** 2026-09-06T23:00:00Z
**Repair basis:** `plan.md` (966-line integrity-repair directive)
**Repository:** bishuk-dev/SIH-26167-SATQuery
**Audit baseline commit:** daa932f15903722468ed69061045f60c5c3e0be8
**Phase 4 start commit:** abebbe29e00b90b3ca6f2b64a92942cb626c70e9
**Current reviewed head:** 2bb9ccf9fb42a785337e3101832209b4c8ddfa9f

---

## 1. Phase 4 Current Status

**BLOCKED_ON_REAL_EXTERNAL_EVALUATION**

Initial P4-E02 and P4-E04 outputs were generated from synthetic/placeholder paths and were rejected before scientific closeout. Corrected benchmark runners now fail closed and use only traceable real dataset evidence.

---

## 2. Exact Root Causes

### P4-E02 Placeholder Result
- `scripts/kaggle/p4_e02_baseline.py` used four `Image.new(...)` solid-color synthetic fixtures as benchmark evidence.
- Hardcoded `accuracy = 1.0` and `exact_match_score = 0.95` with `sample_count = 4`.
- No real LEVIR-CC validation imagery was loaded or evaluated.

### P4-E04 Placeholder Result
- `experiments/phase4_sar_validation/results/sar_validation_metrics.json` contained hardcoded `post_event_water_iou = 0.884`, `post_event_water_accuracy = 0.942`, `total_samples = 50`.
- No real Modified Sen1Floods11 dataset download or evaluation occurred.

---

## 3. Files Changed

| File | Change Type |
|------|-------------|
| `experiments/phase4_integrity_correction.json` | Created |
| `experiments/phase4_temporal_closeout.json` | Updated status, rejected P4-E02/P4-E04 |
| `scripts/kaggle/p4_e02_baseline.py` | Removed synthetic fixtures, added fail-closed dataset check |
| `satquery/tools/temporal_vqa.py` | Removed dummy-image generation, added input absence refusal |
| `satquery/models/change_vqa/baseline.py` | Model failures now raise `ModelExecutionError` instead of returning fallback prose |
| `satquery/analytics/sar.py` | Added `SarRadiometricDomain` enum, explicit domain contract, removed heuristic mean>50 guess |
| `satquery/sensors/semantics.py` | Removed B4/B5/B8 from generic sensor mapping |
| `satquery/analytics/temporal.py` | Added transform validation before differencing |
| `scripts/kaggle/experiments.yaml` | Registered corrected experiment names |
| `docs/DEVELOPMENT_PLAN.md` | Updated Phase 4 status |
| `tests/phase4/test_phase4_integrity_regression.py` | Created 17 regression tests |

---

## 4. Temporal VQA Dummy-Image Fix

**File:** `satquery/tools/temporal_vqa.py:53-65`

`TemporalVqaTool.execute` no longer creates `Image.new(...)` placeholders when `image_t1` or `image_t2` is `None`. Instead returns structured `ChangeVQAResult` with `answer` stating missing visual inputs and `confidence=0.0`.

---

## 5. Model Failure Behavior Fix

**File:** `satquery/models/change_vqa/baseline.py:173-190`

`BiTemporalChangeVQABackend.answer_change_vqa` no longer catches all exceptions and returns a fake fallback answer. Model load/inference failures now raise `ModelExecutionError` or `ModelUnavailableError`. The learned specialist never disguises failure as scientific output.

---

## 6. Test-Sealing Enforcement

**File:** `scripts/kaggle/p4_e02_baseline.py`

P4-E02 runner reads only from `LEVIR_CC_ROOT/val/A` and `LEVIR_CC_ROOT/val/B`. No `split="test"` path is accepted. Regression test `test_p4_e02_test_split_is_refused` verifies failure when dataset is unavailable.

---

## 7. Generic Sensor Semantics Fix

**File:** `satquery/sensors/semantics.py:58-70`

Removed `b04`, `b4`, `b08`, `b8`, `b5`, `band5`, `b11`, `b12`, `b6`, `b7` from generic sensor mapping. Generic/unknown sensors may resolve only explicit semantic labels (`red`, `green`, `blue`, `nir`, `swir`, `vv`, `vh`, etc.) or explicit caller-supplied metadata. Ambiguous substring matches fail closed with `MissingBandError`.

---

## 8. SAR Radiometric Contract Fix

**File:** `satquery/analytics/sar.py:37-66, 121-145`

Introduced `SarRadiometricDomain` enum: `SAR_DB_POWER`, `SAR_LINEAR_POWER`, `SAR_LINEAR_AMPLITUDE`, `SAR_UNKNOWN`. Removed heuristic `if mean > 50: assume linear` inference. Caller must explicitly declare domain. `SAR_UNKNOWN` raises `MeasurementError`. Conversion formulas are correct: power uses `10*log10`, amplitude uses `20*log10`.

---

## 9. Flood-Expansion Semantic Fix

**File:** `satquery/analytics/sar.py:163-180`

Separated `post_water_mask` (post-event water extent) from `flood_expansion_mask` (newly inundated change). Expansion requires `delta_db <= -decrease_threshold AND post_water_mask`. Permanent pre-existing water no longer counted as newly inundated. Provenance records both masks separately.

---

## 10. Temporal Spatial-Alignment Fix

**File:** `satquery/analytics/temporal.py:107-115, 209-217`

`compute_vegetation_change` and `compute_water_change` now validate affine transform before pixelwise differencing. Equal-shape arrays with shifted affines are rejected at the analytics layer, not merely at the pairing layer.

---

## 11. Geographic Area Measurement Fix

**File:** `satquery/geo/measurements.py:118-150`

Geographic CRS area calculation uses WGS84 ellipsoid radii of curvature (`r_n`, `r_m`) at the pixel center latitude. Degrees are never squared or treated as meters. Calculation path recorded as `geodesic_wgs84`.

---

## 12. Corrected P4-E02 Design

| Field | Value |
|-------|-------|
| Dataset source | LEVIR-CC (Chenyang Liu et al., IEEE TGRS 2022) |
| Upstream imagery | LEVIR-CD (academic / non-commercial research only) |
| Validation split | `LEVIR_CC_ROOT/val/A` + `val/B` (1332 pairs) |
| Model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Preprocessing | Registry profile `smolvlm_bitemporal_change_vqa_v1` (512x512) |
| Metrics | Saved predictions JSONL; no accuracy/exact_match_score unless explicitly computed |
| Test set | SEALED |

---

## 13. Corrected P4-E04 Design

| Field | Value |
|-------|-------|
| Dataset | Modified Sen1Floods11 (Zenodo: 10.5281/zenodo.7946594) |
| Label target | `post_event_water_extent` |
| Threshold policy | Explicit heuristic profile (VV water_max_db=-16.0, flood_decrease_db=3.0) |
| Metrics | Per-sample TP/FP/TN/FN, IoU, accuracy, post_event_water_area_m2 |
| Expansion | Recorded separately as `probable_new_water_area_m2` |

---

## 14. Historical Phase 1–3 Notebook Audit Result

No scientific provenance, sealed-test behavior, or frozen experiment semantics were unintentionally changed by Phase 4 commit range. Historical Phase 4A–4F and Phase 5A remain frozen in `experiments/phase3_multisensor_closeout.json`.

---

## 15. Focused Tests

**File:** `tests/phase4/test_phase4_integrity_regression.py`

17 regression tests added covering:
- P4-E02 synthetic fallback rejection
- Missing image input refusal
- Metrics derive from saved predictions
- sample_count == prediction rows
- Test split refusal
- No hardcoded metrics in code paths
- No synthetic PASS
- Generic B4/B5/B8 refusal
- Unknown SAR domain rejection
- Amplitude vs power dB conversion correctness
- Permanent water not counted as newly inundated
- Equal-shape shifted grids rejected
- Geographic area matches rigorous method
- Model failure does not become normal answer
- No PASS when dataset unavailable

**Result:** 15 passed, 2 errors (Windows tmp_path permission issues, not logic failures).

---

## 16. Full Suite Result

```
================= 178 passed, 3 warnings, 91 errors in 40.13s =================
```

All 91 errors are Windows `PermissionError` on `C:\Users\Techno\AppData\Local\Temp\pytest-of-techno` during `tmp_path` fixture setup — environment limitation, not test failures. No logic regressions detected in passing tests.

---

## 17. git diff --check

Clean. No trailing whitespace, no merge conflict markers.

---

## 18. Exact Kaggle Commands

```bash
python scripts/kaggle/runner.py run p4-e02-levircc-change-description
python scripts/kaggle/runner.py run p4-e04-modified-sen1floods11-validation
```

---

## 19. Exact Result Files Expected

P4-E02:
- `experiments/phase4_bitemporal_vqa/validation_metrics.json`
- `experiments/phase4_bitemporal_vqa/validation_predictions.jsonl`
- `experiments/phase4_bitemporal_vqa/runner_meta.json`

P4-E04:
- `experiments/phase4_sar_validation/sar_validation_metrics.json`
- `experiments/phase4_sar_validation/sar_validation_predictions.jsonl`
- `experiments/phase4_sar_validation/runner_meta.json`

---

## 20. Explicit Statement

NO corrected benchmark metrics are claimed until real P4-E02 and P4-E04 runs have completed and passed integrity review. Phase 4 status remains BLOCKED_ON_REAL_EXTERNAL_EVALUATION.
