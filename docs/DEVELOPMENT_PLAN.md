# SatQuery AI — Development Plan

This plan translates the roadmap in `README.md` into implementation gates. The requirements, architecture, evaluation protocol, and failure policy remain the source of truth.

## Canonical roadmap

| Canonical phase | Status | Historical workstreams |
| --- | --- | --- |
| Phase 1 — Core Geospatial Platform | COMPLETE | Phase 0 / Phase 1A–1D |
| Phase 2 — Single-Image Vision Intelligence | COMPLETE | Phase 2A–2C / Phase 3A–3B |
| Phase 3 — Multisensor Intelligence | COMPLETE | Phase 4A–4F / Phase 5A |
| Phase 4 — Temporal + Deterministic Remote-Sensing Analytics | IN_PROGRESS_INTEGRITY_REPAIR | P4-E01–P4-E04 |
| Phase 5 — SatQuery Agent + Evidence Engine | PLANNED | — |
| Phase 6 — Product Integration | PLANNED | — |
| Phase 7 — Demo + Robustness Hardening | PLANNED | — |

Historical identifiers remain immutable. Future scientific experiments use
`P<canonical-phase>-E<number>` (for example, `P4-E01` and `P4-E02`); a Kaggle
run is an experiment, not a product phase. Do not create future product
subphases such as Phase 4A/4B/4C except as historical aliases.

## Canonical phase execution rule

Each canonical phase is implemented from one master prompt. Within that phase,
the agent must:

1. Audit existing code and artifacts.
2. Resolve only necessary research questions.
3. Implement all phase capabilities.
4. Prepare experiments.
5. Execute everything possible locally.
6. Stop only on true external experiment dependencies.
7. Consume returned experiment results.
8. Freeze scientific decisions.
9. Update registries and documentation.
10. Run regression tests.
11. Create `PHASE_N_CLOSEOUT.json`.

## Historical implementation details

The remaining sections retain the detailed historical record. Their old phase
labels are aliases only; the canonical mapping above controls future planning.

**Phase 1A status: complete.** Typed observation contracts, metadata-only GeoTIFF/TIFF inspection, provenance hashing, and configurable header/file limits are implemented and covered by focused tests. Upload handling, persistence, pair compatibility, and all raster transformations remain outside Phase 1A.

**Phase 1B status: complete.** The FastAPI ingestion boundary streams multipart uploads through controlled quarantine, registers immutable originals and filesystem metadata using server-generated IDs, returns sanitized observation metadata, and cleans rejected files. Database persistence and downstream raster processing remain deferred.

**Phase 1C status: complete.** Typed pair compatibility, read-only pair validation, affine/crop coordinate mapping, and CRS bounds transformation are implemented. Pair validation distinguishes exact grid compatibility from pairs that remain analyzable after future reprojection, resampling, or registration; those transformations remain deferred.

**Phase 1D status: complete.** Each accepted observation now receives an immutable, display-only COG derivative with parent provenance and a source-grid-preserving affine. The API returns its display and tile metadata, and serves bounded PNG tiles through Web Mercator XYZ or an explicit pixel-grid fallback for ungeoreferenced imagery. The frontend remains deferred.

**Phase 2A status: complete.** The pinned frozen `HuggingFaceTB/SmolVLM-256M-Instruct` baseline is registered with a versioned 512 × 512 preprocessing profile and checksum verification. Registered observations can produce structured VQA evidence through the API. A deterministic 24-question RSVQA-LR smoke subset is grouped by image-content hash into 14/4/6 train/validation/test questions, and the six-question held-out result is stored under `experiments/phase2a_smolvlm_rsvqa_lr/`. This result establishes plumbing, not model quality.

**Phase 2B status: complete.** A separate 1,767-question RSVQA-LR adaptation manifest contains 70/9/9 scene-grouped train/validation/test scenes and excludes all 12 Phase 2A scenes. Rank-8 attention-projection LoRA improved the reported held-out exact match from 0.26 to 0.57, but question-only, blank-image, and shuffled-image controls around 0.54–0.55 exposed substantial shortcut learning.

**Phase 2C status: complete and rejected.** Train/validation diagnostics identified strong exact-question and answer-template priors. The single visual-contrast sampling experiment did not materially increase the correct-image gap over blank/shuffled controls. Phase 2 is frozen; no further RSVQA training or test evaluation is planned.

**Phase 3A status: complete.** Grounding DINO Tiny is pinned with frozen preprocessing, a production adapter emits model/source/normalized/world-coordinate evidence, and `POST /api/grounding` exposes it for registered observations. A checksum-verified VRSBench subset contains 12 validation scenes / 24 references and 8 untouched test scenes / 16 references. The validation baseline reported mean IoU 0.1286, Acc@0.5 IoU 0.1667, and 17 no-detection references at box/text thresholds 0.40/0.30.

**Phase 3 status: complete and frozen.** Threshold 0.30 plus an exclusive normalized-area cap of 0.80 is the frozen production policy. Boxes at or above the cap are discarded, the highest-scoring remaining model box is selected, and an empty remainder becomes valid abstention evidence. Frozen validation produced mean IoU 0.2058 and Acc@0.5 25.00%. The one-time untouched 8-scene / 16-reference test produced mean IoU 0.1739, Acc@0.5 18.75%, 2 abstentions, 14 detections, detected-only mean IoU 0.1987, and zero oversized selections. The final evidence was produced by a clean reproducible P100 run and is recorded unchanged. The test will not be rerun; Phase 3 validation tuning remains closed.

## Historical Phase 4A–4F + Phase 5A — Canonical Phase 3 evidence

**Status: Canonical Phase 3 is COMPLETE.** Historical Phase 4A–4F and Phase 5A are frozen in [experiments/phase3_multisensor_closeout.json](../experiments/phase3_multisensor_closeout.json). The checksum-verified manifest remains byte-identical at SHA-256 `615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6`, with 12,000 train, 3,000 validation, and 3,001 sealed test pairs. Test was not accessed.

The CROMA decision is `BLOCKED`: its pinned source does not publish positional VV/VH or twelve-band optical semantics, and its batch-dependent README normalization is not established as the checkpoint's pretraining transform. It is not registered. The three official BIFOLD v0.2.0 S1/S2/all safetensors checkpoints and deterministic ConfigILM v0.7.0 `120_nearest` profiles are now pinned. B01 and B09 remain in immutable native data but are excluded from the 10-channel optical and 12-channel joint inputs.

Phase 4D hashed each complete compressed HTTP stream, sequentially decompressed zstd/tar through a strict path/type allowlist, wrote selected members only to quarantine, and atomically promoted each modality after byte-count, publisher-MD5, missing, and duplicate checks passed. Test members use an explicit `sealed_test` namespace and ordinary data access refuses them. The native audit opened all 42 expected rasters from TRAIN only, emitted zero rasters, deleted transient rasters, and preserved test sealing.

## Phase 4 — Temporal + Deterministic Remote-Sensing Analytics

**Phase 4 status: IN_PROGRESS_INTEGRITY_REPAIR.** Initial P4-E02 and P4-E04 outputs were generated from synthetic/placeholder paths and were rejected before scientific closeout. Corrected benchmark runners now fail closed and use only traceable real dataset evidence. Delivered typed core domain contracts (`satquery/core/contracts/temporal.py`, `satquery/core/contracts/evidence.py`), sensor-semantic band resolution across optical and SAR platforms (`satquery/sensors/semantics.py`), deterministic spectral index and spatial measurement analytics (`satquery/analytics/spectral.py`, `satquery/analytics/measurement.py`), bi-temporal change analytics (`satquery/analytics/temporal.py`), deterministic SAR flood inundation mapping on Modified Sen1Floods11 (`satquery/analytics/sar.py`), tool integration (`satquery/tools/temporal_vqa.py`), and learned bi-temporal change description specialist baseline (`satquery/models/change_vqa/baseline.py`, `scripts/kaggle/p4_e02_baseline.py`). Verified with 252 passing unit and integration tests across all layers. Recorded in `experiments/phase4_temporal_closeout.json`.

**Phase 4E gate:** validation is allowed for the separately registered S1 and S2 BIFOLD v0.2.0 experiments. The preprocessing implementation and profile IDs remain frozen: direct float32 casting with no reflectance scaling, nearest-neighbor resize to 120 × 120, official-training-split fixed mean/std normalization, VV/VH S1 order, ten-band S2 model order, native B01/B09 preservation with model exclusion, and fail-closed required pixels. Test remains refused, joint BIFOLD remains prohibited, and manifest/audit mismatches fail closed.

**Phase 4F closeout:** the one registered S2-only head adaptation completed with 38,931 trainable / 23,529,984 frozen parameters. `phase4f_closeout.json` freezes `ADAPTATION_USEFUL_BUT_MODEST`: mAP 0.7496551979598234 (+0.007623044676373758), micro F1 0.7700548081714002 (+0.03747256391723164), macro F1 0.6707376231170578 (+0.02545327777367845). It is one seed, not a statistical replication. Best mAP occurred at epoch 1; validation loss increased overall and mAP declined afterward while train loss decreased, so early stopping completed after epoch 4. No more Phase 4 training is authorized; test remains sealed.

**Historical Phase 5A closeout:** the official joint model completed the same frozen 3,000-scene validation. Its mAP is 0.7425103702661673, a formal `MULTISENSOR_GLOBAL_BENEFIT` under the pre-frozen rule because it exceeds official S2 by 0.0004782169827176608. Its macro F1 declined by 0.006251985052409537, so the scientific interpretation is `MARGINAL_GLOBAL_BENEFIT_WITH_CLASS_CONDITIONAL_COMPLEMENTARITY`, not broad or strong fusion improvement. Joint AP improves for 12 of 19 classes and degrades for 7; the complete paired audit and causal caution are frozen in `phase3_multisensor_closeout.json`. The Phase 4F adapted S2 mAP (0.7496551979598234) is higher than the official joint mAP, but is a secondary, non-controlled reference.

Implementation gates:

1. **Source and split freeze:** verify dataset license, URLs, revisions, checksums, band/polarization metadata, and official exclusions; create a geographically grouped, pair-safe train/validation/test manifest with a small development subset. Keep every S1/S2 pair in one split and do not download the full collection until the subset pipeline is verified.
2. **Modality contracts and preprocessing:** register separate Sentinel-2 multispectral and Sentinel-1 VV/VH preprocessing profiles, including band order, resampling, scaling/backscatter convention, NoData handling, and missing-modality representation. Never treat SAR as ordinary RGB.
3. **Frozen baselines:** select one compact OSS multisensor baseline after license and hardware verification, with CROMA as the leading architecture candidate because its radar-optical encoders match the existing architecture. Measure optical-only and SAR-only performance before adaptation.
4. **One adaptation experiment:** add a reproducible, parameter-efficient adaptation path on the frozen scene-grouped subset. Keep training logic in `ml/`, register the checkpoint/config, CPU-smoke-test locally, and use a thin Kaggle GPU runner for meaningful training.
5. **Evidence and evaluation:** emit modality-attributed structured evidence and report identical-task optical-only versus SAR-only results, cross-region degradation, runtime, and VRAM. Add corrupted- and missing-modality controls now so Phase 5 cannot claim fusion merely because two inputs are accepted.
6. **Phase 4 exit gate:** freeze the dataset manifest, preprocessing profiles, model/checkpoint provenance, and unimodal baselines. Begin Phase 5 fusion only when both branches execute independently and their limitations are documented.

## Foundation decisions

- Keep the implementation as a modular monolith until measured scaling needs justify additional infrastructure.
- Keep API transport in `apps/api`, UI code in `apps/web`, reusable scientific/domain logic in `satquery`, and offline model work in `ml`.
- Add model, tool, and preprocessing registry entries only when a real implementation satisfies their contracts.
- Add Dockerfiles and Compose with the first runnable Phase 1 services; an empty container topology would not be executable or verifiable.
