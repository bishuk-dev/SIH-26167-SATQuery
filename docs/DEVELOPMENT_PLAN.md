# SatQuery AI — Development Plan

This plan translates the roadmap in `README.md` into implementation gates. The requirements, architecture, evaluation protocol, and failure policy remain the source of truth.

## Phase 0 — Project foundation and research freeze

**Status: incomplete.** The repository foundation, task matrix, evaluation protocol, and failure policy are present. Phase 2A now supplies the first pinned VQA model, scene-grouped RSVQA-LR subset, and reproducible frozen-model result. Grounding, fusion, and change baselines are still intentionally unselected, so the broader research freeze remains incomplete.

Exit work:

1. Select the first VQA, grounding, fusion, and change baselines with license/provenance notes.
2. Create scene-grouped dataset split manifests without downloading datasets into Git.
3. Record baseline model, checkpoint, preprocessing, and benchmark identifiers in versioned experiment records.

These research artifacts are not a code dependency for beginning the Phase 1 geospatial foundation.

## Actionable implementation sequence

1. **Phase 1 — Geospatial foundation:** define typed observation/asset schemas; implement secure GeoTIFF/TIFF inspection, immutable storage, pair compatibility, coordinate mapping, COG/tiles, and the first OpenLayers upload/view flow. Test metadata, resource limits, overlap, alignment, and crop-to-world mapping.
2. **Phase 2 — Single-image VQA:** integrate one frozen baseline through a versioned preprocessing profile and canonical evidence adapter; add the benchmark harness and shortcut baselines.
3. **Phase 3 — Grounding:** produce boxes/masks, restore source/world coordinates, render overlays, and evaluate IoU across scale.
4. **Phases 4–5 — Multisensor adaptation and fusion:** add modality-specific optical/SAR preprocessing and encoders, then evaluate optical-only, SAR-only, and fused paths before promoting fusion claims.
5. **Phase 6 — Change analysis:** validate ordered pairs, generate target-specific change evidence before language, and run identity/reversed-pair sanity tests.
6. **Phase 7 — Cross-sensor robustness:** measure region, sensor, and scale degradation; introduce adapters or partial tuning only when results justify them.
7. **Phase 8 — Verification and calibration:** enforce geometric, temporal, physical, provenance, and statistical policies; calibrate supported confidence outputs.
8. **Phase 9 — Agentic orchestration:** implement bounded intent schemas, registry-backed workflow selection, parameter validation, and operational execution traces.
9. **Phase 10 — Product experience:** complete evidence inspection, temporal/multimodal controls, warnings, trace views, and structured report export.
10. **Phases 11–12 — Red-team and hardening:** run adversarial raster/failure suites, freeze checkpoints and preprocessing, profile performance, verify offline deployment, and prepare reproducible demos.

**Phase 1A status: complete.** Typed observation contracts, metadata-only GeoTIFF/TIFF inspection, provenance hashing, and configurable header/file limits are implemented and covered by focused tests. Upload handling, persistence, pair compatibility, and all raster transformations remain outside Phase 1A.

**Phase 1B status: complete.** The FastAPI ingestion boundary streams multipart uploads through controlled quarantine, registers immutable originals and filesystem metadata using server-generated IDs, returns sanitized observation metadata, and cleans rejected files. Database persistence and downstream raster processing remain deferred.

**Phase 1C status: complete.** Typed pair compatibility, read-only pair validation, affine/crop coordinate mapping, and CRS bounds transformation are implemented. Pair validation distinguishes exact grid compatibility from pairs that remain analyzable after future reprojection, resampling, or registration; those transformations remain deferred.

**Phase 1D status: complete.** Each accepted observation now receives an immutable, display-only COG derivative with parent provenance and a source-grid-preserving affine. The API returns its display and tile metadata, and serves bounded PNG tiles through Web Mercator XYZ or an explicit pixel-grid fallback for ungeoreferenced imagery. The frontend remains deferred.

**Phase 2A status: complete.** The pinned frozen `HuggingFaceTB/SmolVLM-256M-Instruct` baseline is registered with a versioned 512 × 512 preprocessing profile and checksum verification. Registered observations can produce structured VQA evidence through the API. A deterministic 24-question RSVQA-LR smoke subset is grouped by image-content hash into 14/4/6 train/validation/test questions, and the six-question held-out result is stored under `experiments/phase2a_smolvlm_rsvqa_lr/`. This result establishes plumbing, not model quality.

**Phase 2B status: complete.** A separate 1,767-question RSVQA-LR adaptation manifest contains 70/9/9 scene-grouped train/validation/test scenes and excludes all 12 Phase 2A scenes. Rank-8 attention-projection LoRA improved the reported held-out exact match from 0.26 to 0.57, but question-only, blank-image, and shuffled-image controls around 0.54–0.55 exposed substantial shortcut learning.

**Phase 2C status: complete and rejected.** Train/validation diagnostics identified strong exact-question and answer-template priors. The single visual-contrast sampling experiment did not materially increase the correct-image gap over blank/shuffled controls. Phase 2 is frozen; no further RSVQA training or test evaluation is planned.

**Phase 3A status: complete.** Grounding DINO Tiny is pinned with frozen preprocessing, a production adapter emits model/source/normalized/world-coordinate evidence, and `POST /api/grounding` exposes it for registered observations. A checksum-verified VRSBench subset contains 12 validation scenes / 24 references and 8 untouched test scenes / 16 references. The validation baseline reported mean IoU 0.1286, Acc@0.5 IoU 0.1667, and 17 no-detection references at box/text thresholds 0.40/0.30.

**Phase 3 status: complete and frozen.** Threshold 0.30 plus an exclusive normalized-area cap of 0.80 is the frozen production policy. Boxes at or above the cap are discarded, the highest-scoring remaining model box is selected, and an empty remainder becomes valid abstention evidence. Frozen validation produced mean IoU 0.2058 and Acc@0.5 25.00%. The one-time untouched 8-scene / 16-reference test produced mean IoU 0.1739, Acc@0.5 18.75%, 2 abstentions, 14 detections, detected-only mean IoU 0.1987, and zero oversized selections. The final evidence was produced by a clean reproducible P100 run and is recorded unchanged. The test will not be rerun; Phase 3 validation tuning remains closed.

## Phase 4 — Multisensor optical/SAR adaptation plan

**Status: Phase 4A–4C audits are complete and the Phase 4D bounded materialization/preprocessing implementation is transfer-ready; the explicit 109.61 GiB transfer and native-raster measurements remain pending.** The checksum-verified manifest remains byte-identical at SHA-256 `615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6`, with 12,000 train, 3,000 validation, and 3,001 untouched test pairs. No archive transfer, checkpoint loading, training, or evaluation occurred in Phase 4D.

The CROMA decision is `BLOCKED`: its pinned source does not publish positional VV/VH or twelve-band optical semantics, and its batch-dependent README normalization is not established as the checkpoint's pretraining transform. It is not registered. The three official BIFOLD v0.2.0 S1/S2/all safetensors checkpoints and deterministic ConfigILM v0.7.0 `120_nearest` profiles are now pinned. B01 and B09 remain in immutable native data but are excluded from the 10-channel optical and 12-channel joint inputs.

Phase 4D hashes each complete compressed HTTP stream, sequentially decompresses zstd/tar through a strict path/type allowlist, writes selected members only to quarantine, and atomically promotes each modality after byte-count, publisher-MD5, missing, and duplicate checks pass. Test members use an explicit `sealed_test` namespace and ordinary data access refuses them. Plan mode confirms 36,002 selected S1 members, 216,012 selected S2 members, a 117,690,863,548-byte network transfer, and an 8 GiB recommended free-disk floor without networking.

**Phase 4D remaining gate:** after explicit transfer approval, run the gated materializer, require both full archive MD5 checks, and inspect only the three frozen train pairs. Native dtype/range/NoData/scale/CRS/affine measurements must replace documentation expectations before Phase 4E. No training or evaluation belongs in this gate.

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
