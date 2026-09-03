# SatQuery AI — Development Plan

This plan translates the roadmap in `README.md` into implementation gates. The requirements, architecture, evaluation protocol, and failure policy remain the source of truth.

## Phase 0 — Project foundation and research freeze

**Status: incomplete.** The repository foundation, task matrix, evaluation protocol, and failure policy are now present. Dataset-specific split manifests and reproducible frozen-model baseline results do not yet exist, so the research-freeze portion of Phase 0 cannot honestly be marked complete.

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

## Foundation decisions

- Keep the implementation as a modular monolith until measured scaling needs justify additional infrastructure.
- Keep API transport in `apps/api`, UI code in `apps/web`, reusable scientific/domain logic in `satquery`, and offline model work in `ml`.
- Keep model, tool, and preprocessing registries empty until a real implementation satisfies their contracts.
- Add Dockerfiles and Compose with the first runnable Phase 1 services; an empty container topology would not be executable or verifiable.
