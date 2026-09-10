# Frontend–Backend Integration Report

Date: 2026-09-11  
Scope: `apps/web` integration with the canonical FastAPI `/api/v1` surface

## Outcome

The active frontend now uses backend-issued observation, analysis, and job identities. The existing three-column analysis workspace was retained, while placeholder scientific content was replaced by persisted API data. The landing page uses the star-field and vignette treatment preferred from `apps/web2` without importing that application's unresolved merge-conflict state.

## Integrated flow

1. The landing page validates that the selected browser file is a TIFF/GeoTIFF.
2. The file is uploaded as multipart form data to `POST /api/v1/observations`.
3. The returned `observation_id` is submitted with the natural-language query to `POST /api/v1/query` using an idempotency key.
4. The frontend navigates using the backend-issued `analysis_id` and `job_id`.
5. The analysis page polls the analysis, job, report, and trace resources until the analysis reaches a terminal state.
6. Observation metadata and the backend-provided display-only tile are rendered separately from the immutable source asset.
7. JSON and HTML exports use the deterministic report endpoints. Unsupported PDF generation is not presented to users.

## Backend data shown in the UI

- durable job and analysis status;
- verified answer or explicit abstention/failure state;
- deterministic measurements and calculation CRS;
- evidence IDs and artifact references;
- warnings, limitations, and verification outcome;
- raster, sensor, temporal, and geospatial metadata;
- source and registry provenance hashes;
- persisted execution trace events.

## Removed placeholder behavior

- filename-derived `sq-*` analysis IDs;
- the non-existent sample GeoTIFF selection;
- timer-driven fake completion;
- hard-coded area, backscatter, confidence, CRS, coordinates, latency, and sensor claims;
- fake PDF/GeoJSON export alerts;
- bundled Sentinel images presented as though they were uploaded evidence.

## Configuration and security

Development requests use Vite's same-origin proxy. Production may use the same-origin deployment or `VITE_SATQUERY_API_BASE_URL` with an exact backend CORS allow-list. Browser-visible environment variables must not contain API keys. If API-key authentication is enabled, credentials should be supplied by a trusted same-origin gateway rather than embedded in the frontend bundle.

## Verification evidence

- `npm run build`: passed.
- `npm run lint`: passed with one pre-existing Fast Refresh warning in `src/components/ui/button.tsx`.
- Backend integration suite: 22 tests passed across observation upload, query submission, jobs/events, and reports.
- Live browser flow: passed using an isolated API data directory and generated GeoTIFF.
  - observation upload succeeded;
  - backend issued real `obs_*`, `ana_*`, and `job_*` identities;
  - analysis reached a terminal state;
  - the display tile rendered;
  - report and trace data rendered;
  - the browser console contained zero errors in the final clean session.

The isolated backend reported degraded readiness because model checkpoint bytes were unavailable. The exercised analysis therefore returned a structured `ABSTAIN` result with `MODEL_UNAVAILABLE`, which the UI displayed without manufacturing an answer. This validates the intended fail-closed behavior; it is not evidence that model inference itself was executed successfully.

## Remaining boundaries

- The landing page currently uploads one observation per submission. Pair creation and two-observation temporal/cross-modal upload remain separate future product work.
- API-key-enabled production deployments require a trusted gateway/session design; secrets are intentionally not accepted through Vite environment variables.
- No frontend unit-test framework is currently configured. TypeScript/build/lint and live Playwright verification cover this change.
