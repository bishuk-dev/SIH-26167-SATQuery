# SatQuery web application

The React frontend is an evidence-grounded client for the canonical FastAPI API. It does not generate analysis identifiers, scientific measurements, sensor metadata, or completion states locally.

## Local development

From the repository root, start the API:

```bash
uvicorn apps.api.app.main:app --reload
```

Then start the frontend:

```bash
cd apps/web
npm ci
npm run dev
```

Vite proxies `/api`, `/health`, and `/tiles` to `http://127.0.0.1:8000` by default. Copy `.env.example` to `.env.local` only when a different development target is required.

For production, serve the frontend and API behind the same origin, or set `VITE_SATQUERY_API_BASE_URL` to the public API origin and configure the API's exact CORS origin. Never place an API secret in a `VITE_*` variable because Vite embeds those values in browser code.

## User flow

```text
GeoTIFF selection
  → POST /api/v1/observations
  → POST /api/v1/query
  → GET /api/v1/jobs/{job_id}
  → GET /api/v1/analyses/{analysis_id}
  → GET /api/v1/reports/{analysis_id}
```

The analysis page also reads the observation display-tile URL, execution trace, evidence links, warnings, measurements, artifacts, and reproducibility hashes returned by the backend. Failed, rejected, or abstained analyses remain explicit and never fall back to demo claims.

## Verification

```bash
npm run build
npm run lint
```
