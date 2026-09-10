# API application

The deployable Phase 5 runtime is the versioned `/api/v1` API. Interactive
Swagger is available at `http://127.0.0.1:8000/docs`; it can upload imagery,
create temporal pairs, submit analyses, inspect jobs/evidence/artifacts, and
render JSON or HTML reports.

This directory contains the FastAPI transport boundary. Run the current API from the repository root with:

```bash
uvicorn apps.api.app.main:app --reload
```

For a production-like local container:

```bash
docker build -t satquery-backend:phase5 .
docker run --rm --name satquery-api -p 8000:8000 \
  -v satquery-data:/data satquery-backend:phase5
```

The container runs as the non-root `satquery` user with one Uvicorn process.
One process is deliberate because the durable in-process job queue owns its
worker threads. Mount `/data` persistently. Optionally mount verified model
checkpoints at `/models`; missing checkpoints remain visible as unavailable
while deterministic registered tools stay ready.

The default image intentionally installs only the deterministic API/GIS
runtime. When verified model checkpoints are mounted and learned inference is
required, build with
`--build-arg SATQUERY_INSTALL_TARGET=.[inference]`.

Set `SATQUERY_API_KEY` through the deployment secret manager to require the
`X-API-Key` header. Configure exact comma-separated origins with
`SATQUERY_CORS_ORIGINS`. `/health/live` checks the process; `/health/ready`
checks required storage, SQLite, registries, and queue capacity without
loading model checkpoints. Shutdown drains the local runner for up to five
seconds and marks interrupted work safely on the next startup.

`POST /api/observations` accepts a multipart GeoTIFF/TIFF file field named `file`. A successful response distinguishes the immutable original from its display-only COG and provides the tile scheme, extent, and URL template.

Raster tiles are available from:

```http
GET /tiles/{visualization_asset_id}/{z}/{x}/{y}.png
```

Georeferenced observations use Web Mercator XYZ tiles. Observations without usable georeferencing use the explicitly reported pixel tile scheme and `pixel_y_axis: down`; the API never invents a geographic placement.

Single-image VQA is available for registered observations:

```http
POST /api/vqa
Content-Type: application/json

{"observation_id":"obs_<server-generated-id>","question":"Is water present?"}
```

The response is structured evidence containing the answer, exact model/checkpoint and preprocessing provenance, source observation and visualization asset, domain status, and warnings. SmolVLM does not expose a meaningful calibrated answer score through this generation path, so `raw_score` is omitted rather than fabricated.

Inference defaults to local-only operation (`ENABLE_REMOTE_NETWORK=false`). Prime and verify the exact registered checkpoint with the Phase 2A evaluation command in `ml/README.md`; a missing checkpoint produces structured `503 MODEL_UNAVAILABLE` evidence-boundary failure.

Text-guided grounding is available for registered observations:

```http
POST /api/grounding
Content-Type: application/json

{"observation_id":"obs_<server-generated-id>","query":"the storage tank"}
```

Each detection includes the model-input box, source-image pixel box, normalized
source box, raw Grounding DINO score, and—only when the observation has valid CRS
and affine metadata—a four-corner world polygon. Missing detections return an
empty evidence list with a warning; the API never invents geometry.

The frozen production policy uses box/text thresholds 0.30/0.30. After mapping
boxes to normalized source-image coordinates, it discards boxes covering at least
80% of the image and returns only the highest-scoring remaining detection. If all
detections are oversized, the response is successful empty evidence with
`GROUNDING_ABSTAINED_OVERSIZED_BOXES`; abstention is not an execution failure.
