# API application

This directory contains the FastAPI transport boundary. Run the current API from the repository root with:

```bash
uvicorn apps.api.app.main:app --reload
```

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
