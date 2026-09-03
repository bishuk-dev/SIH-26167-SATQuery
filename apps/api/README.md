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
