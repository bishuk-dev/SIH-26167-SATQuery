"""Public, read-only artifact metadata, download, GeoJSON, and tile routes.

Every route resolves registered IDs through the verified ``ArtifactStore``,
re-checks metadata and SHA-256 before serving, and exposes only approved
media types and suffixes. No route leaks local filesystem paths.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.openapi import error_responses
from apps.api.app.schemas import ApiModel
from apps.api.app.schemas_v1 import FailureOutcomeV1
from satquery.artifacts import (
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactSummary,
    mask_footprint_geojson,
)
from satquery.visualization.exceptions import (
    InvalidTileRequestError,
    TileRenderingError,
)
from satquery.visualization.tiles import RasterTileService

router = APIRouter()

# Code-owned registered colormaps. The tile alias accepts only these IDs,
# never arbitrary colormap expressions.
_REGISTERED_COLORMAPS: dict[str, dict[int, tuple[int, int, int, int]]] = {
    "binary_mask_red": {0: (0, 0, 0, 0), 1: (227, 26, 28, 210)},
    "binary_mask_grayscale": {0: (0, 0, 0, 0), 1: (64, 64, 64, 255)},
}


class ArtifactV1(ApiModel):
    """Safe public projection of a published derived artifact."""

    artifact_id: str
    storage_key: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: datetime
    analysis_id: str | None = None
    evidence_id: str | None = None
    description: str | None = None
    extra: dict[str, Any]


def _project(summary: ArtifactSummary) -> ArtifactV1:
    return ArtifactV1(
        artifact_id=summary.artifact_id,
        storage_key=summary.storage_key,
        sha256=summary.sha256,
        size_bytes=summary.size_bytes,
        media_type=summary.media_type,
        created_at=summary.created_at,
        analysis_id=summary.analysis_id,
        evidence_id=summary.evidence_id,
        description=summary.description,
        extra=summary.extra,
    )


def _artifact_error(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    return failure_response(
        code=code,
        message=message,
        outcome=FailureOutcomeV1.REJECT,
        status_code=status_code,
        request_id=request_id_from(request),
    )


@router.get(
    "/api/v1/artifacts/{artifact_id}",
    response_model=ArtifactV1,
    operation_id="get_artifact_v1",
    summary="Inspect a published derived artifact.",
    description="Returns public metadata while keeping server filesystem paths private.",
    tags=["Artifacts"],
    responses=error_responses(401, 404),
)
def get_artifact_v1(request: Request, artifact_id: str) -> ArtifactV1:
    store: ArtifactStore = request.app.state.artifact_store
    try:
        record, _path = store.resolve(artifact_id)
    except (ArtifactNotFoundError, ArtifactStoreError):
        raise HTTPException(status_code=404) from None
    return _project(store.summarize(record))


@router.get(
    "/api/v1/artifacts/{artifact_id}/download",
    response_class=FileResponse,
    operation_id="download_artifact_v1",
    summary="Download a hash-verified published artifact.",
    description="Streams one immutable artifact after rechecking its registered SHA-256.",
    tags=["Artifacts"],
    responses={
        200: {
            "description": "The verified artifact bytes.",
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        },
        **error_responses(401, 404),
    },
)
def download_artifact_v1(request: Request, artifact_id: str) -> Response:
    store: ArtifactStore = request.app.state.artifact_store
    try:
        record, path = store.resolve(artifact_id)
        media_type = store.approved_media_type(record)
        filename = store.download_filename(record)
    except (ArtifactNotFoundError, ArtifactStoreError):
        raise HTTPException(status_code=404) from None
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        headers={"X-Artifact-SHA256": record.sha256},
    )


@router.get(
    "/api/v1/evidence/{evidence_id}/geojson",
    operation_id="evidence_geojson_v1",
    summary="Serve the recorded spatial footprint of mask evidence.",
    description="Returns GeoJSON derived from recorded mask grid provenance.",
    tags=["Evidence"],
    responses={
        200: {
            "description": "A GeoJSON spatial footprint.",
            "content": {"application/geo+json": {"schema": {"type": "object"}}},
        },
        **error_responses(401, 404, 409),
    },
)
def evidence_geojson_v1(request: Request, evidence_id: str) -> Response:
    store: ArtifactStore = request.app.state.artifact_store
    try:
        record, _path = store.resolve_for_evidence(evidence_id)
    except (ArtifactNotFoundError, ArtifactStoreError):
        raise HTTPException(status_code=404) from None
    try:
        grid = store.mask_grid(record)
    except ArtifactNotFoundError:
        return _artifact_error(
            request,
            409,
            "ARTIFACT_NOT_SPATIAL",
            "The artifact linked to this evidence has no spatial grid provenance.",
        )
    footprint = mask_footprint_geojson(record, grid)
    return JSONResponse(content=footprint, media_type="application/geo+json")


@router.get(
    "/api/v1/tiles/{artifact_id}/{z}/{x}/{y}.png",
    response_class=Response,
    operation_id="artifact_tile_v1",
    summary="Render a canonical XYZ tile alias for a registered mask artifact.",
    description="Renders a PNG tile using a code-owned registered binary-mask colormap.",
    tags=["Tiles"],
    responses={
        200: {
            "description": "A rendered PNG tile.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        **error_responses(400, 401, 404, 409, 500),
    },
)
def artifact_tile_v1(
    request: Request,
    artifact_id: str,
    z: int,
    x: int,
    y: int,
    colormap_id: str = "binary_mask_red",
) -> Response:
    colormap = _REGISTERED_COLORMAPS.get(colormap_id)
    if colormap is None:
        return _artifact_error(
            request,
            400,
            "INVALID_COLORMAP",
            "The requested colormap_id is not registered.",
        )
    store: ArtifactStore = request.app.state.artifact_store
    try:
        record, path = store.resolve(artifact_id)
        media_type = store.approved_media_type(record)
    except (ArtifactNotFoundError, ArtifactStoreError):
        raise HTTPException(status_code=404) from None
    try:
        store.mask_grid(record)
    except ArtifactNotFoundError:
        return _artifact_error(
            request,
            409,
            "ARTIFACT_NOT_TILEABLE",
            "Only registered raster mask artifacts expose tile aliases.",
        )
    if media_type != "image/tiff":
        return _artifact_error(
            request,
            409,
            "ARTIFACT_NOT_TILEABLE",
            "Only registered raster mask artifacts expose tile aliases.",
        )
    service: RasterTileService = request.app.state.raster_tile_service
    try:
        tile = service.render_binary_mask_tile(path, z, x, y, colormap=colormap)
    except InvalidTileRequestError:
        return _artifact_error(
            request,
            400,
            "INVALID_TILE_REQUEST",
            "The requested tile coordinates are invalid.",
        )
    except TileRenderingError:
        return _artifact_error(
            request,
            500,
            "TILE_RENDERING_FAILED",
            "SatQuery could not render the requested artifact tile.",
        )
    return Response(
        content=tile,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


__all__ = ["router"]
