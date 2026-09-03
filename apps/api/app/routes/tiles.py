"""Read-only raster tile endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from apps.api.app.schemas import ErrorResponse, FailureDetail
from satquery.visualization.exceptions import (
    InvalidTileRequestError,
    TileRenderingError,
    VisualizationAssetNotFoundError,
)
from satquery.visualization.tiles import RasterTileService

router = APIRouter(tags=["tiles"])


@router.get(
    "/tiles/{asset_id}/{z}/{x}/{y}.png",
    response_class=Response,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def raster_tile(
    request: Request,
    asset_id: str,
    z: int,
    x: int,
    y: int,
) -> Response:
    service: RasterTileService = request.app.state.raster_tile_service
    try:
        tile = service.render(asset_id, z, x, y)
    except VisualizationAssetNotFoundError:
        return _tile_error(
            404,
            code="VISUALIZATION_ASSET_NOT_FOUND",
            user_message="The requested visualization asset was not found.",
            recoverable=True,
        )
    except InvalidTileRequestError as exc:
        return _tile_error(
            400,
            code=exc.code,
            user_message="The requested tile coordinates are invalid.",
            recoverable=True,
        )
    except TileRenderingError:
        return _tile_error(
            500,
            code="TILE_RENDERING_FAILED",
            user_message="SatQuery could not render the requested raster tile.",
            recoverable=True,
        )
    return Response(
        content=tile,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def _tile_error(
    status_code: int,
    *,
    code: str,
    user_message: str,
    recoverable: bool,
) -> JSONResponse:
    payload = ErrorResponse(
        error=FailureDetail(
            code=code,
            severity="ERROR",
            outcome="REJECT",
            user_message=user_message,
            technical_message=user_message,
            affected_requirement="R-UI-002",
            recoverable=recoverable,
            required_action=None,
        )
    ).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=payload)
