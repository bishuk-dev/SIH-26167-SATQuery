"""FastAPI application factory and default ASGI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.app.routes.observations import invalid_request_response, router
from apps.api.app.routes.tiles import router as tiles_router
from apps.api.app.services.observations import ObservationIngestionService
from satquery.ingestion import (
    FilesystemObservationStore,
    RasterInspector,
    RasterSafetyLimits,
)
from satquery.visualization.config import VisualizationSettings
from satquery.visualization.derivatives import VisualizationDerivativeGenerator
from satquery.visualization.tiles import RasterTileService


def create_app(
    *,
    data_root: str | Path | None = None,
    limits: RasterSafetyLimits | None = None,
    visualization_settings: VisualizationSettings | None = None,
) -> FastAPI:
    safety_limits = limits or RasterSafetyLimits.from_env()
    display_settings = visualization_settings or VisualizationSettings.from_env()
    storage_root = Path(data_root or os.environ.get("DATA_ROOT", "./data"))
    store = FilesystemObservationStore(storage_root)
    application = FastAPI(title="SatQuery API", version="0.1.0")
    application.state.observation_ingestion_service = ObservationIngestionService(
        inspector=RasterInspector(safety_limits),
        store=store,
        derivative_generator=VisualizationDerivativeGenerator(display_settings),
    )
    application.state.raster_tile_service = RasterTileService(store, display_settings)
    application.include_router(router)
    application.include_router(tiles_router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return invalid_request_response()

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        if request.url.path == "/api/observations" and error.status_code < 500:
            return invalid_request_response()
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    return application


app = create_app()
