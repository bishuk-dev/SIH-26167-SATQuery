"""FastAPI application factory and default ASGI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.app.routes.observations import invalid_request_response, router
from apps.api.app.routes.grounding import (
    invalid_grounding_request_response,
    router as grounding_router,
)
from apps.api.app.routes.tiles import router as tiles_router
from apps.api.app.routes.vqa import invalid_vqa_request_response
from apps.api.app.routes.vqa import router as vqa_router
from apps.api.app.services.observations import ObservationIngestionService
from satquery.inference.config import GroundingRuntimeSettings, VqaRuntimeSettings
from satquery.inference.grounding import GroundingBackend, TextGuidedGroundingService
from satquery.inference.vqa import SingleImageVqaService, VqaBackend
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
    vqa_settings: VqaRuntimeSettings | None = None,
    vqa_backend: VqaBackend | None = None,
    grounding_settings: GroundingRuntimeSettings | None = None,
    grounding_backend: GroundingBackend | None = None,
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
    application.state.single_image_vqa_service = SingleImageVqaService(
        store,
        settings=vqa_settings,
        backend=vqa_backend,
    )
    application.state.text_guided_grounding_service = TextGuidedGroundingService(
        store,
        settings=grounding_settings,
        backend=grounding_backend,
    )
    application.include_router(router)
    application.include_router(tiles_router)
    application.include_router(vqa_router)
    application.include_router(grounding_router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        if request.url.path == "/api/vqa":
            return invalid_vqa_request_response()
        if request.url.path == "/api/grounding":
            return invalid_grounding_request_response()
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
