"""FastAPI application factory and default ASGI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.app import openapi as openapi_metadata
from apps.api.app.errors import (
    failure_response,
    internal_error_response,
    new_request_id,
    request_id_from,
    safe_validation_details,
    with_request_id,
)
from apps.api.app.routes.observations import invalid_request_response, router
from apps.api.app.routes.grounding import (
    invalid_grounding_request_response,
    router as grounding_router,
)
from apps.api.app.routes.tiles import router as tiles_router
from apps.api.app.routes.v1_observations import (
    index_existing_observations,
    router as v1_observations_router,
)
from apps.api.app.routes.v1_pairs import router as v1_pairs_router
from apps.api.app.routes.v1_registry import router as v1_registry_router
from apps.api.app.routes.v1_system import router as v1_system_router
from apps.api.app.routes.vqa import invalid_vqa_request_response
from apps.api.app.routes.vqa import router as vqa_router
from apps.api.app.schemas_v1 import FailureOutcomeV1
from apps.api.app.services.observations import ObservationIngestionService
from satquery.inference.config import GroundingRuntimeSettings, VqaRuntimeSettings
from satquery.inference.grounding import GroundingBackend, TextGuidedGroundingService
from satquery.inference.vqa import SingleImageVqaService, VqaBackend
from satquery.ingestion import (
    FilesystemObservationStore,
    RasterInspector,
    RasterSafetyLimits,
)
from satquery.persistence import Database, MetadataRepository
from satquery.registry import (
    load_model_registry,
    load_runtime_capabilities,
    load_tool_registry,
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
    database = Database(store.data_root / "satquery.db")
    database.migrate()
    repository = MetadataRepository(database)
    index_existing_observations(store, repository)
    tool_registry = load_tool_registry()
    model_registry = load_model_registry()
    runtime_capabilities = load_runtime_capabilities(
        tool_registry, model_registry=model_registry
    )
    application = FastAPI(
        title=openapi_metadata.API_TITLE,
        summary=openapi_metadata.API_SUMMARY,
        description=openapi_metadata.API_DESCRIPTION,
        version=openapi_metadata.API_VERSION,
        openapi_tags=openapi_metadata.TAGS,
        docs_url=openapi_metadata.DOCS_URL,
        redoc_url=openapi_metadata.REDOC_URL,
        openapi_url=openapi_metadata.OPENAPI_URL,
    )
    application.state.observation_repository = repository
    application.state.observation_store = store
    application.state.tool_registry = tool_registry
    application.state.model_registry = model_registry
    application.state.runtime_capabilities = runtime_capabilities
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
    application.include_router(v1_system_router)
    application.include_router(v1_observations_router)
    application.include_router(v1_pairs_router)
    application.include_router(v1_registry_router)

    add_request_id_middleware(application)
    install_v1_error_handlers(application)

    return application


def add_request_id_middleware(application: FastAPI) -> None:
    """Attach a server-generated correlation ID to every exchange.

    Incoming X-Request-ID values are never trusted. The ID is stored on
    ``request.state.request_id`` so handlers, error envelopes, and future
    structured logging share one value.
    """

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = new_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def install_v1_error_handlers(application: FastAPI) -> None:
    """Install the v1/legacy error boundary.

    Paths under ``/api/v1/`` use the frozen v1 envelope; every other path
    keeps its pre-existing legacy behavior exactly.
    """

    def _is_v1(request: Request) -> bool:
        return request.url.path.startswith("/api/v1/")

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        if _is_v1(request):
            request_id = request_id_from(request)
            return failure_response(
                code="INVALID_REQUEST",
                message="The request body failed validation.",
                outcome=FailureOutcomeV1.REQUEST_INPUT,
                status_code=422,
                request_id=request_id,
                details=safe_validation_details(error),
            )
        if request.url.path == "/api/vqa":
            return invalid_vqa_request_response()
        if request.url.path == "/api/grounding":
            return invalid_grounding_request_response()
        return invalid_request_response()

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        if _is_v1(request):
            request_id = request_id_from(request)
            code = {
                404: "NOT_FOUND",
                405: "METHOD_NOT_ALLOWED",
            }.get(error.status_code, f"HTTP_{error.status_code}")
            message = {
                404: "The requested API resource was not found.",
                405: "The HTTP method is not allowed on this resource.",
            }.get(error.status_code, "The request could not be processed.")
            return failure_response(
                code=code,
                message=message,
                outcome=FailureOutcomeV1.REJECT,
                status_code=error.status_code,
                request_id=request_id,
            )
        if request.url.path == "/api/observations" and error.status_code < 500:
            return invalid_request_response()
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    @application.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request, _error: Exception
    ) -> JSONResponse | PlainTextResponse:
        # Starlette routes all unhandled exceptions here; v1 paths get the
        # sanitized envelope while legacy paths keep the default Starlette
        # plaintext response so no legacy behavior changes.
        if _is_v1(request):
            return internal_error_response(request_id_from(request))
        # legacy paths keep the default Starlette plaintext response (with
        # the correlation header added, per the Task-1 every-response rule)
        return with_request_id(
            PlainTextResponse("Internal Server Error", status_code=500),
            request_id_from(request),
        )


app = create_app()
