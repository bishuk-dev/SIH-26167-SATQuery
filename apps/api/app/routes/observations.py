"""Observation upload endpoint and failure mapping."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from apps.api.app.schemas import (
    ErrorResponse,
    FailureDetail,
    ObservationUploadResponse,
)
from apps.api.app.services.observations import ObservationIngestionService
from satquery.ingestion.exceptions import (
    AssetStorageError,
    IngestionError,
    InvalidRasterError,
    InvalidUploadError,
    RasterDriverMismatchError,
    RasterResourceLimitError,
    UnsupportedRasterDriverError,
)
from satquery.visualization.exceptions import (
    VisualizationGenerationError,
    VisualizationResourceLimitError,
)

router = APIRouter(prefix="/api/observations", tags=["observations"])


def get_ingestion_service(request: Request) -> ObservationIngestionService:
    return request.app.state.observation_ingestion_service


@router.post(
    "",
    status_code=201,
    response_model=ObservationUploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_observation(
    request: Request,
    file: UploadFile = File(...),
) -> ObservationUploadResponse | JSONResponse:
    service = get_ingestion_service(request)
    try:
        registration = await service.ingest(file)
    except IngestionError as exc:
        return ingestion_error_response(exc)
    return ObservationUploadResponse.from_registration(registration)


def ingestion_error_response(error: IngestionError) -> JSONResponse:
    status_code, detail = _failure_detail(error)
    payload = ErrorResponse(error=detail).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=payload)


def invalid_request_response() -> JSONResponse:
    return ingestion_error_response(
        InvalidUploadError("A GeoTIFF/TIFF file field named 'file' is required")
    )


def _failure_detail(error: IngestionError) -> tuple[int, FailureDetail]:
    if isinstance(error, VisualizationResourceLimitError):
        return 413, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="The visualization derivative exceeds configured limits.",
            technical_message=str(error),
            affected_requirement="R-SEC-003",
            recoverable=True,
            required_action={"type": "provide_smaller_raster"},
        )
    if isinstance(error, VisualizationGenerationError):
        return 500, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="SatQuery could not prepare this raster for visualization.",
            technical_message="Visualization derivative generation failed.",
            affected_requirement="R-UI-002",
            recoverable=True,
            required_action={"type": "retry_upload"},
        )
    if isinstance(error, RasterResourceLimitError):
        return 413, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="The raster exceeds the configured upload or raster safety limits.",
            technical_message=str(error),
            affected_requirement="R-SEC-003",
            recoverable=True,
            required_action={"type": "provide_smaller_raster"},
        )
    if isinstance(error, UnsupportedRasterDriverError):
        return 415, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="This raster format is unsupported. Upload a GeoTIFF/TIFF file.",
            technical_message=str(error),
            affected_requirement="R-SEC-002",
            recoverable=True,
            required_action={"type": "provide_supported_raster"},
        )
    if isinstance(error, RasterDriverMismatchError):
        return 422, FailureDetail(
            code=error.code,
            severity="CRITICAL",
            outcome="REJECT",
            user_message="The file extension does not match the detected raster format.",
            technical_message=str(error),
            affected_requirement="R-SEC-002",
            recoverable=True,
            required_action={"type": "provide_valid_raster"},
        )
    if isinstance(error, InvalidRasterError):
        return 422, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="The uploaded file is not a readable GeoTIFF/TIFF raster.",
            technical_message=str(error),
            affected_requirement="R-INPUT-004",
            recoverable=True,
            required_action={"type": "provide_valid_raster"},
        )
    if isinstance(error, InvalidUploadError):
        return 400, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="A valid GeoTIFF/TIFF upload is required.",
            technical_message=str(error),
            affected_requirement="R-SEC-001",
            recoverable=True,
            required_action={"type": "provide_valid_upload"},
        )
    if isinstance(error, AssetStorageError):
        return 500, FailureDetail(
            code=error.code,
            severity="ERROR",
            outcome="REJECT",
            user_message="SatQuery could not safely store the uploaded observation.",
            technical_message="Observation storage failed during ingestion.",
            affected_requirement="R-GEO-002",
            recoverable=True,
            required_action={"type": "retry_upload"},
        )
    return 422, FailureDetail(
        code=error.code,
        severity="ERROR",
        outcome="REJECT",
        user_message="SatQuery could not inspect the uploaded raster.",
        technical_message="Raster metadata inspection failed.",
        affected_requirement="R-GEO-001",
        recoverable=True,
        required_action={"type": "provide_valid_raster"},
    )
