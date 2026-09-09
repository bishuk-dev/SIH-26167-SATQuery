"""Canonical v1 system routes: liveness and service version."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from apps.api.app.openapi import API_VERSION
from apps.api.app.schemas import ApiModel
from satquery.observability import readiness_payload

router = APIRouter(tags=["System"])

# Frozen at Phase 5 Task 0 (experiments/phase5_backend/backend_contract.yaml,
# phase5_start.mode). A constant is deliberate: Task 1 performs no registry
# parsing or contract loading at request time.
PHASE5_MODE = "RESTRICTED_CAPABILITY"


class SystemVersionV1(ApiModel):
    api_version: str
    application: str
    application_version: str
    phase5_mode: str


class LivenessV1(ApiModel):
    status: str


class ReadinessV1(ApiModel):
    status: str
    components: dict[str, dict[str, Any]]


class SystemStatusV1(ReadinessV1):
    pass


class SystemLimitsV1(ApiModel):
    raster: dict[str, Any]
    visualization: dict[str, Any]
    query: dict[str, Any]
    queue: dict[str, Any]


@router.get(
    "/api/v1/system/version",
    operation_id="get_system_version_v1",
    response_model=SystemVersionV1,
    summary="Report the canonical API version and Phase 5 mode.",
)
def get_system_version_v1() -> SystemVersionV1:
    return SystemVersionV1(
        api_version="v1",
        application="satquery",
        application_version=API_VERSION,
        phase5_mode=PHASE5_MODE,
    )


@router.get(
    "/health/live",
    operation_id="get_liveness",
    response_model=LivenessV1,
    summary="Report process liveness.",
    description=(
        "Extremely cheap check: no models, checkpoints, network, raster "
        "inspection, or database access. Dependency readiness is reported "
        "separately once implemented."
    ),
)
def get_liveness() -> LivenessV1:
    return LivenessV1(status="alive")


@router.get(
    "/health/ready",
    operation_id="get_readiness",
    response_model=ReadinessV1,
    responses={503: {"model": ReadinessV1}},
    summary="Report dependency readiness without loading model checkpoints.",
    description=(
        "Checks SQLite, writable storage, loaded registries, queue capacity, "
        "and checkpoint bytes. It never runs inference or loads a checkpoint."
    ),
)
def get_readiness(request: Request) -> ReadinessV1 | JSONResponse:
    payload = readiness_payload(request.app)
    if payload["status"] != "ready":
        return JSONResponse(status_code=503, content=payload)
    return ReadinessV1.model_validate(payload)


@router.get(
    "/api/v1/system/status",
    operation_id="get_system_status_v1",
    response_model=SystemStatusV1,
    summary="Report dependency and capability status.",
)
def get_system_status_v1(request: Request) -> SystemStatusV1:
    return SystemStatusV1.model_validate(readiness_payload(request.app))


def _limits_payload(request: Request) -> dict[str, Any]:
    safety = request.app.state.safety_limits
    visualization = request.app.state.visualization_settings
    queue = request.app.state.job_runner._queue
    return {
        "raster": {
            "max_file_size_bytes": safety.max_file_size_bytes,
            "max_width": safety.max_width,
            "max_height": safety.max_height,
            "max_pixel_count": safety.max_pixel_count,
            "max_band_count": safety.max_band_count,
            "allowed_drivers": sorted(safety.allowed_drivers),
            "allowed_extensions": sorted(safety.allowed_extensions),
        },
        "visualization": {
            "max_derivative_size_bytes": visualization.max_derivative_size_bytes,
            "max_tile_zoom": visualization.max_tile_zoom,
            "tile_size": visualization.tile_size,
            "derivative_block_size": visualization.derivative_block_size,
            "statistics_sample_size": visualization.statistics_sample_size,
        },
        "query": {"max_characters": 500, "max_plan_steps": 8},
        "queue": {
            "max_queued_jobs": queue.maxsize,
            "worker_count": request.app.state.job_runner._worker_count,
        },
    }


@router.get(
    "/api/v1/system/limits",
    operation_id="get_system_limits_v1",
    response_model=SystemLimitsV1,
    summary="Report configured resource limits.",
)
def get_system_limits_v1(request: Request) -> SystemLimitsV1:
    return SystemLimitsV1.model_validate(_limits_payload(request))


@router.get(
    "/limits",
    operation_id="get_limits",
    response_model=SystemLimitsV1,
    summary="Report configured resource limits.",
)
def get_limits(request: Request) -> SystemLimitsV1:
    return SystemLimitsV1.model_validate(_limits_payload(request))
