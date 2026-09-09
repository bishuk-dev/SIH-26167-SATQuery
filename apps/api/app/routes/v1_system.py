"""Canonical v1 system routes: liveness and service version."""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.app.openapi import API_VERSION
from apps.api.app.schemas import ApiModel

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
