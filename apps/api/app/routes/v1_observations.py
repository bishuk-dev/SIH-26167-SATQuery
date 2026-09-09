"""Canonical v1 observation lifecycle routes and metadata index."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.routes.observations import _failure_detail as legacy_failure_detail
from apps.api.app.schemas import (
    ApiModel,
    AssetResponse,
    ObservationMetadataResponse,
    ObservationUploadResponse,
    VisualizationAssetResponse,
)
from apps.api.app.schemas_v1 import FailureOutcomeV1
from apps.api.app.services.observations import ObservationIngestionService
from satquery.ingestion import FilesystemObservationStore
from satquery.ingestion.exceptions import AssetStorageError, IngestionError
from satquery.persistence import (
    MetadataRepository,
    ObservationRecord,
    PageCursor,
    PersistenceError,
    RecordAlreadyExistsError,
)
from satquery.visualization.models import ObservationRegistration

router = APIRouter(prefix="/api/v1/observations", tags=["Observations"])


class ObservationPage(ApiModel):
    items: tuple[ObservationUploadResponse, ...]
    next_cursor: str | None = None


class ObservationAssets(ApiModel):
    items: tuple[AssetResponse | VisualizationAssetResponse, ...]


def index_registration(
    repository: MetadataRepository, registration: ObservationRegistration
) -> None:
    """Index immutable filesystem metadata without moving or rewriting assets."""

    observation = registration.observation
    if repository.get_observation(observation.observation_id) is not None:
        return
    try:
        repository.create_observation(
            ObservationRecord(
                observation_id=observation.observation_id,
                created_at=observation.provenance.created_at,
                payload={
                    "observation": observation.model_dump(mode="json"),
                    "visualization": registration.visualization.model_dump(mode="json"),
                },
            )
        )
    except RecordAlreadyExistsError:
        # A concurrent startup/upload indexed the same immutable registration.
        return
    except PersistenceError as exc:
        raise AssetStorageError("Could not index observation metadata") from exc


def index_existing_observations(
    store: FilesystemObservationStore, repository: MetadataRepository
) -> None:
    """Index all existing registrations during application startup."""

    for registration in store.list_registrations():
        index_registration(repository, registration)


def registration_from_record(record: ObservationRecord) -> ObservationRegistration:
    try:
        return ObservationRegistration.model_validate_json(json.dumps(record.payload))
    except ValueError as exc:
        raise AssetStorageError("Indexed observation metadata is invalid") from exc


def get_observation_registration(
    request: Request, observation_id: str
) -> ObservationRegistration:
    repository: MetadataRepository = request.app.state.observation_repository
    record = repository.get_observation(observation_id)
    if record is None:
        raise HTTPException(status_code=404)
    return registration_from_record(record)


def _encode_cursor(cursor: PageCursor) -> str:
    payload = json.dumps(
        {
            "created_at": cursor.created_at.isoformat(),
            "record_id": cursor.record_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> PageCursor | None:
    if value is None:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        return PageCursor(
            created_at=datetime.fromisoformat(payload["created_at"]),
            record_id=payload["record_id"],
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(status_code=422) from None


def _to_response(record: ObservationRecord) -> ObservationUploadResponse:
    return ObservationUploadResponse.from_registration(registration_from_record(record))


@router.post(
    "",
    status_code=201,
    response_model=ObservationUploadResponse,
)
async def create_observation_v1(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> ObservationUploadResponse | JSONResponse:
    service: ObservationIngestionService = request.app.state.observation_ingestion_service
    repository: MetadataRepository = request.app.state.observation_repository
    try:
        registration = await service.ingest(file)
        index_registration(repository, registration)
    except IngestionError as exc:
        return v1_ingestion_error_response(request, exc)
    return ObservationUploadResponse.from_registration(registration)


def v1_ingestion_error_response(
    request: Request, error: IngestionError
) -> JSONResponse:
    status_code, detail = legacy_failure_detail(error)
    return failure_response(
        code=detail.code,
        message=detail.user_message,
        outcome=FailureOutcomeV1(detail.outcome),
        status_code=status_code,
        request_id=request_id_from(request),
        details={
            "affected_requirement": detail.affected_requirement,
            "recoverable": detail.recoverable,
            "required_action": detail.required_action,
            "evidence_ids": list(detail.evidence_ids),
            "warnings": list(detail.warnings),
        },
    )


@router.get("", response_model=ObservationPage)
def list_observations_v1(
    request: Request,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> ObservationPage:
    repository: MetadataRepository = request.app.state.observation_repository
    records = repository.list_observations(
        cursor=_decode_cursor(cursor),
        limit=limit,
    )
    next_cursor = None
    if len(records) == limit:
        last = records[-1]
        next_cursor = _encode_cursor(
            PageCursor(created_at=last.created_at, record_id=last.observation_id)
        )
    return ObservationPage(
        items=tuple(_to_response(record) for record in records),
        next_cursor=next_cursor,
    )


@router.get("/{observation_id}", response_model=ObservationUploadResponse)
def get_observation_v1(
    request: Request, observation_id: str
) -> ObservationUploadResponse:
    repository: MetadataRepository = request.app.state.observation_repository
    record = repository.get_observation(observation_id)
    if record is None:
        raise HTTPException(status_code=404)
    return _to_response(record)


@router.get("/{observation_id}/metadata", response_model=ObservationMetadataResponse)
def get_observation_metadata_v1(
    request: Request, observation_id: str
) -> ObservationMetadataResponse:
    registration = get_observation_registration(request, observation_id)
    return ObservationMetadataResponse(
        raster=registration.observation.raster,
        sensor=registration.observation.sensor,
        geo=registration.observation.geo,
        temporal=registration.observation.temporal,
        provenance=registration.observation.provenance,
    )


@router.get("/{observation_id}/assets", response_model=ObservationAssets)
def get_observation_assets_v1(
    request: Request, observation_id: str
) -> ObservationAssets:
    registration = get_observation_registration(request, observation_id)
    projection = ObservationUploadResponse.from_registration(registration)
    return ObservationAssets(items=(projection.asset, projection.visualization))


@router.delete("/{observation_id}")
def delete_observation_v1(observation_id: str) -> None:
    """Source observations are immutable and cannot be deleted."""

    raise HTTPException(status_code=405)
