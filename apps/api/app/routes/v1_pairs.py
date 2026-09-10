"""Canonical v1 observation-pair validation and persistence routes."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.openapi import error_responses
from apps.api.app.schemas_v1 import (
    FailureOutcomeV1,
    PairCreateRequest,
    PairPage,
    PairTemporalOrder,
    PairTypeV1,
    PairValidationResponse,
)
from satquery.geo import PairValidator
from satquery.geo.models import CompatibilityStatus, ModalityPairType, PairCompatibility
from satquery.ingestion.exceptions import ObservationNotFoundError
from satquery.persistence import (
    MetadataRepository,
    PageCursor,
    PairRecord,
    PersistenceError,
)
from satquery.visualization.models import ObservationRegistration

router = APIRouter(prefix="/api/v1/pairs", tags=["Pairs"])


def _load_pair_observations(
    request: Request, payload: PairCreateRequest
) -> tuple[ObservationRegistration, ObservationRegistration]:
    store = request.app.state.observation_store
    try:
        first, _ = store.load_registration(payload.observation_a)
        second, _ = store.load_registration(payload.observation_b)
    except ObservationNotFoundError:
        raise HTTPException(status_code=404) from None
    return first, second


def _temporal_order(
    validation: PairCompatibility, payload: PairCreateRequest
) -> PairTemporalOrder | None:
    if payload.explicit_t1_id is not None:
        t2_id = (
            payload.observation_b
            if payload.explicit_t1_id == payload.observation_a
            else payload.observation_a
        )
        return PairTemporalOrder(
            t1_observation_id=payload.explicit_t1_id,
            t2_observation_id=t2_id,
        )
    if validation.temporal.order_known:
        return PairTemporalOrder(
            t1_observation_id=validation.temporal.first,  # type: ignore[arg-type]
            t2_observation_id=validation.temporal.second,  # type: ignore[arg-type]
        )
    return None


def _failure(
    request: Request,
    *,
    code: str,
    message: str,
    outcome: FailureOutcomeV1,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return failure_response(
        code=code,
        message=message,
        outcome=outcome,
        status_code=422,
        request_id=request_id_from(request),
        details=details,
    )


def _validate(
    request: Request, payload: PairCreateRequest
) -> tuple[PairCompatibility, PairTemporalOrder | None, FailureOutcomeV1, tuple[str, ...]] | JSONResponse:
    if payload.observation_a == payload.observation_b:
        return _failure(
            request,
            code="DUPLICATE_OBSERVATION",
            message="A pair must contain two distinct observations.",
            outcome=FailureOutcomeV1.REJECT,
        )

    first, second = _load_pair_observations(request, payload)
    validation = PairValidator().validate(first.observation, second.observation)
    reasons = validation.result.reasons

    if "DUPLICATE_SOURCE_ASSET" in reasons:
        return _failure(
            request,
            code="DUPLICATE_SOURCE_ASSET",
            message="The pair observations refer to the same immutable source asset.",
            outcome=FailureOutcomeV1.REJECT,
            details={"reasons": list(reasons)},
        )
    if "NO_SPATIAL_OVERLAP" in reasons:
        return _failure(
            request,
            code="NO_SPATIAL_OVERLAP",
            message="The observations do not overlap geographically.",
            outcome=FailureOutcomeV1.REJECT,
            details={"reasons": list(reasons)},
        )

    if (
        payload.explicit_t1_id is not None
        and validation.temporal.order_known
        and payload.explicit_t1_id != validation.temporal.first
    ):
        return _failure(
            request,
            code="TEMPORAL_ORDER_CONFLICT",
            message="The explicit T1 observation conflicts with acquisition-time ordering.",
            outcome=FailureOutcomeV1.REJECT,
            details={"reasons": list(reasons)},
        )

    temporal_order = _temporal_order(validation, payload)
    is_temporal = payload.pair_type in {
        PairTypeV1.TEMPORAL,
        PairTypeV1.TEMPORAL_SAME_MODALITY,
        PairTypeV1.TEMPORAL_CROSS_MODAL,
    }
    if is_temporal and temporal_order is None:
        return _failure(
            request,
            code="TEMPORAL_ORDER_UNKNOWN",
            message="Temporal pairing requires acquisition dates or an explicit T1 observation.",
            outcome=FailureOutcomeV1.REQUEST_INPUT,
            details={"reasons": list(reasons)},
        )
    expected_modality = {
        PairTypeV1.TEMPORAL_SAME_MODALITY: ModalityPairType.TEMPORAL_SAME_MODALITY,
        PairTypeV1.TEMPORAL_CROSS_MODAL: ModalityPairType.TEMPORAL_CROSS_MODAL,
        PairTypeV1.OPTICAL_SAR: ModalityPairType.OPTICAL_SAR,
    }.get(payload.pair_type)
    if expected_modality is not None and validation.modality.pair_type is not expected_modality:
        return _failure(
            request,
            code="PAIR_TYPE_MISMATCH",
            message="The observations do not match the requested pair type.",
            outcome=FailureOutcomeV1.REJECT,
            details={"actual_pair_type": validation.modality.pair_type.value},
        )
    if validation.result.status is CompatibilityStatus.FAIL:
        return _failure(
            request,
            code="PAIR_VALIDATION_FAILED",
            message="The observations failed compatibility validation.",
            outcome=FailureOutcomeV1.REJECT,
            details={"reasons": list(reasons)},
        )

    warnings = tuple(
        reason
        for reason in reasons
        if not (reason == "TEMPORAL_ORDER_UNKNOWN" and temporal_order is not None)
    )
    outcome = (
        FailureOutcomeV1.ALLOW_WITH_WARNING
        if warnings or validation.result.status is CompatibilityStatus.WARN
        else FailureOutcomeV1.ALLOW
    )
    return validation, temporal_order, outcome, warnings


def _response(
    payload: PairCreateRequest,
    validation: PairCompatibility,
    temporal_order: PairTemporalOrder | None,
    outcome: FailureOutcomeV1,
    warnings: tuple[str, ...],
    *,
    pair_id: str | None = None,
    created_at: datetime | None = None,
) -> PairValidationResponse:
    return PairValidationResponse(
        pair_id=pair_id,
        observation_a=payload.observation_a,
        observation_b=payload.observation_b,
        pair_type=payload.pair_type,
        explicit_t1_id=payload.explicit_t1_id,
        temporal_order=temporal_order,
        validation=validation.model_dump(mode="json"),
        outcome=outcome,
        warnings=warnings,
        created_at=created_at,
    )


def _payload_from_response(response: PairValidationResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")


def _response_from_record(record: PairRecord) -> PairValidationResponse:
    try:
        return PairValidationResponse.model_validate(record.payload)
    except ValueError:
        raise PersistenceError("Persisted pair validation snapshot is invalid") from None


def _encode_cursor(cursor: PageCursor) -> str:
    value = json.dumps(
        {
            "created_at": cursor.created_at.isoformat(),
            "record_id": cursor.record_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


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


@router.post(
    "/validate",
    response_model=PairValidationResponse,
    operation_id="validate_pair_v1",
    summary="Validate two observations as a supported pair.",
    description="Checks spatial, temporal, modality, and grid compatibility without persisting the pair.",
    responses=error_responses(401, 404, 422),
)
def validate_pair_v1(
    request: Request, payload: PairCreateRequest
) -> PairValidationResponse | JSONResponse:
    result = _validate(request, payload)
    if isinstance(result, JSONResponse):
        return result
    validation, temporal_order, outcome, warnings = result
    return _response(payload, validation, temporal_order, outcome, warnings)


@router.post(
    "",
    status_code=201,
    response_model=PairValidationResponse,
    operation_id="create_pair_v1",
    summary="Create a validated observation pair.",
    description="Validates and persists a pair snapshot for later analysis planning.",
    responses={
        201: {"description": "Pair validated and persisted."},
        **error_responses(401, 404, 409, 422),
    },
)
def create_pair_v1(
    request: Request, payload: PairCreateRequest
) -> PairValidationResponse | JSONResponse:
    result = _validate(request, payload)
    if isinstance(result, JSONResponse):
        return result
    validation, temporal_order, outcome, warnings = result
    now = datetime.now(timezone.utc)
    pair_id = f"pair_{uuid4().hex}"
    response = _response(
        payload,
        validation,
        temporal_order,
        outcome,
        warnings,
        pair_id=pair_id,
        created_at=now,
    )
    repository: MetadataRepository = request.app.state.observation_repository
    try:
        repository.create_pair(
            PairRecord(
                pair_id=pair_id,
                observation_a_id=payload.observation_a,
                observation_b_id=payload.observation_b,
                created_at=now,
                payload=_payload_from_response(response),
            )
        )
    except PersistenceError:
        return _failure(
            request,
            code="PAIR_STORAGE_FAILED",
            message="The validated pair could not be persisted.",
            outcome=FailureOutcomeV1.REJECT,
        )
    return response


@router.get(
    "",
    response_model=PairPage,
    operation_id="list_pairs_v1",
    summary="List validated observation pairs.",
    description="Returns keyset-paginated persisted pair validation snapshots.",
    responses=error_responses(401, 422),
)
def list_pairs_v1(
    request: Request,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> PairPage:
    repository: MetadataRepository = request.app.state.observation_repository
    records = repository.list_pairs(cursor=_decode_cursor(cursor), limit=limit)
    next_cursor = None
    if len(records) == limit:
        last = records[-1]
        next_cursor = _encode_cursor(
            PageCursor(created_at=last.created_at, record_id=last.pair_id)
        )
    return PairPage(
        items=tuple(_response_from_record(record) for record in records),
        next_cursor=next_cursor,
    )


@router.get(
    "/{pair_id}",
    response_model=PairValidationResponse,
    operation_id="get_pair_v1",
    summary="Inspect one validated observation pair.",
    description="Returns the immutable validation snapshot for a server-issued pair ID.",
    responses=error_responses(401, 404),
)
def get_pair_v1(request: Request, pair_id: str) -> PairValidationResponse:
    repository: MetadataRepository = request.app.state.observation_repository
    record = repository.get_pair(pair_id)
    if record is None:
        raise HTTPException(status_code=404)
    return _response_from_record(record)
