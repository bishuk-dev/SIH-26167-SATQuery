"""Text-guided grounding endpoint backed by registered model evidence."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field, field_validator

from apps.api.app.schemas import ApiModel, ErrorResponse, FailureDetail
from satquery.evidence.models import GroundingEvidence
from satquery.inference.exceptions import (
    EvidenceGeometryError,
    ModelExecutionError,
    ModelInputUnsupportedError,
    ModelUnavailableError,
)
from satquery.inference.grounding import TextGuidedGroundingService
from satquery.ingestion.exceptions import ObservationNotFoundError

router = APIRouter(prefix="/api/grounding", tags=["grounding"])


class GroundingRequest(ApiModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def require_meaningful_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not any(character.isalnum() for character in cleaned):
            raise ValueError("query must contain letters or numbers")
        return cleaned


@router.post(
    "",
    response_model=GroundingEvidence,
    response_model_exclude_none=True,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def text_guided_grounding(
    request: Request,
    payload: GroundingRequest,
) -> GroundingEvidence | JSONResponse:
    service: TextGuidedGroundingService = (
        request.app.state.text_guided_grounding_service
    )
    try:
        return service.ground(payload.observation_id, payload.query)
    except ObservationNotFoundError:
        return _grounding_error(
            404,
            code="OBSERVATION_NOT_FOUND",
            user_message="The requested observation was not found.",
            action={"type": "provide_registered_observation"},
        )
    except ModelInputUnsupportedError:
        return _grounding_error(
            422,
            code="MODEL_INPUT_UNSUPPORTED",
            user_message="This observation cannot be prepared for grounding.",
            action={"type": "provide_supported_observation"},
        )
    except EvidenceGeometryError:
        return _grounding_error(
            422,
            code="INVALID_EVIDENCE_GEOMETRY",
            user_message="The model geometry could not be mapped to this raster.",
            action={"type": "retry_or_inspect_source_mapping"},
        )
    except ModelUnavailableError:
        return _grounding_error(
            503,
            code="MODEL_UNAVAILABLE",
            user_message="The registered grounding model is unavailable.",
            action={"type": "install_registered_model"},
        )
    except ModelExecutionError:
        return _grounding_error(
            500,
            code="MODEL_EXECUTION_FAILED",
            user_message="The registered grounding model could not process the query.",
            action={"type": "retry_request"},
        )


def invalid_grounding_request_response() -> JSONResponse:
    return _grounding_error(
        422,
        code="INVALID_GROUNDING_REQUEST",
        user_message="A registered observation ID and non-empty text query are required.",
        action={"type": "correct_request"},
    )


def _grounding_error(
    status_code: int,
    *,
    code: str,
    user_message: str,
    action: dict[str, object],
) -> JSONResponse:
    payload = ErrorResponse(
        error=FailureDetail(
            code=code,
            severity="ERROR",
            outcome="REJECT",
            user_message=user_message,
            technical_message=user_message,
            affected_requirement="R-EVIDENCE-002",
            recoverable=True,
            required_action=action,
        )
    ).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=payload)
