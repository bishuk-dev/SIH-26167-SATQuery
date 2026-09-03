"""Single-image VQA endpoint backed by the frozen registered model."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from apps.api.app.schemas import ApiModel, ErrorResponse, FailureDetail
from satquery.evidence.models import VqaEvidence
from satquery.inference.exceptions import (
    ModelExecutionError,
    ModelInputUnsupportedError,
    ModelUnavailableError,
)
from satquery.inference.vqa import SingleImageVqaService
from satquery.ingestion.exceptions import ObservationNotFoundError

router = APIRouter(prefix="/api/vqa", tags=["vqa"])


class VqaRequest(ApiModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    question: str = Field(min_length=1, max_length=500)


@router.post(
    "",
    response_model=VqaEvidence,
    response_model_exclude_none=True,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def single_image_vqa(
    request: Request,
    payload: VqaRequest,
) -> VqaEvidence | JSONResponse:
    service: SingleImageVqaService = request.app.state.single_image_vqa_service
    try:
        return service.answer(payload.observation_id, payload.question)
    except ObservationNotFoundError:
        return _vqa_error(
            404,
            code="OBSERVATION_NOT_FOUND",
            user_message="The requested observation was not found.",
            action={"type": "provide_registered_observation"},
        )
    except ModelInputUnsupportedError:
        return _vqa_error(
            422,
            code="MODEL_INPUT_UNSUPPORTED",
            user_message="This observation cannot be prepared for the VQA model.",
            action={"type": "provide_supported_observation"},
        )
    except ModelUnavailableError:
        return _vqa_error(
            503,
            code="MODEL_UNAVAILABLE",
            user_message="The registered VQA model is unavailable on this server.",
            action={"type": "install_registered_model"},
        )
    except ModelExecutionError:
        return _vqa_error(
            500,
            code="MODEL_EXECUTION_FAILED",
            user_message="The registered VQA model could not answer this question.",
            action={"type": "retry_request"},
        )


def invalid_vqa_request_response() -> JSONResponse:
    return _vqa_error(
        422,
        code="INVALID_VQA_REQUEST",
        user_message=(
            "A registered observation ID and a non-empty question are required."
        ),
        action={"type": "correct_request"},
    )


def _vqa_error(
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
            affected_requirement="R-EVIDENCE-001",
            recoverable=True,
            required_action=action,
        )
    ).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=payload)
