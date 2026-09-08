"""Analysis orchestration endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from apps.api.app.services.analysis import AnalysisService
from apps.api.app.services.open_model import OpenModelAdapter
from satquery.ingestion.exceptions import ObservationNotFoundError


router = APIRouter(
    prefix="/api/analyses",
    tags=["analyses"],
)


class AnalysisRequest(BaseModel):
    observation_ids: list[str] = Field(
        min_length=1,
        max_length=2,
    )
    query: str = Field(
        min_length=1,
        max_length=500,
    )


class AnalysisResponse(BaseModel):
    task: str
    observation_ids: list[str]
    query: str
    status: str
    execution_trace: dict[str, object]
    evidence: dict[str, object] | None = None
    answer: str | None = None
    model_name: str | None = None
    confidence: float | None = None


@router.post(
    "",
    response_model=AnalysisResponse,
)
def create_analysis(
    request: Request,
    payload: AnalysisRequest,
) -> AnalysisResponse | JSONResponse:

    analysis_service = AnalysisService(
        observation_store=request.app.state.observation_store,
        open_model=OpenModelAdapter(),
    )

    try:
        result = analysis_service.analyze(
            observation_ids=payload.observation_ids,
            query=payload.query,
        )
    except ObservationNotFoundError:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "OBSERVATION_NOT_FOUND",
                    "message": "One or more observation IDs are not registered.",
                }
            },
        )

    return AnalysisResponse(**result)