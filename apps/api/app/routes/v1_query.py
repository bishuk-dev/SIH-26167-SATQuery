"""Canonical plan-only query endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.routes.v1_observations import registration_from_record
from apps.api.app.schemas import ApiModel
from apps.api.app.schemas_v1 import FailureOutcomeV1
from satquery.agent.feasibility import FeasibilityValidator
from satquery.agent.interpreter import DeterministicQueryInterpreter
from satquery.agent.models import FeasibilityResult, QueryIntent
from satquery.agent.planner import BoundedPlanner, ExecutionPlan, PlannerError
from satquery.geo import PairValidator
from satquery.geo.models import PairCompatibility
from satquery.persistence import MetadataRepository

router = APIRouter(prefix="/api/v1/query", tags=["Query"])


class QueryPlanRequest(ApiModel):
    query: str = Field(min_length=1, max_length=4096)
    observation_ids: tuple[str, ...] = Field(default=(), min_length=0, max_length=2)
    pair_id: str | None = Field(default=None, pattern=r"^pair_[0-9a-f]{32}$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    roi: dict[str, Any] | None = None


class QueryPlanResponse(ApiModel):
    intent: QueryIntent
    feasibility: FeasibilityResult
    plan: ExecutionPlan
    reasons: tuple[str, ...] = ()


def _load_inputs(request: Request, payload: QueryPlanRequest) -> tuple[tuple[Any, ...], PairCompatibility | None, tuple[str, ...]]:
    repository: MetadataRepository = request.app.state.observation_repository
    observation_ids = list(payload.observation_ids)
    pair: PairCompatibility | None = None
    if payload.pair_id is not None:
        record = repository.get_pair(payload.pair_id)
        if record is None:
            raise HTTPException(status_code=404)
        pair = PairCompatibility.model_validate(record.payload["validation"])
        pair_ids = (record.observation_a_id, record.observation_b_id)
        if observation_ids and tuple(observation_ids) != pair_ids:
            raise HTTPException(status_code=422)
        observation_ids = list(pair_ids)
    if len(set(observation_ids)) != len(observation_ids):
        raise HTTPException(status_code=422)
    observations = []
    for observation_id in observation_ids:
        record = repository.get_observation(observation_id)
        if record is None:
            raise HTTPException(status_code=404)
        observations.append(registration_from_record(record).observation)
    if pair is None and len(observations) == 2:
        pair = PairValidator().validate(observations[0], observations[1])
    return tuple(observations), pair, tuple(observation_ids)


def _failure(
    request: Request,
    *,
    code: str,
    message: str,
    outcome: FailureOutcomeV1,
    details: dict[str, Any],
) -> JSONResponse:
    return failure_response(
        code=code,
        message=message,
        outcome=outcome,
        status_code=422,
        request_id=request_id_from(request),
        details=details,
    )


@router.post(
    "/plan",
    response_model=QueryPlanResponse,
    operation_id="plan_query_v1",
    summary="Validate and plan a bounded analysis without execution.",
    description=(
        "Interprets a query, loads immutable observations by server-issued IDs, "
        "and returns a registry-constrained dry-run plan. The route never "
        "persists or executes a plan."
    ),
)
def plan_query_v1(
    request: Request, payload: QueryPlanRequest
) -> QueryPlanResponse | JSONResponse:
    observations, pair, observation_ids = _load_inputs(request, payload)
    intent = DeterministicQueryInterpreter().interpret(payload.query)
    feasibility = FeasibilityValidator().validate(
        intent,
        observations,
        pair,
        payload.roi,
        request.app.state.runtime_capabilities,
    )
    if feasibility.outcome not in {"ALLOW", "ALLOW_WITH_WARNING"}:
        issue = feasibility.failures[0] if feasibility.failures else None
        return _failure(
            request,
            code=issue.code if issue else "QUERY_NOT_FEASIBLE",
            message=issue.message if issue else "The request is not feasible under the registered contracts.",
            outcome=FailureOutcomeV1(feasibility.outcome),
            details={
                "reasons": [item.model_dump(mode="json") for item in feasibility.failures],
                "warnings": [item.model_dump(mode="json") for item in feasibility.warnings],
                "checks": [item.model_dump(mode="json") for item in feasibility.checks],
            },
        )
    try:
        plan = BoundedPlanner(request.app.state.tool_registry).plan(
            intent,
            feasibility,
            {"observation_ids": observation_ids, "parameters": payload.parameters},
        )
    except PlannerError as exc:
        return _failure(
            request,
            code=exc.code,
            message=exc.message,
            outcome=FailureOutcomeV1.ABSTAIN,
            details={"reasons": [exc.message]},
        )
    return QueryPlanResponse(
        intent=intent,
        feasibility=feasibility,
        plan=plan,
        reasons=tuple(item.message for item in feasibility.warnings),
    )


__all__ = ["QueryPlanRequest", "QueryPlanResponse", "router"]
