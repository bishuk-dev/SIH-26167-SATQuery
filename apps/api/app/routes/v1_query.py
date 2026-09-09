"""Canonical plan-only query endpoint."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request
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
from satquery.execution import ExecutionPlan as RuntimeExecutionPlan
from satquery.execution import JobQueueFullError, ModelBusyError, PlanStep as RuntimePlanStep
from satquery.geo import PairValidator
from satquery.geo.models import PairCompatibility
from satquery.persistence import (
    AnalysisRecord,
    AnalysisStatus,
    JobRecord,
    JobStatus,
    MetadataRepository,
)
from satquery.persistence.repositories import canonical_json, encode_timestamp

router = APIRouter(prefix="/api/v1/query", tags=["Query"])

# client idempotency keys are bounded, URL-safe tokens; only their SHA-256 is stored
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]{16,128}$")


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


class QuerySubmissionResponse(ApiModel):
    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    status: JobStatus
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    status_code: int = 422,
) -> JSONResponse:
    return failure_response(
        code=code,
        message=message,
        outcome=outcome,
        status_code=status_code,
        request_id=request_id_from(request),
        details=details,
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan_or_empty(
    planner: BoundedPlanner,
    intent: QueryIntent,
    feasibility: FeasibilityResult,
    inputs: dict[str, Any],
) -> tuple[ExecutionPlan, tuple[str, ...]]:
    if feasibility.outcome in {"ALLOW", "ALLOW_WITH_WARNING"}:
        try:
            return planner.plan(intent, feasibility, inputs), ()
        except PlannerError as exc:
            empty_feasibility = FeasibilityResult(checks=(), outcome="ABSTAIN")
            return planner.plan(intent, empty_feasibility, inputs), (f"{exc.code}: {exc.message}",)
    return planner.plan(intent, feasibility, inputs), tuple(
        f"{issue.code}: {issue.message}" for issue in feasibility.failures
    )


def _runtime_plan(plan: ExecutionPlan) -> RuntimeExecutionPlan:
    return RuntimeExecutionPlan(
        planner_version=plan.planner_version,
        steps=tuple(
            RuntimePlanStep(
                step_id=step.step_id,
                tool_id=step.tool_id,
                input_bindings=dict(step.input_bindings),
                parameters=dict(step.parameters),
                depends_on=step.depends_on,
                expected_evidence_type=step.expected_evidence_type,
            )
            for step in plan.steps
        ),
    )


def _input_hashes(
    observations: tuple[Any, ...], pair: PairCompatibility | None, pair_id: str | None
) -> dict[str, str]:
    hashes = {
        observation.observation_id: observation.source_asset.sha256
        for observation in observations
    }
    if pair is not None and pair_id is not None:
        hashes[pair_id] = hashlib.sha256(
            canonical_json(pair.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()
    return hashes


def _verified_common_grid_pairs(
    observations: tuple[Any, ...], pair: PairCompatibility | None
) -> list[list[str]]:
    if pair is None or pair.grid.aligned is not True or len(observations) != 2:
        return []
    temporal = pair.temporal
    if temporal.order_known and temporal.first and temporal.second:
        return [[temporal.first, temporal.second]]
    return [[observations[0].observation_id, observations[1].observation_id]]


def _persist_submission(
    repository: MetadataRepository,
    *,
    analysis: AnalysisRecord,
    job: JobRecord,
    observation_ids: tuple[str, ...],
    pair_id: str | None,
    plan: ExecutionPlan,
    idempotency_key_hash: str | None = None,
    request_hash: str | None = None,
) -> dict[str, Any] | None:
    """Atomically persist the submission transaction.

    With an idempotency key, the key check, the idempotency record, and the
    analysis/job rows commit together. Returns ``{"conflict": ...}`` when the
    key was already used with a different body, a ``{"replay": ...}`` payload
    when the same key/body pair is resubmitted, and None for a fresh
    submission. GET routes are never idempotency-guarded.
    """

    with repository._db.transaction() as connection:
        if idempotency_key_hash is not None:
            existing = connection.execute(
                "SELECT request_hash, analysis_id FROM idempotency_keys WHERE key_hash = ?",
                (idempotency_key_hash,),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    return {"conflict": True}
                analysis_row = connection.execute(
                    "SELECT status, payload_json FROM analyses WHERE analysis_id = ?",
                    (existing["analysis_id"],),
                ).fetchone()
                job_row = connection.execute(
                    "SELECT job_id, status FROM jobs WHERE analysis_id = ?",
                    (existing["analysis_id"],),
                ).fetchone()
                if analysis_row is None or job_row is None:
                    # record without its submission would be corrupt; fail closed
                    return {"conflict": True}
                return {
                    "replay": {
                        "analysis_id": existing["analysis_id"],
                        "job_id": job_row["job_id"],
                        "status": job_row["status"],
                        "payload": json.loads(analysis_row["payload_json"]),
                    }
                }
        connection.execute(
            "INSERT INTO analyses(analysis_id, status, created_at, updated_at, payload_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                analysis.status.value,
                encode_timestamp(analysis.created_at),
                encode_timestamp(analysis.updated_at),
                canonical_json(analysis.payload),
            ),
        )
        position = 0
        if pair_id is not None:
            connection.execute(
                "INSERT INTO analysis_inputs(analysis_id, position, observation_id, pair_id)"
                " VALUES (?, ?, NULL, ?)",
                (analysis.analysis_id, position, pair_id),
            )
            position += 1
        for observation_id in observation_ids:
            connection.execute(
                "INSERT INTO analysis_inputs(analysis_id, position, observation_id, pair_id)"
                " VALUES (?, ?, ?, NULL)",
                (analysis.analysis_id, position, observation_id),
            )
            position += 1
        for index, step in enumerate(plan.steps):
            connection.execute(
                "INSERT INTO plan_steps(analysis_id, step_index, tool_id, payload_json)"
                " VALUES (?, ?, ?, ?)",
                (
                    analysis.analysis_id,
                    index,
                    step.tool_id,
                    canonical_json(step.model_dump(mode="json")),
                ),
            )
        connection.execute(
            "INSERT INTO jobs(job_id, analysis_id, status, created_at, updated_at, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                job.job_id,
                job.analysis_id,
                job.status.value,
                encode_timestamp(job.created_at),
                encode_timestamp(job.updated_at),
                canonical_json(job.payload),
            ),
        )
        if idempotency_key_hash is not None:
            connection.execute(
                "INSERT INTO idempotency_keys(key_hash, request_hash, analysis_id, created_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    idempotency_key_hash,
                    request_hash,
                    analysis.analysis_id,
                    encode_timestamp(analysis.created_at),
                ),
            )
    return None


def _mark_submission_failed(repository: MetadataRepository, analysis_id: str, job_id: str) -> None:
    now = datetime.now(timezone.utc)
    repository.transition_job(job_id, JobStatus.QUEUED, JobStatus.FAILED, updated_at=now)
    repository.transition_analysis(
        analysis_id, AnalysisStatus.PENDING, AnalysisStatus.FAILED, updated_at=now
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


@router.post(
    "",
    status_code=202,
    response_model=QuerySubmissionResponse,
    operation_id="submit_query_v1",
    summary="Submit a bounded analysis for durable execution.",
    description=(
        "Creates one immutable analysis submission and a queued job. The frozen "
        "plan is derived only from server-side registries and server-issued "
        "observation identifiers; query text never selects executable tools."
    ),
)
def submit_query_v1(
    request: Request,
    payload: QueryPlanRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> QuerySubmissionResponse | JSONResponse:
    idempotency_key_hash: str | None = None
    request_hash: str | None = None
    if idempotency_key is not None:
        if not _IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key):
            return _failure(
                request,
                code="INVALID_IDEMPOTENCY_KEY",
                message=(
                    "The Idempotency-Key header must be 16-128 URL-safe characters."
                ),
                outcome=FailureOutcomeV1.REJECT,
                status_code=400,
                details={},
            )
        idempotency_key_hash = _hash_text(idempotency_key)
        request_hash = _hash_text(canonical_json(payload.model_dump(mode="json")))
    if idempotency_key_hash is None and not request.app.state.job_runner.has_queue_capacity():
        return _failure(
            request,
            code="RESOURCE_BUSY",
            message="The analysis job could not be queued because the local worker queue is full.",
            outcome=FailureOutcomeV1.ABSTAIN,
            status_code=429,
            details={},
        )
    observations, pair, observation_ids = _load_inputs(request, payload)
    intent = DeterministicQueryInterpreter().interpret(payload.query)
    feasibility = FeasibilityValidator().validate(
        intent,
        observations,
        pair,
        payload.roi,
        request.app.state.runtime_capabilities,
    )
    planner = BoundedPlanner(request.app.state.tool_registry)
    plan, submission_reasons = _plan_or_empty(
        planner,
        intent,
        feasibility,
        {"observation_ids": observation_ids, "parameters": payload.parameters},
    )
    runtime_plan = _runtime_plan(plan)

    analysis_id = f"ana_{uuid4().hex}"
    job_id = f"job_{uuid4().hex}"
    now = datetime.now(timezone.utc)
    verified_pairs = _verified_common_grid_pairs(observations, pair)
    analysis_payload = {
        "schema_version": 1,
        "query": payload.query,
        "query_sha256": _hash_text(payload.query),
        "intent": intent.model_dump(mode="json"),
        "feasibility": feasibility.model_dump(mode="json"),
        "reasons": list(submission_reasons),
        "observation_ids": list(observation_ids),
        "pair_id": payload.pair_id,
        "input_hashes": _input_hashes(observations, pair, payload.pair_id),
        "plan": plan.model_dump(mode="json"),
        "plan_hash": plan.plan_hash,
        "registry_hash": plan.registry_hash,
    }
    job_payload = {
        "plan": runtime_plan.as_payload(),
        "submission": {
            "analysis_plan_hash": plan.plan_hash,
            "registry_hash": plan.registry_hash,
        },
        "metadata": {"verified_common_grid_pairs": verified_pairs},
    }
    analysis = AnalysisRecord(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        created_at=now,
        updated_at=now,
        payload=analysis_payload,
    )
    job = JobRecord(
        job_id=job_id,
        analysis_id=analysis_id,
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        payload=job_payload,
    )
    outcome = _persist_submission(
        request.app.state.observation_repository,
        analysis=analysis,
        job=job,
        observation_ids=observation_ids,
        pair_id=payload.pair_id,
        plan=plan,
        idempotency_key_hash=idempotency_key_hash,
        request_hash=request_hash,
    )
    if outcome is not None and outcome.get("conflict"):
        return _failure(
            request,
            code="IDEMPOTENCY_CONFLICT",
            message=(
                "This Idempotency-Key was already used with a different request body."
            ),
            outcome=FailureOutcomeV1.REJECT,
            status_code=409,
            details={},
        )
    if outcome is not None and "replay" in outcome:
        replay = outcome["replay"]
        payload_json = replay["payload"]
        return JSONResponse(
            status_code=200,
            content=QuerySubmissionResponse(
                analysis_id=replay["analysis_id"],
                job_id=replay["job_id"],
                status=JobStatus(replay["status"]),
                plan_hash=payload_json["plan_hash"],
                registry_hash=payload_json["registry_hash"],
            ).model_dump(mode="json"),
        )
    try:
        request.app.state.job_runner.enqueue_existing(job_id)
    except JobQueueFullError:
        _mark_submission_failed(request.app.state.observation_repository, analysis_id, job_id)
        return _failure(
            request,
            code="RESOURCE_BUSY",
            message="The analysis job could not be queued because the local worker queue is full.",
            outcome=FailureOutcomeV1.ABSTAIN,
            status_code=429,
            details={"analysis_id": analysis_id, "job_id": job_id},
        )
    except ModelBusyError:
        _mark_submission_failed(request.app.state.observation_repository, analysis_id, job_id)
        return _failure(
            request,
            code="MODEL_BUSY",
            message="The registered model is temporarily busy.",
            outcome=FailureOutcomeV1.ABSTAIN,
            status_code=503,
            details={"analysis_id": analysis_id, "job_id": job_id},
        )
    return QuerySubmissionResponse(
        analysis_id=analysis_id,
        job_id=job_id,
        status=JobStatus.QUEUED,
        plan_hash=plan.plan_hash,
        registry_hash=plan.registry_hash,
    )


__all__ = ["QueryPlanRequest", "QueryPlanResponse", "QuerySubmissionResponse", "router"]
