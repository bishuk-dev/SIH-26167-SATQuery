"""Analysis history, evidence, trace, reproducibility, and rerun routes.

Read routes project persisted, immutable analysis records. Projections never
expose local filesystem paths; rerun re-executes the original frozen plan
bit-for-bit or refuses with ``WORKFLOW_VERSION_UNAVAILABLE`` — it never
re-plans the workflow and never mutates the original analysis.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.routes.v1_observations import _decode_cursor, _encode_cursor
from apps.api.app.schemas_v1 import (
    AnalysisArtifactV1,
    AnalysisArtifactsV1,
    AnalysisDetailV1,
    AnalysisEvidenceV1,
    AnalysisPageV1,
    AnalysisReproducibilityV1,
    AnalysisRerunResponseV1,
    AnalysisSummaryV1,
    AnalysisTraceV1,
    EvidenceEdgeV1,
    EvidenceItemV1,
    FailureOutcomeV1,
    TraceEventV1,
)
from satquery.artifacts import ArtifactNotFoundError
from satquery.execution import JobQueueFullError
from satquery.persistence import AnalysisStatus, JobStatus, MetadataRepository, PageCursor
from satquery.persistence.repositories import (
    canonical_json,
    decode_timestamp,
    encode_timestamp,
)

router = APIRouter(prefix="/api/v1/analyses", tags=["Analyses"])

_ARTIFACT_ID = re.compile(r"^artifact_[0-9a-f]{32}$")


def _get_analysis_or_404(repository: MetadataRepository, analysis_id: str):
    record = repository.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404)
    return record


def _observation_ids(repository: MetadataRepository, analysis_id: str) -> tuple[str, ...]:
    with repository._db.read_transaction() as connection:
        rows = connection.execute(
            "SELECT observation_id FROM analysis_inputs"
            " WHERE analysis_id = ? AND observation_id IS NOT NULL ORDER BY position",
            (analysis_id,),
        ).fetchall()
    return tuple(row["observation_id"] for row in rows)


def _summary(repository: MetadataRepository, record) -> AnalysisSummaryV1:
    payload = record.payload
    intent = payload.get("intent", {})
    return AnalysisSummaryV1(
        analysis_id=record.analysis_id,
        status=record.status.value,
        intent=intent.get("task_family", "") if isinstance(intent, dict) else "",
        created_at=record.created_at,
        updated_at=record.updated_at,
        rerun_of=payload.get("rerun_of"),
    )


def _strip_local_paths(value: Any) -> Any:
    """Recursively drop ``path`` keys so projections stay path-free.

    Evidence payloads embed ``MaskAsset`` records whose ``path`` is a server
    filesystem location; artifacts are addressed publicly by ID and hash.
    """

    if isinstance(value, dict):
        return {
            key: _strip_local_paths(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_strip_local_paths(item) for item in value]
    return value


def _artifact_projection(request: Request, analysis_id: str) -> list[AnalysisArtifactV1]:
    store = request.app.state.artifact_store
    items: list[AnalysisArtifactV1] = []
    for child in sorted(store.artifacts_root.iterdir()):
        if not child.is_dir() or not _ARTIFACT_ID.fullmatch(child.name):
            continue
        try:
            record, _path = store.resolve(child.name)
        except ArtifactNotFoundError:
            continue
        if record.metadata.analysis_id != analysis_id:
            continue
        items.append(
            AnalysisArtifactV1(
                artifact_id=record.artifact_id,
                evidence_id=record.metadata.evidence_id,
                media_type=record.media_type,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
            )
        )
    return items


@router.get("", response_model=AnalysisPageV1, operation_id="list_analyses_v1")
def list_analyses_v1(
    request: Request,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    status: AnalysisStatus | None = None,
    intent: str | None = Query(default=None, min_length=1, max_length=64),
    observation_id: str | None = Query(
        default=None, pattern=r"^obs_[0-9a-f]{32}$"
    ),
) -> AnalysisPageV1:
    """Stable keyset-paginated history with status/intent/observation filters."""

    repository: MetadataRepository = request.app.state.observation_repository
    conditions: list[str] = []
    parameters: list[Any] = []
    if status is not None:
        conditions.append("a.status = ?")
        parameters.append(status.value)
    if intent is not None:
        conditions.append("json_extract(a.payload_json, '$.intent.task_family') = ?")
        parameters.append(intent)
    if observation_id is not None:
        conditions.append(
            "EXISTS (SELECT 1 FROM analysis_inputs ai"
            " WHERE ai.analysis_id = a.analysis_id AND ai.observation_id = ?)"
        )
        parameters.append(observation_id)
    decoded = _decode_cursor(cursor)
    if decoded is not None:
        conditions.append("(a.created_at < ? OR (a.created_at = ? AND a.analysis_id < ?))")
        parameters.extend(
            [
                encode_timestamp(decoded.created_at),
                encode_timestamp(decoded.created_at),
                decoded.record_id,
            ]
        )
    clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with repository._db.read_transaction() as connection:
        rows = connection.execute(
            "SELECT a.analysis_id, a.status, a.created_at, a.updated_at, a.payload_json"
            f" FROM analyses a {clause}"
            " ORDER BY a.created_at DESC, a.analysis_id DESC LIMIT ?",
            [*parameters, limit],
        ).fetchall()
    records = [
        repository._analysis_from_row(row)
        for row in rows
    ]
    next_cursor = None
    if len(records) == limit:
        last = records[-1]
        next_cursor = _encode_cursor(
            PageCursor(created_at=last.created_at, record_id=last.analysis_id)
        )
    return AnalysisPageV1(
        items=tuple(_summary(repository, record) for record in records),
        next_cursor=next_cursor,
    )


@router.get("/{analysis_id}", response_model=AnalysisDetailV1, operation_id="get_analysis_v1")
def get_analysis_v1(request: Request, analysis_id: str) -> AnalysisDetailV1:
    repository: MetadataRepository = request.app.state.observation_repository
    record = _get_analysis_or_404(repository, analysis_id)
    summary = _summary(repository, record)
    payload = record.payload
    return AnalysisDetailV1(
        **summary.model_dump(),
        observation_ids=_observation_ids(repository, analysis_id),
        plan_hash=payload.get("plan_hash"),
        registry_hash=payload.get("registry_hash"),
    )


@router.get(
    "/{analysis_id}/evidence",
    response_model=AnalysisEvidenceV1,
    operation_id="get_analysis_evidence_v1",
)
def get_analysis_evidence_v1(request: Request, analysis_id: str) -> AnalysisEvidenceV1:
    repository: MetadataRepository = request.app.state.observation_repository
    _get_analysis_or_404(repository, analysis_id)
    with repository._db.read_transaction() as connection:
        rows = connection.execute(
            "SELECT evidence_id, created_at, payload_json FROM evidence"
            " WHERE analysis_id = ? ORDER BY created_at, evidence_id",
            (analysis_id,),
        ).fetchall()
        edges = connection.execute(
            "SELECT source_evidence_id, target_evidence_id, edge_type"
            " FROM evidence_edges WHERE analysis_id = ?"
            " ORDER BY target_evidence_id, source_evidence_id, edge_type",
            (analysis_id,),
        ).fetchall()
    items = tuple(
        EvidenceItemV1(
            evidence_id=row["evidence_id"],
            created_at=decode_timestamp(row["created_at"]),
            evidence=_strip_local_paths(json.loads(row["payload_json"])),
        )
        for row in rows
    )
    edge_items = tuple(
        EvidenceEdgeV1(
            source_evidence_id=row["source_evidence_id"],
            target_evidence_id=row["target_evidence_id"],
            edge_type=row["edge_type"],
        )
        for row in edges
    )
    return AnalysisEvidenceV1(items=items, edges=edge_items)


@router.get(
    "/{analysis_id}/trace",
    response_model=AnalysisTraceV1,
    operation_id="get_analysis_trace_v1",
)
def get_analysis_trace_v1(request: Request, analysis_id: str) -> AnalysisTraceV1:
    repository: MetadataRepository = request.app.state.observation_repository
    _get_analysis_or_404(repository, analysis_id)
    with repository._db.read_transaction() as connection:
        rows = connection.execute(
            "SELECT e.job_id, e.sequence, e.event_type, e.created_at, e.payload_json"
            " FROM execution_events e JOIN jobs j ON j.job_id = e.job_id"
            " WHERE j.analysis_id = ?"
            " ORDER BY e.created_at, e.job_id, e.sequence",
            (analysis_id,),
        ).fetchall()
    return AnalysisTraceV1(
        items=tuple(
            TraceEventV1(
                job_id=row["job_id"],
                sequence=row["sequence"],
                event_type=row["event_type"],
                created_at=decode_timestamp(row["created_at"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )
    )


@router.get(
    "/{analysis_id}/artifacts",
    response_model=AnalysisArtifactsV1,
    operation_id="get_analysis_artifacts_v1",
)
def get_analysis_artifacts_v1(request: Request, analysis_id: str) -> AnalysisArtifactsV1:
    repository: MetadataRepository = request.app.state.observation_repository
    _get_analysis_or_404(repository, analysis_id)
    return AnalysisArtifactsV1(items=tuple(_artifact_projection(request, analysis_id)))


@router.get(
    "/{analysis_id}/reproducibility",
    response_model=AnalysisReproducibilityV1,
    operation_id="get_analysis_reproducibility_v1",
)
def get_analysis_reproducibility_v1(
    request: Request, analysis_id: str
) -> AnalysisReproducibilityV1:
    repository: MetadataRepository = request.app.state.observation_repository
    record = _get_analysis_or_404(repository, analysis_id)
    payload = record.payload
    steps = tuple(
        {
            "step_id": step.get("step_id"),
            "tool_id": step.get("tool_id"),
            "parameters": step.get("parameters", {}),
        }
        for step in payload.get("plan", {}).get("steps", [])
    )
    return AnalysisReproducibilityV1(
        analysis_id=record.analysis_id,
        status=record.status.value,
        plan_hash=payload.get("plan_hash"),
        registry_hash=payload.get("registry_hash"),
        observation_ids=_observation_ids(repository, analysis_id),
        input_hashes={
            key: value
            for key, value in payload.get("input_hashes", {}).items()
            if isinstance(value, str)
        },
        steps=steps,
        artifacts=tuple(_artifact_projection(request, analysis_id)),
    )


@router.post(
    "/{analysis_id}/rerun",
    status_code=202,
    response_model=AnalysisRerunResponseV1,
    operation_id="rerun_analysis_v1",
    summary="Re-execute one frozen analysis plan under a new identity.",
    description=(
        "Creates a new analysis and job that reuse the original frozen plan, "
        "input hashes, and execution payload without re-planning. Returns 409 "
        "WORKFLOW_VERSION_UNAVAILABLE when any original tool registration is "
        "no longer loadable. The original analysis is never overwritten."
    ),
)
def rerun_analysis_v1(request: Request, analysis_id: str) -> AnalysisRerunResponseV1 | JSONResponse:
    repository: MetadataRepository = request.app.state.observation_repository
    record = _get_analysis_or_404(repository, analysis_id)

    with repository._db.read_transaction() as connection:
        step_rows = connection.execute(
            "SELECT tool_id FROM plan_steps WHERE analysis_id = ? ORDER BY step_index",
            (analysis_id,),
        ).fetchall()
        job_row = connection.execute(
            "SELECT payload_json FROM jobs WHERE analysis_id = ?"
            " ORDER BY created_at, job_id LIMIT 1",
            (analysis_id,),
        ).fetchone()

    tool_ids = sorted({row["tool_id"] for row in step_rows})
    tool_registry = request.app.state.tool_registry
    unavailable = [tool_id for tool_id in tool_ids if tool_registry.get(tool_id) is None]
    if unavailable or job_row is None:
        request_id = request_id_from(request)
        return failure_response(
            code="WORKFLOW_VERSION_UNAVAILABLE",
            message=(
                "The original workflow can no longer be executed because its "
                "registered tools are unavailable; the frozen plan is never "
                "silently upgraded."
            ),
            outcome=FailureOutcomeV1.ABSTAIN,
            status_code=409,
            request_id=request_id,
            details={"unavailable_tools": unavailable},
        )

    new_analysis_id = f"ana_{uuid4().hex}"
    new_job_id = f"job_{uuid4().hex}"
    now = datetime.now(timezone.utc)
    rerun_payload = dict(record.payload)
    rerun_payload["rerun_of"] = analysis_id
    with repository._db.transaction() as connection:
        connection.execute(
            "INSERT INTO analyses(analysis_id, status, created_at, updated_at, payload_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                new_analysis_id,
                AnalysisStatus.PENDING.value,
                encode_timestamp(now),
                encode_timestamp(now),
                canonical_json(rerun_payload),
            ),
        )
        # copy immutable inputs and the frozen plan rows byte-for-byte
        connection.execute(
            "INSERT INTO analysis_inputs(analysis_id, position, observation_id, pair_id)"
            " SELECT ?, position, observation_id, pair_id"
            " FROM analysis_inputs WHERE analysis_id = ?",
            (new_analysis_id, analysis_id),
        )
        connection.execute(
            "INSERT INTO plan_steps(analysis_id, step_index, tool_id, payload_json)"
            " SELECT ?, step_index, tool_id, payload_json"
            " FROM plan_steps WHERE analysis_id = ?",
            (new_analysis_id, analysis_id),
        )
        connection.execute(
            "INSERT INTO jobs(job_id, analysis_id, status, created_at, updated_at, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                new_job_id,
                new_analysis_id,
                JobStatus.QUEUED.value,
                encode_timestamp(now),
                encode_timestamp(now),
                job_row["payload_json"],
            ),
        )
    try:
        request.app.state.job_runner.enqueue_existing(new_job_id)
    except JobQueueFullError:
        repository.transition_job(
            new_job_id, JobStatus.QUEUED, JobStatus.FAILED, updated_at=now
        )
        repository.transition_analysis(
            new_analysis_id,
            AnalysisStatus.PENDING,
            AnalysisStatus.FAILED,
            updated_at=now,
        )
        return failure_response(
            code="RESOURCE_BUSY",
            message="The rerun job could not be queued because the local worker queue is full.",
            outcome=FailureOutcomeV1.ABSTAIN,
            status_code=503,
            request_id=request_id_from(request),
            details={"analysis_id": new_analysis_id, "job_id": new_job_id},
        )
    return AnalysisRerunResponseV1(
        analysis_id=new_analysis_id,
        job_id=new_job_id,
        status=JobStatus.QUEUED.value,
        rerun_of=analysis_id,
        plan_hash=record.payload.get("plan_hash"),
        registry_hash=record.payload.get("registry_hash"),
    )


__all__ = ["router"]
