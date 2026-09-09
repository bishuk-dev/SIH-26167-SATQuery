"""Persistent job inspection, progress streaming, and cancellation routes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from apps.api.app.schemas import ApiModel
from satquery.execution.jobs import TERMINAL_EVENT_TYPES, TERMINAL_JOB_STATUSES
from satquery.persistence import ExecutionEvent, JobRecord, JobStatus

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

# SSE poll/heartbeat pacing. Events are read from SQLite on each poll, so the
# stream survives process restarts and reconnects without an in-memory broker.
SSE_POLL_SECONDS = 0.05
SSE_HEARTBEAT_SECONDS = 1.0
# Terminal statuses are always followed by a terminal event; this bounds how
# long the stream keeps draining if that append has not landed yet.
SSE_TERMINAL_DRAIN_SECONDS = 2.0


class JobV1(ApiModel):
    job_id: str
    analysis_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime


class JobCancelV1(ApiModel):
    job: JobV1


def _project(record: JobRecord) -> JobV1:
    return JobV1(
        job_id=record.job_id,
        analysis_id=record.analysis_id,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/{job_id}", response_model=JobV1, operation_id="get_job_v1")
def get_job_v1(request: Request, job_id: str) -> JobV1:
    record = request.app.state.job_runner.get(job_id)
    if record is None:
        raise HTTPException(status_code=404)
    return _project(record)


@router.get("/{job_id}/events", operation_id="stream_job_events_v1")
async def stream_job_events_v1(request: Request, job_id: str) -> StreamingResponse:
    runner = request.app.state.job_runner
    if runner.get(job_id) is None:
        raise HTTPException(status_code=404)
    last_event_id = request.headers.get("last-event-id")
    after_sequence = -1
    if last_event_id is not None:
        try:
            after_sequence = int(last_event_id.strip())
        except ValueError:
            raise HTTPException(status_code=422) from None
        if after_sequence < 0:
            raise HTTPException(status_code=422)
    return StreamingResponse(
        _job_event_stream(runner, job_id, after_sequence, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


def _sse_frame(event: ExecutionEvent) -> str:
    data = json.dumps(event.payload, separators=(",", ":"), allow_nan=False)
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {data}\n\n"


def _terminal_event_seen(runner: Any, job_id: str, cursor: int) -> bool:
    return any(
        event.sequence <= cursor and event.event_type in TERMINAL_EVENT_TYPES
        for event in runner.repository.list_events(job_id)
    )


async def _job_event_stream(
    runner: Any, job_id: str, after_sequence: int, request: Request
) -> AsyncIterator[str]:
    cursor = after_sequence
    next_heartbeat = time.monotonic() + SSE_HEARTBEAT_SECONDS
    while True:
        for event in runner.events_after(job_id, cursor):
            cursor = event.sequence
            yield _sse_frame(event)
        record = runner.get(job_id)
        status = record.status if record is not None else None
        if status is None or status is JobStatus.INTERRUPTED:
            return
        if status in TERMINAL_JOB_STATUSES:
            # The terminal event is appended right after the terminal status
            # transition. Drain until the client receives it (or the client
            # already resumed past it) so the terminal outcome is never lost.
            deadline = time.monotonic() + SSE_TERMINAL_DRAIN_SECONDS
            while True:
                for event in runner.events_after(job_id, cursor):
                    cursor = event.sequence
                    yield _sse_frame(event)
                    if event.event_type in TERMINAL_EVENT_TYPES:
                        return
                if _terminal_event_seen(runner, job_id, cursor) or time.monotonic() >= deadline:
                    return
                await asyncio.sleep(SSE_POLL_SECONDS)
        if await request.is_disconnected():
            return
        now = time.monotonic()
        if now >= next_heartbeat:
            next_heartbeat = now + SSE_HEARTBEAT_SECONDS
            yield ": heartbeat\n\n"
        await asyncio.sleep(SSE_POLL_SECONDS)


@router.post("/{job_id}/cancel", response_model=JobCancelV1, operation_id="cancel_job_v1")
def cancel_job_v1(request: Request, job_id: str) -> JobCancelV1:
    runner = request.app.state.job_runner
    try:
        record = runner.request_cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404) from None
    except ValueError:
        raise HTTPException(status_code=409) from None
    return JobCancelV1(job=_project(record))


__all__ = ["router"]
