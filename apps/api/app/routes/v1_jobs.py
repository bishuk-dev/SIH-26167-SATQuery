"""Persistent job inspection and cancellation routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.app.schemas import ApiModel
from satquery.persistence import JobRecord, JobStatus

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


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
