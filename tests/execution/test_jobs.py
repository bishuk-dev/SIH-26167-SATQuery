from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

from satquery.execution import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionPlan,
    JobQueueFullError,
    JobRunner,
    PlanStep,
    ToolResult,
)
from satquery.persistence import AnalysisRecord, AnalysisStatus, Database, JobStatus, MetadataRepository


class FakeAdapter:
    def __init__(self, *, started: Event | None = None, release: Event | None = None):
        self.started, self.release = started, release

    def execute(self, call, context):
        if self.started:
            self.started.set()
        if self.release:
            self.release.wait(timeout=2)
        return ToolResult(output=call.step_id)


def _runner(tmp_path: Path, adapter: object, *, queue_size: int = 4) -> tuple[JobRunner, MetadataRepository]:
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    repo = MetadataRepository(db)
    now = datetime.now(timezone.utc)
    repo.create_analysis(AnalysisRecord(
        analysis_id="ana_" + "a" * 32,
        status=AnalysisStatus.PENDING,
        created_at=now,
        updated_at=now,
        payload={},
    ))
    engine = ExecutionEngine({"fake": adapter})
    return JobRunner(repo, engine, max_queued_jobs=queue_size), repo


def _plan() -> ExecutionPlan:
    return ExecutionPlan(steps=(PlanStep("step", "fake"),))


def test_runner_persists_ordered_events_and_succeeds(tmp_path: Path):
    runner, repo = _runner(tmp_path, FakeAdapter())
    runner.start()
    job = runner.submit("ana_" + "a" * 32, _plan())
    deadline = time.time() + 2
    while time.time() < deadline and repo.get_job(job.job_id).status not in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
        time.sleep(0.01)
    runner.stop()
    assert repo.get_job(job.job_id).status is JobStatus.SUCCEEDED
    assert [event.event_type for event in repo.list_events(job.job_id)] == [
        "JOB_SUBMITTED", "JOB_STARTED", "STEP_STARTED", "STEP_SUCCEEDED", "JOB_SUCCEEDED"
    ]


def test_runner_cancellation_between_claim_and_execution(tmp_path: Path):
    started, release = Event(), Event()
    runner, repo = _runner(tmp_path, FakeAdapter(started=started, release=release))
    runner.start()
    job = runner.submit("ana_" + "a" * 32, _plan())
    assert started.wait(2)
    assert runner.request_cancel(job.job_id).status is JobStatus.CANCEL_REQUESTED
    release.set()
    deadline = time.time() + 2
    while time.time() < deadline and repo.get_job(job.job_id).status not in {JobStatus.CANCELLED, JobStatus.FAILED}:
        time.sleep(0.01)
    runner.stop()
    assert repo.get_job(job.job_id).status is JobStatus.CANCELLED


def test_runner_queue_is_bounded(tmp_path: Path):
    started, release = Event(), Event()
    runner, _ = _runner(tmp_path, FakeAdapter(started=started, release=release), queue_size=1)
    first = runner.submit("ana_" + "a" * 32, _plan())
    with pytest.raises(Exception):
        # a second persisted job is allowed to fail closed when the queue has no slot
        runner.submit("ana_" + "a" * 32, _plan())
    assert first.status is JobStatus.QUEUED
