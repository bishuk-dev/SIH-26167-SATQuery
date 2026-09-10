"""SSE progress streaming and cancellation semantics for persistent jobs.

These tests run the FastAPI application under a real threaded uvicorn server:
the in-process TestClient buffers whole responses, which can never observe a
long-lived SSE stream, heartbeats, or client disconnects.
"""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from threading import Event, Thread
from uuid import uuid4

import httpx
import uvicorn

import apps.api.app.routes.v1_jobs as v1_jobs
from apps.api.app.main import create_app
from satquery.execution import (
    ExecutionEngine,
    ExecutionPlan,
    JobRunner,
    PlanStep,
    ToolResult,
)
from satquery.persistence import AnalysisRecord, AnalysisStatus, JobStatus


class BlockingAdapter:
    """Test tool that signals start and blocks execution until released."""

    def __init__(self, started: Event, release: Event) -> None:
        self.started = started
        self.release = release

    def execute(self, call, context):  # type: ignore[no-untyped-def]
        self.started.set()
        self.release.wait(timeout=5)
        return ToolResult(output=call.step_id)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RunningServer:
    """Threaded uvicorn server with a real httpx client for SSE streaming."""

    def __init__(self, application, *, lifespan: str = "off") -> None:  # type: ignore[no-untyped-def]
        self.port = _free_port()
        config = uvicorn.Config(
            application,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan=lifespan,
        )
        self.server = uvicorn.Server(config)
        self.thread = Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while time.time() < deadline and not self.server.started:
            time.sleep(0.02)
        assert self.server.started, "uvicorn server did not start"
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            timeout=httpx.Timeout(10.0),
        )

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.client.close()


def _app_with_runner(tmp_path, adapter=None):  # type: ignore[no-untyped-def]
    application = create_app(data_root=tmp_path / "data")
    if adapter is not None:
        application.state.job_runner = JobRunner(
            application.state.observation_repository,
            ExecutionEngine({"fake": adapter}),
        )
    return application


def _create_analysis(application) -> str:  # type: ignore[no-untyped-def]
    repository = application.state.observation_repository
    now = datetime.now(timezone.utc)
    analysis_id = f"ana_{uuid4().hex}"
    repository.create_analysis(
        AnalysisRecord(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=now,
            updated_at=now,
            payload={},
        )
    )
    return analysis_id


def _plan() -> ExecutionPlan:
    return ExecutionPlan(steps=(PlanStep("s1", "fake"), PlanStep("s2", "fake")))


def _read_stream(
    client,  # type: ignore[no-untyped-def]
    job_id,
    *,  # type: ignore[no-untyped-def]
    headers=None,  # type: ignore[no-untyped-def]
    stop=None,  # type: ignore[no-untyped-def]
    read_timeout=2.0,
    max_seconds=5.0,
):  # type: ignore[no-untyped-def]
    """Read an SSE stream until ``stop(text)`` is true or the server closes.

    ``stop`` receives the accumulated raw stream text. Every read is bounded
    twice: httpx aborts if no bytes arrive within ``read_timeout``, and the
    loop fails if ``max_seconds`` elapse even while frames keep arriving, so
    a misbehaving stream can never hang the suite.
    """

    text = ""
    started = time.monotonic()
    with client.stream(
        "GET",
        f"/api/v1/jobs/{job_id}/events",
        headers=headers or {},
        timeout=httpx.Timeout(10.0, read=read_timeout),
    ) as response:
        assert response.status_code == 200, response.status_code
        content_type = response.headers["content-type"]
        for chunk in response.iter_text():
            text += chunk
            if stop is not None and stop(text):
                break
            if time.monotonic() - started > max_seconds:
                raise AssertionError(
                    f"stream did not satisfy its stop condition in time: {text!r}"
                )
    return text, content_type


def _complete_frames(text: str, count: int) -> bool:
    """True once ``count`` blank-line-terminated SSE messages have arrived."""

    return text.count("\n\n") >= count


def _parse_sse(text: str) -> list[dict]:
    frames: list[dict] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        frame: dict = {}
        comments: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                comments.append(line)
            else:
                field, _, value = line.partition(":")
                frame[field] = value.lstrip(" ")
        if comments:
            frame["comments"] = comments
        frames.append(frame)
    return frames


def _stop_on_heartbeat(text: str) -> bool:
    return ": heartbeat\n\n" in text


def test_event_stream_replays_persisted_events_in_id_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    application = _app_with_runner(tmp_path)
    server = RunningServer(application)
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        runner = application.state.job_runner
        runner._append_event(job.job_id, "STEP_STARTED", {"step_id": "s1"})
        runner._append_event(job.job_id, "STEP_SUCCEEDED", {"step_id": "s1"})
        lines, content_type = _read_stream(
            server.client,
            job.job_id,
            stop=lambda text: _complete_frames(text, 3),
        )
    finally:
        server.close()

    assert content_type.startswith("text/event-stream")
    frames = _parse_sse(lines)
    assert [frame["id"] for frame in frames] == ["0", "1", "2"]
    assert [frame["event"] for frame in frames] == [
        "JOB_SUBMITTED",
        "STEP_STARTED",
        "STEP_SUCCEEDED",
    ]
    assert json.loads(frames[0]["data"]) == {"analysis_id": analysis_id}
    assert json.loads(frames[1]["data"]) == {"step_id": "s1"}


def test_event_stream_resumes_from_last_event_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    application = _app_with_runner(tmp_path)
    server = RunningServer(application)
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        runner = application.state.job_runner
        for step_id in ("s1", "s2", "s3", "s4"):
            runner._append_event(job.job_id, "STEP_SUCCEEDED", {"step_id": step_id})
        lines, _ = _read_stream(
            server.client,
            job.job_id,
            headers={"Last-Event-ID": "1"},
            stop=lambda text: _complete_frames(text, 3),
        )
    finally:
        server.close()

    frames = _parse_sse(lines)
    assert [frame["id"] for frame in frames] == ["2", "3", "4"]
    assert [frame["event"] for frame in frames] == ["STEP_SUCCEEDED"] * 3
    assert [json.loads(frame["data"])["step_id"] for frame in frames] == ["s2", "s3", "s4"]


def test_heartbeat_comments_contain_no_data(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(v1_jobs, "SSE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(v1_jobs, "SSE_POLL_SECONDS", 0.01)
    application = _app_with_runner(tmp_path)
    server = RunningServer(application)
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        # No workers run (lifespan off), so the job stays QUEUED and the
        # stream stays open long enough for heartbeat comments to appear.
        lines, _ = _read_stream(server.client, job.job_id, stop=_stop_on_heartbeat)
    finally:
        server.close()

    frames = _parse_sse(lines)
    heartbeats = [frame for frame in frames if "comments" in frame]
    assert heartbeats, lines
    for heartbeat in heartbeats:
        # Comment frames carry no id/event/data fields.
        assert set(heartbeat) == {"comments"}
    data_frames = [frame for frame in frames if "comments" not in frame]
    assert [frame["event"] for frame in data_frames] == ["JOB_SUBMITTED"]


def test_stream_disconnect_does_not_cancel_job(tmp_path) -> None:  # type: ignore[no-untyped-def]
    application = _app_with_runner(tmp_path)
    server = RunningServer(application)
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        _read_stream(
            server.client,
            job.job_id,
            stop=lambda text: _complete_frames(text, 1),
        )
        time.sleep(0.2)  # bounded wait for the server to observe disconnect
        record = application.state.job_runner.get(job.job_id)
        event_types = [
            event.event_type
            for event in application.state.job_runner.repository.list_events(job.job_id)
        ]
    finally:
        server.close()

    assert record is not None
    assert record.status is JobStatus.QUEUED
    assert "CANCEL_REQUESTED" not in event_types
    assert "JOB_CANCELLED" not in event_types


def test_queued_cancellation_is_immediate_and_streamed_until_close(tmp_path) -> None:  # type: ignore[no-untyped-def]
    application = _app_with_runner(tmp_path)  # no workers: the job stays QUEUED
    server = RunningServer(application)
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        response = server.client.post(f"/api/v1/jobs/{job.job_id}/cancel")
        assert response.status_code == 200, response.text
        assert response.json()["job"]["status"] == "CANCELLED"
        assert application.state.job_runner.get(job.job_id).status is JobStatus.CANCELLED
        # The stream replays the terminal event and then closes on its own;
        # a heartbeat would only appear if the server failed to exit.
        lines, _ = _read_stream(server.client, job.job_id, stop=_stop_on_heartbeat)
    finally:
        server.close()

    frames = _parse_sse(lines)
    assert [frame["event"] for frame in frames] == ["JOB_SUBMITTED", "JOB_CANCELLED"]


def test_running_cancellation_takes_effect_at_safe_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    started, release = Event(), Event()
    application = _app_with_runner(tmp_path, adapter=BlockingAdapter(started, release))
    server = RunningServer(application, lifespan="on")  # lifespan starts the worker
    try:
        analysis_id = _create_analysis(application)
        job = application.state.job_runner.submit(analysis_id, _plan())
        assert started.wait(5)
        response = server.client.post(f"/api/v1/jobs/{job.job_id}/cancel")
        assert response.status_code == 200, response.text
        assert response.json()["job"]["status"] == "CANCEL_REQUESTED"
        # The in-flight step is not killed mid-execution.
        time.sleep(0.1)
        assert (
            application.state.job_runner.get(job.job_id).status
            is JobStatus.CANCEL_REQUESTED
        )
        release.set()
        deadline = time.time() + 5
        while (
            time.time() < deadline
            and application.state.job_runner.get(job.job_id).status
            is not JobStatus.CANCELLED
        ):
            time.sleep(0.01)
        assert application.state.job_runner.get(job.job_id).status is JobStatus.CANCELLED
        # SSE replays the full ordered history, then closes on the terminal
        # event (a heartbeat would appear only if the server failed to exit).
        lines, _ = _read_stream(server.client, job.job_id, stop=_stop_on_heartbeat)
    finally:
        server.close()

    frames = _parse_sse(lines)
    events = [frame["event"] for frame in frames]
    step_ids = [json.loads(frame["data"]).get("step_id") for frame in frames]
    assert events == [
        "JOB_SUBMITTED",
        "JOB_STARTED",
        "STEP_STARTED",
        "CANCEL_REQUESTED",
        "JOB_CANCELLED",
    ]
    # Cancellation landed at the checkpoint after s1; s2 never started.
    assert "s2" not in step_ids
