"""Durable bounded local worker queue."""

from __future__ import annotations

import json
import os
import queue
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from uuid import uuid4

from satquery.agent.composer import AnswerComposer
from satquery.agent.models import QueryIntent
from satquery.evidence.graph import (
    EVIDENCE_NODE_ADAPTER,
    EdgeType,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
)
from satquery.evidence.models import AgreementEvidence, MeasurementEvidence
from satquery.execution.engine import ExecutionEngine, JobCancelledError
from satquery.execution.models import (
    ExecutionContext,
    ExecutionEventType,
    ExecutionPlan,
    ToolResult,
)
from satquery.persistence import (
    AnalysisStatus,
    ExecutionEvent,
    JobRecord,
    JobStatus,
    MetadataRepository,
)
from satquery.verification.analysis import AnalysisVerifier


class JobQueueFullError(RuntimeError):
    """The bounded local queue has no capacity."""


def _graph_from_results(
    results: tuple[ToolResult, ...], *, input_ids: tuple[str, ...]
) -> EvidenceGraph:
    nodes = _evidence_nodes(results)
    return EvidenceGraph(nodes=nodes, edges=_evidence_edges(nodes), input_ids=input_ids)


def _evidence_nodes(results: tuple[ToolResult, ...]) -> tuple[EvidenceNode, ...]:
    nodes: list[EvidenceNode] = []
    for result in results:
        raw = result.evidence
        if raw is None:
            continue
        values = raw if isinstance(raw, (list, tuple)) else (raw,)
        for item in values:
            nodes.append(EVIDENCE_NODE_ADAPTER.validate_python(item))
    return tuple(nodes)


def _evidence_edges(nodes: tuple[EvidenceNode, ...]) -> tuple[EvidenceEdge, ...]:
    node_ids = {node.evidence_id for node in nodes}
    edges: set[tuple[str, str, EdgeType]] = set()
    for node in nodes:
        if isinstance(node, MeasurementEvidence):
            edges.add((node.source_evidence_id, node.evidence_id, EdgeType.MEASURED_FROM))
        elif isinstance(node, AgreementEvidence):
            edges.add((node.first_evidence_id, node.evidence_id, EdgeType.DERIVED_FROM))
            edges.add((node.second_evidence_id, node.evidence_id, EdgeType.DERIVED_FROM))
        if not isinstance(node, MeasurementEvidence):
            for parent_id in node.provenance.parent_evidence_ids:
                if parent_id in node_ids and parent_id != node.evidence_id:
                    edges.add((parent_id, node.evidence_id, EdgeType.DERIVED_FROM))
    return tuple(
        EvidenceEdge(
            source_evidence_id=source_id,
            target_evidence_id=target_id,
            edge_type=edge_type,
        )
        for source_id, target_id, edge_type in sorted(
            edges, key=lambda edge: (edge[1], edge[0], edge[2].value)
        )
    )


class JobRunner:
    def __init__(
        self,
        repository: MetadataRepository,
        engine: ExecutionEngine,
        *,
        max_queued_jobs: int = 32,
        worker_count: int = 1,
    ) -> None:
        if max_queued_jobs <= 0 or worker_count <= 0:
            raise ValueError("queue and worker counts must be positive")
        self.repository = repository
        self.engine = engine
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queued_jobs)
        self._worker_count = worker_count
        self._workers: list[Thread] = []
        self._stop_event = Event()
        self._lock = Lock()
        self._cancel_events: dict[str, Event] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _append_event(self, job_id: str, event_type: ExecutionEventType | str, payload: dict) -> None:
        # Sequence allocation and insertion must be one runner-serialized
        # operation because engine callbacks and API cancellation can emit
        # events concurrently for the same job.
        with self._lock:
            events = self.repository.list_events(job_id)
            value = event_type.value if isinstance(event_type, ExecutionEventType) else event_type
            self.repository.append_event(
                ExecutionEvent(
                    event_id=f"event_{uuid4().hex}",
                    job_id=job_id,
                    sequence=len(events),
                    event_type=value,
                    created_at=self._now(),
                    payload=payload,
                )
            )

    def submit(self, analysis_id: str, plan: ExecutionPlan) -> JobRecord:
        job_id = f"job_{uuid4().hex}"
        now = self._now()
        record = JobRecord(
            job_id=job_id,
            analysis_id=analysis_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            payload={"plan": plan.as_payload()},
        )
        self.repository.create_job(record)
        return self.enqueue_existing(job_id)

    def enqueue_existing(self, job_id: str) -> JobRecord:
        """Submit an already-persisted QUEUED job to the in-process queue.

        This supports API submission transactions that create the analysis,
        immutable inputs, frozen plan, and job atomically before making the job
        visible to workers.
        """

        record = self.repository.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.status is not JobStatus.QUEUED:
            raise ValueError("only QUEUED jobs can be submitted")
        self._append_event(job_id, ExecutionEventType.SUBMITTED, {"analysis_id": record.analysis_id})
        try:
            self._queue.put_nowait(job_id)
        except queue.Full as exc:
            self.repository.transition_job(
                job_id, JobStatus.QUEUED, JobStatus.FAILED, updated_at=self._now()
            )
            self._append_event(
                job_id,
                ExecutionEventType.FAILED,
                {"error_type": "JobQueueFullError"},
            )
            raise JobQueueFullError("job queue is full") from exc
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.repository.get_job(job_id)

    def request_cancel(self, job_id: str) -> JobRecord:
        job = self.repository.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        while True:
            now = self._now()
            if job.status is JobStatus.QUEUED:
                if self.repository.transition_job(
                    job_id, JobStatus.QUEUED, JobStatus.CANCELLED, updated_at=now
                ):
                    self._append_event(job_id, ExecutionEventType.CANCELLED, {})
                    return self.repository.get_job(job_id) or job
            elif job.status is JobStatus.RUNNING:
                if self.repository.transition_job(
                    job_id, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, updated_at=now
                ):
                    with self._lock:
                        self._cancel_events.setdefault(job_id, Event()).set()
                    self._append_event(job_id, ExecutionEventType.CANCEL_REQUESTED, {})
                    return self.repository.get_job(job_id) or job
            else:
                raise ValueError("job is already terminal")
            # A concurrent worker changed the state between the read and CAS.
            # Re-read and either cancel the new RUNNING state or report the
            # terminal conflict instead of returning a stale success.
            refreshed = self.repository.get_job(job_id)
            if refreshed is None:
                raise KeyError(job_id)
            job = refreshed

    def start(self) -> None:
        with self._lock:
            if self._workers:
                return
            self._stop_event.clear()
            self._workers = [
                Thread(target=self._worker, name=f"satquery-job-{index}", daemon=True)
                for index in range(self._worker_count)
            ]
            for worker in self._workers:
                worker.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        for _ in self._workers:
            try:
                self._queue.put_nowait("")
            except queue.Full:
                break
        for worker in self._workers:
            worker.join(timeout=max(timeout_seconds, 0.0))
        with self._lock:
            self._workers = []

    def _claim(self, job_id: str) -> JobRecord | None:
        job = self.repository.get_job(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return None
        if not self.repository.transition_job(
            job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=self._now()
        ):
            return None
        claimed = self.repository.get_job(job_id)
        if claimed is None:
            return None
        return claimed

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not job_id:
                    continue
                job = self._claim(job_id)
                if job is None:
                    continue
                self._run(job)
            finally:
                self._queue.task_done()

    def _run(self, job: JobRecord) -> None:
        self._append_event(job.job_id, ExecutionEventType.STARTED, {})
        self.repository.transition_analysis(
            job.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=self._now()
        )
        cancel_event = Event()
        with self._lock:
            cancel_event = self._cancel_events.setdefault(job.job_id, cancel_event)
        try:
            plan = ExecutionPlan.from_payload(job.payload["plan"])
            metadata = dict(job.payload.get("metadata", {})) if isinstance(job.payload.get("metadata"), dict) else {}
            metadata["event_callback"] = lambda event_type, payload: self._append_event(
                job.job_id, event_type, payload
            )
            context = ExecutionContext(
                job_id=job.job_id,
                analysis_id=job.analysis_id,
                repository=self.repository,
                cancel_event=cancel_event,
                metadata=metadata,
            )
            results = self.engine.execute(plan, context)
            current = self.repository.get_job(job.job_id)
            if current is not None and current.status is JobStatus.CANCEL_REQUESTED:
                raise JobCancelledError("job cancellation requested")
            self._verify_and_persist_answer(job, results)
            succeeded = self.repository.transition_job(
                job.job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, updated_at=self._now()
            )
            if succeeded:
                self._append_event(job.job_id, ExecutionEventType.SUCCEEDED, {})
            else:
                current = self.repository.get_job(job.job_id)
                if current is not None and current.status is JobStatus.CANCEL_REQUESTED:
                    if self.repository.transition_job(
                        job.job_id,
                        JobStatus.CANCEL_REQUESTED,
                        JobStatus.CANCELLED,
                        updated_at=self._now(),
                    ):
                        self._mark_analysis_terminal(job.analysis_id, AnalysisStatus.CANCELLED)
                        self._append_event(job.job_id, ExecutionEventType.CANCELLED, {})
        except JobCancelledError:
            current = self.repository.get_job(job.job_id)
            if current is not None and current.status is JobStatus.RUNNING:
                self.repository.transition_job(
                    job.job_id, JobStatus.RUNNING, JobStatus.CANCELLED, updated_at=self._now()
                )
            elif current is not None and current.status is JobStatus.CANCEL_REQUESTED:
                self.repository.transition_job(
                    job.job_id, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED, updated_at=self._now()
                )
            self._mark_analysis_terminal(job.analysis_id, AnalysisStatus.CANCELLED)
            self._append_event(job.job_id, ExecutionEventType.CANCELLED, {})
        except BaseException as exc:
            current = self.repository.get_job(job.job_id)
            if current is not None and current.status is JobStatus.CANCEL_REQUESTED:
                self.repository.transition_job(
                    job.job_id, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED, updated_at=self._now()
                )
                self._mark_analysis_terminal(job.analysis_id, AnalysisStatus.CANCELLED)
                self._append_event(job.job_id, ExecutionEventType.CANCELLED, {})
            elif current is not None and current.status is JobStatus.RUNNING:
                self.repository.transition_job(
                    job.job_id, JobStatus.RUNNING, JobStatus.FAILED, updated_at=self._now()
                )
                self._mark_analysis_terminal(job.analysis_id, AnalysisStatus.FAILED)
                self._append_event(
                    job.job_id,
                    ExecutionEventType.FAILED,
                    {"error_type": type(exc).__name__},
                )
        finally:
            with self._lock:
                self._cancel_events.pop(job.job_id, None)

    def _verify_and_persist_answer(self, job: JobRecord, results: tuple[ToolResult, ...]) -> None:
        analysis = self.repository.get_analysis(job.analysis_id)
        if analysis is None:
            raise RuntimeError("analysis record is missing")
        intent = QueryIntent.model_validate_json(json.dumps(analysis.payload["intent"]))
        graph = _graph_from_results(
            results,
            input_ids=tuple(str(item) for item in analysis.payload.get("observation_ids", ())),
        )
        report = AnalysisVerifier().verify(intent, graph)
        answer = AnswerComposer().compose(
            str(analysis.payload.get("query", "")), intent, graph, report
        )
        target_status = AnalysisStatus.SUCCEEDED if answer.answered else AnalysisStatus.ABSTAINED
        persisted = self.repository.complete_analysis_with_evidence(
            job.analysis_id,
            expected=AnalysisStatus.RUNNING,
            target=target_status,
            updated_at=self._now(),
            evidence_rows=graph.to_evidence_rows(analysis_id=job.analysis_id),
            edge_rows=graph.to_edge_rows(analysis_id=job.analysis_id),
            verification_payload=report.model_dump(mode="json"),
            answer_payload=answer.model_dump(mode="json"),
        )
        if not persisted:
            raise RuntimeError("analysis completion could not be persisted")

    def _mark_analysis_terminal(self, analysis_id: str, target: AnalysisStatus) -> None:
        now = self._now()
        if self.repository.transition_analysis(
            analysis_id, AnalysisStatus.RUNNING, target, updated_at=now
        ):
            return
        self.repository.transition_analysis(
            analysis_id, AnalysisStatus.PENDING, target, updated_at=now
        )


__all__ = ["JobQueueFullError", "JobRunner"]
