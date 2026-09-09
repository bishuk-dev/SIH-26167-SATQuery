"""Typed boundaries shared by the execution engine and registered adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from satquery.persistence import MetadataRepository


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    tool_id: str
    input_bindings: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    expected_evidence_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]
    planner_version: str = "1"

    def as_payload(self) -> dict[str, Any]:
        return {
            "planner_version": self.planner_version,
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool_id": step.tool_id,
                    "input_bindings": dict(step.input_bindings),
                    "parameters": dict(step.parameters),
                    "depends_on": list(step.depends_on),
                    "expected_evidence_type": step.expected_evidence_type,
                }
                for step in self.steps
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionPlan":
        return cls(
            planner_version=str(payload.get("planner_version", "1")),
            steps=tuple(
                PlanStep(
                    step_id=str(item["step_id"]),
                    tool_id=str(item["tool_id"]),
                    input_bindings=dict(item.get("input_bindings", {})),
                    parameters=dict(item.get("parameters", {})),
                    depends_on=tuple(item.get("depends_on", ())),
                    expected_evidence_type=item.get("expected_evidence_type"),
                )
                for item in payload.get("steps", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    step_id: str
    tool_id: str
    input_bindings: dict[str, str]
    parameters: dict[str, Any]
    prior_results: dict[str, "ToolResult"] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    analysis_id: str | None = None
    evidence_id: str | None = None
    media_type: str = "application/octet-stream"
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    artifact_id: str
    path: Path
    suffix: str


@dataclass(frozen=True, slots=True)
class ArtifactOutput:
    staged: StagedArtifact
    metadata: ArtifactMetadata


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    path: Path
    storage_key: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: datetime
    metadata: ArtifactMetadata


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Validated adapter result; artifacts are published only after return."""

    output: Any = None
    artifacts: tuple[ArtifactOutput, ...] = ()
    evidence: Any = None


class ToolAdapter(Protocol):
    def execute(self, call: ToolCall, context: "ExecutionContext") -> ToolResult: ...


@dataclass(slots=True)
class ExecutionContext:
    job_id: str
    analysis_id: str
    repository: "MetadataRepository | None" = None
    cancel_event: Event = field(default_factory=Event)
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_cancelled(self) -> bool:
        if self.cancel_event.is_set():
            return True
        if self.repository is None:
            return False
        from satquery.persistence import JobStatus

        job = self.repository.get_job(self.job_id)
        return job is not None and job.status is JobStatus.CANCEL_REQUESTED


class ExecutionEventType(StrEnum):
    SUBMITTED = "JOB_SUBMITTED"
    STARTED = "JOB_STARTED"
    STEP_STARTED = "STEP_STARTED"
    STEP_SUCCEEDED = "STEP_SUCCEEDED"
    STEP_FAILED = "STEP_FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "JOB_CANCELLED"
    SUCCEEDED = "JOB_SUCCEEDED"
    FAILED = "JOB_FAILED"


__all__ = [
    "ArtifactMetadata",
    "ArtifactOutput",
    "ArtifactRecord",
    "ExecutionContext",
    "ExecutionEventType",
    "ExecutionPlan",
    "PlanStep",
    "StagedArtifact",
    "ToolAdapter",
    "ToolCall",
    "ToolResult",
]
