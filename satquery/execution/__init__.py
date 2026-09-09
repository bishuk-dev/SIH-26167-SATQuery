"""Bounded execution primitives and durable local job runner."""

from satquery.execution.models import (
    ArtifactMetadata,
    ArtifactOutput,
    ArtifactRecord,
    ExecutionContext,
    ExecutionEventType,
    ExecutionPlan,
    PlanStep,
    StagedArtifact,
    ToolAdapter,
    ToolCall,
    ToolResult,
)

__all__ = [
    "ArtifactMetadata",
    "ArtifactOutput",
    "ArtifactRecord",
    "ExecutionContext",
    "ExecutionEngine",
    "ExecutionError",
    "ExecutionEventType",
    "ExecutionPlan",
    "JobCancelledError",
    "JobQueueFullError",
    "JobRunner",
    "PlanStep",
    "StagedArtifact",
    "ToolAdapter",
    "ToolCall",
    "ToolExecutionError",
    "ToolResult",
    "UnknownToolError",
]


def __getattr__(name: str):
    if name in {"ExecutionEngine", "ExecutionError", "JobCancelledError", "ToolExecutionError", "UnknownToolError"}:
        from satquery.execution.engine import (
            ExecutionEngine,
            ExecutionError,
            JobCancelledError,
            ToolExecutionError,
            UnknownToolError,
        )
        return locals()[name]
    if name in {"JobQueueFullError", "JobRunner"}:
        from satquery.execution.jobs import JobQueueFullError, JobRunner
        return locals()[name]
    raise AttributeError(name)
