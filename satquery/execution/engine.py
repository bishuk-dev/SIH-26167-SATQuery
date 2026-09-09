"""Topological execution of an already validated bounded plan."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import BoundedSemaphore
from typing import Any, Mapping

from satquery.artifacts import ArtifactStore
from satquery.execution.models import (
    ExecutionContext,
    ExecutionPlan,
    PlanStep,
    ToolAdapter,
    ToolCall,
    ToolResult,
)


class ExecutionError(RuntimeError):
    """Base execution failure."""


class UnknownToolError(ExecutionError):
    """A plan referenced an adapter not registered at startup."""


class ToolExecutionError(ExecutionError):
    """A registered tool failed or returned an invalid result."""


class JobCancelledError(ExecutionError):
    """Execution stopped at a cancellation checkpoint."""


class ExecutionEngine:
    def __init__(
        self,
        adapters: Mapping[str, ToolAdapter],
        *,
        artifact_store: ArtifactStore | None = None,
        tool_registry: Any | None = None,
        timeout_seconds: float = 300.0,
        tool_concurrency: Mapping[str, int] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._adapters = dict(adapters)
        self._artifact_store = artifact_store
        self._tool_registry = tool_registry
        self._timeout_seconds = timeout_seconds
        self._semaphores = {
            tool_id: BoundedSemaphore(limit)
            for tool_id, limit in (tool_concurrency or {}).items()
            if limit > 0
        }
        if any(limit <= 0 for limit in (tool_concurrency or {}).values()):
            raise ValueError("tool concurrency limits must be positive")

    def _validate_step(self, step: PlanStep) -> ToolAdapter:
        adapter = self._adapters.get(step.tool_id)
        if adapter is None:
            raise UnknownToolError(f"tool is not registered: {step.tool_id}")
        if self._tool_registry is not None and self._tool_registry.get(step.tool_id) is None:
            raise UnknownToolError(f"tool is not registered: {step.tool_id}")
        return adapter

    @staticmethod
    def _ordered_steps(plan: ExecutionPlan) -> tuple[PlanStep, ...]:
        steps = {step.step_id: step for step in plan.steps}
        if len(steps) != len(plan.steps):
            raise ExecutionError("execution plan contains duplicate step IDs")
        for step in plan.steps:
            missing = set(step.depends_on) - steps.keys()
            if missing:
                raise ExecutionError(f"step {step.step_id} has unknown dependencies")
        remaining = set(steps)
        ordered: list[PlanStep] = []
        while remaining:
            ready = [
                steps[step_id]
                for step_id in remaining
                if set(steps[step_id].depends_on).issubset({step.step_id for step in ordered})
            ]
            if not ready:
                raise ExecutionError("execution plan contains a dependency cycle")
            ready.sort(key=lambda step: step.step_id)
            ordered.extend(ready)
            remaining.difference_update(step.step_id for step in ready)
        return tuple(ordered)

    def _execute_adapter(
        self,
        adapter: ToolAdapter,
        call: ToolCall,
        context: ExecutionContext,
        timeout_seconds: float,
    ) -> ToolResult:
        semaphore = self._semaphores.get(call.tool_id)
        if semaphore is not None and not semaphore.acquire(timeout=timeout_seconds):
            raise ToolExecutionError(f"tool concurrency limit timed out: {call.tool_id}")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="satquery-tool")
        future = executor.submit(adapter.execute, call, context)
        release_now = True
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            # Python cannot forcibly stop a running thread. Keep the semaphore
            # held until that call actually exits, preventing an over-capacity
            # retry while the timed-out call is still executing.
            release_now = False
            if semaphore is not None:
                future.add_done_callback(lambda _future: semaphore.release())
            future.cancel()
            raise ToolExecutionError(f"tool timed out: {call.tool_id}") from exc
        except JobCancelledError:
            raise
        except BaseException as exc:
            raise ToolExecutionError(f"tool failed: {call.tool_id}") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if semaphore is not None and release_now:
                semaphore.release()
        if not isinstance(result, ToolResult):
            raise ToolExecutionError(f"tool returned an invalid result: {call.tool_id}")
        return result

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> tuple[ToolResult, ...]:
        ordered = self._ordered_steps(plan)
        for step in ordered:
            self._validate_step(step)
        results: dict[str, ToolResult] = {}
        for step in ordered:
            if context.is_cancelled():
                raise JobCancelledError("job cancellation requested")
            self._emit(context, "STEP_STARTED", {"step_id": step.step_id, "tool_id": step.tool_id})
            call = ToolCall(
                step_id=step.step_id,
                tool_id=step.tool_id,
                input_bindings=dict(step.input_bindings),
                parameters=dict(step.parameters),
                prior_results={key: results[key] for key in step.depends_on},
            )
            try:
                result = self._execute_adapter(
                    self._adapters[step.tool_id],
                    call,
                    context,
                    context.timeout_seconds or self._timeout_seconds,
                )
            except BaseException as exc:
                self._emit(
                    context,
                    "STEP_FAILED",
                    {"step_id": step.step_id, "tool_id": step.tool_id, "error_type": type(exc).__name__},
                )
                raise
            if context.is_cancelled():
                self._discard_outputs(result)
                raise JobCancelledError("job cancellation requested")
            self._publish_outputs(result)
            results[step.step_id] = result
            self._emit(context, "STEP_SUCCEEDED", {"step_id": step.step_id, "tool_id": step.tool_id})
        return tuple(results[step.step_id] for step in ordered)

    @staticmethod
    def _emit(context: ExecutionContext, event_type: str, payload: dict[str, Any]) -> None:
        callback = context.metadata.get("event_callback")
        if callback is not None:
            callback(event_type, payload)

    def _publish_outputs(self, result: ToolResult) -> None:
        if not result.artifacts:
            return
        if self._artifact_store is None:
            raise ToolExecutionError("tool returned artifacts without an artifact store")
        published = []
        try:
            for output in result.artifacts:
                published.append(self._artifact_store.publish(output.staged, output.metadata))
        except BaseException as exc:
            for output in result.artifacts:
                self._artifact_store.discard(output.staged)
            raise ToolExecutionError("artifact publication failed") from exc

    def _discard_outputs(self, result: ToolResult) -> None:
        if self._artifact_store is not None:
            for output in result.artifacts:
                self._artifact_store.discard(output.staged)


__all__ = [
    "ExecutionEngine",
    "ExecutionError",
    "JobCancelledError",
    "ToolExecutionError",
    "UnknownToolError",
]
