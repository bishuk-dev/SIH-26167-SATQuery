from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import pytest

from satquery.artifacts import ArtifactStore
from satquery.execution import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionPlan,
    PlanStep,
    ToolExecutionError,
    ToolResult,
    UnknownToolError,
)


class RecordingAdapter:
    def __init__(self, name: str, order: list[str], *, delay: float = 0.0) -> None:
        self.name, self.order, self.delay = name, order, delay

    def execute(self, call, context):
        self.order.append(self.name)
        if self.delay:
            time.sleep(self.delay)
        return ToolResult(output=self.name)


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(steps=tuple(steps))


def test_engine_executes_dependency_order_and_passes_prior_results():
    order: list[str] = []
    first = RecordingAdapter("first", order)
    second = RecordingAdapter("second", order)
    engine = ExecutionEngine({"first": first, "second": second})
    results = engine.execute(
        _plan(
            PlanStep("b", "second", depends_on=("a",)),
            PlanStep("a", "first"),
        ),
        ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32),
    )
    assert order == ["first", "second"]
    assert [result.output for result in results] == ["first", "second"]


def test_engine_rejects_unknown_tools_and_cycles():
    engine = ExecutionEngine({})
    with pytest.raises(UnknownToolError):
        engine.execute(_plan(PlanStep("a", "missing")), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    with pytest.raises(Exception, match="cycle"):
        ExecutionEngine({"x": RecordingAdapter("x", [])}).execute(
            _plan(PlanStep("a", "x", depends_on=("b",)), PlanStep("b", "x", depends_on=("a",))),
            ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32),
        )


def test_engine_timeout_fails_and_does_not_publish(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    engine = ExecutionEngine(
        {"slow": RecordingAdapter("slow", [], delay=0.1)},
        artifact_store=store,
        timeout_seconds=0.01,
    )
    with pytest.raises(ToolExecutionError, match="timed out"):
        engine.execute(_plan(PlanStep("a", "slow")), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    assert list(store.artifacts_root.iterdir()) == [store.staging_root]


def test_engine_cancellation_between_steps_discards_outputs(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    cancel = Event()

    class CancellingAdapter:
        def execute(self, call, context):
            cancel.set()
            return ToolResult(output="not published")

    engine = ExecutionEngine({"x": CancellingAdapter()}, artifact_store=store)
    with pytest.raises(Exception, match="cancellation"):
        engine.execute(_plan(PlanStep("a", "x"), PlanStep("b", "x", depends_on=("a",))), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32, cancel_event=cancel))
    assert not any(store.artifacts_root.glob("artifact_*"))
