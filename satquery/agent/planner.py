"""Registry-constrained, deterministic workflow planning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import Field, model_validator

from satquery.agent.models import FeasibilityResult, QueryIntent
from satquery.ingestion.models import ContractModel
from satquery.registry.tools import ToolParameter, ToolRegistration, ToolRegistry

_MAX_PLAN_STEPS = 8
_DEFAULT_PLANNER_VERSION = "phase5-planner-v1"
_ID_PATTERN = re.compile(r"^(?:obs|pair|artifact|mask|ana|job|step)_[a-z0-9_]{1,64}$")


class PlannerError(ValueError):
    """A safe, operational planning refusal."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PlanStep(ContractModel):
    """One bounded registered-tool invocation."""

    step_id: str = Field(pattern=r"^step_[a-z0-9_]+$")
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    input_bindings: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    expected_evidence_type: str = Field(min_length=1)


class ExecutionPlan(ContractModel):
    """Immutable dry-run plan; it contains no query text or executable code."""

    planner_version: str = Field(min_length=1)
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    steps: tuple[PlanStep, ...] = Field(max_length=_MAX_PLAN_STEPS)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_dependencies(self) -> "ExecutionPlan":
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("plan step IDs must be unique")
        known = set(step_ids)
        edges = {step.step_id: set(step.depends_on) for step in self.steps}
        for step_id, dependencies in edges.items():
            if step_id in dependencies:
                raise ValueError("a plan step cannot depend on itself")
            if not dependencies <= known:
                raise ValueError("plan dependency refers to an unknown step")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("plan dependencies must be acyclic")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in edges[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            visit(step_id)
        return self


@dataclass(frozen=True)
class _StepSpec:
    tool_id: str
    input_bindings: Mapping[str, str]
    parameters: Mapping[str, Any]
    depends_on: tuple[str, ...]


# Policy values are factories over normalized server-side IDs. The wildcard
# matching keeps the policy table composable without exposing arbitrary tools.
PolicyFactory = Callable[[QueryIntent, Mapping[str, Any]], tuple[_StepSpec, ...]]


def _ids(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    value = inputs.get("observation_ids", ())
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _sar_change(intent: QueryIntent, inputs: Mapping[str, Any]) -> tuple[_StepSpec, ...]:
    observation_ids = _ids(inputs)
    if len(observation_ids) != 2:
        raise PlannerError("MISSING_TEMPORAL_INPUTS", "A SAR temporal plan requires two observations.")
    parameters = dict(inputs.get("parameters", {}))
    return (
        _StepSpec(
            tool_id="sar_temporal_change_v1",
            input_bindings={"t1": observation_ids[0], "t2": observation_ids[1]},
            parameters=parameters,
            depends_on=(),
        ),
    )


def _sar_change_then_area(intent: QueryIntent, inputs: Mapping[str, Any]) -> tuple[_StepSpec, ...]:
    steps = _sar_change(intent, inputs)
    return steps + (
        _StepSpec(
            tool_id="compute_mask_area_v1",
            input_bindings={"mask": "step_sar_temporal_change_v1"},
            parameters={},
            depends_on=("step_sar_temporal_change_v1",),
        ),
    )


_POLICIES: dict[tuple[str, str | None, str | None, str | None, bool | None], PolicyFactory] = {
    ("CHANGE_LOCALIZE", "sar", None, None, None): _sar_change,
    ("CHANGE_MEASURE", "sar", "area", None, None): _sar_change_then_area,
}


def _policy_for(intent: QueryIntent) -> PolicyFactory:
    key = (
        intent.task_family,
        intent.target_semantic,
        intent.requested_measurement,
        intent.temporal_direction,
        intent.spatial_request,
    )
    factory = _POLICIES.get(key)
    if factory is not None:
        return factory
    # Temporal direction and spatial request are orthogonal to tool choice for
    # the currently registered deterministic SAR specialist.
    for candidate, value in _POLICIES.items():
        if candidate[0] != intent.task_family:
            continue
        if candidate[1] not in {None, intent.target_semantic}:
            continue
        if candidate[2] not in {None, intent.requested_measurement}:
            continue
        return value
    raise PlannerError(
        "WORKFLOW_UNSUPPORTED",
        "No bounded registered workflow supports this request.",
    )


def _validate_parameter(name: str, value: Any, definition: ToolParameter) -> Any:
    kind = definition.type
    if kind == "enum":
        if not isinstance(value, str) or value not in definition.values:
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' is outside its registered enum.")
        return value
    if kind == "enum_list":
        if not isinstance(value, (list, tuple)):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be a bounded list.")
        if not definition.min_items <= len(value) <= definition.max_items:  # type: ignore[operator]
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' has an invalid item count.")
        if any(not isinstance(item, str) or item not in definition.values for item in value):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' contains an unregistered value.")
        return list(value)
    if kind in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be numeric.")
        if kind == "integer" and not isinstance(value, int):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be an integer.")
        if definition.minimum is not None and value < definition.minimum:
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' is below its minimum.")
        if definition.maximum is not None and value > definition.maximum:
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' exceeds its maximum.")
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be boolean.")
        return value
    if kind in {"server_id", "mask"}:
        if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be a server-issued ID.")
        return value
    if kind == "roi":
        if not isinstance(value, Mapping):
            raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' must be a bounded ROI object.")
        return dict(value)
    raise PlannerError("PARAMETER_OUT_OF_BOUNDS", f"Parameter '{name}' has an unsupported registered type.")


def _validate_step(step: _StepSpec, registry: ToolRegistry) -> PlanStep:
    registration = registry.get(step.tool_id)
    if registration is None:
        raise PlannerError("TOOL_NOT_REGISTERED", "The requested workflow references an unavailable registered tool.")
    registered_inputs = {item.name: item for item in registration.inputs}
    if set(step.input_bindings) - set(registered_inputs):
        raise PlannerError("INPUT_BINDING_INVALID", "The workflow contains an input not accepted by its registered tool.")
    for value in step.input_bindings.values():
        if not _ID_PATTERN.fullmatch(value):
            raise PlannerError("INPUT_BINDING_INVALID", "Workflow inputs must be server-issued IDs.")
    unknown_parameters = set(step.parameters) - set(registration.parameters)
    if unknown_parameters:
        raise PlannerError("PARAMETER_NOT_REGISTERED", "The workflow contains an unregistered parameter.")
    missing_parameters = set(registration.parameters) - set(step.parameters)
    if missing_parameters:
        raise PlannerError("PARAMETER_REQUIRED", "The workflow omitted a registered parameter.")
    parameters = {
        name: _validate_parameter(name, value, registration.parameters[name])
        for name, value in step.parameters.items()
    }
    return PlanStep(
        step_id=f"step_{step.tool_id}",
        tool_id=step.tool_id,
        input_bindings=dict(sorted(step.input_bindings.items())),
        parameters=parameters,
        depends_on=step.depends_on,
        expected_evidence_type=registration.evidence.type,
    )


class BoundedPlanner:
    """Build only deterministic plans backed by the supplied ToolRegistry."""

    def __init__(self, registry: ToolRegistry, *, planner_version: str = _DEFAULT_PLANNER_VERSION) -> None:
        self.registry = registry
        self.planner_version = planner_version

    def plan(
        self,
        intent: QueryIntent,
        feasibility: FeasibilityResult,
        inputs: Mapping[str, Any],
    ) -> ExecutionPlan:
        if feasibility.outcome not in {"ALLOW", "ALLOW_WITH_WARNING"}:
            return self._empty_plan()
        if not isinstance(inputs, Mapping):
            raise PlannerError("INVALID_PLAN_INPUTS", "Plan inputs must be server-resolved identifiers.")
        specs = _policy_for(intent)(intent, inputs)
        if len(specs) > _MAX_PLAN_STEPS:
            raise PlannerError("PLAN_TOO_LARGE", "The bounded workflow exceeds the maximum step count.")
        steps = tuple(_validate_step(spec, self.registry) for spec in specs)
        payload = {
            "planner_version": self.planner_version,
            "registry_hash": self.registry.registry_hash,
            "steps": [step.model_dump(mode="json") for step in steps],
        }
        plan_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        return ExecutionPlan(
            planner_version=self.planner_version,
            registry_hash=self.registry.registry_hash,
            steps=steps,
            plan_hash=plan_hash,
        )

    def _empty_plan(self) -> ExecutionPlan:
        payload = {
            "planner_version": self.planner_version,
            "registry_hash": self.registry.registry_hash,
            "steps": (),
        }
        plan_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        return ExecutionPlan(plan_hash=plan_hash, **payload)


__all__ = ["BoundedPlanner", "ExecutionPlan", "PlanStep", "PlannerError"]
