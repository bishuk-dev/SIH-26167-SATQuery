from __future__ import annotations

import pytest
from pydantic import ValidationError

from satquery.agent.models import FeasibilityResult, QueryIntent
from satquery.agent.planner import BoundedPlanner, ExecutionPlan, PlannerError
from satquery.registry.tools import load_tool_registry


def _intent(**updates: object) -> QueryIntent:
    values: dict[str, object] = {
        "task_family": "CHANGE_LOCALIZE",
        "target_semantic": "sar",
        "temporal_direction": "T1_TO_T2",
        "matched_rule": "test",
    }
    values.update(updates)
    return QueryIntent.model_validate(values)


def _feasibility(outcome: str = "ALLOW") -> FeasibilityResult:
    return FeasibilityResult.model_validate(
        {"checks": (), "outcome": outcome, "allowed_capability_ids": ()}
    )


def _inputs() -> dict[str, object]:
    return {
        "observation_ids": ("obs_" + "1" * 32, "obs_" + "2" * 32),
        "parameters": {
            "radiometric_domain": "backscatter_db",
            "polarizations": ["VV"],
            "threshold": 3.0,
        },
    }


def test_plan_is_deterministic_and_registry_bound() -> None:
    planner = BoundedPlanner(load_tool_registry())
    first = planner.plan(_intent(), _feasibility(), _inputs())
    second = planner.plan(_intent(), _feasibility(), _inputs())
    assert first == second
    assert first.steps[0].tool_id == "sar_temporal_change_v1"
    assert first.steps[0].expected_evidence_type == "change_mask"
    assert first.plan_hash
    assert "query" not in first.model_dump_json()


def test_parameter_bounds_are_enforced() -> None:
    planner = BoundedPlanner(load_tool_registry())
    invalid = _inputs()
    invalid["parameters"] = {
        "radiometric_domain": "backscatter_db",
        "polarizations": ["VV"],
        "threshold": -1,
    }
    with pytest.raises(PlannerError, match="below its minimum"):
        planner.plan(_intent(), _feasibility(), invalid)


def test_denied_feasibility_returns_no_steps() -> None:
    plan = BoundedPlanner(load_tool_registry()).plan(
        _intent(), _feasibility("REJECT"), _inputs()
    )
    assert plan.steps == ()


def test_execution_plan_rejects_cycles_and_more_than_eight_steps() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        ExecutionPlan.model_validate(
            {
                "planner_version": "test",
                "registry_hash": "a" * 64,
                "steps": (
                    {
                        "step_id": "step_a",
                        "tool_id": "tool_a",
                        "input_bindings": {},
                        "parameters": {},
                        "depends_on": ("step_b",),
                        "expected_evidence_type": "x",
                    },
                    {
                        "step_id": "step_b",
                        "tool_id": "tool_b",
                        "input_bindings": {},
                        "parameters": {},
                        "depends_on": ("step_a",),
                        "expected_evidence_type": "x",
                    },
                ),
                "plan_hash": "b" * 64,
            }
        )
    with pytest.raises(ValidationError, match="at most 8"):
        ExecutionPlan.model_validate(
            {
                "planner_version": "test",
                "registry_hash": "a" * 64,
                "steps": tuple(
                    {
                        "step_id": f"step_{index}",
                        "tool_id": "tool_a",
                        "input_bindings": {},
                        "parameters": {},
                        "depends_on": (),
                        "expected_evidence_type": "x",
                    }
                    for index in range(9)
                ),
                "plan_hash": "b" * 64,
            }
        )


def test_unsupported_workflow_fails_closed() -> None:
    planner = BoundedPlanner(load_tool_registry())
    with pytest.raises(PlannerError, match="supports"):
        planner.plan(
            _intent(task_family="MEASURE", target_semantic=None, requested_measurement="area"),
            _feasibility(),
            _inputs(),
        )
