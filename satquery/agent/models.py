"""Typed query-intent contracts for the deterministic agent boundary."""

from __future__ import annotations

from typing import Literal

from satquery.ingestion.models import ContractModel


TaskFamily = Literal[
    "SINGLE_VQA",
    "GROUND_OBJECT",
    "METADATA_QUERY",
    "MEASURE",
    "CHANGE_VQA",
    "CHANGE_LOCALIZE",
    "CHANGE_MEASURE",
    "CHANGE_DESCRIPTION",
    "CROSS_MODAL_VQA",
    "CAPABILITY_QUERY",
    "AMBIGUOUS",
]
TemporalDirection = Literal["T1_TO_T2", "T2_TO_T1", "UNKNOWN"]
FailureOutcome = Literal[
    "ALLOW",
    "ALLOW_WITH_WARNING",
    "REQUEST_INPUT",
    "ABSTAIN",
    "REJECT",
]
FailureSeverity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]


class QueryIntent(ContractModel):
    """A bounded interpretation that contains no executable instruction."""

    task_family: TaskFamily
    target_semantic: str | None = None
    requested_measurement: str | None = None
    temporal_direction: TemporalDirection | None = None
    spatial_request: bool = False
    matched_rule: str
    ambiguities: tuple[str, ...] = ()


class FeasibilityCheck(ContractModel):
    """One ordered, named feasibility check and its policy contribution."""

    check_id: str
    passed: bool
    severity: FailureSeverity
    outcome: FailureOutcome
    code: str | None = None
    message: str | None = None


class FeasibilityIssue(ContractModel):
    """A user-safe failure or warning retained by feasibility evaluation."""

    code: str
    severity: FailureSeverity
    outcome: FailureOutcome
    message: str


class FeasibilityResult(ContractModel):
    """Complete feasibility decision; no check is discarded for precedence."""

    checks: tuple[FeasibilityCheck, ...]
    outcome: FailureOutcome
    failures: tuple[FeasibilityIssue, ...] = ()
    warnings: tuple[FeasibilityIssue, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()

    @property
    def ordered_checks(self) -> tuple[FeasibilityCheck, ...]:
        return self.checks

    @property
    def policy_outcome(self) -> FailureOutcome:
        return self.outcome


__all__ = [
    "FailureOutcome",
    "FailureSeverity",
    "FeasibilityCheck",
    "FeasibilityIssue",
    "FeasibilityResult",
    "QueryIntent",
    "TaskFamily",
    "TemporalDirection",
]
