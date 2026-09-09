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


class QueryIntent(ContractModel):
    """A bounded interpretation that contains no executable instruction."""

    task_family: TaskFamily
    target_semantic: str | None = None
    requested_measurement: str | None = None
    temporal_direction: TemporalDirection | None = None
    spatial_request: bool = False
    matched_rule: str
    ambiguities: tuple[str, ...] = ()


__all__ = ["QueryIntent", "TaskFamily", "TemporalDirection"]
