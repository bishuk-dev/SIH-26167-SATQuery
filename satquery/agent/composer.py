"""Deterministic, evidence-only answer composition."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from satquery.agent.models import FailureOutcome, QueryIntent
from satquery.evidence.graph import EvidenceGraph, EvidenceNode, evidence_id_of
from satquery.evidence.models import (
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    DomainStatus,
    FloodMaskEvidence,
    GroundingEvidence,
    MeasurementEvidence,
    VqaEvidence,
)
from satquery.ingestion.models import ContractModel
from satquery.verification.analysis import VerificationIssue, VerificationReport


class AnswerMeasurement(ContractModel):
    """A numeric claim copied from deterministic MeasurementEvidence."""

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    source_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    measurement_type: Literal["area"]
    value: float
    unit: Literal["m2", "ha", "km2"]
    display_value: str
    method: str
    calculation_crs: str
    positive_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(ge=0)


class UncalibratedScore(ContractModel):
    """A raw model score explicitly labeled as not calibrated confidence."""

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    value: float
    label: Literal["raw_model_score_uncalibrated"] = "raw_model_score_uncalibrated"


class AnalysisAnswer(ContractModel):
    """User-facing deterministic answer plus machine-readable evidence data."""

    answered: bool
    outcome: FailureOutcome
    answer: str
    evidence_ids: tuple[str, ...] = ()
    measurements: tuple[AnswerMeasurement, ...] = ()
    limitations: tuple[str, ...] = ()
    uncalibrated_scores: tuple[UncalibratedScore, ...] = ()
    data: dict[str, Any] = Field(default_factory=dict)


class AnswerComposer:
    """Compose answers from verified evidence records and nothing else.

    Query text and model prose are never parsed for numeric values. Numeric
    response fields are copied only from valid ``MeasurementEvidence`` records.
    """

    def __init__(self, *, display_precision: int = 2) -> None:
        if display_precision < 0:
            raise ValueError("display_precision must be non-negative")
        self.display_precision = display_precision

    def compose(
        self,
        query: str,
        intent: QueryIntent,
        graph: EvidenceGraph,
        report: VerificationReport,
    ) -> AnalysisAnswer:
        del query  # answer content is derived from typed evidence only.
        valid_nodes = _valid_nodes(graph, report)
        limitations = _limitations(valid_nodes, report)
        scores = _uncalibrated_scores(valid_nodes)

        if not report.answered or report.issues:
            return AnalysisAnswer(
                answered=False,
                outcome="ABSTAIN",
                answer="I cannot provide a verified answer because the evidence did not pass verification.",
                evidence_ids=tuple(evidence_id_of(node) for node in valid_nodes),
                limitations=limitations,
                uncalibrated_scores=scores,
                data={
                    "verification": {
                        "answered": report.answered,
                        "issue_codes": [issue.code for issue in report.issues],
                    }
                },
            )

        measurements = _measurements(valid_nodes, self.display_precision)
        if _numeric_intent(intent) and measurements:
            text = _measurement_text(intent, measurements)
        elif intent.task_family == "CHANGE_DESCRIPTION":
            text = _caption_text(valid_nodes)
        elif intent.task_family in {"SINGLE_VQA", "CHANGE_VQA", "CROSS_MODAL_VQA"}:
            text = _vqa_text(valid_nodes, scores)
        elif intent.task_family == "GROUND_OBJECT":
            text = _grounding_text(valid_nodes)
        elif intent.task_family == "CHANGE_LOCALIZE":
            text = _localization_text(valid_nodes)
        else:
            text = "The request is supported by verified evidence."

        return AnalysisAnswer(
            answered=True,
            outcome="ALLOW",
            answer=text,
            evidence_ids=tuple(evidence_id_of(node) for node in valid_nodes),
            measurements=measurements,
            limitations=limitations,
            uncalibrated_scores=scores,
            data={
                "verification": {"answered": True, "issue_codes": []},
                "measurement_values": [
                    measurement.model_dump(mode="json") for measurement in measurements
                ],
            },
        )


def _numeric_intent(intent: QueryIntent) -> bool:
    return intent.requested_measurement is not None or intent.task_family in {
        "MEASURE",
        "CHANGE_MEASURE",
    }


def _valid_nodes(graph: EvidenceGraph, report: VerificationReport) -> tuple[EvidenceNode, ...]:
    valid_ids = set(report.valid_evidence_ids)
    nodes = [
        node for node in graph.node_by_id.values() if evidence_id_of(node) in valid_ids
    ]
    nodes.sort(key=evidence_id_of)
    return tuple(nodes)


def _measurements(
    nodes: tuple[EvidenceNode, ...], display_precision: int
) -> tuple[AnswerMeasurement, ...]:
    result: list[AnswerMeasurement] = []
    for node in nodes:
        if not isinstance(node, MeasurementEvidence):
            continue
        result.append(
            AnswerMeasurement(
                evidence_id=node.evidence_id,
                source_evidence_id=node.source_evidence_id,
                measurement_type=node.measurement_type,
                value=node.value,
                unit=node.unit,
                display_value=f"{node.value:.{display_precision}f}",
                method=node.method,
                calculation_crs=node.calculation_crs,
                positive_pixel_count=node.positive_pixel_count,
                valid_pixel_count=node.valid_pixel_count,
            )
        )
    return tuple(result)


def _limitations(
    nodes: tuple[EvidenceNode, ...], report: VerificationReport
) -> tuple[str, ...]:
    values: list[str] = []
    for issue in (*report.issues, *report.warnings):
        values.append(_issue_text(issue))
    for node in nodes:
        values.extend(getattr(node, "warnings", ()))
        domain = getattr(node, "domain", None)
        if domain is not None and domain.status is not DomainStatus.IN_DOMAIN:
            values.extend(domain.reasons)
    return _dedupe(values)


def _issue_text(issue: VerificationIssue) -> str:
    return f"{issue.code}: {issue.message}"


def _dedupe(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return tuple(result)


def _uncalibrated_scores(nodes: tuple[EvidenceNode, ...]) -> tuple[UncalibratedScore, ...]:
    scores: list[UncalibratedScore] = []
    for node in nodes:
        if isinstance(node, (ChangeMaskEvidence, FloodMaskEvidence)):
            if node.raw_model_score is not None:
                scores.append(
                    UncalibratedScore(evidence_id=node.evidence_id, value=node.raw_model_score)
                )
        elif isinstance(node, VqaEvidence):
            if node.prediction.raw_score is not None:
                scores.append(
                    UncalibratedScore(evidence_id=node.evidence_id, value=node.prediction.raw_score)
                )
        elif isinstance(node, GroundingEvidence):
            for detection in node.detections:
                scores.append(
                    UncalibratedScore(evidence_id=node.evidence_id, value=detection.raw_score)
                )
    return tuple(scores)


def _measurement_text(intent: QueryIntent, measurements: tuple[AnswerMeasurement, ...]) -> str:
    first = measurements[0]
    subject = f" for {intent.target_semantic}" if intent.target_semantic else ""
    return (
        f"Verified {first.measurement_type}{subject}: "
        f"{first.display_value} {first.unit}. "
        f"The full stored value is available in the response data and was computed by "
        f"{first.method} using {first.calculation_crs}."
    )


def _caption_text(nodes: tuple[EvidenceNode, ...]) -> str:
    captions = [node for node in nodes if isinstance(node, ChangeCaptionEvidence)]
    if not captions:
        return "The verified evidence contains no change description text."
    return (
        "Verified change description (text-only, not a measurement source): "
        f"{captions[0].caption}"
    )


def _vqa_text(nodes: tuple[EvidenceNode, ...], scores: tuple[UncalibratedScore, ...]) -> str:
    vqas = [node for node in nodes if isinstance(node, VqaEvidence)]
    if not vqas:
        return "The verified evidence contains no VQA answer text."
    suffix = ""
    if scores:
        suffix = " The associated value is an uncalibrated raw model score, not calibrated confidence."
    return f"Verified model answer: {vqas[0].prediction.answer}.{suffix}"


def _grounding_text(nodes: tuple[EvidenceNode, ...]) -> str:
    groundings = [node for node in nodes if isinstance(node, GroundingEvidence)]
    if not groundings:
        return "The verified evidence contains no grounding detections."
    count = sum(len(node.detections) for node in groundings)
    return f"Verified grounding evidence contains {count} detection record(s)."


def _localization_text(nodes: tuple[EvidenceNode, ...]) -> str:
    masks = [node for node in nodes if isinstance(node, (ChangeMaskEvidence, FloodMaskEvidence))]
    if not masks:
        return "The verified evidence contains no localization mask."
    return f"Verified localization evidence is available from {len(masks)} mask record(s)."


__all__ = [
    "AnalysisAnswer",
    "AnswerComposer",
    "AnswerMeasurement",
    "UncalibratedScore",
]
