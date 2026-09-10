"""Deterministic verification for analysis evidence graphs.

Verification operates on typed evidence, artifact bytes, and deterministic GIS
recalculation. It never treats prose as proof for masks, measurements, or
numeric claims.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pydantic import Field

from satquery.agent.models import QueryIntent
from satquery.analytics.measurement import measure_mask_area
from satquery.evidence.graph import EvidenceGraph, EvidenceNode, evidence_id_of
from satquery.evidence.models import (
    AgreementEvidence,
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    FloodMaskEvidence,
    GroundingEvidence,
    MaskAsset,
    MeasurementEvidence,
    VqaEvidence,
)
from satquery.ingestion.models import ContractModel

IssueSeverity = Literal["ERROR", "WARNING"]
EvidenceStatus = Literal["valid", "invalid"]


class VerificationIssue(ContractModel):
    code: str = Field(min_length=1)
    severity: IssueSeverity
    message: str = Field(min_length=1)
    evidence_id: str | None = Field(default=None, pattern=r"^evidence_[0-9a-f]{32}$")


class EvidenceVerification(ContractModel):
    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: str = Field(min_length=1)
    status: EvidenceStatus
    issues: tuple[VerificationIssue, ...] = ()


class VerificationReport(ContractModel):
    answered: bool
    evidence: tuple[EvidenceVerification, ...]
    issues: tuple[VerificationIssue, ...] = ()
    warnings: tuple[VerificationIssue, ...] = ()

    @property
    def valid_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence if item.status == "valid")

    @property
    def invalid_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence if item.status == "invalid")

    @property
    def passed(self) -> bool:
        return self.answered and not self.issues


class AnalysisVerifier:
    """Verify graph evidence can answer the requested intent.

    Valid independent branches remain valid even when unrelated branches fail;
    final ``answered`` status depends only on whether the intent has at least
    one admissible, matching evidence path.
    """

    def verify(self, intent: QueryIntent, graph: EvidenceGraph) -> VerificationReport:
        issues_by_evidence: dict[str, list[VerificationIssue]] = {
            evidence_id: [] for evidence_id in graph.node_by_id
        }
        global_issues: list[VerificationIssue] = []

        for node in graph.node_by_id.values():
            node_id = evidence_id_of(node)
            issues_by_evidence[node_id].extend(
                _unknown_input_issues(node, graph.input_ids)
            )
            issues_by_evidence[node_id].extend(_artifact_issues(node))

        # Source/type checks and deterministic recalculation are evaluated after
        # base node checks so one invalid branch does not poison all others.
        for node in graph.node_by_id.values():
            if isinstance(node, MeasurementEvidence):
                issues_by_evidence[node.evidence_id].extend(
                    self._measurement_issues(node, graph, issues_by_evidence)
                )
            elif isinstance(node, AgreementEvidence):
                issues_by_evidence[node.evidence_id].extend(
                    self._agreement_issues(node, graph)
                )

        evidence_results = tuple(
            EvidenceVerification(
                evidence_id=node_id,
                task=graph.node_by_id[node_id].task,
                status=("invalid" if node_issues else "valid"),
                issues=tuple(node_issues),
            )
            for node_id, node_issues in issues_by_evidence.items()
        )

        global_issues.extend(issue for issues in issues_by_evidence.values() for issue in issues)

        if _numeric_intent(intent) and not _has_valid_measurement(graph, evidence_results):
            global_issues.append(
                VerificationIssue(
                    code="NUMERIC_CLAIM_WITHOUT_MEASUREMENT",
                    severity="ERROR",
                    message="numeric or area requests require valid MeasurementEvidence",
                )
            )

        answered = _intent_answered(intent, graph, evidence_results)
        if not answered:
            global_issues.append(
                VerificationIssue(
                    code="INTENT_NOT_ANSWERED",
                    severity="ERROR",
                    message="the evidence graph contains no valid evidence matching the requested intent",
                )
            )

        warnings = tuple(issue for issue in global_issues if issue.severity == "WARNING")
        errors = tuple(issue for issue in global_issues if issue.severity == "ERROR")
        return VerificationReport(
            answered=answered,
            evidence=evidence_results,
            issues=errors,
            warnings=warnings,
        )

    def _measurement_issues(
        self,
        measurement: MeasurementEvidence,
        graph: EvidenceGraph,
        issues_by_evidence: dict[str, list[VerificationIssue]],
    ) -> tuple[VerificationIssue, ...]:
        issues: list[VerificationIssue] = []
        source = graph.node_by_id.get(measurement.source_evidence_id)
        if source is None:
            return (
                VerificationIssue(
                    code="UNKNOWN_EVIDENCE_REFERENCE",
                    severity="ERROR",
                    message="measurement references a missing source evidence ID",
                    evidence_id=measurement.evidence_id,
                ),
            )
        if not isinstance(source, (ChangeMaskEvidence, FloodMaskEvidence)):
            return (
                VerificationIssue(
                    code="MEASUREMENT_SOURCE_NOT_MASK",
                    severity="ERROR",
                    message="measurement source must be spatial mask evidence, not caption or VQA evidence",
                    evidence_id=measurement.evidence_id,
                ),
            )
        if issues_by_evidence.get(source.evidence_id):
            issues.append(
                VerificationIssue(
                    code="SOURCE_EVIDENCE_INVALID",
                    severity="ERROR",
                    message="measurement source failed verification",
                    evidence_id=measurement.evidence_id,
                )
            )
            return tuple(issues)

        issues.extend(_deterministic_area_issues(measurement, source.mask))
        return tuple(issues)

    def _agreement_issues(
        self, agreement: AgreementEvidence, graph: EvidenceGraph
    ) -> tuple[VerificationIssue, ...]:
        issues: list[VerificationIssue] = []
        for source_id in (agreement.first_evidence_id, agreement.second_evidence_id):
            source = graph.node_by_id.get(source_id)
            if source is None:
                issues.append(
                    VerificationIssue(
                        code="UNKNOWN_EVIDENCE_REFERENCE",
                        severity="ERROR",
                        message="agreement references a missing source evidence ID",
                        evidence_id=agreement.evidence_id,
                    )
                )
            elif not isinstance(source, (ChangeMaskEvidence, FloodMaskEvidence)):
                issues.append(
                    VerificationIssue(
                        code="AGREEMENT_SOURCE_NOT_MASK",
                        severity="ERROR",
                        message="agreement source must be spatial mask evidence",
                        evidence_id=agreement.evidence_id,
                    )
                )
        return tuple(issues)


def _numeric_intent(intent: QueryIntent) -> bool:
    if intent.requested_measurement is not None:
        return True
    return intent.task_family in {"MEASURE", "CHANGE_MEASURE"}


def _has_valid_measurement(
    graph: EvidenceGraph, evidence_results: tuple[EvidenceVerification, ...]
) -> bool:
    valid_ids = {item.evidence_id for item in evidence_results if item.status == "valid"}
    return any(
        isinstance(node, MeasurementEvidence) and node.evidence_id in valid_ids
        for node in graph.node_by_id.values()
    )


def _intent_answered(
    intent: QueryIntent, graph: EvidenceGraph, evidence_results: tuple[EvidenceVerification, ...]
) -> bool:
    valid_ids = {item.evidence_id for item in evidence_results if item.status == "valid"}
    valid_nodes = [node for node in graph.node_by_id.values() if evidence_id_of(node) in valid_ids]

    if _numeric_intent(intent):
        return any(isinstance(node, MeasurementEvidence) for node in valid_nodes)
    if intent.task_family == "CHANGE_DESCRIPTION":
        return any(isinstance(node, ChangeCaptionEvidence) for node in valid_nodes)
    if intent.task_family == "CHANGE_LOCALIZE":
        return any(isinstance(node, (ChangeMaskEvidence, FloodMaskEvidence)) for node in valid_nodes)
    if intent.task_family in {"SINGLE_VQA", "CHANGE_VQA", "CROSS_MODAL_VQA"}:
        return any(isinstance(node, VqaEvidence) for node in valid_nodes)
    if intent.task_family == "GROUND_OBJECT":
        return any(isinstance(node, GroundingEvidence) for node in valid_nodes)
    if intent.task_family in {"METADATA_QUERY", "CAPABILITY_QUERY"}:
        return True
    return False


def _artifact_issues(node: EvidenceNode) -> tuple[VerificationIssue, ...]:
    if isinstance(node, (ChangeMaskEvidence, FloodMaskEvidence)):
        return _mask_artifact_issues(node.evidence_id, node.mask)
    return ()


def _mask_artifact_issues(evidence_id: str, mask: MaskAsset) -> tuple[VerificationIssue, ...]:
    path = Path(mask.path)
    if not path.is_file():
        return (
            VerificationIssue(
                code="ARTIFACT_HASH_MISSING",
                severity="ERROR",
                message="mask artifact bytes are missing and cannot be hash-verified",
                evidence_id=evidence_id,
            ),
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != mask.sha256:
        return (
            VerificationIssue(
                code="ARTIFACT_HASH_MISMATCH",
                severity="ERROR",
                message="mask artifact hash does not match evidence metadata",
                evidence_id=evidence_id,
            ),
        )
    return ()


def _deterministic_area_issues(
    measurement: MeasurementEvidence, mask: MaskAsset
) -> tuple[VerificationIssue, ...]:
    if measurement.measurement_type != "area":
        return ()
    if mask.crs is None or mask.transform is None:
        return (
            VerificationIssue(
                code="MEASUREMENT_GRID_MISSING",
                severity="ERROR",
                message="area measurement requires source mask CRS and transform",
                evidence_id=measurement.evidence_id,
            ),
        )
    try:
        with rasterio.open(mask.path) as dataset:
            if dataset.count != 1:
                raise ValueError("measurement masks must be single-band")
            values = dataset.read(1)
        if not np.isin(values, (0, 1)).all():
            raise ValueError("measurement mask must be binary 0/1")
        result = measure_mask_area(
            values.astype(bool),
            mask.transform,
            mask.crs,
            unit=measurement.unit,
        )
    except Exception as exc:
        return (
            VerificationIssue(
                code="MEASUREMENT_RECALCULATION_FAILED",
                severity="ERROR",
                message=f"area measurement could not be recalculated deterministically: {exc}",
                evidence_id=measurement.evidence_id,
            ),
        )

    mismatches: list[str] = []
    if not math.isclose(result.value, measurement.value, rel_tol=1e-9, abs_tol=1e-9):
        mismatches.append(
            f"value expected {result.value!r} {measurement.unit}, got {measurement.value!r}"
        )
    if result.positive_pixel_count != measurement.positive_pixel_count:
        mismatches.append(
            f"positive_pixel_count expected {result.positive_pixel_count}, got {measurement.positive_pixel_count}"
        )
    if result.valid_pixel_count != measurement.valid_pixel_count:
        mismatches.append(
            f"valid_pixel_count expected {result.valid_pixel_count}, got {measurement.valid_pixel_count}"
        )
    if result.method != measurement.method:
        mismatches.append(f"method expected {result.method!r}, got {measurement.method!r}")
    if mismatches:
        return (
            VerificationIssue(
                code="MEASUREMENT_VALUE_MISMATCH",
                severity="ERROR",
                message="; ".join(mismatches),
                evidence_id=measurement.evidence_id,
            ),
        )
    return ()


def _unknown_input_issues(node: EvidenceNode, input_ids: tuple[str, ...]) -> tuple[VerificationIssue, ...]:
    if not input_ids:
        return ()
    allowed = set(input_ids)
    referenced = _referenced_input_ids(node)
    unknown = tuple(sorted(item for item in referenced if item not in allowed))
    if not unknown:
        return ()
    return (
        VerificationIssue(
            code="UNKNOWN_INPUT_ID",
            severity="ERROR",
            message=f"evidence references input IDs not present in the analysis: {', '.join(unknown)}",
            evidence_id=evidence_id_of(node),
        ),
    )


def _referenced_input_ids(node: EvidenceNode) -> tuple[str, ...]:
    if isinstance(node, (ChangeMaskEvidence, ChangeCaptionEvidence)):
        ids = [node.temporal.t1_observation_id, node.temporal.t2_observation_id]
        if isinstance(node, ChangeMaskEvidence):
            ids.append(node.mask.source_grid_observation_id)
        return tuple(ids)
    if isinstance(node, FloodMaskEvidence):
        return (node.source_observation_id, node.mask.source_grid_observation_id)
    if isinstance(node, (VqaEvidence, GroundingEvidence)):
        return tuple(node.source_observations)
    return ()


__all__ = [
    "AnalysisVerifier",
    "EvidenceVerification",
    "VerificationIssue",
    "VerificationReport",
]
