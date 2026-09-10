"""Deterministic JSON and HTML rendering of auditable analysis reports.

The report is a typed projection of persisted, immutable analysis records:
query, sanitized inputs, workflow, verified answer, evidence links,
measurements, warnings, verification, trace summary, and reproducibility
identifiers. Rendering never invents scientific claims, never includes local
filesystem paths, hidden prompts, or chain-of-thought, and HTML output escapes
every dynamic value with the standard library only.
"""

from __future__ import annotations

import html
import json
from typing import Any, Literal

from pydantic import Field

from satquery.artifacts import ArtifactNotFoundError, ArtifactStore
from satquery.ingestion.models import ContractModel
from satquery.persistence import AnalysisRecord, MetadataRepository
from satquery.persistence.repositories import decode_timestamp

_REPORT_SCHEMA_VERSION: Literal[1] = 1

_ARTIFACT_ID_PATTERN = r"^artifact_[0-9a-f]{32}$"
_EVIDENCE_ID_PATTERN = r"^evidence_[0-9a-f]{32}$"
_ANALYSIS_ID_PATTERN = r"^ana_[0-9a-f]{32}$"
_JOB_ID_PATTERN = r"^job_[0-9a-f]{32}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ReportArtifact(ContractModel):
    """Immutable published artifact reference with its verified hash."""

    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    evidence_id: str | None = Field(default=None, pattern=_EVIDENCE_ID_PATTERN)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class ReportEvidenceLink(ContractModel):
    """A path-free link to one persisted evidence record."""

    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    task: str = Field(min_length=1)
    artifact_ids: tuple[str, ...] = ()


class ReportTraceEvent(ContractModel):
    """One execution-trace event in the report's trace summary."""

    job_id: str = Field(pattern=_JOB_ID_PATTERN)
    sequence: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class ReportIssue(ContractModel):
    code: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence_id: str | None = Field(default=None, pattern=_EVIDENCE_ID_PATTERN)


class ReportVerification(ContractModel):
    answered: bool
    passed: bool
    valid_evidence_ids: tuple[str, ...] = ()
    invalid_evidence_ids: tuple[str, ...] = ()
    issues: tuple[ReportIssue, ...] = ()
    warnings: tuple[ReportIssue, ...] = ()


class ReportMeasurement(ContractModel):
    """A numeric claim copied verbatim from deterministic measurement evidence."""

    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    source_evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    measurement_type: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    display_value: str = Field(min_length=1)
    method: str = Field(min_length=1)
    calculation_crs: str = Field(min_length=1)
    positive_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(ge=0)


class ReportUncalibratedScore(ContractModel):
    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    value: float
    label: str = Field(min_length=1)


class ReportAnswer(ContractModel):
    answered: bool
    outcome: str = Field(min_length=1)
    answer: str
    evidence_ids: tuple[str, ...] = ()
    measurements: tuple[ReportMeasurement, ...] = ()
    limitations: tuple[str, ...] = ()
    uncalibrated_scores: tuple[ReportUncalibratedScore, ...] = ()


class ReportWorkflowStep(ContractModel):
    step_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    expected_evidence_type: str = Field(min_length=1)


class ReportInputs(ContractModel):
    observation_ids: tuple[str, ...] = ()
    pair_id: str | None = Field(default=None, pattern=r"^pair_[0-9a-f]{32}$")
    input_hashes: dict[str, str] = Field(default_factory=dict)


class ReportWorkflow(ContractModel):
    steps: tuple[ReportWorkflowStep, ...] = ()
    plan_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    registry_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class AnalysisReport(ContractModel):
    """The complete, deterministic report document for one analysis."""

    schema_version: Literal[1] = _REPORT_SCHEMA_VERSION
    report_type: Literal["analysis_report"] = "analysis_report"
    analysis_id: str = Field(pattern=_ANALYSIS_ID_PATTERN)
    status: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    query: str = ""
    query_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    intent: dict[str, Any] = Field(default_factory=dict)
    feasibility_outcome: str | None = None
    inputs: ReportInputs = Field(default_factory=ReportInputs)
    workflow: ReportWorkflow = Field(default_factory=ReportWorkflow)
    answer: ReportAnswer | None = None
    evidence: tuple[ReportEvidenceLink, ...] = ()
    verification: ReportVerification | None = None
    warnings: tuple[str, ...] = ()
    trace: tuple[ReportTraceEvent, ...] = ()
    artifacts: tuple[ReportArtifact, ...] = ()
    rerun_of: str | None = Field(default=None, pattern=_ANALYSIS_ID_PATTERN)


def _iso(value: Any) -> str:
    return decode_timestamp(value).isoformat() if isinstance(value, str) else value.isoformat()


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}


def _intent_projection(intent: Any) -> dict[str, Any]:
    if not isinstance(intent, dict):
        return {}
    allowed = (
        "task_family",
        "target_semantic",
        "requested_measurement",
        "temporal_direction",
        "spatial_request",
        "matched_rule",
        "ambiguities",
    )
    return {key: intent[key] for key in allowed if key in intent}


def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    feasibility = payload.get("feasibility")
    if isinstance(feasibility, dict):
        for issue in feasibility.get("warnings", ()):
            if isinstance(issue, dict) and issue.get("message"):
                warnings.append(f"{issue.get('code', 'WARNING')}: {issue['message']}")
    for reason in payload.get("reasons", ()):
        if isinstance(reason, str) and reason:
            warnings.append(reason)
    verification = payload.get("verification")
    if isinstance(verification, dict):
        for issue in verification.get("warnings", ()):
            if isinstance(issue, dict) and issue.get("message"):
                warnings.append(f"{issue.get('code', 'WARNING')}: {issue['message']}")
    deduped: list[str] = []
    for warning in warnings:
        if warning not in deduped:
            deduped.append(warning)
    return tuple(deduped)


def _verification_from_payload(payload: dict[str, Any]) -> ReportVerification | None:
    verification = payload.get("verification")
    if not isinstance(verification, dict):
        return None
    issues = tuple(
        ReportIssue.model_validate(issue)
        for issue in verification.get("issues", ())
        if isinstance(issue, dict)
    )
    warnings = tuple(
        ReportIssue.model_validate(issue)
        for issue in verification.get("warnings", ())
        if isinstance(issue, dict)
    )
    valid = tuple(
        item["evidence_id"]
        for item in verification.get("evidence", ())
        if isinstance(item, dict) and item.get("status") == "valid"
    )
    invalid = tuple(
        item["evidence_id"]
        for item in verification.get("evidence", ())
        if isinstance(item, dict) and item.get("status") == "invalid"
    )
    answered = bool(verification.get("answered", False))
    return ReportVerification(
        answered=answered,
        passed=answered and not issues,
        valid_evidence_ids=valid,
        invalid_evidence_ids=invalid,
        issues=issues,
        warnings=warnings,
    )


def _answer_from_payload(payload: dict[str, Any]) -> ReportAnswer | None:
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        return None
    return ReportAnswer(
        answered=bool(answer.get("answered", False)),
        outcome=str(answer.get("outcome", "")),
        answer=str(answer.get("answer", "")),
        evidence_ids=tuple(
            item for item in answer.get("evidence_ids", ()) if isinstance(item, str)
        ),
        measurements=tuple(
            ReportMeasurement.model_validate(item)
            for item in answer.get("measurements", ())
            if isinstance(item, dict)
        ),
        limitations=tuple(
            item for item in answer.get("limitations", ()) if isinstance(item, str)
        ),
        uncalibrated_scores=tuple(
            ReportUncalibratedScore.model_validate(item)
            for item in answer.get("uncalibrated_scores", ())
            if isinstance(item, dict)
        ),
    )


class ReportBuilder:
    """Build one :class:`AnalysisReport` from persisted immutable records."""

    def __init__(
        self,
        repository: MetadataRepository,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._repository = repository
        self._artifact_store = artifact_store

    def build(self, analysis: AnalysisRecord) -> AnalysisReport:
        payload = analysis.payload
        if not isinstance(payload, dict):
            payload = {}
        artifacts = self._artifacts(analysis.analysis_id)
        evidence = self._evidence(analysis.analysis_id, artifacts)
        return AnalysisReport(
            analysis_id=analysis.analysis_id,
            status=analysis.status.value,
            created_at=_iso(analysis.created_at),
            updated_at=_iso(analysis.updated_at),
            query=str(payload.get("query", "")),
            query_sha256=payload.get("query_sha256")
            if isinstance(payload.get("query_sha256"), str)
            else None,
            intent=_intent_projection(payload.get("intent")),
            feasibility_outcome=(
                str(payload["feasibility"]["outcome"])
                if isinstance(payload.get("feasibility"), dict)
                and isinstance(payload["feasibility"].get("outcome"), str)
                else None
            ),
            inputs=ReportInputs(
                observation_ids=tuple(
                    item
                    for item in payload.get("observation_ids", ())
                    if isinstance(item, str)
                ),
                pair_id=payload.get("pair_id")
                if isinstance(payload.get("pair_id"), str)
                else None,
                input_hashes=_string_map(payload.get("input_hashes")),
            ),
            workflow=ReportWorkflow(
                steps=tuple(
                    ReportWorkflowStep(
                        step_id=str(step.get("step_id", "")),
                        tool_id=str(step.get("tool_id", "")),
                        depends_on=tuple(
                            item for item in step.get("depends_on", ()) if isinstance(item, str)
                        ),
                        expected_evidence_type=str(step.get("expected_evidence_type", "")),
                    )
                    for step in payload.get("plan", {}).get("steps", ())
                    if isinstance(step, dict)
                )
                if isinstance(payload.get("plan"), dict)
                else (),
                plan_hash=payload.get("plan_hash")
                if isinstance(payload.get("plan_hash"), str)
                else None,
                registry_hash=payload.get("registry_hash")
                if isinstance(payload.get("registry_hash"), str)
                else None,
            ),
            answer=_answer_from_payload(payload),
            evidence=evidence,
            verification=_verification_from_payload(payload),
            warnings=_warnings_from_payload(payload),
            trace=self._trace(analysis.analysis_id),
            artifacts=artifacts,
            rerun_of=payload.get("rerun_of")
            if isinstance(payload.get("rerun_of"), str)
            else None,
        )

    def _evidence(
        self, analysis_id: str, artifacts: tuple[ReportArtifact, ...]
    ) -> tuple[ReportEvidenceLink, ...]:
        artifact_ids_by_evidence: dict[str, list[str]] = {}
        for artifact in artifacts:
            if artifact.evidence_id is not None:
                artifact_ids_by_evidence.setdefault(artifact.evidence_id, []).append(
                    artifact.artifact_id
                )
        with self._repository._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT evidence_id, payload_json FROM evidence"
                " WHERE analysis_id = ? ORDER BY created_at, evidence_id",
                (analysis_id,),
            ).fetchall()
        links: list[ReportEvidenceLink] = []
        for row in rows:
            try:
                evidence_payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                evidence_payload = {}
            task = (
                evidence_payload.get("task")
                if isinstance(evidence_payload, dict) and evidence_payload.get("task")
                else "unknown"
            )
            links.append(
                ReportEvidenceLink(
                    evidence_id=row["evidence_id"],
                    task=str(task),
                    artifact_ids=tuple(artifact_ids_by_evidence.get(row["evidence_id"], ())),
                )
            )
        return tuple(links)

    def _trace(self, analysis_id: str) -> tuple[ReportTraceEvent, ...]:
        with self._repository._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT e.job_id, e.sequence, e.event_type, e.created_at"
                " FROM execution_events e JOIN jobs j ON j.job_id = e.job_id"
                " WHERE j.analysis_id = ?"
                " ORDER BY e.created_at, e.job_id, e.sequence",
                (analysis_id,),
            ).fetchall()
        return tuple(
            ReportTraceEvent(
                job_id=row["job_id"],
                sequence=row["sequence"],
                event_type=row["event_type"],
                created_at=decode_timestamp(row["created_at"]).isoformat(),
            )
            for row in rows
        )

    def _artifacts(self, analysis_id: str) -> tuple[ReportArtifact, ...]:
        if self._artifact_store is None:
            return ()
        items: list[ReportArtifact] = []
        for child in sorted(self._artifact_store.artifacts_root.iterdir()):
            if not child.is_dir():
                continue
            try:
                record, _path = self._artifact_store.resolve(child.name)
            except ArtifactNotFoundError:
                continue
            if record.metadata.analysis_id != analysis_id:
                continue
            items.append(
                ReportArtifact(
                    artifact_id=record.artifact_id,
                    evidence_id=record.metadata.evidence_id,
                    media_type=record.media_type,
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                )
            )
        return tuple(items)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _definition_list(entries: list[tuple[str, str]]) -> list[str]:
    lines = ["<dl>"]
    for name, value in entries:
        lines.append(f"<dt>{_esc(name)}</dt><dd>{_esc(value)}</dd>")
    lines.append("</dl>")
    return lines


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["<table>", "<thead><tr>"]
    lines.extend(f"<th>{_esc(header)}</th>" for header in headers)
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>")
        lines.extend(f"<td>{_esc(cell)}</td>" for cell in row)
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return lines


def render_html(report: AnalysisReport) -> str:
    """Render the report as deterministic, fully escaped HTML."""

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>SatQuery Analysis Report {_esc(report.analysis_id)}</title>",
        "</head>",
        "<body>",
        "<h1>SatQuery Analysis Report</h1>",
        '<section id="summary">',
        "<h2>Summary</h2>",
        *_definition_list(
            [
                ("Analysis ID", report.analysis_id),
                ("Status", report.status),
                ("Created", report.created_at),
                ("Updated", report.updated_at),
            ]
        ),
        "</section>",
        '<section id="query">',
        "<h2>Query</h2>",
        f"<p>{_esc(report.query)}</p>",
        *_definition_list(
            [("Query SHA-256", report.query_sha256 or "not recorded")],
        ),
        "</section>",
        '<section id="intent">',
        "<h2>Interpreted Intent</h2>",
    ]
    if report.intent:
        parts.extend(
            _definition_list(
                [(key, report.intent[key]) for key in sorted(report.intent)]
            )
        )
    else:
        parts.append("<p>No intent recorded.</p>")
    parts.extend(
        [
            "</section>",
            '<section id="inputs">',
            "<h2>Inputs</h2>",
        ]
    )
    input_entries: list[tuple[str, str]] = []
    if report.inputs.pair_id is not None:
        input_entries.append(("Pair ID", report.inputs.pair_id))
    for observation_id in report.inputs.observation_ids:
        input_entries.append(("Observation ID", observation_id))
    for source_id, source_hash in sorted(report.inputs.input_hashes.items()):
        input_entries.append((f"Input SHA-256 ({source_id})", source_hash))
    parts.extend(_definition_list(input_entries) if input_entries else ["<p>No inputs recorded.</p>"])
    parts.extend(['</section>', '<section id="workflow">', "<h2>Workflow</h2>"])
    workflow_entries = [
        ("Plan SHA-256", report.workflow.plan_hash or "not recorded"),
        ("Registry SHA-256", report.workflow.registry_hash or "not recorded"),
    ]
    if report.feasibility_outcome:
        workflow_entries.append(("Feasibility outcome", report.feasibility_outcome))
    parts.extend(_definition_list(workflow_entries))
    if report.workflow.steps:
        parts.extend(
            _table(
                ["Step ID", "Tool ID", "Depends on", "Expected evidence type"],
                [
                    [
                        step.step_id,
                        step.tool_id,
                        ", ".join(step.depends_on),
                        step.expected_evidence_type,
                    ]
                    for step in report.workflow.steps
                ],
            )
        )
    else:
        parts.append("<p>No workflow steps recorded.</p>")
    parts.extend(["</section>", '<section id="answer">', "<h2>Answer</h2>"])
    if report.answer is None:
        parts.append("<p>No answer: the analysis has not completed verification.</p>")
    else:
        parts.extend(
            _definition_list(
                [
                    ("Answered", str(report.answer.answered)),
                    ("Outcome", report.answer.outcome),
                ]
            )
        )
        parts.append(f"<p>{_esc(report.answer.answer)}</p>")
        if report.answer.measurements:
            parts.append("<h3>Measurements</h3>")
            parts.extend(
                _table(
                    [
                        "Evidence ID",
                        "Type",
                        "Value",
                        "Unit",
                        "Display value",
                        "Method",
                        "Calculation CRS",
                        "Positive pixels",
                        "Valid pixels",
                    ],
                    [
                        [
                            item.evidence_id,
                            item.measurement_type,
                            repr(item.value),
                            item.unit,
                            item.display_value,
                            item.method,
                            item.calculation_crs,
                            str(item.positive_pixel_count),
                            str(item.valid_pixel_count),
                        ]
                        for item in report.answer.measurements
                    ],
                )
            )
        if report.answer.uncalibrated_scores:
            parts.append("<h3>Uncalibrated model scores</h3>")
            parts.extend(
                _table(
                    ["Evidence ID", "Value", "Label"],
                    [
                        [item.evidence_id, repr(item.value), item.label]
                        for item in report.answer.uncalibrated_scores
                    ],
                )
            )
        if report.answer.limitations:
            parts.append("<h3>Limitations</h3><ul>")
            parts.extend(f"<li>{_esc(item)}</li>" for item in report.answer.limitations)
            parts.append("</ul>")
    parts.extend(
        [
            "</section>",
            '<section id="evidence">',
            "<h2>Evidence</h2>",
        ]
    )
    if report.evidence:
        parts.extend(
            _table(
                ["Evidence ID", "Task", "Artifact IDs"],
                [
                    [
                        item.evidence_id,
                        item.task,
                        ", ".join(item.artifact_ids) if item.artifact_ids else "-",
                    ]
                    for item in report.evidence
                ],
            )
        )
    else:
        parts.append("<p>No evidence recorded.</p>")
    parts.extend(
        [
            "</section>",
            '<section id="verification">',
            "<h2>Verification</h2>",
        ]
    )
    if report.verification is None:
        parts.append("<p>Verification has not completed.</p>")
    else:
        parts.extend(
            _definition_list(
                [
                    ("Answered", str(report.verification.answered)),
                    ("Passed", str(report.verification.passed)),
                    (
                        "Valid evidence",
                        ", ".join(report.verification.valid_evidence_ids) or "-",
                    ),
                    (
                        "Invalid evidence",
                        ", ".join(report.verification.invalid_evidence_ids) or "-",
                    ),
                ]
            )
        )
        if report.verification.issues or report.verification.warnings:
            parts.extend(
                _table(
                    ["Kind", "Severity", "Code", "Message", "Evidence ID"],
                    [
                        [
                            "issue" if item in report.verification.issues else "warning",
                            item.severity,
                            item.code,
                            item.message,
                            item.evidence_id or "-",
                        ]
                        for item in (*report.verification.issues, *report.verification.warnings)
                    ],
                )
            )
    parts.extend(
        [
            "</section>",
            '<section id="trace">',
            "<h2>Execution Trace</h2>",
        ]
    )
    if report.trace:
        parts.extend(
            _table(
                ["Job ID", "Sequence", "Event", "Created"],
                [
                    [item.job_id, str(item.sequence), item.event_type, item.created_at]
                    for item in report.trace
                ],
            )
        )
    else:
        parts.append("<p>No execution events recorded.</p>")
    parts.extend(
        [
            "</section>",
            '<section id="artifacts">',
            "<h2>Artifacts</h2>",
        ]
    )
    if report.artifacts:
        parts.extend(
            _table(
                ["Artifact ID", "Evidence ID", "Media type", "SHA-256", "Size (bytes)"],
                [
                    [
                        item.artifact_id,
                        item.evidence_id or "-",
                        item.media_type,
                        item.sha256,
                        str(item.size_bytes),
                    ]
                    for item in report.artifacts
                ],
            )
        )
    else:
        parts.append("<p>No published artifacts.</p>")
    parts.extend(["</section>"])
    if report.warnings:
        parts.extend(
            [
                '<section id="warnings">',
                "<h2>Warnings</h2>",
                "<ul>",
                *(f"<li>{_esc(item)}</li>" for item in report.warnings),
                "</ul>",
                "</section>",
            ]
        )
    if report.rerun_of is not None:
        parts.extend(
            [
                '<section id="rerun">',
                f"<p>Rerun of analysis {_esc(report.rerun_of)}.</p>",
                "</section>",
            ]
        )
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


__all__ = [
    "AnalysisReport",
    "ReportAnswer",
    "ReportArtifact",
    "ReportBuilder",
    "ReportEvidenceLink",
    "ReportInputs",
    "ReportIssue",
    "ReportMeasurement",
    "ReportTraceEvent",
    "ReportUncalibratedScore",
    "ReportVerification",
    "ReportWorkflow",
    "ReportWorkflowStep",
    "render_html",
]
