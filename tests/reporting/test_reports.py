"""Unit tests for deterministic JSON/HTML analysis report rendering."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from satquery.artifacts import ArtifactStore
from satquery.execution.models import ArtifactMetadata
from satquery.persistence import (
    AnalysisRecord,
    AnalysisStatus,
    Database,
    ExecutionEvent,
    JobRecord,
    JobStatus,
    MetadataRepository,
)
from satquery.persistence.repositories import encode_timestamp
from satquery.reporting import AnalysisReport, ReportBuilder, render_html

_OBS_ID = "obs_" + "a" * 32
_EVIDENCE_ID = "evidence_" + "1" * 32
_MEASUREMENT_ID = "evidence_" + "2" * 32
_ARTIFACT_ID = "artifact_" + "3" * 32
_QUERY = 'calculate the area of water change between T1 and T2 <script>alert("x")</script>'


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _payload(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "query": _QUERY,
        "query_sha256": hashlib.sha256(_QUERY.encode("utf-8")).hexdigest(),
        "intent": {
            "task_family": "CHANGE_MEASURE",
            "target_semantic": "water",
            "requested_measurement": "area",
            "temporal_direction": "T1_TO_T2",
            "spatial_request": False,
            "matched_rule": "change_measure_area",
            "ambiguities": (),
        },
        "feasibility": {
            "checks": [],
            "outcome": "ALLOW_WITH_WARNING",
            "failures": [],
            "warnings": [
                {
                    "code": "INPUT_RESOLUTION_MIXED",
                    "severity": "WARNING",
                    "outcome": "ALLOW_WITH_WARNING",
                    "message": "Observation resolutions differ.",
                }
            ],
            "allowed_capability_ids": ["compute_mask_area_v1"],
        },
        "reasons": [],
        "observation_ids": [_OBS_ID],
        "pair_id": None,
        "input_hashes": {_OBS_ID: "f" * 64},
        "plan": {
            "planner_version": "test-1",
            "steps": [
                {
                    "step_id": "step_1",
                    "tool_id": "compute_mask_area_v1",
                    "input_bindings": {"mask": _EVIDENCE_ID},
                    "parameters": {"unit": "ha"},
                    "depends_on": [],
                    "expected_evidence_type": "MeasurementEvidence",
                }
            ],
        },
        "plan_hash": "b" * 64,
        "registry_hash": "c" * 64,
        "verification": {
            "answered": True,
            "evidence": [
                {
                    "evidence_id": _EVIDENCE_ID,
                    "task": "change_mask",
                    "status": "valid",
                    "issues": [],
                },
                {
                    "evidence_id": _MEASUREMENT_ID,
                    "task": "measure_area",
                    "status": "valid",
                    "issues": [],
                },
            ],
            "issues": [],
            "warnings": [
                {
                    "code": "AGREEMENT_DIAGNOSTIC_ONLY",
                    "severity": "WARNING",
                    "message": "Mask agreement is diagnostic, not accuracy.",
                    "evidence_id": None,
                }
            ],
        },
        "answer": {
            "answered": True,
            "outcome": "ALLOW_WITH_WARNING",
            "answer": "Water area changed by 1.50 ha & records <verified>.",
            "evidence_ids": [_MEASUREMENT_ID],
            "measurements": [
                {
                    "evidence_id": _MEASUREMENT_ID,
                    "source_evidence_id": _EVIDENCE_ID,
                    "measurement_type": "area",
                    "value": 15000.0,
                    "unit": "ha",
                    "display_value": "1.50 ha",
                    "method": "equal_area_reprojection_epsg_6933",
                    "calculation_crs": "EPSG:6933",
                    "positive_pixel_count": 15,
                    "valid_pixel_count": 16,
                }
            ],
            "limitations": ["Raw model scores are not calibrated confidence."],
            "uncalibrated_scores": [
                {"evidence_id": _EVIDENCE_ID, "value": 0.75, "label": "raw_model_score_uncalibrated"}
            ],
            "data": {},
        },
        "evidence_side_channel": {"path": str(tmp_path / "must-not-leak.tif")},
    }


def _insert_analysis(repository: MetadataRepository, tmp_path: Path, *, status: AnalysisStatus):
    now = _now()
    analysis_id = "ana_" + "4" * 32
    job_id = "job_" + "5" * 32
    completed = status is AnalysisStatus.SUCCEEDED
    payload = _payload(tmp_path)
    if not completed:
        # only a terminal, verified analysis carries verification/answer payloads
        payload.pop("verification", None)
        payload.pop("answer", None)
    repository.create_analysis(
        AnalysisRecord(
            analysis_id=analysis_id,
            status=status,
            created_at=now,
            updated_at=now,
            payload=payload,
        )
    )
    if not completed:
        return analysis_id
    repository.create_job(
        JobRecord(
            job_id=job_id,
            analysis_id=analysis_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            payload={"plan": {}},
        )
    )
    repository.transition_job(
        job_id,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        updated_at=now,
    )
    repository.transition_job(
        job_id,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        updated_at=now,
    )
    repository.append_event(
        ExecutionEvent(
            event_id="event_" + "6" * 32,
            job_id=job_id,
            sequence=0,
            event_type="JOB_SUBMITTED",
            created_at=now,
            payload={},
        )
    )
    repository.append_event(
        ExecutionEvent(
            event_id="event_" + "7" * 32,
            job_id=job_id,
            sequence=1,
            event_type="JOB_SUCCEEDED",
            created_at=now,
            payload={},
        )
    )
    with repository._db.transaction() as connection:
        for suffix, evidence_id, task in (
            ("a", _EVIDENCE_ID, "change_mask"),
            ("b", _MEASUREMENT_ID, "measure_area"),
        ):
            connection.execute(
                "INSERT INTO evidence(evidence_id, analysis_id, created_at, payload_json)"
                " VALUES (?, ?, ?, ?)",
                (
                    evidence_id,
                    analysis_id,
                    encode_timestamp(now),
                    json.dumps(
                        {
                            "evidence_id": evidence_id,
                            "task": task,
                            # a local path must never reach the report
                            "mask": {"path": str(tmp_path / f"{suffix}.tif")},
                        }
                    ),
                ),
            )
    return analysis_id


def _publish_artifact(store: ArtifactStore, analysis_id: str, evidence_id: str) -> str:
    staged = store.stage("tif")
    staged.path.write_bytes(b"mask-bytes")
    record = store.publish(
        staged,
        ArtifactMetadata(
            analysis_id=analysis_id,
            evidence_id=evidence_id,
            media_type="image/tiff",
        ),
    )
    return record.artifact_id


@pytest.fixture()
def environment(tmp_path: Path):
    database = Database(tmp_path / "satquery.db")
    database.migrate()
    repository = MetadataRepository(database)
    store = ArtifactStore(tmp_path / "artifacts")
    analysis_id = _insert_analysis(repository, tmp_path, status=AnalysisStatus.SUCCEEDED)
    artifact_id = _publish_artifact(store, analysis_id, _EVIDENCE_ID)
    builder = ReportBuilder(repository, artifact_store=store)
    record = repository.get_analysis(analysis_id)
    assert record is not None
    return builder, record, artifact_id, tmp_path


def test_report_contains_all_required_sections(environment) -> None:
    builder, record, artifact_id, _tmp_path = environment
    report = builder.build(record)
    assert isinstance(report, AnalysisReport)
    assert report.analysis_id == record.analysis_id
    assert report.status == "SUCCEEDED"
    assert "<script>" in report.query and "alert" in report.query
    assert report.intent["task_family"] == "CHANGE_MEASURE"
    assert report.inputs.observation_ids == (_OBS_ID,)
    assert report.inputs.input_hashes[_OBS_ID] == "f" * 64
    assert report.workflow.plan_hash == "b" * 64
    assert report.workflow.registry_hash == "c" * 64
    assert report.workflow.steps[0].tool_id == "compute_mask_area_v1"
    assert report.answer is not None
    assert report.answer.measurements[0].value == 15000.0
    assert report.answer.measurements[0].unit == "ha"
    assert report.verification is not None
    assert report.verification.passed is True
    assert set(report.verification.valid_evidence_ids) == {_EVIDENCE_ID, _MEASUREMENT_ID}
    evidence_ids = {item.evidence_id for item in report.evidence}
    assert evidence_ids == {_EVIDENCE_ID, _MEASUREMENT_ID}
    assert report.artifacts[0].artifact_id == artifact_id
    assert report.artifacts[0].sha256
    assert len(report.artifacts[0].sha256) == 64
    assert [(event.job_id, event.sequence, event.event_type) for event in report.trace] == [
        ("job_" + "5" * 32, 0, "JOB_SUBMITTED"),
        ("job_" + "5" * 32, 1, "JOB_SUCCEEDED"),
]
    assert "INPUT_RESOLUTION_MIXED: Observation resolutions differ." in report.warnings
    assert (
        "AGREEMENT_DIAGNOSTIC_ONLY: Mask agreement is diagnostic, not accuracy."
        in report.warnings
    )


def test_report_never_contains_local_paths_or_hidden_prompt_sections(environment) -> None:
    builder, record, _artifact_id, tmp_path = environment
    report = builder.build(record)
    rendered = json.dumps(report.model_dump(mode="json"))
    assert str(tmp_path) not in rendered
    assert ".tif" not in rendered
    lowered = rendered.casefold()
    for forbidden in ("system_prompt", "chain_of_thought", "hidden_prompt", "instructions"):
        assert forbidden not in lowered


def test_html_escapes_query_model_and_provenance_text(environment) -> None:
    builder, record, artifact_id, tmp_path = environment
    html = render_html(builder.build(record))
    assert html.startswith("<!DOCTYPE html>")
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "1.50 ha &amp; records &lt;verified&gt;." in html
    assert artifact_id in html
    assert f"SatQuery Analysis Report {record.analysis_id}" in html
    assert str(tmp_path) not in html


def test_html_rendering_is_deterministic(environment) -> None:
    builder, record, _artifact_id, _tmp_path = environment
    assert render_html(builder.build(record)) == render_html(builder.build(record))


def test_pending_analysis_builds_with_empty_dynamic_sections(tmp_path: Path) -> None:
    database = Database(tmp_path / "satquery.db")
    database.migrate()
    repository = MetadataRepository(database)
    analysis_id = _insert_analysis(repository, tmp_path, status=AnalysisStatus.PENDING)
    record = repository.get_analysis(analysis_id)
    assert record is not None
    report = ReportBuilder(repository).build(record)
    assert report.status == "PENDING"
    assert report.answer is None
    assert report.verification is None
    assert report.evidence == ()
    assert report.artifacts == ()
    assert report.trace == ()


def test_builder_rejects_analysis_with_malformed_payload(tmp_path: Path) -> None:
    database = Database(tmp_path / "satquery.db")
    database.migrate()
    repository = MetadataRepository(database)
    now = _now()
    analysis_id = "ana_" + "8" * 32
    repository.create_analysis(
        AnalysisRecord(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=now,
            updated_at=now,
            payload={"schema_version": 1, "not_a_report_input": True},
        )
    )
    record = repository.get_analysis(analysis_id)
    assert record is not None
    report = ReportBuilder(repository).build(record)
    assert report.query == ""
    assert report.intent == {}
    assert report.inputs.observation_ids == ()
