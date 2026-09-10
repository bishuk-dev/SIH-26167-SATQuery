"""Integration tests for analysis report JSON/HTML endpoints."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from satquery.persistence import JobStatus


def _sar_raster_bytes(
    tmp_path: Path,
    name: str,
    *,
    value: int,
    acquisition_time: str,
) -> bytes:
    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=Affine(10, 0, 0, 0, -10, 40),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((1, 4, 4), value, dtype=np.float32))
        dataset.set_band_description(1, "VV")
        dataset.update_tags(
            MODALITY="sar",
            SENSOR_NAME="TestSAR",
            POLARIZATIONS="VV",
            ACQUISITION_TIME=acquisition_time,
        )
        dataset.update_tags(1, radiometric_domain="backscatter_db", calibration="sigma0")
    return path.read_bytes()


def _upload(client: TestClient, content: bytes, name: str) -> str:
    response = client.post(
        "/api/v1/observations",
        files={"file": (name, content, "image/tiff")},
    )
    assert response.status_code == 201, response.text
    return response.json()["observation_id"]


def _submit(client: TestClient, first: str, second: str, query: str | None = None) -> dict:
    response = client.post(
        "/api/v1/query",
        json={
            "query": query
            or "calculate the area of SAR change between T1 and T2",
            "observation_ids": [first, second],
            "parameters": {
                "radiometric_domain": "backscatter_db",
                "polarizations": ["VV"],
                "threshold": 2.0,
            },
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def _submit_pair(
    client: TestClient, tmp_path: Path, stem: str, query: str | None = None
) -> tuple[dict, str, str]:
    first = _upload(
        client,
        _sar_raster_bytes(
            tmp_path,
            f"{stem}-first.tif",
            value=1,
            acquisition_time="2026-01-01T00:00:00+00:00",
        ),
        f"{stem}-first.tif",
    )
    second = _upload(
        client,
        _sar_raster_bytes(
            tmp_path,
            f"{stem}-second.tif",
            value=5,
            acquisition_time="2026-01-02T00:00:00+00:00",
        ),
        f"{stem}-second.tif",
    )
    return _submit(client, first, second, query=query), first, second


def _wait_terminal(repository, job_id: str) -> None:
    for _ in range(200):
        job = repository.get_job(job_id)
        if job is not None and job.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        }:
            return
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal state")


def test_report_json_contains_sections_and_exact_provenance(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    with TestClient(application) as client:
        submitted, _first, _second = _submit_pair(client, tmp_path, "report")
        repository = application.state.observation_repository
        _wait_terminal(repository, submitted["job_id"])
        analysis_id = submitted["analysis_id"]
        response = client.get(f"/api/v1/reports/{analysis_id}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["report_type"] == "analysis_report"
    assert body["schema_version"] == 1
    assert body["analysis_id"] == analysis_id
    assert body["status"] == "SUCCEEDED"
    assert "calculate the area of SAR change" in body["query"]
    assert body["intent"]["task_family"] == "CHANGE_MEASURE"
    assert body["inputs"]["observation_ids"]
    assert body["workflow"]["plan_hash"] == submitted["plan_hash"]
    assert body["workflow"]["registry_hash"] == submitted["registry_hash"]
    assert body["verification"] is not None
    assert body["verification"]["answered"] is True
    assert body["answer"] is not None
    assert body["answer"]["answered"] is True
    assert body["evidence"]
    for event in body["trace"]:
        assert set(event) == {"job_id", "sequence", "event_type", "created_at"}
    for artifact in body["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["artifact_id"].startswith("artifact_")
    assert str(tmp_path) not in response.text
    assert "path" not in body["inputs"]


def test_report_html_sets_content_type_and_escapes_query(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    with TestClient(application) as client:
        submitted, _first, _second = _submit_pair(client, tmp_path, "html")
        repository = application.state.observation_repository
        _wait_terminal(repository, submitted["job_id"])
        json_report = client.get(f"/api/v1/reports/{submitted['analysis_id']}")
        html_report = client.get(f"/api/v1/reports/{submitted['analysis_id']}/html")
    assert json_report.status_code == 200
    assert html_report.status_code == 200, html_report.text
    assert html_report.headers["content-type"].startswith("text/html")
    body = html_report.text
    assert body.startswith("<!DOCTYPE html>")
    assert "SatQuery Analysis Report" in body
    assert "Verification" in body
    assert "Measurements" in body
    assert "Artifacts" in body
    assert str(tmp_path) not in body


def test_report_html_escapes_injected_query_markup(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    with TestClient(application) as client:
        submitted, _first, _second = _submit_pair(
            client,
            tmp_path,
            "injected",
            query='<script>alert("x")</script> calculate the area of SAR change',
        )
        repository = application.state.observation_repository
        _wait_terminal(repository, submitted["job_id"])
        response = client.get(f"/api/v1/reports/{submitted['analysis_id']}/html")
    assert response.status_code == 200, response.text
    assert "&lt;script&gt;" in response.text
    assert "<script>" not in response.text


def test_pending_analysis_report_has_empty_dynamic_sections(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    application.state.job_runner.start = lambda: None
    application.state.job_runner.stop = lambda timeout_seconds=5.0: None
    client = TestClient(application)
    try:
        submitted, _first, _second = _submit_pair(client, tmp_path, "pending")
        response = client.get(f"/api/v1/reports/{submitted['analysis_id']}")
    finally:
        client.close()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "PENDING"
    assert body["answer"] is None
    assert body["verification"] is None
    assert body["evidence"] == []
    assert body["artifacts"] == []


def test_pdf_endpoint_is_disabled_with_single_envelope(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    application.state.job_runner.start = lambda: None
    application.state.job_runner.stop = lambda timeout_seconds=5.0: None
    client = TestClient(application)
    try:
        submitted, _first, _second = _submit_pair(client, tmp_path, "pdf")
        response = client.get(f"/api/v1/reports/{submitted['analysis_id']}/pdf")
    finally:
        client.close()
    assert response.status_code == 501, response.text
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "PDF_REPORT_DISABLED"


def test_report_endpoints_return_404_for_unknown_analysis(tmp_path: Path) -> None:
    application = create_app(data_root=tmp_path / "data")
    application.state.job_runner.start = lambda: None
    application.state.job_runner.stop = lambda timeout_seconds=5.0: None
    client = TestClient(application)
    try:
        analysis_id = "ana_" + "0" * 32
        for path in (
            f"/api/v1/reports/{analysis_id}",
            f"/api/v1/reports/{analysis_id}/html",
            f"/api/v1/reports/{analysis_id}/pdf",
        ):
            response = client.get(path)
            assert response.status_code == 404, (path, response.text)
            assert response.json()["error"]["code"] == "NOT_FOUND"
    finally:
        client.close()
