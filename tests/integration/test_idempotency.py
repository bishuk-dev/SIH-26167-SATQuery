"""Task 17 — Idempotency-Key handling for query submission.

Same key + same body replays the original analysis/job IDs; same key with a
different body is a 409 IDEMPOTENCY_CONFLICT. Only hashed keys and body
hashes are persisted. Idempotency applies to submission only; other routes
(including the dry-run plan route) are never guarded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app


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


def _client_without_started_worker(tmp_path: Path) -> TestClient:
    application = create_app(data_root=tmp_path / "data")
    application.state.job_runner.start = lambda: None
    application.state.job_runner.stop = lambda timeout_seconds=5.0: None
    return TestClient(application)


def _submission_body(first: str, second: str, *, threshold: float) -> dict[str, object]:
    return {
        "query": "calculate the area of SAR change between T1 and T2",
        "observation_ids": [first, second],
        "parameters": {
            "radiometric_domain": "backscatter_db",
            "polarizations": ["VV"],
            "threshold": threshold,
        },
    }


def _setup_pair(tmp_path: Path) -> tuple[TestClient, str, str]:
    client = _client_without_started_worker(tmp_path)
    first = _upload(
        client,
        _sar_raster_bytes(tmp_path, "first.tif", value=1, acquisition_time="2026-01-01T00:00:00+00:00"),
        "first.tif",
    )
    second = _upload(
        client,
        _sar_raster_bytes(tmp_path, "second.tif", value=5, acquisition_time="2026-01-02T00:00:00+00:00"),
        "second.tif",
    )
    return client, first, second


def test_same_key_and_body_replays_original_ids(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        headers = {"Idempotency-Key": "idem-key-0123456789abcdef"}
        original = client.post("/api/v1/query", json=body, headers=headers)
        assert original.status_code == 202, original.text
        replay = client.post("/api/v1/query", json=body, headers=headers)
        assert replay.status_code == 200, replay.text
        assert replay.json()["analysis_id"] == original.json()["analysis_id"]
        assert replay.json()["job_id"] == original.json()["job_id"]
        assert replay.json()["plan_hash"] == original.json()["plan_hash"]
        assert replay.json()["registry_hash"] == original.json()["registry_hash"]
    finally:
        client.close()


def test_same_key_with_different_body_conflicts(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        headers = {"Idempotency-Key": "idem-key-0123456789abcdef"}
        original = client.post(
            "/api/v1/query", json=_submission_body(first, second, threshold=2.0), headers=headers
        )
        assert original.status_code == 202, original.text
        conflict = client.post(
            "/api/v1/query", json=_submission_body(first, second, threshold=3.5), headers=headers
        )
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    finally:
        client.close()


def test_absent_key_creates_independent_submissions(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        first_response = client.post("/api/v1/query", json=body)
        second_response = client.post("/api/v1/query", json=body)
        assert first_response.status_code == 202
        assert second_response.status_code == 202
        assert first_response.json()["analysis_id"] != second_response.json()["analysis_id"]
    finally:
        client.close()


def test_different_keys_with_same_body_get_distinct_submissions(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        one = client.post("/api/v1/query", json=body, headers={"Idempotency-Key": "idem-key-a-0123456"})
        two = client.post("/api/v1/query", json=body, headers={"Idempotency-Key": "idem-key-b-0123456"})
        assert one.status_code == 202
        assert two.status_code == 202
        assert one.json()["analysis_id"] != two.json()["analysis_id"]
    finally:
        client.close()


def test_malformed_idempotency_key_is_rejected(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        response = client.post("/api/v1/query", json=body, headers={"Idempotency-Key": "short"})
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "INVALID_IDEMPOTENCY_KEY"
    finally:
        client.close()


def test_idempotency_table_stores_only_hashes(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        headers = {"Idempotency-Key": "idem-key-0123456789abcdef"}
        submitted = client.post("/api/v1/query", json=body, headers=headers)
        assert submitted.status_code == 202, submitted.text
    finally:
        client.close()

    application = _client_without_started_worker(tmp_path).app
    with application.state.observation_repository._db.transaction() as connection:
        rows = list(connection.execute("SELECT key_hash, request_hash FROM idempotency_keys"))
    assert len(rows) == 1
    assert rows[0]["key_hash"] != "idem-key-0123456789abcdef"
    assert len(rows[0]["key_hash"]) == 64
    assert len(rows[0]["request_hash"]) == 64


def test_plan_route_is_never_idempotency_guarded(tmp_path: Path) -> None:
    client, first, second = _setup_pair(tmp_path)
    try:
        body = _submission_body(first, second, threshold=2.0)
        headers = {"Idempotency-Key": "idem-key-0123456789abcdef"}
        first_plan = client.post("/api/v1/query/plan", json=body, headers=headers)
        assert first_plan.status_code == 200, first_plan.text
        # same key, different body: the dry-run plan route must not conflict
        other_plan = client.post(
            "/api/v1/query/plan", json=_submission_body(first, second, threshold=3.5), headers=headers
        )
        assert other_plan.status_code == 200, other_plan.text
    finally:
        client.close()
