from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app


def _write_sar(path: Path, *, value: float, acquired: str) -> None:
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
        dataset.write(np.full((1, 4, 4), value, dtype="float32"))
        dataset.set_band_description(1, "VV")
        dataset.update_tags(
            MODALITY="sar",
            SENSOR_NAME="DemoSAR",
            POLARIZATIONS="VV",
            ACQUISITION_TIME=acquired,
        )
        dataset.update_tags(1, radiometric_domain="backscatter_db", calibration="sigma0")


def _upload(client: TestClient, path: Path) -> str:
    with path.open("rb") as handle:
        response = client.post(
            "/api/v1/observations",
            files={"file": (path.name, handle, "image/tiff")},
        )
    assert response.status_code == 201, response.text
    return response.json()["observation_id"]


def test_public_api_upload_analysis_evidence_report_and_restart(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-data"
    before = tmp_path / "before.tif"
    after = tmp_path / "after.tif"
    _write_sar(before, value=1, acquired="2026-01-01T00:00:00+00:00")
    _write_sar(after, value=5, acquired="2026-01-02T00:00:00+00:00")

    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(client, before)
        second = _upload(client, after)
        assert client.get(f"/api/v1/observations/{first}").status_code == 200

        pair = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": first,
                "observation_b": second,
                "pair_type": "temporal_same_modality",
            },
        )
        assert pair.status_code == 201, pair.text
        pair_id = pair.json()["pair_id"]

        submitted = client.post(
            "/api/v1/query",
            json={
                "query": "calculate the area of SAR change between T1 and T2",
                "pair_id": pair_id,
                "parameters": {
                    "radiometric_domain": "backscatter_db",
                    "polarizations": ["VV"],
                    "threshold": 2.0,
                },
            },
        )
        assert submitted.status_code == 202, submitted.text
        identifiers = submitted.json()

        for _ in range(100):
            job = client.get(f"/api/v1/jobs/{identifiers['job_id']}")
            assert job.status_code == 200, job.text
            if job.json()["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}:
                break
            time.sleep(0.05)
        assert job.json()["status"] == "SUCCEEDED"

        analysis_id = identifiers["analysis_id"]
        evidence = client.get(f"/api/v1/analyses/{analysis_id}/evidence")
        trace = client.get(f"/api/v1/analyses/{analysis_id}/trace")
        artifacts = client.get(f"/api/v1/analyses/{analysis_id}/artifacts")
        report = client.get(f"/api/v1/reports/{analysis_id}")
        html = client.get(f"/api/v1/reports/{analysis_id}/html")

        assert len(evidence.json()["items"]) == 2
        assert evidence.json()["edges"][0]["edge_type"] == "MEASURED_FROM"
        assert trace.json()["items"][-1]["event_type"] == "JOB_SUCCEEDED"
        assert len(artifacts.json()["items"]) == 1
        assert report.json()["answer"]["answered"] is True
        assert report.json()["answer"]["measurements"][0]["value"] == 0.16
        assert html.status_code == 200

    with TestClient(create_app(data_root=data_root)) as restarted:
        observations = restarted.get("/api/v1/observations").json()["items"]
        analyses = restarted.get("/api/v1/analyses").json()["items"]
        assert {item["observation_id"] for item in observations} == {first, second}
        assert any(item["analysis_id"] == analysis_id for item in analyses)
