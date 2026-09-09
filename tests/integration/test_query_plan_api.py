from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app


def _raster_bytes(tmp_path: Path, name: str, acquisition_time: str | None) -> bytes:
    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=Affine(10, 0, 0, 0, -10, 40),
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((1, 4, 4), dtype=np.uint8))
        dataset.update_tags(MODALITY="optical")
        if acquisition_time is not None:
            dataset.update_tags(ACQUISITION_TIME=acquisition_time)
    return path.read_bytes()


def _upload(client: TestClient, content: bytes, name: str) -> str:
    response = client.post(
        "/api/v1/observations",
        files={"file": (name, content, "image/tiff")},
    )
    assert response.status_code == 201, response.text
    return response.json()["observation_id"]


def test_plan_route_returns_structured_refusal_without_observations(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.post(
            "/api/v1/query/plan",
            json={"query": "show temporal change"},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["outcome"] == "REQUEST_INPUT"
    assert body["error"]["code"] == "MISSING_TEMPORAL_PAIR"
    assert "steps" not in response.text


def test_plan_route_loads_ids_server_side_and_never_persists_dry_run(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        observation_id = _upload(
            client,
            _raster_bytes(tmp_path, "one.tif", "2026-01-01T00:00:00+00:00"),
            "one.tif",
        )
        response = client.post(
            "/api/v1/query/plan",
            json={
                "query": "calculate area",
                "observation_ids": [observation_id],
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["outcome"] == "ABSTAIN"
        assert body["error"]["code"] == "WORKFLOW_UNSUPPORTED"
        assert client.get("/api/v1/pairs").json()["items"] == []


def test_plan_route_is_documented_with_stable_operation_id(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/query/plan"]["post"]
    assert operation["operationId"] == "plan_query_v1"
    assert operation["tags"] == ["Query"]
