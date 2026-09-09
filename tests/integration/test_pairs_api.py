from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app


def _raster_bytes(
    tmp_path: Path,
    name: str,
    *,
    value: int,
    modality: str = "optical",
    acquisition_time: str | None = "2026-01-01T00:00:00+00:00",
    origin_x: float = 0,
) -> bytes:
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
        transform=Affine(10, 0, origin_x, 0, -10, 40),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((1, 4, 4), value, dtype=np.uint8))
        dataset.update_tags(MODALITY=modality)
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


def test_exact_grid_temporal_pair_is_validated_and_survives_restart(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(
            client,
            _raster_bytes(
                tmp_path, "first.tif", value=1, acquisition_time="2026-01-01T00:00:00+00:00"
            ),
            "first.tif",
        )
        second = _upload(
            client,
            _raster_bytes(
                tmp_path, "second.tif", value=2, acquisition_time="2026-01-02T00:00:00+00:00"
            ),
            "second.tif",
        )
        request = {
            "observation_a": first,
            "observation_b": second,
            "pair_type": "temporal",
        }
        validated = client.post("/api/v1/pairs/validate", json=request)
        assert validated.status_code == 200, validated.text
        body = validated.json()
        assert body["outcome"] == "ALLOW"
        assert body["validation"]["result"]["status"] == "PASS"
        assert body["validation"]["grid"]["aligned"] is True

        created = client.post("/api/v1/pairs", json=request)
        assert created.status_code == 201, created.text
        pair = created.json()
        assert re.fullmatch(r"pair_[0-9a-f]{32}", pair["pair_id"])
        assert pair["validation"] == body["validation"]

    with TestClient(create_app(data_root=data_root)) as client:
        fetched = client.get(f"/api/v1/pairs/{pair['pair_id']}")
        assert fetched.status_code == 200
        assert fetched.json() == pair
        listed = client.get("/api/v1/pairs")
        assert listed.status_code == 200
        assert listed.json()["items"] == [pair]


def test_explicit_t1_allows_temporal_pair_without_acquisition_dates(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(
            client,
            _raster_bytes(tmp_path, "first.tif", value=1, acquisition_time=None),
            "first.tif",
        )
        second = _upload(
            client,
            _raster_bytes(tmp_path, "second.tif", value=2, acquisition_time=None),
            "second.tif",
        )
        request = {
            "observation_a": first,
            "observation_b": second,
            "pair_type": "temporal",
            "explicit_t1_id": first,
        }
        created = client.post("/api/v1/pairs", json=request)
        assert created.status_code == 201, created.text
        assert created.json()["explicit_t1_id"] == first
        assert created.json()["temporal_order"] == {
            "t1_observation_id": first,
            "t2_observation_id": second,
        }


def test_missing_temporal_order_requests_input(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(
            client,
            _raster_bytes(tmp_path, "first.tif", value=1, acquisition_time=None),
            "first.tif",
        )
        second = _upload(
            client,
            _raster_bytes(tmp_path, "second.tif", value=2, acquisition_time=None),
            "second.tif",
        )
        response = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": first,
                "observation_b": second,
                "pair_type": "temporal",
            },
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TEMPORAL_ORDER_UNKNOWN"
    assert response.json()["error"]["outcome"] == "REQUEST_INPUT"


def test_optical_sar_pair_classification_is_persisted(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        optical = _upload(
            client,
            _raster_bytes(tmp_path, "optical.tif", value=1, modality="optical"),
            "optical.tif",
        )
        sar = _upload(
            client,
            _raster_bytes(
                tmp_path,
                "sar.tif",
                value=2,
                modality="sar",
                acquisition_time="2026-01-02T00:00:00+00:00",
            ),
            "sar.tif",
        )
        response = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": optical,
                "observation_b": sar,
                "pair_type": "optical_sar",
            },
        )
    assert response.status_code == 201, response.text
    assert response.json()["validation"]["modality"]["pair_type"] == "optical_sar"


def test_explicit_t1_conflicting_with_acquisition_order_is_rejected(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(
            client,
            _raster_bytes(
                tmp_path, "first.tif", value=1, acquisition_time="2026-01-01T00:00:00+00:00"
            ),
            "first.tif",
        )
        second = _upload(
            client,
            _raster_bytes(
                tmp_path, "second.tif", value=2, acquisition_time="2026-01-02T00:00:00+00:00"
            ),
            "second.tif",
        )
        response = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": first,
                "observation_b": second,
                "pair_type": "temporal",
                "explicit_t1_id": second,
            },
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TEMPORAL_ORDER_CONFLICT"
    assert response.json()["error"]["outcome"] == "REJECT"


def test_temporal_pair_type_must_match_modalities(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        optical = _upload(
            client,
            _raster_bytes(tmp_path, "optical.tif", value=1, modality="optical"),
            "optical.tif",
        )
        sar = _upload(
            client,
            _raster_bytes(
                tmp_path,
                "sar.tif",
                value=2,
                modality="sar",
                acquisition_time="2026-01-02T00:00:00+00:00",
            ),
            "sar.tif",
        )
        response = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": optical,
                "observation_b": sar,
                "pair_type": "temporal_same_modality",
            },
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "PAIR_TYPE_MISMATCH"
    assert response.json()["error"]["outcome"] == "REJECT"


def test_duplicate_observation_and_no_overlap_are_rejected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        first = _upload(
            client,
            _raster_bytes(tmp_path, "first.tif", value=1),
            "first.tif",
        )
        duplicate = client.post(
            "/api/v1/pairs",
            json={
                "observation_a": first,
                "observation_b": first,
                "pair_type": "temporal",
            },
        )
        assert duplicate.status_code == 422
        assert duplicate.json()["error"]["code"] == "DUPLICATE_OBSERVATION"

        distant = _upload(
            client,
            _raster_bytes(
                tmp_path,
                "distant.tif",
                value=2,
                acquisition_time="2026-01-02T00:00:00+00:00",
                origin_x=1_000,
            ),
            "distant.tif",
        )
        no_overlap = client.post(
            "/api/v1/pairs/validate",
            json={
                "observation_a": first,
                "observation_b": distant,
                "pair_type": "temporal",
            },
        )
    assert no_overlap.status_code == 422
    assert no_overlap.json()["error"]["code"] == "NO_SPATIAL_OVERLAP"
    assert no_overlap.json()["error"]["outcome"] == "REJECT"
