from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from satquery.ingestion import FilesystemObservationStore
from satquery.ingestion.exceptions import ObservationNotFoundError
from satquery.persistence import PersistenceError


@pytest.fixture
def geotiff_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "fixture.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="uint16",
        crs="EPSG:32643",
        transform=Affine(10, 0, 500_000, 0, -10, 2_000_000),
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((1, 3, 4), dtype=np.uint16))
        dataset.set_band_description(1, "red")
        dataset.update_tags(MODALITY="multispectral", SENSOR_NAME="Test Sensor")
    return path.read_bytes()


def test_v1_upload_indexes_and_survives_restart(
    tmp_path: Path, geotiff_bytes: bytes
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        uploaded = client.post(
            "/api/v1/observations",
            files={"file": ("scene.tif", geotiff_bytes, "image/tiff")},
        )
        assert uploaded.status_code == 201
        body = uploaded.json()
        observation_id = body["observation_id"]
        source_hash = body["asset"]["sha256"]
        assert re.fullmatch(r"obs_[0-9a-f]{32}", observation_id)
        assert "path" not in body["asset"]

    with TestClient(create_app(data_root=data_root)) as client:
        listed = client.get("/api/v1/observations")
        assert listed.status_code == 200
        assert listed.json()["items"][0]["observation_id"] == observation_id
        assert listed.json()["items"][0]["asset"]["sha256"] == source_hash
        detail = client.get(f"/api/v1/observations/{observation_id}")
        assert detail.status_code == 200
        assert "path" not in detail.text


def test_v1_observation_subresources_pagination_and_immutable_delete(
    tmp_path: Path, geotiff_bytes: bytes
) -> None:
    data_root = tmp_path / "data"
    with TestClient(create_app(data_root=data_root)) as client:
        ids = []
        for name in ("one.tif", "two.tif"):
            response = client.post(
                "/api/v1/observations",
                files={"file": (name, geotiff_bytes, "image/tiff")},
            )
            assert response.status_code == 201
            ids.append(response.json()["observation_id"])

        first_page = client.get("/api/v1/observations?limit=1")
        assert first_page.status_code == 200
        page = first_page.json()
        assert len(page["items"]) == 1
        assert page["next_cursor"]

        second_page = client.get(
            "/api/v1/observations", params={"limit": 1, "cursor": page["next_cursor"]}
        )
        assert second_page.status_code == 200
        assert len(second_page.json()["items"]) == 1
        assert second_page.json()["items"][0]["observation_id"] != page["items"][0]["observation_id"]

        observation_id = ids[0]
        assert client.get(f"/api/v1/observations/{observation_id}/metadata").status_code == 200
        assets = client.get(f"/api/v1/observations/{observation_id}/assets")
        assert assets.status_code == 200
        assert {item["kind"] for item in assets.json()["items"]} == {
            "original",
            "visualization",
        }
        deleted = client.delete(f"/api/v1/observations/{observation_id}")
        assert deleted.status_code == 405
        assert deleted.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_v1_upload_errors_use_the_frozen_error_envelope(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.post(
            "/api/v1/observations",
            files={"file": ("not-a-raster.tif", b"not a raster", "image/tiff")},
        )

    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert set(response.json()["error"]) == {
        "code",
        "message",
        "outcome",
        "details",
        "request_id",
    }
    assert response.json()["error"]["code"] == "INVALID_RASTER"
    assert response.json()["error"]["outcome"] == "REJECT"
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


def test_v1_indexing_failure_keeps_registration_for_restart_recovery(
    tmp_path: Path,
    geotiff_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    application = create_app(data_root=data_root)
    repository = application.state.observation_repository

    def fail_indexing(*_args: object, **_kwargs: object) -> None:
        raise PersistenceError("simulated indexing failure")

    monkeypatch.setattr(repository, "create_observation", fail_indexing)
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/observations",
            files={"file": ("recoverable.tif", geotiff_bytes, "image/tiff")},
        )

    assert response.status_code == 500
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "ASSET_STORAGE_FAILED"
    assert response.json()["error"]["outcome"] == "REJECT"
    observation_dirs = list((data_root / "observations").iterdir())
    assert len(observation_dirs) == 1

    with TestClient(create_app(data_root=data_root)) as client:
        listed = client.get("/api/v1/observations")
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1


def test_v1_upload_does_not_remove_legacy_route(
    tmp_path: Path, geotiff_bytes: bytes
) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("legacy.tif", geotiff_bytes, "image/tiff")},
        )
    assert response.status_code == 201


def test_list_registrations_does_not_hide_malformed_registration(
    tmp_path: Path,
) -> None:
    store = FilesystemObservationStore(tmp_path / "data")
    malformed = store.observations_root / ("obs_" + "a" * 32)
    malformed.mkdir(parents=True)
    (malformed / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ObservationNotFoundError):
        store.list_registrations()
