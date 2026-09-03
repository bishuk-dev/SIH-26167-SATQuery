from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.errors import NotGeoreferencedWarning

from apps.api.app.main import create_app
from satquery.ingestion import RasterSafetyLimits
from satquery.ingestion.exceptions import AssetStorageError


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


def _client(data_root: Path, limits: RasterSafetyLimits | None = None) -> TestClient:
    return TestClient(create_app(data_root=data_root, limits=limits))


def _assert_quarantine_empty(data_root: Path) -> None:
    quarantine = data_root / "quarantine"
    assert quarantine.is_dir()
    assert list(quarantine.iterdir()) == []


def test_upload_registers_immutable_observation_in_controlled_storage(
    tmp_path: Path,
    geotiff_bytes: bytes,
) -> None:
    data_root = tmp_path / "data"
    with _client(data_root) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("../../client-scene.tiff", geotiff_bytes, "image/tiff")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "READY"
    assert re.fullmatch(r"obs_[0-9a-f]{32}", body["observation_id"])
    assert re.fullmatch(r"asset_[0-9a-f]{32}", body["asset"]["asset_id"])
    assert body["asset"]["original_name"] == "client-scene.tiff"
    assert body["asset"]["immutable"] is True
    assert "path" not in body["asset"]
    assert body["metadata"]["raster"]["driver"] == "GTiff"
    assert body["metadata"]["geo"]["crs"] == "EPSG:32643"
    assert body["metadata"]["sensor"]["sensor_name"] == "Test Sensor"
    assert body["validity"]["has_crs"] is True
    assert body["warnings"] == []

    observation_dir = data_root / "observations" / body["observation_id"]
    original = observation_dir / "original.tif"
    metadata = observation_dir / "metadata.json"
    assert original.read_bytes() == geotiff_bytes
    assert metadata.is_file()
    stored_state = json.loads(metadata.read_text(encoding="utf-8"))
    assert stored_state["source_asset"]["original_name"] == "client-scene.tiff"
    assert stored_state["source_asset"]["path"] == (
        f"observations/{body['observation_id']}/original.tif"
    )
    assert original.stat().st_mode & stat.S_IWRITE == 0
    _assert_quarantine_empty(data_root)

    # Restore write permission so pytest can remove its temporary directory on Windows.
    original.chmod(stat.S_IWRITE | stat.S_IREAD)


def test_upload_rejects_oversized_body_and_cleans_quarantine(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    limits = RasterSafetyLimits(max_file_size_bytes=8)
    with _client(data_root, limits) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("too-large.tif", b"0123456789", "image/tiff")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "RASTER_RESOURCE_LIMIT_EXCEEDED"
    _assert_quarantine_empty(data_root)
    assert list((data_root / "observations").iterdir()) == []


def test_upload_rejects_raster_over_header_limit(
    tmp_path: Path,
    geotiff_bytes: bytes,
) -> None:
    data_root = tmp_path / "data"
    limits = RasterSafetyLimits(max_width=3)
    with _client(data_root, limits) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("wide.tif", geotiff_bytes, "image/tiff")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "RASTER_RESOURCE_LIMIT_EXCEEDED"
    _assert_quarantine_empty(data_root)
    assert list((data_root / "observations").iterdir()) == []


def test_upload_rejects_corrupt_tiff_and_cleans_quarantine(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with _client(data_root) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("corrupt.tif", b"not a raster", "image/tiff")},
        )

    assert response.status_code == 422
    failure = response.json()["error"]
    assert failure["code"] == "INVALID_RASTER"
    assert failure["outcome"] == "REJECT"
    assert failure["affected_requirement"] == "R-INPUT-004"
    _assert_quarantine_empty(data_root)
    assert list((data_root / "observations").iterdir()) == []


def test_upload_rejects_non_tiff_driver(tmp_path: Path) -> None:
    png_path = tmp_path / "fixture.png"
    with pytest.warns(NotGeoreferencedWarning):
        with rasterio.open(
            png_path,
            "w",
            driver="PNG",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
        ) as dataset:
            dataset.write(np.zeros((1, 2, 2), dtype=np.uint8))

    data_root = tmp_path / "data"
    with _client(data_root) as client:
        with pytest.warns(NotGeoreferencedWarning):
            response = client.post(
                "/api/observations",
                files={"file": ("fixture.png", png_path.read_bytes(), "image/png")},
            )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_RASTER_DRIVER"
    _assert_quarantine_empty(data_root)


def test_upload_requires_file_field_with_structured_error(tmp_path: Path) -> None:
    with _client(tmp_path / "data") as client:
        response = client.post("/api/observations")

    assert response.status_code == 400
    failure = response.json()["error"]
    assert failure["code"] == "INVALID_UPLOAD"
    assert failure["severity"] == "ERROR"
    assert failure["outcome"] == "REJECT"


def test_storage_failure_returns_structured_error_and_cleans_quarantine(
    tmp_path: Path,
    geotiff_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    application = create_app(data_root=data_root)
    service = application.state.observation_ingestion_service

    def fail_registration(*_args: object, **_kwargs: object) -> None:
        raise AssetStorageError("simulated storage failure")

    monkeypatch.setattr(service._store, "register", fail_registration)
    with TestClient(application) as client:
        response = client.post(
            "/api/observations",
            files={"file": ("scene.tif", geotiff_bytes, "image/tiff")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ASSET_STORAGE_FAILED"
    _assert_quarantine_empty(data_root)
    assert list((data_root / "observations").iterdir()) == []
