from __future__ import annotations

import json
import re
import stat
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from apps.api.app.main import create_app
from satquery.visualization.exceptions import VisualizationGenerationError


def _raster_bytes(
    tmp_path: Path,
    name: str,
    data: np.ndarray,
    *,
    crs: str | None = "EPSG:3857",
    transform: Affine = Affine(10, 0, 0, 0, -10, 100),
    nodata: int | float | None = None,
    modality: str,
    descriptions: tuple[str, ...] | None = None,
    color_interpretation: tuple[ColorInterp, ...] | None = None,
) -> bytes:
    path = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            count=data.shape[0],
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(data)
            dataset.update_tags(MODALITY=modality)
            if color_interpretation is not None:
                dataset.colorinterp = color_interpretation
            if descriptions is not None:
                for index, description in enumerate(descriptions, start=1):
                    dataset.set_band_description(index, description)
    return path.read_bytes()


def _upload(client: TestClient, payload: bytes, name: str = "scene.tif"):
    return client.post(
        "/api/observations",
        files={"file": (name, payload, "image/tiff")},
    )


def test_upload_creates_immutable_cog_and_exposes_frontend_contract(
    tmp_path: Path,
) -> None:
    data = np.stack(
        [
            np.full((16, 16), 10, dtype=np.uint16),
            np.full((16, 16), 20, dtype=np.uint16),
            np.full((16, 16), 30, dtype=np.uint16),
            np.full((16, 16), 40, dtype=np.uint16),
        ]
    )
    original_bytes = _raster_bytes(
        tmp_path,
        "multispectral.tif",
        data,
        modality="multispectral",
        descriptions=("blue", "green", "red", "near infrared"),
    )
    data_root = tmp_path / "data"

    with TestClient(create_app(data_root=data_root)) as client:
        response = _upload(client, original_bytes)
        body = response.json()
        tile_response = client.get(
            body["visualization"]["tile_url_template"].format(z=0, x=0, y=0)
        )

    assert response.status_code == 201
    assert body["asset"]["kind"] == "original"
    visualization = body["visualization"]
    assert re.fullmatch(r"asset_[0-9a-f]{32}", visualization["asset_id"])
    assert visualization["asset_id"] != body["asset"]["asset_id"]
    assert visualization["parent_asset_id"] == body["asset"]["asset_id"]
    assert visualization["kind"] == "visualization"
    assert visualization["format"] == "COG"
    assert visualization["rendering"] == "rgb"
    assert visualization["source_band_indexes"] == [3, 2, 1]
    assert visualization["source_grid_preserved"] is True
    assert visualization["tile_scheme"] == "web_mercator"
    assert visualization["tile_crs"] == "EPSG:3857"
    assert visualization["tile_url_template"].endswith("/{z}/{x}/{y}.png")
    assert tile_response.status_code == 200
    assert tile_response.headers["content-type"] == "image/png"
    assert tile_response.headers["cache-control"].endswith("immutable")

    observation_dir = data_root / "observations" / body["observation_id"]
    original = observation_dir / "original.tif"
    derivative = observation_dir / "visualization.tif"
    derivative_record = json.loads(
        (observation_dir / "visualization.json").read_text(encoding="utf-8")
    )
    index_record = json.loads(
        (
            data_root / "assets" / f"{visualization['asset_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert original.read_bytes() == original_bytes
    assert original.stat().st_mode & stat.S_IWRITE == 0
    assert derivative.stat().st_mode & stat.S_IWRITE == 0
    assert derivative_record == index_record
    assert derivative_record["parent_asset_id"] == body["asset"]["asset_id"]
    assert derivative_record["path"].endswith("/visualization.tif")
    with rasterio.open(derivative) as dataset:
        assert dataset.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        assert dataset.count == 4
        assert dataset.dtypes == ("uint8",) * 4
        assert dataset.colorinterp == (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
        assert dataset.transform == Affine(10, 0, 0, 0, -10, 100)
        assert dataset.crs.to_string() == "EPSG:3857"


def test_sar_derivative_uses_log_stretched_grayscale(tmp_path: Path) -> None:
    values = np.arange(1, 33, dtype=np.float32).reshape(2, 4, 4)
    payload = _raster_bytes(
        tmp_path,
        "sar.tif",
        values,
        modality="sar",
        descriptions=("VV", "VH"),
    )
    data_root = tmp_path / "data"

    with TestClient(create_app(data_root=data_root)) as client:
        response = _upload(client, payload)

    assert response.status_code == 201
    body = response.json()
    assert body["visualization"]["rendering"] == "grayscale"
    assert body["visualization"]["source_band_indexes"] == [1]
    record = json.loads(
        (
            data_root
            / "observations"
            / body["observation_id"]
            / "visualization.json"
        ).read_text(encoding="utf-8")
    )
    assert record["stretches"][0]["scale"] == "log1p"
    with rasterio.open(
        data_root / "observations" / body["observation_id"] / "visualization.tif"
    ) as dataset:
        assert dataset.count == 2
        assert dataset.colorinterp == (ColorInterp.gray, ColorInterp.alpha)


def test_rgb_derivative_respects_declared_color_band_order(tmp_path: Path) -> None:
    data = np.stack(
        [
            np.full((4, 4), 10, dtype=np.uint8),
            np.full((4, 4), 20, dtype=np.uint8),
            np.full((4, 4), 30, dtype=np.uint8),
        ]
    )
    payload = _raster_bytes(
        tmp_path,
        "rgb.tif",
        data,
        modality="optical",
        color_interpretation=(
            ColorInterp.blue,
            ColorInterp.green,
            ColorInterp.red,
        ),
    )

    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = _upload(client, payload)

    assert response.status_code == 201
    assert response.json()["visualization"]["rendering"] == "rgb"
    assert response.json()["visualization"]["source_band_indexes"] == [3, 2, 1]


def test_pixel_grid_tile_preserves_nodata_as_png_transparency(tmp_path: Path) -> None:
    data = np.ones((1, 4, 4), dtype=np.uint8)
    data[0, :2, :2] = 0
    payload = _raster_bytes(
        tmp_path,
        "unreferenced.tif",
        data,
        crs=None,
        transform=Affine.identity(),
        nodata=0,
        modality="optical",
    )
    data_root = tmp_path / "data"

    with TestClient(create_app(data_root=data_root)) as client:
        upload = _upload(client, payload)
        visualization = upload.json()["visualization"]
        tile = client.get(
            visualization["tile_url_template"].format(z=0, x=0, y=0)
        )

    assert upload.status_code == 201
    assert visualization["tile_scheme"] == "pixel"
    assert visualization["tile_crs"] is None
    assert visualization["pixel_y_axis"] == "down"
    assert tile.status_code == 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(tile.content) as memory:
            with memory.open() as dataset:
                alpha = dataset.read(4)
                assert (dataset.width, dataset.height) == (256, 256)
                assert alpha[16, 16] == 0
                assert alpha[-16, -16] == 255


def test_tile_endpoint_rejects_unknown_and_malformed_asset_ids(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        malformed = client.get("/tiles/not-an-asset/0/0/0.png")
        missing = client.get(f"/tiles/asset_{'a' * 32}/0/0/0.png")

    assert malformed.status_code == 404
    assert malformed.json()["error"]["code"] == "VISUALIZATION_ASSET_NOT_FOUND"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "VISUALIZATION_ASSET_NOT_FOUND"


def test_derivative_failure_rejects_upload_and_cleans_all_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _raster_bytes(
        tmp_path,
        "scene.tif",
        np.ones((1, 4, 4), dtype=np.uint8),
        modality="optical",
    )
    data_root = tmp_path / "data"
    application = create_app(data_root=data_root)
    generator = (
        application.state.observation_ingestion_service._derivative_generator
    )

    def fail_derivative(*_args: object, **_kwargs: object) -> None:
        raise VisualizationGenerationError("simulated derivative failure")

    monkeypatch.setattr(generator, "create", fail_derivative)
    with TestClient(application) as client:
        response = _upload(client, payload)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "VISUALIZATION_GENERATION_FAILED"
    assert list((data_root / "quarantine").iterdir()) == []
    assert list((data_root / "observations").iterdir()) == []
    assert list((data_root / "assets").iterdir()) == []
