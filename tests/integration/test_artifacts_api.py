"""Integration tests for the public artifact, GeoJSON, and tile-alias routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from satquery.execution import ArtifactMetadata

_EVIDENCE_ID = "evidence_" + "a" * 32
_OBSERVATION_ID = "obs_" + "1" * 32
_GRID_GEO = {
    "width": 2,
    "height": 2,
    "crs": "EPSG:32643",
    "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
    "bounds": [500000.0, 1999980.0, 500020.0, 2000000.0],
    "source_grid_observation_id": _OBSERVATION_ID,
    "value_semantics": "binary_0_1",
}
_GRID_PIXEL = {**_GRID_GEO, "crs": None}


def _write_mask_tif(path: Path, values: np.ndarray, *, crs: str | None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=rasterio.Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
    ) as dataset:
        dataset.write(values, 1)


def _publish_mask_artifact(
    store,
    *,
    grid: dict = _GRID_GEO,
    evidence_id: str | None = _EVIDENCE_ID,
) -> str:
    staged = store.stage("tif")
    _write_mask_tif(staged.path, np.array([[0, 1], [1, 0]], dtype="uint8"), crs=grid["crs"])
    record = store.publish(
        staged,
        ArtifactMetadata(
            evidence_id=evidence_id,
            media_type="image/tiff",
            description="Test change mask",
            extra={"grid": grid},
        ),
    )
    return record.artifact_id


def _publish_json_artifact(store, *, evidence_id: str | None = None) -> str:
    staged = store.stage("json")
    staged.path.write_text('{"value": 12.5}', encoding="utf-8")
    record = store.publish(
        staged,
        ArtifactMetadata(
            evidence_id=evidence_id,
            media_type="application/json",
            description="Test measurement",
        ),
    )
    return record.artifact_id


def test_artifact_metadata_route_projects_safe_summary_without_local_paths(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_mask_artifact(store)

        response = client.get(f"/api/v1/artifacts/{artifact_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["artifact_id"] == artifact_id
        assert body["media_type"] == "image/tiff"
        assert body["sha256"]
        assert body["evidence_id"] == _EVIDENCE_ID
        assert "path" not in body
        assert str(tmp_path) not in json.dumps(body)

        missing = client.get("/api/v1/artifacts/" + "artifact_" + "b" * 32)
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "NOT_FOUND"
        assert str(tmp_path) not in json.dumps(missing.json())


def test_artifact_download_serves_verified_bytes_with_safe_disposition(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_mask_artifact(store)

        response = client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/tiff"
        disposition = response.headers["content-disposition"]
        assert disposition == f'attachment; filename="{artifact_id}.tif"'
        assert str(tmp_path) not in disposition
        record, path = store.resolve(artifact_id)
        assert response.content == path.read_bytes()
        assert response.headers["x-artifact-sha256"] == record.sha256

        # Any on-disk tampering must fail closed before bytes are served.
        record.path.write_bytes(b"tampered")
        tampered = client.get(f"/api/v1/artifacts/{artifact_id}/download")
        assert tampered.status_code == 404


def test_artifact_routes_reject_traversal_style_ids(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        for candidate in (
            "artifact_%2e%2e%2fescape",
            "artifact_" + "a" * 32 + "%2f..%2fmetadata.json",
            ".%2e%2fescape",
        ):
            response = client.get(f"/api/v1/artifacts/{candidate}/download")
            assert response.status_code == 404, candidate


def test_evidence_geojson_serves_recorded_grid_provenance(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_mask_artifact(store)

        response = client.get(f"/api/v1/evidence/{_EVIDENCE_ID}/geojson")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/geo+json")
        footprint = response.json()
        assert footprint["type"] == "Feature"
        assert footprint["crs"] == {"type": "name", "name": "EPSG:32643"}
        ring = footprint["geometry"]["coordinates"][0]
        assert ring == [
            [500000.0, 1999980.0],
            [500020.0, 1999980.0],
            [500020.0, 2000000.0],
            [500000.0, 2000000.0],
            [500000.0, 1999980.0],
        ]
        properties = footprint["properties"]
        assert properties["artifact_id"] == artifact_id
        assert properties["evidence_id"] == _EVIDENCE_ID
        assert properties["coordinate_space"] == "grid"
        assert properties["source_grid_observation_id"] == _OBSERVATION_ID
        assert str(tmp_path) not in json.dumps(footprint)

        missing = client.get("/api/v1/evidence/" + "evidence_" + "b" * 32 + "/geojson")
        assert missing.status_code == 404


def test_evidence_geojson_labels_pixel_space_without_inventing_a_crs(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        _publish_mask_artifact(store, grid=_GRID_PIXEL)

        response = client.get(f"/api/v1/evidence/{_EVIDENCE_ID}/geojson")
        assert response.status_code == 200, response.text
        footprint = response.json()
        assert footprint["crs"] == {"type": "name", "name": "PIXEL_SPACE"}
        assert footprint["properties"]["coordinate_space"] == "pixel"
        ring = footprint["geometry"]["coordinates"][0]
        assert ring == [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
        assert "EPSG:4326" not in json.dumps(footprint)


def test_evidence_geojson_rejects_artifacts_without_spatial_grid(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        _publish_json_artifact(store, evidence_id=_EVIDENCE_ID)

        response = client.get(f"/api/v1/evidence/{_EVIDENCE_ID}/geojson")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ARTIFACT_NOT_SPATIAL"


def test_tile_alias_renders_registered_mask_artifacts(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_mask_artifact(store)

        response = client.get(f"/api/v1/tiles/{artifact_id}/0/0/0.png")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"

        grayscale = client.get(
            f"/api/v1/tiles/{artifact_id}/0/0/0.png",
            params={"colormap_id": "binary_mask_grayscale"},
        )
        assert grayscale.status_code == 200

        pixel_mask = _publish_mask_artifact(
            store, grid={**_GRID_PIXEL, "source_grid_observation_id": "obs_" + "2" * 32}
        )
        store.resolve(pixel_mask)
        pixel_response = client.get(f"/api/v1/tiles/{pixel_mask}/1/0/0.png")
        assert pixel_response.status_code == 200, pixel_response.text


def test_tile_alias_rejects_unknown_artifacts_coords_and_colormaps(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_mask_artifact(store)

        missing = client.get("/api/v1/tiles/" + "artifact_" + "c" * 32 + "/0/0/0.png")
        assert missing.status_code == 404

        bad_zoom = client.get(f"/api/v1/tiles/{artifact_id}/30/0/0.png")
        assert bad_zoom.status_code == 400
        assert bad_zoom.json()["error"]["code"] == "INVALID_TILE_REQUEST"

        bad_colormap = client.get(
            f"/api/v1/tiles/{artifact_id}/0/0/0.png",
            params={"colormap_id": "expression:not(a-b)"},
        )
        assert bad_colormap.status_code == 400
        assert bad_colormap.json()["error"]["code"] == "INVALID_COLORMAP"


def test_tile_alias_rejects_non_raster_artifacts(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        store = client.app.state.artifact_store
        artifact_id = _publish_json_artifact(store)

        response = client.get(f"/api/v1/tiles/{artifact_id}/0/0/0.png")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ARTIFACT_NOT_TILEABLE"


@pytest.mark.parametrize("route", ["/api/v1/artifacts", "/api/v1/evidence", "/api/v1/tiles"])
def test_openapi_declares_new_route_families(tmp_path, route: str) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        schema = client.get("/openapi.json").json()
    assert any(path.startswith(route) for path in schema["paths"])
