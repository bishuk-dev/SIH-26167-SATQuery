from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.app.main import create_app
from satquery.inference.exceptions import ModelUnavailableError
from satquery.registry import load_model_registry, load_preprocessing_registry


class StubBackend:
    def __init__(self, answer: str = "forest") -> None:
        self.answer_text = answer
        self.calls: list[tuple[tuple[int, int], str]] = []

    def answer(self, image: Image.Image, question: str) -> str:
        self.calls.append((image.size, question))
        return self.answer_text


class UnavailableBackend:
    def answer(self, image: Image.Image, question: str) -> str:
        raise ModelUnavailableError("checkpoint missing")


def _geotiff(tmp_path: Path) -> bytes:
    path = tmp_path / "scene.tif"
    data = np.stack(
        [np.full((8, 16), value, dtype=np.uint8) for value in (30, 60, 90)]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 70, 0, -0.01, 20),
        nodata=0,
    ) as dataset:
        dataset.write(data)
        dataset.update_tags(MODALITY="multispectral", SENSOR_NAME="FixtureSat")
    return path.read_bytes()


def _register(client: TestClient, payload: bytes) -> dict[str, object]:
    response = client.post(
        "/api/observations",
        files={"file": ("scene.tif", payload, "image/tiff")},
    )
    assert response.status_code == 201
    return response.json()


def test_registries_pin_frozen_model_and_preprocessing() -> None:
    registration = load_model_registry().models["smolvlm_256m_instruct_v1"]
    profile = load_preprocessing_registry().profiles[
        registration.preprocessing_profile
    ]

    assert registration.model_id == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert registration.frozen is True
    assert registration.allow_remote_code is False
    assert len(registration.revision) == 40
    assert len(registration.checkpoint_sha256) == 64
    assert profile.resize == "fit_pad"
    assert (profile.width, profile.height) == (512, 512)
    assert profile.processor_resize == "disabled"


def test_vqa_endpoint_uses_registered_observation_and_returns_evidence(
    tmp_path: Path,
) -> None:
    backend = StubBackend()
    app = create_app(data_root=tmp_path / "data", vqa_backend=backend)
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/vqa",
            json={
                "observation_id": observation["observation_id"],
                "question": "What is the dominant land cover?",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "single_image_vqa"
    assert body["prediction"] == {"answer": "forest"}
    assert body["source_observations"] == [observation["observation_id"]]
    assert body["model"]["registry_id"] == "smolvlm_256m_instruct_v1"
    assert body["model"]["preprocessing_profile"] == "smolvlm_single_image_v1"
    assert body["model"]["preprocessing_version"] == "1.0.0"
    assert body["domain"]["status"] == "shifted"
    assert "MODEL_NOT_REMOTE_SENSING_ADAPTED" in body["domain"]["reasons"]
    assert body["provenance"]["input_asset_id"] == observation["visualization"][
        "asset_id"
    ]
    assert backend.calls == [((512, 512), "What is the dominant land cover?")]


def test_vqa_endpoint_rejects_unknown_observation_without_running_model(
    tmp_path: Path,
) -> None:
    backend = StubBackend()
    with TestClient(
        create_app(data_root=tmp_path / "data", vqa_backend=backend)
    ) as client:
        response = client.post(
            "/api/vqa",
            json={
                "observation_id": "obs_" + "0" * 32,
                "question": "What is visible?",
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "OBSERVATION_NOT_FOUND"
    assert backend.calls == []


def test_vqa_endpoint_reports_unavailable_checkpoint(tmp_path: Path) -> None:
    app = create_app(
        data_root=tmp_path / "data", vqa_backend=UnavailableBackend()
    )
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/vqa",
            json={
                "observation_id": observation["observation_id"],
                "question": "What is visible?",
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_vqa_endpoint_validates_request_before_inference(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.post(
            "/api/vqa",
            json={"observation_id": "caller-path.tif", "question": ""},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_VQA_REQUEST"
