from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.app.main import create_app
from satquery.inference.exceptions import ModelUnavailableError
from satquery.inference.grounding import (
    DEFAULT_GROUNDING_MODEL_REGISTRY_ID,
    GroundingDinoBackend,
    GroundingBackendResult,
    GroundingSelectionCandidate,
    RawGroundingDetection,
    select_grounding_candidate,
)
from satquery.inference.config import GroundingRuntimeSettings
from satquery.inference.grounding_preprocessing import grounding_input_size
from satquery.registry.models import (
    GroundingModelRegistration,
    GroundingPreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)


class FractionalBoxBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], str]] = []

    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        self.calls.append((image.size, query))
        width, height = image.size
        return GroundingBackendResult(
            input_width=width,
            input_height=height,
            detections=(
                RawGroundingDetection(
                    phrase="storage tank",
                    score=0.81,
                    x_min=width * 0.25,
                    y_min=height * 0.25,
                    x_max=width * 0.75,
                    y_max=height * 0.75,
                ),
            ),
        )


class EmptyBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        return GroundingBackendResult(image.width, image.height, ())


class UnavailableBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        raise ModelUnavailableError("checkpoint missing")


class InconsistentCoordinateBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        return GroundingBackendResult(image.width + 1, image.height, ())


class OversizedBackend:
    def detect(self, image: Image.Image, query: str) -> GroundingBackendResult:
        return GroundingBackendResult(
            image.width,
            image.height,
            (
                RawGroundingDetection(
                    phrase="scene",
                    score=0.9,
                    x_min=0,
                    y_min=0,
                    x_max=image.width,
                    y_max=image.height,
                ),
            ),
        )


class _TensorValues:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _ProcessorInputs(dict):
    input_ids = object()

    def to(self, _device: str) -> _ProcessorInputs:
        return self


class _ProcessorStub:
    def __init__(self) -> None:
        self.thresholds: tuple[float, float] | None = None

    def __call__(self, **_kwargs: object) -> _ProcessorInputs:
        return _ProcessorInputs()

    def post_process_grounded_object_detection(
        self,
        _outputs: object,
        _input_ids: object,
        *,
        threshold: float,
        text_threshold: float,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict[str, object]]:
        self.thresholds = (threshold, text_threshold)
        assert target_sizes == [(8, 16)]
        return [
            {
                "scores": _TensorValues([0.9]),
                "boxes": _TensorValues([[1.0, 2.0, 8.0, 6.0]]),
                "text_labels": ["ship"],
            }
        ]


class _TorchStub:
    @staticmethod
    def inference_mode():
        return nullcontext()


def _geotiff(tmp_path: Path, *, georeferenced: bool = True) -> bytes:
    path = tmp_path / "grounding-scene.tif"
    kwargs = (
        {"crs": "EPSG:4326", "transform": Affine(0.01, 0, 70, 0, -0.01, 20)}
        if georeferenced
        else {}
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=8,
        count=3,
        dtype="uint8",
        nodata=0,
        **kwargs,
    ) as dataset:
        dataset.write(
            np.stack(
                [np.full((8, 16), value, dtype=np.uint8) for value in (30, 60, 90)]
            )
        )
        dataset.update_tags(MODALITY="multispectral")
    return path.read_bytes()


def _register(client: TestClient, payload: bytes) -> dict[str, object]:
    response = client.post(
        "/api/observations",
        files={"file": ("scene.tif", payload, "image/tiff")},
    )
    assert response.status_code == 201
    return response.json()


def test_grounding_registry_is_frozen_and_task_typed() -> None:
    registration = load_model_registry().models["grounding_dino_tiny_v1"]
    assert isinstance(registration, GroundingModelRegistration)
    profile = load_preprocessing_registry().profiles[
        registration.preprocessing_profile
    ]
    assert isinstance(profile, GroundingPreprocessingProfile)
    assert registration.frozen is True
    assert registration.allow_remote_code is False
    assert profile.processor_resize == "disabled"
    assert (profile.shortest_edge, profile.longest_edge) == (800, 1333)
    assert (profile.box_threshold, profile.text_threshold) == (0.4, 0.3)

    final_registration = load_model_registry().models[
        DEFAULT_GROUNDING_MODEL_REGISTRY_ID
    ]
    assert isinstance(final_registration, GroundingModelRegistration)
    final_profile = load_preprocessing_registry().profiles[
        final_registration.preprocessing_profile
    ]
    assert isinstance(final_profile, GroundingPreprocessingProfile)
    assert final_registration.model_id == registration.model_id
    assert final_registration.revision == registration.revision
    assert final_registration.checkpoint_sha256 == registration.checkpoint_sha256
    assert (final_profile.box_threshold, final_profile.text_threshold) == (0.3, 0.3)
    assert final_profile.max_normalized_box_area == 0.8


def test_backend_uses_transformers_threshold_contract() -> None:
    registration = load_model_registry().models["grounding_dino_tiny_v1"]
    profile = load_preprocessing_registry().profiles["grounding_dino_tiny_v1"]
    assert isinstance(registration, GroundingModelRegistration)
    assert isinstance(profile, GroundingPreprocessingProfile)
    backend = GroundingDinoBackend(
        registration, profile, GroundingRuntimeSettings(device="cpu")
    )
    processor = _ProcessorStub()
    backend._processor = processor
    backend._model = lambda **_inputs: object()
    backend._torch = _TorchStub()

    result = backend.detect(Image.new("RGB", (16, 8)), "The ship")

    assert processor.thresholds == (0.4, 0.3)
    assert result.detections[0].phrase == "ship"


def test_grounding_endpoint_maps_model_box_to_source_and_world_coordinates(
    tmp_path: Path,
) -> None:
    backend = FractionalBoxBackend()
    app = create_app(data_root=tmp_path / "data", grounding_backend=backend)
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/grounding",
            json={
                "observation_id": observation["observation_id"],
                "query": "the storage tank",
            },
        )

    assert response.status_code == 200
    body = response.json()
    detection = body["detections"][0]
    assert body["task"] == "text_guided_grounding"
    assert body["model"]["registry_id"] == DEFAULT_GROUNDING_MODEL_REGISTRY_ID
    assert body["model"]["preprocessing_profile"] == (
        "grounding_dino_tiny_phase3_final_v1"
    )
    assert body["provenance"]["input_asset_id"] == observation["visualization"][
        "asset_id"
    ]
    assert detection["phrase"] == "storage tank"
    assert detection["raw_score"] == 0.81
    assert detection["model_input_box"]["coordinate_space"] == "model_input"
    assert detection["source_pixel_box"] == {
        "coordinate_space": "source_image",
        "x_min": 4.0,
        "y_min": 2.0,
        "x_max": 12.0,
        "y_max": 6.0,
        "image_width": 16,
        "image_height": 8,
    }
    assert detection["normalized_box"] == {
        "coordinate_space": "source_normalized",
        "x_min": 0.25,
        "y_min": 0.25,
        "x_max": 0.75,
        "y_max": 0.75,
    }
    assert detection["world_polygon"]["crs"] == "EPSG:4326"
    assert np.asarray(detection["world_polygon"]["coordinates"]) == pytest.approx(
        np.asarray(
            [[70.04, 19.98], [70.12, 19.98], [70.12, 19.94], [70.04, 19.94]]
        )
    )
    assert backend.calls == [((1333, 666), "the storage tank")]


def test_empty_detection_is_valid_evidence_without_fake_geometry(
    tmp_path: Path,
) -> None:
    app = create_app(data_root=tmp_path / "data", grounding_backend=EmptyBackend())
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/grounding",
            json={"observation_id": observation["observation_id"], "query": "ship"},
        )

    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert "NO_GROUNDING_DETECTIONS" in response.json()["warnings"]


def test_all_oversized_detections_produce_valid_abstention(tmp_path: Path) -> None:
    app = create_app(data_root=tmp_path / "data", grounding_backend=OversizedBackend())
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/grounding",
            json={"observation_id": observation["observation_id"], "query": "ship"},
        )

    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert "GROUNDING_ABSTAINED_OVERSIZED_BOXES" in response.json()["warnings"]
    assert "NO_GROUNDING_DETECTIONS" in response.json()["warnings"]


def test_grounding_rejections_use_structured_failure_contract(tmp_path: Path) -> None:
    backend = FractionalBoxBackend()
    with TestClient(
        create_app(data_root=tmp_path / "data", grounding_backend=backend)
    ) as client:
        missing = client.post(
            "/api/grounding",
            json={"observation_id": "obs_" + "0" * 32, "query": "ship"},
        )
        invalid = client.post(
            "/api/grounding",
            json={"observation_id": "caller-path.tif", "query": "   "},
        )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "OBSERVATION_NOT_FOUND"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INVALID_GROUNDING_REQUEST"
    assert backend.calls == []


def test_grounding_reports_unavailable_model(tmp_path: Path) -> None:
    app = create_app(
        data_root=tmp_path / "data", grounding_backend=UnavailableBackend()
    )
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/grounding",
            json={"observation_id": observation["observation_id"], "query": "ship"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_grounding_rejects_geometry_that_cannot_map_to_source(tmp_path: Path) -> None:
    app = create_app(
        data_root=tmp_path / "data", grounding_backend=InconsistentCoordinateBackend()
    )
    with TestClient(app) as client:
        observation = _register(client, _geotiff(tmp_path))
        response = client.post(
            "/api/grounding",
            json={"observation_id": observation["observation_id"], "query": "ship"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_EVIDENCE_GEOMETRY"


def test_grounding_resize_preserves_aspect_ratio_and_long_edge_cap() -> None:
    assert grounding_input_size(16, 8, 800, 1333) == (1333, 666)
    assert grounding_input_size(512, 512, 800, 1333) == (800, 800)


def _candidate(
    index: int, score: float, box: tuple[float, float, float, float]
) -> GroundingSelectionCandidate:
    return GroundingSelectionCandidate(index, score, box)


def test_guardrail_rejects_area_equal_to_limit() -> None:
    candidate = _candidate(0, 0.9, (0.0, 0.0, 0.8, 1.0))
    assert select_grounding_candidate((candidate,), max_area=0.8) is None


def test_guardrail_retains_area_below_limit() -> None:
    candidate = _candidate(0, 0.9, (0.0, 0.0, 0.79, 1.0))
    assert select_grounding_candidate((candidate,), max_area=0.8) == candidate


def test_guardrail_selects_next_highest_valid_detection() -> None:
    oversized = _candidate(0, 0.95, (0.0, 0.0, 1.0, 1.0))
    next_valid = _candidate(1, 0.80, (0.1, 0.1, 0.5, 0.5))
    lower_valid = _candidate(2, 0.70, (0.2, 0.2, 0.4, 0.4))
    assert (
        select_grounding_candidate(
            (oversized, next_valid, lower_valid), max_area=0.8
        )
        == next_valid
    )


def test_guardrail_abstains_when_every_detection_is_oversized() -> None:
    candidates = (
        _candidate(0, 0.9, (0.0, 0.0, 1.0, 1.0)),
        _candidate(1, 0.8, (0.0, 0.0, 0.9, 0.9)),
    )
    assert select_grounding_candidate(candidates, max_area=0.8) is None


def test_guardrail_preserves_highest_score_policy_among_valid_detections() -> None:
    lower = _candidate(0, 0.31, (0.0, 0.0, 0.1, 0.1))
    higher = _candidate(1, 0.85, (0.2, 0.2, 0.6, 0.6))
    assert select_grounding_candidate((lower, higher), max_area=0.8) == higher
