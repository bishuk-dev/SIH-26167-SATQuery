from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.inference.change_detection import ChangerExBackend, StructuralChangeService
from satquery.inference.exceptions import ModelInputUnsupportedError, ModelUnavailableError
from satquery.ingestion.models import (
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    MetadataQuality,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)
from satquery.evidence.models import DomainAssessment, EvidenceModelProvenance


class FakeBackend:
    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
        assert t1_rgb.shape == t2_rgb.shape == (3, 2, 2)
        return np.array([[0.9, 0.1], [0.6, 0.2]], dtype=np.float32)


def _observation(path: Path, observation_id: str) -> ObservationState:
    transform = Affine(10, 0, 0, 0, -10, 20)
    return ObservationState(
        observation_id=observation_id,
        source_asset=SourceAsset(
            asset_id=observation_id + "-asset",
            original_name=path.name,
            path=str(path),
            sha256="0" * 64,
        ),
        raster=RasterMetadata(
            driver="GTiff",
            width=2,
            height=2,
            band_count=3,
            dtypes=("uint8",) * 3,
            nodata=(None,) * 3,
        ),
        sensor=SensorMetadata(
            modality=Modality.OPTICAL,
            sensor_name="FixtureSat",
            bands=tuple(
                BandMetadata(index=index, description=name, dtype="uint8")
                for index, name in enumerate(("R", "G", "B"), 1)
            ),
        ),
        geo=GeoMetadata(
            crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": -10.0, "f": 20.0},
            bounds=GeoBounds(left=0, bottom=0, right=20, top=20),
            native_gsd_x=10,
            native_gsd_y=10,
            units="m",
        ),
        temporal=TemporalMetadata(acquisition_time=datetime(2020, 1, 1, tzinfo=timezone.utc)),
        validity=ValidityMetadata(
            has_crs=True,
            has_transform=True,
            has_nodata=False,
            metadata_quality=MetadataQuality.HIGH,
        ),
        provenance=ObservationProvenance(
            created_at=datetime.now(timezone.utc), ingestion_version="test"
        ),
    )


def _write_image(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=Affine(10, 0, 0, 0, -10, 20),
    ) as dataset:
        dataset.write(np.ones((3, 2, 2), dtype="uint8"))


def test_structural_change_service_emits_source_grid_mask(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")
    provenance = EvidenceModelProvenance(
        registry_id="changerex_fixture",
        model_id="fixture/model",
        revision="0" * 40,
        checkpoint_sha256="0" * 64,
        preprocessing_profile="changerex_fixture_v1",
        preprocessing_version="1.0.0",
    )

    evidence = StructuralChangeService(FakeBackend(), tmp_path / "out", provenance).detect(t1, t2)

    assert evidence.target_class == "high_res_structural_change"
    assert evidence.mask.source_grid_observation_id == t1.observation_id
    assert evidence.domain.status.value == "in_domain"


def test_structural_model_rejects_sar_and_unverified_alignment(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")
    provenance = EvidenceModelProvenance(
        registry_id="changerex_fixture",
        model_id="fixture/model",
        revision="0" * 40,
        checkpoint_sha256="0" * 64,
        preprocessing_profile="changerex_fixture_v1",
        preprocessing_version="1.0.0",
    )
    service = StructuralChangeService(FakeBackend(), tmp_path / "out", provenance)

    sar = t1.model_copy(update={"sensor": t1.sensor.model_copy(update={"modality": Modality.SAR})})
    with pytest.raises(ModelInputUnsupportedError):
        service.detect(sar, t2)

    shifted = t2.model_copy(
        update={"geo": t2.geo.model_copy(update={"transform": t2.geo.transform.model_copy(update={"c": 10.0})})}
    )
    with pytest.raises(ModelInputUnsupportedError):
        service.detect(t1, shifted)


def test_changerex_backend_checkpoint_verification(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    t1_rgb = np.ones((3, 2, 2), dtype=np.float32)
    t2_rgb = np.ones((3, 2, 2), dtype=np.float32)
    with pytest.raises(ModelUnavailableError, match="ChangerEx checkpoint is unavailable"):
        ChangerExBackend(missing, "0" * 64, predictor=lambda a, b: np.zeros((2, 2), dtype=np.float32)).predict(t1_rgb, t2_rgb)

    dummy = tmp_path / "checkpoint.pth"
    dummy.write_bytes(b"bad-checkpoint")
    with pytest.raises(ModelUnavailableError, match="ChangerEx checkpoint hash is invalid"):
        ChangerExBackend(dummy, "0" * 64, predictor=lambda a, b: np.zeros((2, 2), dtype=np.float32)).predict(t1_rgb, t2_rgb)
