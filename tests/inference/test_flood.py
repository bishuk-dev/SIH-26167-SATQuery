from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.evidence.models import DomainStatus, EvidenceModelProvenance
from satquery.inference.exceptions import ModelInputUnsupportedError, ModelUnavailableError
from satquery.inference.flood import FloodSegmentationService, SturmS1Backend
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


class FakeFloodBackend:
    def segment(self, image: np.ndarray) -> np.ndarray:
        assert image.shape == (2, 128, 128)
        return np.full((128, 128), 0.75, dtype=np.float32)


def _observation(path: Path, *, polarizations=("VV", "VH"), sensor="Sentinel-1 C-SAR", domain="backscatter_db") -> ObservationState:
    transform = Affine(10, 0, 0, 0, -10, 1280)
    return ObservationState(
        observation_id="s1-observation",
        source_asset=SourceAsset(asset_id="s1-asset", original_name=path.name, path=str(path), sha256="0" * 64),
        raster=RasterMetadata(
            driver="GTiff", width=128, height=128, band_count=2,
            dtypes=("float32", "float32"), nodata=(None, None),
            tags={"RADIOMETRIC_DOMAIN": domain, "CALIBRATION": "calibrated"},
        ),
        sensor=SensorMetadata(
            modality=Modality.SAR, sensor_name=sensor,
            polarizations=polarizations,
            bands=tuple(BandMetadata(index=i, description=p, dtype="float32") for i, p in enumerate(("VV", "VH"), 1)),
        ),
        geo=GeoMetadata(
            crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": -10.0, "f": 1280.0},
            bounds=GeoBounds(left=0, bottom=0, right=1280, top=1280),
            native_gsd_x=10, native_gsd_y=10, units="m",
        ),
        temporal=TemporalMetadata(acquisition_time=datetime(2020, 1, 1, tzinfo=timezone.utc)),
        validity=ValidityMetadata(has_crs=True, has_transform=True, has_nodata=False, metadata_quality=MetadataQuality.HIGH),
        provenance=ObservationProvenance(created_at=datetime.now(timezone.utc), ingestion_version="test"),
    )


def _write(path: Path) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=2, dtype="float32", crs="EPSG:32643", transform=Affine(10, 0, 0, 0, -10, 1280)) as dataset:
        dataset.write(np.ones((2, 128, 128), dtype="float32"))


def _model() -> EvidenceModelProvenance:
    return EvidenceModelProvenance(registry_id="sturm_fixture", model_id="fixture/sturm", revision="0" * 40, checkpoint_sha256="0" * 64, preprocessing_profile="sturm_fixture_v1", preprocessing_version="1.0.0")


def test_sturm_accepts_only_exact_audited_sentinel1_contract(tmp_path: Path) -> None:
    path = tmp_path / "s1.tif"
    _write(path)

    evidence = FloodSegmentationService(FakeFloodBackend(), tmp_path / "out", _model()).segment(_observation(path))

    assert evidence.domain.status is DomainStatus.IN_DOMAIN
    assert evidence.target_class == "water_or_flood_extent"


@pytest.mark.parametrize("polarizations", [("HH", "HV"), ("VV",), ()])
def test_sturm_rejects_incompatible_polarization(tmp_path: Path, polarizations: tuple[str, ...]) -> None:
    path = tmp_path / "s1.tif"
    _write(path)
    with pytest.raises(ModelInputUnsupportedError):
        FloodSegmentationService(FakeFloodBackend(), tmp_path / "out", _model()).segment(
            _observation(path, polarizations=polarizations)
        )


def test_risat_never_falls_through_to_sturm(tmp_path: Path) -> None:
    path = tmp_path / "s1.tif"
    _write(path)
    with pytest.raises(ModelInputUnsupportedError):
        FloodSegmentationService(FakeFloodBackend(), tmp_path / "out", _model()).segment(
            _observation(path, sensor="RISAT-1")
        )


def test_sturm_backend_checkpoint_verification(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    image = np.ones((2, 128, 128), dtype=np.float32)
    with pytest.raises(ModelUnavailableError, match="STURM checkpoint is unavailable"):
        SturmS1Backend(missing, "0" * 64, segmenter=lambda img: np.zeros((128, 128), dtype=np.float32)).segment(image)

    dummy = tmp_path / "checkpoint.pth"
    dummy.write_bytes(b"bad-checkpoint")
    with pytest.raises(ModelUnavailableError, match="STURM checkpoint hash is invalid"):
        SturmS1Backend(dummy, "0" * 64, segmenter=lambda img: np.zeros((128, 128), dtype=np.float32)).segment(image)
