"""Strict Sentinel-1 flood segmentation specialist boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import rasterio

from satquery.evidence.models import (
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    FloodMaskEvidence,
    MaskAsset,
)
from satquery.inference.checkpoints import require_checkpoint, sha256_file
from satquery.inference.exceptions import ModelExecutionError, ModelInputUnsupportedError, ModelUnavailableError
from satquery.ingestion.models import Modality, ObservationState
from satquery.verification.domain import require_domain


class FloodBackend(Protocol):
    def segment(self, image: np.ndarray) -> np.ndarray: ...


class SturmS1Backend:
    """Lazy adapter seam for the audited STURM Sentinel-1 U-Net checkpoint."""

    def __init__(self, checkpoint_path: Path, checkpoint_sha256: str, *, segmenter: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.segmenter = segmenter

    def segment(self, image: np.ndarray) -> np.ndarray:
        require_checkpoint(self.checkpoint_path, self.checkpoint_sha256, model_name="STURM")
        if self.segmenter is not None:
            return self.segmenter(image)
        raise ModelUnavailableError("Audited STURM runtime is not installed")


class FloodSegmentationService:
    def __init__(self, backend: FloodBackend, output_dir: Path, model: EvidenceModelProvenance, *, threshold: float = 0.5) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("flood threshold must be between zero and one")
        self.backend = backend
        self.output_dir = output_dir
        self.model = model
        self.threshold = threshold

    def segment(self, observation: ObservationState) -> FloodMaskEvidence:
        _validate_observation(observation)
        with rasterio.open(observation.source_asset.path) as source:
            if source.count < 2 or (source.width, source.height) != (128, 128):
                raise ModelInputUnsupportedError("STURM requires a 128x128 VV/VH input")
            if observation.validity.has_nodata and not np.all(source.read_masks((1, 2)) > 0):
                raise ModelInputUnsupportedError("STURM refuses masked input pixels")
            image = source.read((1, 2)).astype("float32")
            profile = source.profile.copy()
        try:
            scores = np.asarray(self.backend.segment(image), dtype="float32")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelExecutionError("STURM execution failed") from exc
        if scores.shape != (128, 128) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ModelExecutionError("STURM returned invalid segmentation scores")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        asset_id = f"mask_{uuid4().hex}"
        mask_path = self.output_dir / f"{asset_id}.tif"
        profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(mask_path, "w", **profile) as target:
            target.write((scores >= self.threshold).astype("uint8"), 1)
        transform = observation.geo.transform
        if transform is None or observation.geo.crs is None:
            raise ModelInputUnsupportedError("flood evidence requires georeferencing")
        return FloodMaskEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            target_class="water_or_flood_extent",
            source_observation_id=observation.observation_id,
            mask=MaskAsset(
                asset_id=asset_id,
                path=str(mask_path),
                sha256=sha256_file(mask_path),
                width=128,
                height=128,
                crs=observation.geo.crs,
                transform=transform,
                source_grid_observation_id=observation.observation_id,
                value_semantics="binary_0_1",
            ),
            raw_model_score=float(scores.max()),
            model=self.model,
            domain=DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
            warnings=(),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id="flood_segmentation",
                input_asset_id=observation.source_asset.asset_id,
            ),
        )


def _validate_observation(observation: ObservationState) -> None:
    require_domain(
        observation,
        supported_modalities=(Modality.SAR,),
        required_sensor_names=("Sentinel-1", "Sentinel-1 C-SAR"),
        required_polarizations=("VV", "VH"),
    )
    if observation.sensor.modality is not Modality.SAR:
        raise ModelInputUnsupportedError("STURM requires SAR")
    if observation.sensor.sensor_name not in {"Sentinel-1", "Sentinel-1 C-SAR"}:
        raise ModelInputUnsupportedError("STURM only accepts audited Sentinel-1")
    if observation.sensor.polarizations != ("VV", "VH"):
        raise ModelInputUnsupportedError("STURM requires ordered VV/VH polarization")
    if observation.raster.tags.get("RADIOMETRIC_DOMAIN") not in {"backscatter_db", "backscatter_linear"}:
        raise ModelInputUnsupportedError("STURM requires an explicit radiometric domain")
    if observation.raster.tags.get("CALIBRATION") != "calibrated":
        raise ModelInputUnsupportedError("STURM requires calibrated SAR input")
    if observation.geo.native_gsd_x is None or observation.geo.native_gsd_y is None or abs(observation.geo.native_gsd_x - 10) > 1 or abs(observation.geo.native_gsd_y - 10) > 1:
        raise ModelInputUnsupportedError("STURM requires approximately 10 m GSD")
    if observation.geo.crs is None or observation.geo.transform is None:
        raise ModelInputUnsupportedError("STURM requires georeferencing")
