"""Strict high-resolution structural change specialist boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import rasterio

from satquery.evidence.models import (
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    MaskAsset,
    TemporalPairEvidence,
)
from satquery.inference.checkpoints import require_checkpoint, sha256_file
from satquery.inference.exceptions import (
    ModelExecutionError,
    ModelInputUnsupportedError,
    ModelUnavailableError,
)
from satquery.inference.temporal_inputs import read_aligned_rgb_pair
from satquery.ingestion.models import ObservationState


class StructuralChangeBackend(Protocol):
    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray: ...


class ChangerExBackend:
    """Lazy adapter seam for the audited Open-CD ChangerEx checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        *,
        predictor: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.predictor = predictor

    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
        require_checkpoint(self.checkpoint_path, self.checkpoint_sha256, model_name="ChangerEx")
        if self.predictor is not None:
            return self.predictor(t1_rgb, t2_rgb)
        raise ModelUnavailableError(
            "Audited Open-CD ChangerEx runtime is not installed"
        )


class StructuralChangeService:
    def __init__(
        self,
        backend: StructuralChangeBackend,
        output_dir: Path,
        model: EvidenceModelProvenance,
        *,
        threshold: float = 0.5,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("change threshold must be between zero and one")
        self.backend = backend
        self.output_dir = output_dir
        self.model = model
        self.threshold = threshold

    def detect(self, t1: ObservationState, t2: ObservationState) -> ChangeMaskEvidence:
        first, second = read_aligned_rgb_pair(t1, t2)
        try:
            scores = np.asarray(self.backend.predict(first, second), dtype="float32")
        except ModelInputUnsupportedError:
            raise
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelExecutionError("ChangerEx execution failed") from exc
        if scores.shape != first.shape[1:] or not np.isfinite(scores).all():
            raise ModelExecutionError("ChangerEx returned invalid score geometry")
        if (scores < 0).any() or (scores > 1).any():
            raise ModelExecutionError("ChangerEx scores must be probabilities")
        binary = scores >= self.threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)
        asset_id = f"mask_{uuid4().hex}"
        mask_path = self.output_dir / f"{asset_id}.tif"
        with rasterio.open(t1.source_asset.path) as source:
            profile = source.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0)
            with rasterio.open(mask_path, "w", **profile) as target:
                target.write(binary.astype("uint8"), 1)
        transform = t1.geo.transform
        if transform is None or t1.geo.crs is None:
            raise ModelInputUnsupportedError("change evidence requires georeferencing")
        return ChangeMaskEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            target_class="high_res_structural_change",
            change_kind="symmetric_change",
            temporal=TemporalPairEvidence(
                t1_observation_id=t1.observation_id,
                t2_observation_id=t2.observation_id,
                order_source="explicit_user_mapping",
            ),
            mask=MaskAsset(
                asset_id=asset_id,
                path=str(mask_path),
                sha256=sha256_file(mask_path),
                width=t1.raster.width,
                height=t1.raster.height,
                crs=t1.geo.crs,
                transform=transform,
                source_grid_observation_id=t1.observation_id,
                value_semantics="binary_0_1",
            ),
            raw_model_score=float(scores.max()) if scores.size else None,
            model=self.model,
            domain=DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
            warnings=(),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id="structural_change_segmentation",
                input_asset_id=t1.source_asset.asset_id,
            ),
        )
