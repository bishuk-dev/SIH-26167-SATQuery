"""Strict text-only change captioning specialist boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import rasterio

from satquery.evidence.models import (
    ChangeCaptionEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    TemporalPairEvidence,
)
from satquery.inference.checkpoints import require_checkpoint
from satquery.inference.exceptions import (
    ModelExecutionError,
    ModelInputUnsupportedError,
    ModelUnavailableError,
    TemporalOrderUnknownError,
)
from satquery.inference.temporal_inputs import read_aligned_rgb_pair
from satquery.ingestion.models import Modality, ObservationState


class ChangeCaptionBackend(Protocol):
    def caption(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> str: ...


class Chg2CapBackend:
    """Lazy adapter seam for the audited Chg2Cap checkpoint and vocabulary."""

    def __init__(
        self,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        *,
        captioner: Callable[[np.ndarray, np.ndarray], str] | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.captioner = captioner

    def caption(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> str:
        require_checkpoint(self.checkpoint_path, self.checkpoint_sha256, model_name="Chg2Cap")
        if self.captioner is not None:
            return self.captioner(t1_rgb, t2_rgb)
        raise ModelUnavailableError("Audited Chg2Cap runtime is not installed")


class ChangeCaptionService:
    def __init__(self, backend: ChangeCaptionBackend, model: EvidenceModelProvenance) -> None:
        self.backend = backend
        self.model = model

    def describe(self, t1: ObservationState, t2: ObservationState) -> ChangeCaptionEvidence:
        _validate_temporal_order(t1, t2)
        first, second = read_aligned_rgb_pair(t1, t2)
        try:
            caption = self.backend.caption(first, second).strip()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelExecutionError("Chg2Cap execution failed") from exc
        if not caption:
            raise ModelExecutionError("Chg2Cap returned an empty caption")
        return ChangeCaptionEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            caption=caption,
            temporal=TemporalPairEvidence(
                t1_observation_id=t1.observation_id,
                t2_observation_id=t2.observation_id,
                order_source="metadata",
            ),
            model=self.model,
            domain=DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
            warnings=(),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id="change_captioning",
                input_asset_id=t1.source_asset.asset_id,
            ),
        )


def _validate_temporal_order(t1: ObservationState, t2: ObservationState) -> None:
    if t1.temporal.acquisition_time is None or t2.temporal.acquisition_time is None:
        raise TemporalOrderUnknownError("temporal order is unknown")
    if t1.temporal.acquisition_time >= t2.temporal.acquisition_time:
        raise TemporalOrderUnknownError("temporal order is not T1 before T2")
