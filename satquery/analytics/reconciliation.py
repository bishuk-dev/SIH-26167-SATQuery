"""Fail-closed reconciliation of independent temporal mask evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import rasterio

from satquery.analytics.sar import mask_agreement
from satquery.evidence.models import (
    AgreementEvidence,
    ChangeMaskEvidence,
    FloodMaskEvidence,
)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    outcome: Literal["ALLOW", "ALLOW_WITH_WARNING", "ABSTAIN", "REJECT"]
    evidence_ids: tuple[str, ...]
    agreement: AgreementEvidence | None
    warnings: tuple[str, ...]
    aggregate_confidence: None = None


def reconcile_masks(
    learned: FloodMaskEvidence | ChangeMaskEvidence | None,
    deterministic: ChangeMaskEvidence | None,
) -> ReconciliationResult:
    evidence_ids = tuple(
        evidence.evidence_id
        for evidence in (learned, deterministic)
        if evidence is not None
    )
    if learned is None and deterministic is None:
        return ReconciliationResult("REJECT", (), None, ("NO_MASK_EVIDENCE",))
    if learned is None:
        return ReconciliationResult(
            "ALLOW_WITH_WARNING",
            evidence_ids,
            None,
            ("LEARNED_MODEL_NOT_APPLICABLE",),
        )
    if deterministic is None:
        return ReconciliationResult(
            "ALLOW_WITH_WARNING",
            evidence_ids,
            None,
            ("DETERMINISTIC_CHECK_NOT_AVAILABLE",),
        )
    try:
        first, first_grid = _read_mask(learned.mask.path)
        second, second_grid = _read_mask(deterministic.mask.path)
    except (OSError, rasterio.errors.RasterioIOError, ValueError) as exc:
        return ReconciliationResult(
            "REJECT", evidence_ids, None, (f"INVALID_MASK_EVIDENCE: {exc}",)
        )
    if first.shape != second.shape or first_grid != second_grid:
        return ReconciliationResult(
            "REJECT", evidence_ids, None, ("MASK_GRIDS_INCOMPATIBLE",)
        )
    agreement = mask_agreement(first, second, np.ones(first.shape, dtype=bool))
    agreement_evidence = AgreementEvidence(
        evidence_id=f"evidence_{uuid4().hex}",
        first_evidence_id=learned.evidence_id,
        second_evidence_id=deterministic.evidence_id,
        metric="mask_iou",
        value=agreement.iou,
        interpretation="agreement_not_accuracy",
    )
    if agreement.iou is None or agreement.iou < 0.1:
        return ReconciliationResult(
            "ABSTAIN",
            evidence_ids,
            agreement_evidence,
            ("MASK_AGREEMENT_STRONG_CONFLICT",),
        )
    if agreement.iou < 0.5:
        return ReconciliationResult(
            "ALLOW_WITH_WARNING",
            evidence_ids,
            agreement_evidence,
            ("MASK_AGREEMENT_PARTIAL",),
        )
    return ReconciliationResult("ALLOW", evidence_ids, agreement_evidence, ())


def _read_mask(path: str) -> tuple[np.ndarray, tuple[object, ...]]:
    with rasterio.open(Path(path)) as dataset:
        values = dataset.read(1)
        grid = (dataset.width, dataset.height, dataset.crs, dataset.transform)
    return values.astype(bool), grid
