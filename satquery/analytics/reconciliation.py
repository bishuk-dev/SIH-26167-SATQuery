"""Mask-evidence compatibility assessment and diagnostic agreement.

Phase 4 scope: compatibility is a factual grid/phenomenon/time/value check and
agreement is diagnostic IoU only. No decision policy (ALLOW/WARN/ABSTAIN) is
derived from IoU values, and agreement never claims either producer is correct.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from satquery.analytics.exceptions import AnalyticsError
from satquery.analytics.sar import mask_agreement
from satquery.evidence.models import (
    AgreementEvidence,
    ChangeMaskEvidence,
    EvidenceProvenance,
    FloodMaskEvidence,
    MaskAsset,
)

MaskEvidence = FloodMaskEvidence | ChangeMaskEvidence

_FLOOD_PHENOMENA = {"water_extent", "flood_extent", "water_or_flood_extent"}


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    comparable: bool
    reasons: tuple[str, ...]


class MaskCompatibilityError(AnalyticsError):
    """Raised when diagnostic agreement is requested for non-comparable masks."""

    def __init__(self, result: CompatibilityResult) -> None:
        self.result = result
        super().__init__("; ".join(result.reasons) or "masks are not comparable")


def _grid_reasons(first: MaskAsset, second: MaskAsset) -> list[str]:
    reasons: list[str] = []
    first_georef = first.crs is not None
    second_georef = second.crs is not None
    if first_georef != second_georef:
        reasons.append(
            "grid semantics differ: one mask is georeferenced and the other is "
            "pixel-space; they cannot be compared pixelwise"
        )
        return reasons
    if first_georef:
        if first.crs != second.crs:
            reasons.append("grid semantics differ: CRS values differ")
        if first.transform != second.transform:
            reasons.append(
                "grid semantics differ: transforms differ and no explicit "
                "alignment evidence exists"
            )
    if (first.width, first.height) != (second.width, second.height):
        reasons.append("grid semantics differ: mask dimensions differ")
    return reasons


def _time_semantics(evidence: MaskEvidence) -> tuple[str, ...]:
    if isinstance(evidence, FloodMaskEvidence):
        return ("single_observation", evidence.source_observation_id)
    return (
        "temporal_pair",
        evidence.temporal.t1_observation_id,
        evidence.temporal.t2_observation_id,
    )


def _phenomenon(evidence: MaskEvidence) -> str | None:
    if isinstance(evidence, FloodMaskEvidence):
        return evidence.target_class if evidence.target_class in _FLOOD_PHENOMENA else None
    return f"change:{evidence.target_class}:{evidence.change_kind}"


def assess_mask_compatibility(first: MaskEvidence, second: MaskEvidence) -> CompatibilityResult:
    """Collect every reason two mask evidences cannot be compared pixelwise."""

    reasons: list[str] = []

    if type(first) is not type(second):
        reasons.append(
            "time semantics differ: single-observation and temporal-pair masks "
            "describe different time structures"
        )
    else:
        if _time_semantics(first) != _time_semantics(second):
            reasons.append(
                "time semantics differ: masks come from different observations "
                "or temporal pairs"
            )
        first_phenomenon = _phenomenon(first)
        second_phenomenon = _phenomenon(second)
        if first_phenomenon is None or first_phenomenon != second_phenomenon:
            reasons.append(
                "target phenomena differ: the masks do not describe the same "
                "semantic quantity"
            )

    reasons.extend(_grid_reasons(first.mask, second.mask))

    if first.mask.value_semantics != second.mask.value_semantics:
        reasons.append("mask value semantics differ")

    return CompatibilityResult(comparable=not reasons, reasons=tuple(reasons))


def _load_binary_mask(asset: MaskAsset) -> np.ndarray:
    digest = hashlib.sha256(Path(asset.path).read_bytes()).hexdigest()
    if digest != asset.sha256:
        raise MaskCompatibilityError(
            CompatibilityResult(
                comparable=False,
                reasons=(f"mask asset {asset.asset_id} hash mismatch: immutable "
                         "asset bytes changed after registration",),
            )
        )
    with rasterio.open(asset.path) as dataset:
        if dataset.count != 1:
            raise MaskCompatibilityError(
                CompatibilityResult(
                    comparable=False,
                    reasons=(f"mask asset {asset.asset_id} must be single-band",),
                )
            )
        values = dataset.read(1)
    if not np.isin(values, (0, 1)).all():
        raise MaskCompatibilityError(
            CompatibilityResult(
                comparable=False,
                reasons=(f"mask asset {asset.asset_id} is not strictly binary 0/1",),
            )
        )
    return values.astype(bool)


def diagnostic_mask_agreement(
    first: MaskEvidence,
    second: MaskEvidence,
    valid: np.ndarray | None,
    *,
    tool_id: str,
    operation_id: str,
) -> AgreementEvidence:
    """Build diagnostic IoU evidence between two compatible mask evidences.

    Raises :class:`MaskCompatibilityError` when the masks are not comparable —
    the caller records ``not_comparable`` rather than fabricating a decision.
    """

    compatibility = assess_mask_compatibility(first, second)
    if not compatibility.comparable:
        raise MaskCompatibilityError(compatibility)

    first_values = _load_binary_mask(first.mask)
    second_values = _load_binary_mask(second.mask)
    valid_array = (
        np.ones(first_values.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    )
    if valid_array.shape != first_values.shape:
        raise MaskCompatibilityError(
            CompatibilityResult(
                comparable=False,
                reasons=("valid mask shape does not match the mask grid",),
            )
        )

    result = mask_agreement(first_values, second_values, valid_array)
    return AgreementEvidence(
        evidence_id=f"evidence_{uuid.uuid4().hex}",
        first_evidence_id=first.evidence_id,
        second_evidence_id=second.evidence_id,
        metric="mask_iou",
        value=result.iou,
        intersection_pixel_count=result.intersection,
        union_pixel_count=result.union,
        valid_pixel_count=result.valid_pixel_count,
        tool_id=tool_id,
        provenance=EvidenceProvenance(
            created_at=datetime.now(timezone.utc),
            operation_id=operation_id,
            input_asset_id=first.mask.asset_id,
            parent_evidence_ids=(first.evidence_id, second.evidence_id),
        ),
    )
