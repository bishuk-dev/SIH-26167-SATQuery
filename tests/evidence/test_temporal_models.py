from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from satquery.ingestion.models import AffineTransform
from satquery.evidence.models import (
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    MaskAsset,
    TemporalPairEvidence,
)


def _temporal() -> TemporalPairEvidence:
    return TemporalPairEvidence(
        t1_observation_id="obs-t1",
        t2_observation_id="obs-t2",
        order_source="metadata",
    )


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=datetime.now(timezone.utc),
        operation_id="operation-1",
        input_asset_id="asset-1",
    )


def _mask() -> MaskAsset:
    return MaskAsset(
        asset_id="mask-1",
        path="derived/mask.tif",
        sha256="0" * 64,
        width=2,
        height=2,
        crs="EPSG:32643",
        transform=AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=0.0),
        source_grid_observation_id="obs-t1",
        value_semantics="binary_0_1",
    )


def _change_mask() -> dict[str, object]:
    return {
        "evidence_id": "evidence_" + "1" * 32,
        "target_class": "high_res_structural_change",
        "change_kind": "symmetric_change",
        "temporal": _temporal(),
        "mask": _mask(),
        "raw_model_score": 0.8,
        "model": None,
        "tool_id": "temporal_difference_v1",
        "domain": DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
        "warnings": (),
        "provenance": _provenance(),
    }


def _caption() -> dict[str, object]:
    return {
        "evidence_id": "evidence_" + "2" * 32,
        "caption": "New structures appear in the scene.",
        "temporal": _temporal(),
        "model": None,
        "domain": DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
        "warnings": (),
        "provenance": _provenance(),
    }


def test_change_mask_requires_ordered_distinct_observations() -> None:
    with pytest.raises(ValidationError):
        TemporalPairEvidence(
            t1_observation_id="obs-1",
            t2_observation_id="obs-1",
            order_source="metadata",
        )


def test_caption_cannot_carry_measurement_fields() -> None:
    with pytest.raises(ValidationError):
        ChangeCaptionEvidence.model_validate({**_caption(), "area_ha": 4.2})


def test_mask_requires_binary_semantics_and_source_grid() -> None:
    evidence = ChangeMaskEvidence.model_validate(_change_mask())
    assert evidence.mask.value_semantics == "binary_0_1"
    assert evidence.mask.source_grid_observation_id == evidence.temporal.t1_observation_id
