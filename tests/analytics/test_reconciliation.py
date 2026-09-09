"""Mask-evidence compatibility and diagnostic-agreement tests (Phase 4 Task 10).

Agreement stays diagnostic: no confidence, no ALLOW/WARN/ABSTAIN mapping, no
claim that either producer is correct.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.analytics.reconciliation import (
    MaskCompatibilityError,
    assess_mask_compatibility,
    diagnostic_mask_agreement,
)
from satquery.evidence.models import (
    Modality,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    FloodMaskEvidence,
    ChangeMaskEvidence,
    MaskAsset,
    TemporalPairEvidence,
)

_AFFINE = {"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 4000000.0}
_NOW = datetime.now(timezone.utc)


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=_NOW, operation_id="op_test", input_asset_id="asset_test"
    )


def _mask_asset(tmp_path: Path, name: str, values: np.ndarray, *, georef: bool = True) -> MaskAsset:
    path = tmp_path / f"{name}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="uint8",
        crs="EPSG:32633" if georef else None,
        transform=rasterio.Affine(*(_AFFINE[k] for k in "abcdef")) if georef else None,
    ) as dataset:
        dataset.write(values.astype("uint8"), 1)
    import hashlib

    from satquery.ingestion.models import GeoBounds, Modality

    bounds = None
    if georef:
        bounds = GeoBounds(
            left=_AFFINE["c"],
            top=_AFFINE["f"],
            right=_AFFINE["c"] + values.shape[1] * _AFFINE["a"],
            bottom=_AFFINE["f"] + values.shape[0] * _AFFINE["e"],
        )
    return MaskAsset(
        asset_id=f"asset_{name}",
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        width=values.shape[1],
        height=values.shape[0],
        crs="EPSG:32633" if georef else None,
        transform=_AFFINE if georef else None,
        bounds=bounds,
        source_grid_observation_id="obs_shared",
    )


def _flood(
    tmp_path: Path,
    evidence_id: str,
    values: np.ndarray,
    *,
    observation_id: str = "obs_t2",
    target_class: str = "water_extent",
    georef: bool = True,
) -> FloodMaskEvidence:
    return FloodMaskEvidence(
        evidence_id=evidence_id,
        source_observation_id=observation_id,
        source_modality=Modality.SAR,
        target_class=target_class,  # type: ignore[arg-type]
        mask=_mask_asset(tmp_path, evidence_id, values, georef=georef),
        tool_id="flood_backend_v1",
        domain=DomainAssessment(status=DomainStatus.IN_DOMAIN),
        provenance=_provenance(),
    )


def _change(
    tmp_path: Path,
    evidence_id: str,
    values: np.ndarray,
    *,
    t1: str = "obs_t1",
    t2: str = "obs_t2",
    target_class: str = "water_gain",
    change_kind: str = "gain",
) -> ChangeMaskEvidence:
    return ChangeMaskEvidence(
        evidence_id=evidence_id,
        target_class=target_class,
        change_kind=change_kind,  # type: ignore[arg-type]
        temporal=TemporalPairEvidence(
            t1_observation_id=t1, t2_observation_id=t2, order_source="metadata"
        ),
        mask=_mask_asset(tmp_path, evidence_id, values),
        tool_id="deterministic_change_v1",
        domain=DomainAssessment(status=DomainStatus.IN_DOMAIN),
        provenance=_provenance(),
    )


# ---------------------------------------------------------------------------
# compatibility
# ---------------------------------------------------------------------------


def test_two_flood_masks_same_observation_are_compatible(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.array([[1, 0]], dtype="uint8"))
    b = _flood(tmp_path, "evidence_" + "b" * 32, np.array([[0, 1]], dtype="uint8"))
    result = assess_mask_compatibility(a, b)
    assert result.comparable
    assert result.reasons == ()


def test_flood_vs_change_is_not_comparable(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _change(tmp_path, "evidence_" + "b" * 32, np.ones((1, 2), dtype="uint8"))
    result = assess_mask_compatibility(a, b)
    assert not result.comparable
    assert any("phenomen" in reason or "time" in reason for reason in result.reasons)


def test_flood_masks_from_different_observations_not_comparable(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _flood(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((1, 2), dtype="uint8"),
        observation_id="obs_other",
    )
    result = assess_mask_compatibility(a, b)
    assert not result.comparable
    assert any("observation" in reason for reason in result.reasons)


def test_change_masks_with_different_temporal_pairs_not_comparable(tmp_path: Path) -> None:
    a = _change(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _change(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((1, 2), dtype="uint8"),
        t2="obs_t3",
    )
    result = assess_mask_compatibility(a, b)
    assert not result.comparable


def test_change_masks_with_different_phenomena_not_comparable(tmp_path: Path) -> None:
    a = _change(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _change(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((1, 2), dtype="uint8"),
        target_class="building_gain",
    )
    result = assess_mask_compatibility(a, b)
    assert not result.comparable
    assert any("phenomen" in reason for reason in result.reasons)


def test_differing_change_kind_not_comparable(tmp_path: Path) -> None:
    a = _change(
        tmp_path,
        "evidence_" + "a" * 32,
        np.ones((1, 2), dtype="uint8"),
        target_class="water_gain",
        change_kind="gain",
    )
    b = _change(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((1, 2), dtype="uint8"),
        target_class="water_gain",
        change_kind="loss",
    )
    result = assess_mask_compatibility(a, b)
    assert not result.comparable


def test_georeferenced_vs_pixel_space_not_comparable(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _flood(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((1, 2), dtype="uint8"),
        georef=False,
    )
    result = assess_mask_compatibility(a, b)
    assert not result.comparable
    assert any("grid" in reason for reason in result.reasons)


def test_shape_mismatch_not_comparable(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _flood(tmp_path, "evidence_" + "b" * 32, np.ones((2, 2), dtype="uint8"))
    result = assess_mask_compatibility(a, b)
    assert not result.comparable


def test_all_reasons_are_collected_not_first_only(tmp_path: Path) -> None:
    a = _flood(
        tmp_path,
        "evidence_" + "a" * 32,
        np.ones((1, 2), dtype="uint8"),
        target_class="water_extent",
    )
    b = _flood(
        tmp_path,
        "evidence_" + "b" * 32,
        np.ones((2, 3), dtype="uint8"),
        target_class="flood_extent",
    )
    result = assess_mask_compatibility(a, b)
    assert len(result.reasons) >= 2


# ---------------------------------------------------------------------------
# diagnostic agreement
# ---------------------------------------------------------------------------


def test_diagnostic_agreement_is_agreement_not_accuracy(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.array([[1, 1, 0]], dtype="uint8"))
    b = _flood(tmp_path, "evidence_" + "b" * 32, np.array([[1, 0, 0]], dtype="uint8"))
    evidence = diagnostic_mask_agreement(
        a,
        b,
        np.array([[True, True, True]]),
        tool_id="mask_agreement_v1",
        operation_id="op_agree",
    )
    assert evidence.metric == "mask_iou"
    assert evidence.interpretation == "agreement_not_accuracy"
    assert evidence.value == pytest.approx(1 / 2)
    assert evidence.intersection_pixel_count == 1
    assert evidence.union_pixel_count == 2
    assert evidence.valid_pixel_count == 3
    assert not hasattr(evidence, "confidence")


def test_diagnostic_agreement_rejects_incompatible_masks(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _change(tmp_path, "evidence_" + "b" * 32, np.ones((1, 2), dtype="uint8"))
    with pytest.raises(MaskCompatibilityError):
        diagnostic_mask_agreement(
            a,
            b,
            np.ones((1, 2), dtype=bool),
            tool_id="mask_agreement_v1",
            operation_id="op_agree",
        )


def test_diagnostic_agreement_empty_union_has_null_value(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.zeros((1, 2), dtype="uint8"))
    b = _flood(tmp_path, "evidence_" + "b" * 32, np.zeros((1, 2), dtype="uint8"))
    evidence = diagnostic_mask_agreement(
        a,
        b,
        np.ones((1, 2), dtype=bool),
        tool_id="mask_agreement_v1",
        operation_id="op_agree",
    )
    assert evidence.value is None
    assert evidence.union_pixel_count == 0


def test_diagnostic_agreement_detects_corrupted_mask_asset(tmp_path: Path) -> None:
    a = _flood(tmp_path, "evidence_" + "a" * 32, np.ones((1, 2), dtype="uint8"))
    b = _flood(tmp_path, "evidence_" + "b" * 32, np.ones((1, 2), dtype="uint8"))
    # corrupt the second mask file after registration
    with open(b.mask.path, "ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(MaskCompatibilityError, match="hash"):
        diagnostic_mask_agreement(
            a,
            b,
            np.ones((1, 2), dtype=bool),
            tool_id="mask_agreement_v1",
            operation_id="op_agree",
        )
