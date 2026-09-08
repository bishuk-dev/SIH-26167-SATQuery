from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from satquery.analytics.reconciliation import reconcile_masks
from satquery.evidence.models import (
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    MaskAsset,
    TemporalPairEvidence,
)


def _evidence(root: Path, name: str, values: np.ndarray) -> ChangeMaskEvidence:
    path = root / f"{name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8", crs="EPSG:32643", transform=(10, 0, 0, 0, -10, 20)) as dataset:
        dataset.write(values.astype("uint8"), 1)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return ChangeMaskEvidence(
        evidence_id="evidence_" + hashlib.sha256(name.encode()).hexdigest()[:32],
        target_class="high_res_structural_change",
        change_kind="symmetric_change",
        temporal=TemporalPairEvidence(t1_observation_id="t1", t2_observation_id="t2", order_source="metadata"),
        mask=MaskAsset(asset_id=name, path=str(path), sha256=digest, width=2, height=2, crs="EPSG:32643", transform={"a": 10.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": -10.0, "f": 20.0}, source_grid_observation_id="t1", value_semantics="binary_0_1"),
        raw_model_score=None,
        model=None,
        tool_id="fixture",
        domain=DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
        warnings=(),
        provenance=EvidenceProvenance(created_at=datetime.now(timezone.utc), operation_id="fixture", input_asset_id="asset"),
    )


def test_compatible_agreeing_masks_allow_with_separate_scores(tmp_path: Path) -> None:
    learned = _evidence(tmp_path, "learned", np.array([[1, 0], [0, 0]]))
    deterministic = _evidence(tmp_path, "deterministic", np.array([[1, 0], [0, 0]]))

    result = reconcile_masks(learned, deterministic)

    assert result.outcome == "ALLOW"
    assert result.agreement is not None
    assert result.agreement.interpretation == "agreement_not_accuracy"
    assert result.aggregate_confidence is None


def test_strong_conflict_abstains_without_destroying_independent_evidence(tmp_path: Path) -> None:
    learned = _evidence(tmp_path, "learned", np.array([[1, 1], [1, 1]]))
    deterministic = _evidence(tmp_path, "deterministic", np.zeros((2, 2)))

    result = reconcile_masks(learned, deterministic)

    assert result.outcome == "ABSTAIN"
    assert result.evidence_ids == (learned.evidence_id, deterministic.evidence_id)


def test_missing_learned_model_degrades_to_deterministic_with_warning(tmp_path: Path) -> None:
    deterministic = _evidence(tmp_path, "deterministic", np.ones((2, 2)))

    result = reconcile_masks(None, deterministic)

    assert result.outcome == "ALLOW_WITH_WARNING"
    assert "LEARNED_MODEL_NOT_APPLICABLE" in result.warnings
