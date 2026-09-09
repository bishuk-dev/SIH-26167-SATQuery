"""Verification rules for requested scientific claims."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from satquery.agent.models import QueryIntent
from satquery.evidence.graph import EvidenceGraph
from satquery.evidence.models import (
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    MaskAsset,
    MeasurementEvidence,
    TemporalPairEvidence,
    VqaEvidence,
    VqaPrediction,
)
from satquery.ingestion.models import AffineTransform, GeoBounds, Modality
from satquery.verification.analysis import AnalysisVerifier


def _eid(index: int) -> str:
    return f"evidence_{index:032x}"


def _provenance(*parents: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation_id="test_operation",
        input_asset_id="asset_1",
        parent_evidence_ids=tuple(parents),
    )


def _domain() -> DomainAssessment:
    return DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=())


def _model() -> EvidenceModelProvenance:
    return EvidenceModelProvenance(
        registry_id="registry_v1",
        model_id="org/model",
        revision="a" * 40,
        checkpoint_sha256="b" * 64,
        preprocessing_profile="profile_v1",
        preprocessing_version="1.0.0",
    )


def _write_mask(path, values: np.ndarray) -> str:
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dataset:
        dataset.write(values.astype("uint8"), 1)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_asset(tmp_path, *, asset_id: str = "mask_1", missing: bool = False) -> MaskAsset:
    path = tmp_path / f"{asset_id}.tif"
    if missing:
        sha256 = "c" * 64
    else:
        sha256 = _write_mask(path, np.array([[1, 0], [1, 1]], dtype="uint8"))
    return MaskAsset(
        asset_id=asset_id,
        path=str(path),
        sha256=sha256,
        width=2,
        height=2,
        crs="EPSG:32643",
        transform=AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=20.0),
        bounds=GeoBounds(left=0.0, bottom=0.0, right=20.0, top=20.0),
        source_grid_observation_id="obs_t1",
    )


def _pair(t1: str = "obs_t1", t2: str = "obs_t2") -> TemporalPairEvidence:
    return TemporalPairEvidence(
        t1_observation_id=t1,
        t2_observation_id=t2,
        order_source="metadata",
    )


def _change_mask(
    tmp_path,
    *,
    evidence_id: str = _eid(1),
    asset_id: str = "mask_1",
    missing_artifact: bool = False,
    t1: str = "obs_t1",
    t2: str = "obs_t2",
) -> ChangeMaskEvidence:
    return ChangeMaskEvidence(
        evidence_id=evidence_id,
        target_class="vegetation",
        change_kind="gain",
        temporal=_pair(t1, t2),
        mask=_mask_asset(tmp_path, asset_id=asset_id, missing=missing_artifact),
        tool_id="temporal_difference_v1",
        domain=_domain(),
        provenance=_provenance(),
    )


def _measurement(
    source_id: str,
    *,
    evidence_id: str = _eid(2),
    value: float = 300.0,
    positive_pixels: int = 3,
) -> MeasurementEvidence:
    return MeasurementEvidence(
        evidence_id=evidence_id,
        source_evidence_id=source_id,
        value=value,
        unit="m2",
        method="projected_affine_determinant",
        calculation_crs="EPSG:32643",
        positive_pixel_count=positive_pixels,
        valid_pixel_count=4,
        tool_id="compute_mask_area_v1",
        provenance=_provenance(source_id),
    )


def _caption(*, evidence_id: str = _eid(3)) -> ChangeCaptionEvidence:
    return ChangeCaptionEvidence(
        evidence_id=evidence_id,
        temporal=_pair(),
        caption="Vegetation increased in the northern field.",
        model=_model(),
        domain=_domain(),
        provenance=_provenance(),
    )


def _vqa(*, evidence_id: str = _eid(4)) -> VqaEvidence:
    return VqaEvidence(
        evidence_id=evidence_id,
        prediction=VqaPrediction(answer="about three hectares", raw_score=0.7),
        source_observations=("obs_t1",),
        source_modalities=(Modality.OPTICAL,),
        model=_model(),
        domain=_domain(),
        provenance=_provenance(),
    )


def _intent(task_family: str = "CHANGE_MEASURE", requested_measurement: str | None = "area") -> QueryIntent:
    return QueryIntent(
        task_family=task_family,
        target_semantic="vegetation",
        requested_measurement=requested_measurement,
        temporal_direction="T1_TO_T2",
        spatial_request=True,
        matched_rule="test_rule",
        ambiguities=(),
    )


def _verify(intent: QueryIntent, graph: EvidenceGraph):
    return AnalysisVerifier().verify(intent, graph)


def _issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_verifier_rejects_hash_missing_artifacts(tmp_path) -> None:
    mask = _change_mask(tmp_path, missing_artifact=True)
    graph = EvidenceGraph(nodes=(mask,), edges=(), input_ids=("obs_t1", "obs_t2"))

    report = _verify(_intent(task_family="CHANGE_LOCALIZE", requested_measurement=None), graph)

    assert mask.evidence_id in report.invalid_evidence_ids
    assert "ARTIFACT_HASH_MISSING" in _issue_codes(report)


def test_measurement_sourced_from_caption_or_vqa_is_rejected(tmp_path) -> None:
    caption = _caption(evidence_id=_eid(3))
    vqa = _vqa(evidence_id=_eid(4))
    caption_measurement = _measurement(caption.evidence_id, evidence_id=_eid(5))
    vqa_measurement = _measurement(vqa.evidence_id, evidence_id=_eid(6))
    graph = EvidenceGraph(
        nodes=(caption, vqa, caption_measurement, vqa_measurement),
        edges=(),
        input_ids=("obs_t1", "obs_t2"),
    )

    report = _verify(_intent(), graph)

    assert caption_measurement.evidence_id in report.invalid_evidence_ids
    assert vqa_measurement.evidence_id in report.invalid_evidence_ids
    assert "MEASUREMENT_SOURCE_NOT_MASK" in _issue_codes(report)


def test_area_value_must_match_deterministic_mask_measurement(tmp_path) -> None:
    mask = _change_mask(tmp_path)
    measurement = _measurement(mask.evidence_id, value=42.0)
    graph = EvidenceGraph(nodes=(mask, measurement), edges=(), input_ids=("obs_t1", "obs_t2"))

    report = _verify(_intent(), graph)

    assert measurement.evidence_id in report.invalid_evidence_ids
    assert "MEASUREMENT_VALUE_MISMATCH" in _issue_codes(report)


def test_numeric_intent_requires_measurement_evidence(tmp_path) -> None:
    caption = _caption()
    graph = EvidenceGraph(nodes=(caption,), edges=(), input_ids=("obs_t1", "obs_t2"))

    report = _verify(_intent(), graph)

    assert not report.answered
    assert "NUMERIC_CLAIM_WITHOUT_MEASUREMENT" in _issue_codes(report)


def test_unknown_input_ids_reject_evidence(tmp_path) -> None:
    mask = _change_mask(tmp_path, t1="obs_unknown", t2="obs_t2")
    graph = EvidenceGraph(nodes=(mask,), edges=(), input_ids=("obs_t1", "obs_t2"))

    report = _verify(_intent(task_family="CHANGE_LOCALIZE", requested_measurement=None), graph)

    assert mask.evidence_id in report.invalid_evidence_ids
    assert "UNKNOWN_INPUT_ID" in _issue_codes(report)


def test_intent_not_answered_when_matching_evidence_is_absent(tmp_path) -> None:
    mask = _change_mask(tmp_path)
    graph = EvidenceGraph(nodes=(mask,), edges=(), input_ids=("obs_t1", "obs_t2"))

    report = _verify(_intent(task_family="CHANGE_DESCRIPTION", requested_measurement=None), graph)

    assert not report.answered
    assert "INTENT_NOT_ANSWERED" in _issue_codes(report)


def test_preserves_valid_independent_evidence_when_another_branch_fails(tmp_path) -> None:
    invalid_mask = _change_mask(tmp_path, evidence_id=_eid(1), asset_id="missing", missing_artifact=True)
    valid_mask = _change_mask(tmp_path, evidence_id=_eid(2), asset_id="valid")
    valid_measurement = _measurement(valid_mask.evidence_id, evidence_id=_eid(3))
    graph = EvidenceGraph(
        nodes=(invalid_mask, valid_mask, valid_measurement),
        edges=(),
        input_ids=("obs_t1", "obs_t2"),
    )

    report = _verify(_intent(), graph)

    assert invalid_mask.evidence_id in report.invalid_evidence_ids
    assert valid_mask.evidence_id in report.valid_evidence_ids
    assert valid_measurement.evidence_id in report.valid_evidence_ids
    assert report.answered
