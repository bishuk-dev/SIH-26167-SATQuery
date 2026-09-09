from __future__ import annotations

from datetime import datetime, timezone

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
from satquery.ingestion.models import Modality
from satquery.verification.analysis import AnalysisVerifier, VerificationReport
from satquery.agent.composer import AnswerComposer


def _eid(index: int) -> str:
    return f"evidence_{index:032x}"


def _provenance(*parents: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation_id="test_operation",
        input_asset_id="asset_1",
        parent_evidence_ids=tuple(parents),
    )


def _domain(status: DomainStatus = DomainStatus.IN_DOMAIN) -> DomainAssessment:
    return DomainAssessment(status=status, reasons=("verified test fixture",))


def _model() -> EvidenceModelProvenance:
    return EvidenceModelProvenance(
        registry_id="registry_v1",
        model_id="org/model",
        revision="a" * 40,
        checkpoint_sha256="b" * 64,
        preprocessing_profile="profile_v1",
        preprocessing_version="1.0.0",
    )


def _intent(
    task_family: str = "CHANGE_MEASURE",
    requested_measurement: str | None = "area",
) -> QueryIntent:
    return QueryIntent(
        task_family=task_family,
        target_semantic="water",
        requested_measurement=requested_measurement,
        temporal_direction="T1_TO_T2",
        spatial_request=True,
        matched_rule="test_rule",
    )


def _pair() -> TemporalPairEvidence:
    return TemporalPairEvidence(
        t1_observation_id="obs_t1",
        t2_observation_id="obs_t2",
        order_source="metadata",
    )


def _mask() -> ChangeMaskEvidence:
    return ChangeMaskEvidence(
        evidence_id=_eid(1),
        target_class="water",
        change_kind="gain",
        temporal=_pair(),
        mask=MaskAsset(
            asset_id="mask_1",
            path="/not/read/by/composer.tif",
            sha256="c" * 64,
            width=2,
            height=2,
            source_grid_observation_id="obs_t1",
        ),
        raw_model_score=0.83,
        tool_id="threshold_temporal_difference_v1",
        domain=_domain(DomainStatus.SHIFTED),
        warnings=("limited to verified mask extent",),
        provenance=_provenance(),
    )


def _measurement(source_id: str) -> MeasurementEvidence:
    return MeasurementEvidence(
        evidence_id=_eid(2),
        source_evidence_id=source_id,
        value=12.3456789,
        unit="ha",
        method="projected_affine_determinant",
        calculation_crs="EPSG:32643",
        positive_pixel_count=123,
        valid_pixel_count=456,
        tool_id="compute_mask_area_v1",
        warnings=("area excludes nodata",),
        provenance=_provenance(source_id),
    )


def _caption() -> ChangeCaptionEvidence:
    return ChangeCaptionEvidence(
        evidence_id=_eid(3),
        temporal=_pair(),
        caption="The model prose says approximately 999 ha changed.",
        model=_model(),
        domain=_domain(),
        provenance=_provenance(),
    )


def _vqa() -> VqaEvidence:
    return VqaEvidence(
        evidence_id=_eid(4),
        prediction=VqaPrediction(answer="about 888 hectares", raw_score=0.61),
        source_observations=("obs_t1",),
        source_modalities=(Modality.OPTICAL,),
        model=_model(),
        domain=_domain(),
        provenance=_provenance(),
    )


def _report(graph: EvidenceGraph, intent: QueryIntent) -> VerificationReport:
    # Composer must trust only the typed verification result it receives; for
    # this isolated template test we mark the evidence valid without requiring
    # mask bytes on disk.
    verified = tuple(
        {
            "evidence_id": node.evidence_id,
            "task": node.task,
            "status": "valid",
            "issues": (),
        }
        for node in graph.nodes
    )
    answered = intent.task_family in {"CHANGE_MEASURE", "SINGLE_VQA", "CHANGE_DESCRIPTION"}
    return VerificationReport(answered=answered, evidence=verified)


def test_measurement_answer_uses_typed_measurement_not_model_prose() -> None:
    mask = _mask()
    measurement = _measurement(mask.evidence_id)
    graph = EvidenceGraph(nodes=(mask, measurement, _caption(), _vqa()))
    intent = _intent()

    answer = AnswerComposer(display_precision=2).compose(
        "How many hectares changed?", intent, graph, _report(graph, intent)
    )

    assert answer.answered is True
    assert answer.outcome == "ALLOW"
    assert "12.35 ha" in answer.answer
    assert "999" not in answer.answer
    assert "888" not in answer.answer
    assert answer.measurements[0].value == 12.3456789
    assert answer.measurements[0].unit == "ha"
    assert answer.measurements[0].source_evidence_id == mask.evidence_id
    assert "limited to verified mask extent" in answer.limitations
    assert "area excludes nodata" in answer.limitations
    assert answer.uncalibrated_scores[0].label == "raw_model_score_uncalibrated"


def test_abstention_contains_no_positive_scientific_claim() -> None:
    graph = EvidenceGraph(nodes=())
    intent = _intent()
    report = VerificationReport(
        answered=False,
        evidence=(),
        issues=(
            {
                "code": "NUMERIC_CLAIM_WITHOUT_MEASUREMENT",
                "severity": "ERROR",
                "message": "area requests require MeasurementEvidence",
            },
        ),
    )

    answer = AnswerComposer().compose("How much area changed?", intent, graph, report)

    assert answer.answered is False
    assert answer.outcome == "ABSTAIN"
    assert answer.measurements == ()
    assert "verified area" not in answer.answer.casefold()
    assert "changed" not in answer.answer.casefold()
    assert answer.limitations == ("NUMERIC_CLAIM_WITHOUT_MEASUREMENT: area requests require MeasurementEvidence",)


def test_raw_model_score_is_labeled_uncalibrated_for_vqa() -> None:
    graph = EvidenceGraph(nodes=(_vqa(),))
    intent = _intent(task_family="SINGLE_VQA", requested_measurement=None)

    answer = AnswerComposer().compose(
        "What is visible?", intent, graph, _report(graph, intent)
    )

    assert "uncalibrated raw model score" in answer.answer
    assert answer.uncalibrated_scores[0].evidence_id == _eid(4)
    assert answer.uncalibrated_scores[0].value == 0.61
