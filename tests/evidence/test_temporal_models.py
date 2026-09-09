"""Phase 4 temporal/scientific evidence contract tests (Task 3 scope only)."""

from __future__ import annotations

from datetime import datetime, timezone
from math import inf, nan

import pytest
from pydantic import ValidationError

from satquery.evidence.models import (
    AgreementEvidence,
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    FloodMaskEvidence,
    GroundingDetection,
    GroundingEvidence,
    MaskAsset,
    MeasurementEvidence,
    NormalizedBoundingBox,
    PixelBoundingBox,
    TemporalPairEvidence,
    VqaEvidence,
    VqaPrediction,
    WorldBoundingPolygon,
)
from satquery.ingestion.models import AffineTransform, GeoBounds, Modality


def _provenance(operation_id: str = "test_operation") -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation_id=operation_id,
        input_asset_id="asset_1",
    )


def _model_provenance() -> EvidenceModelProvenance:
    return EvidenceModelProvenance(
        registry_id="registry_v1",
        model_id="org/model",
        revision="a" * 40,
        checkpoint_sha256="b" * 64,
        preprocessing_profile="profile_v1",
        preprocessing_version="1.0.0",
    )


def _domain() -> DomainAssessment:
    return DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=())


def _mask(asset_id: str = "mask_1", **overrides) -> MaskAsset:
    values: dict = {
        "asset_id": asset_id,
        "path": "masks/mask_1.tif",
        "sha256": "c" * 64,
        "width": 128,
        "height": 128,
        "source_grid_observation_id": "obs_1",
        "value_semantics": "binary_0_1",
    }
    values.update(overrides)
    return MaskAsset.model_validate(values)


def _temporal_pair() -> TemporalPairEvidence:
    return TemporalPairEvidence(
        t1_observation_id="obs_t1",
        t2_observation_id="obs_t2",
        order_source="metadata",
    )


# ---------------------------------------------------------------------------
# MaskAsset
# ---------------------------------------------------------------------------


def test_mask_asset_accepts_pixel_space_mask() -> None:
    mask = _mask(crs=None, transform=None, bounds=None)

    assert mask.crs is None
    assert mask.transform is None
    assert mask.bounds is None
    assert mask.immutable is True


def test_mask_asset_accepts_georeferenced_mask() -> None:
    mask = _mask(
        crs="EPSG:32643",
        transform=AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=20.0),
        bounds=GeoBounds(left=0, bottom=0, right=1280, top=1280),
    )

    assert mask.crs == "EPSG:32643"


def test_mask_asset_rejects_partial_georeferencing() -> None:
    with pytest.raises(ValidationError):
        _mask(crs="EPSG:32643", transform=None, bounds=None)

    with pytest.raises(ValidationError):
        _mask(
            crs=None,
            transform=AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=20.0),
            bounds=None,
        )

    with pytest.raises(ValidationError):
        _mask(
            crs=None,
            transform=None,
            bounds=GeoBounds(left=0, bottom=0, right=1280, top=1280),
        )


def test_mask_asset_rejects_malformed_sha256() -> None:
    with pytest.raises(ValidationError):
        _mask(sha256="nothex")


def test_mask_asset_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValidationError):
        _mask(width=0)

    with pytest.raises(ValidationError):
        _mask(height=-5)


def test_mask_asset_rejects_empty_identity() -> None:
    with pytest.raises(ValidationError):
        _mask(asset_id="")

    with pytest.raises(ValidationError):
        _mask(path="")


# ---------------------------------------------------------------------------
# TemporalPairEvidence
# ---------------------------------------------------------------------------


def test_temporal_pair_accepts_valid_ordered_distinct_ids() -> None:
    pair = _temporal_pair()

    assert pair.t1_observation_id == "obs_t1"
    assert pair.order_source == "metadata"


def test_temporal_pair_rejects_same_observation() -> None:
    with pytest.raises(ValidationError):
        TemporalPairEvidence(
            t1_observation_id="obs_1",
            t2_observation_id="obs_1",
            order_source="metadata",
        )


def test_temporal_pair_rejects_unknown_order_source() -> None:
    with pytest.raises(ValidationError):
        TemporalPairEvidence(
            t1_observation_id="obs_t1",
            t2_observation_id="obs_t2",
            order_source="visual_inspection",
        )


def test_temporal_pair_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError):
        TemporalPairEvidence(
            t1_observation_id="", t2_observation_id="obs_2", order_source="metadata"
        )


# ---------------------------------------------------------------------------
# ChangeMaskEvidence
# ---------------------------------------------------------------------------


def _change_mask(**overrides) -> ChangeMaskEvidence:
    values: dict = {
        "evidence_id": "evidence_" + "0" * 32,
        "target_class": "high_res_structural_change",
        "change_kind": "symmetric_change",
        "temporal": _temporal_pair(),
        "mask": _mask(),
        "model": _model_provenance(),
        "tool_id": None,
        "domain": _domain(),
        "warnings": (),
        "provenance": _provenance("structural_change_segmentation"),
    }
    values.update(overrides)
    return ChangeMaskEvidence.model_validate(values)


def test_change_mask_accepts_model_producer() -> None:
    evidence = _change_mask()

    assert evidence.task == "change_localize"
    assert evidence.model is not None
    assert evidence.tool_id is None


def test_change_mask_accepts_deterministic_tool_producer() -> None:
    evidence = _change_mask(
        model=None,
        tool_id="change_vector_analysis_v1",
        change_kind="gain",
    )

    assert evidence.model is None
    assert evidence.tool_id == "change_vector_analysis_v1"


def test_change_mask_accepts_model_and_tool_producers() -> None:
    evidence = _change_mask(tool_id="compute_mask_area_v1")

    assert evidence.model is not None and evidence.tool_id is not None


def test_change_mask_rejects_missing_producer() -> None:
    with pytest.raises(ValidationError, match="producer"):
        _change_mask(model=None, tool_id=None)


def test_change_mask_rejects_nonfinite_raw_model_score() -> None:
    with pytest.raises(ValidationError):
        _change_mask(raw_model_score=nan)

    with pytest.raises(ValidationError):
        _change_mask(raw_model_score=inf)


def test_change_mask_allows_scores_outside_probability_range() -> None:
    evidence = _change_mask(raw_model_score=7.5)

    assert evidence.raw_model_score == pytest.approx(7.5)


def test_change_mask_rejects_unknown_change_kind() -> None:
    with pytest.raises(ValidationError):
        _change_mask(change_kind="mystery_change")


def test_change_mask_rejects_malformed_evidence_id() -> None:
    with pytest.raises(ValidationError):
        _change_mask(evidence_id="evidence_short")


# ---------------------------------------------------------------------------
# FloodMaskEvidence
# ---------------------------------------------------------------------------


def _flood_mask(**overrides) -> FloodMaskEvidence:
    values: dict = {
        "evidence_id": "evidence_" + "1" * 32,
        "source_observation_id": "obs_flood",
        "source_modality": Modality.SAR,
        "target_class": "water_or_flood_extent",
        "mask": _mask(asset_id="mask_2", path="masks/mask_2.tif"),
        "raw_model_score": 0.9,
        "model": _model_provenance(),
        "tool_id": None,
        "domain": _domain(),
        "warnings": (),
        "provenance": _provenance("flood_segmentation"),
    }
    values.update(overrides)
    return FloodMaskEvidence.model_validate(values)


def test_flood_mask_is_single_observation_evidence() -> None:
    evidence = _flood_mask()

    assert evidence.source_observation_id == "obs_flood"
    assert not hasattr(evidence, "temporal")


def test_flood_mask_target_semantics_are_explicit() -> None:
    assert _flood_mask(target_class="water_extent").target_class == "water_extent"
    assert _flood_mask(target_class="flood_extent").target_class == "flood_extent"
    with pytest.raises(ValidationError):
        _flood_mask(target_class="everything_wet")


def test_flood_mask_does_not_require_sentinel1_sar() -> None:
    evidence = _flood_mask(
        source_modality=Modality.OPTICAL,
        model=None,
        tool_id="water_index_tool_v1",
    )

    assert evidence.source_modality == Modality.OPTICAL


def test_flood_mask_requires_producer() -> None:
    with pytest.raises(ValidationError, match="producer"):
        _flood_mask(model=None, tool_id=None)


def test_flood_mask_rejects_nonfinite_raw_model_score() -> None:
    with pytest.raises(ValidationError):
        _flood_mask(raw_model_score=nan)


# ---------------------------------------------------------------------------
# ChangeCaptionEvidence
# ---------------------------------------------------------------------------


def _caption(**overrides) -> ChangeCaptionEvidence:
    values: dict = {
        "evidence_id": "evidence_" + "2" * 32,
        "temporal": _temporal_pair(),
        "caption": "A new building appears between the two scenes.",
        "model": _model_provenance(),
        "domain": _domain(),
        "warnings": (),
        "provenance": _provenance("change_captioning"),
    }
    values.update(overrides)
    return ChangeCaptionEvidence.model_validate(values)


def test_caption_preserves_pair_order() -> None:
    evidence = _caption()

    assert evidence.temporal.t1_observation_id == "obs_t1"
    assert evidence.temporal.t2_observation_id == "obs_t2"
    assert evidence.task == "change_captioning"


def test_caption_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        _caption(caption="   ")


def test_caption_rejects_measurement_like_fields() -> None:
    for field, value in (
        ("area_ha", 4.2),
        ("measurement", {"value": 1, "unit": "ha"}),
        ("mask", _mask(asset_id="mask_3").model_dump()),
        ("confidence", 0.9),
    ):
        payload = _caption().model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            ChangeCaptionEvidence.model_validate(payload)


def test_caption_requires_model_provenance() -> None:
    with pytest.raises(ValidationError):
        _caption(model=None)


# ---------------------------------------------------------------------------
# MeasurementEvidence
# ---------------------------------------------------------------------------


def _measurement(**overrides) -> MeasurementEvidence:
    values: dict = {
        "evidence_id": "evidence_" + "3" * 32,
        "measurement_type": "area",
        "source_evidence_id": "evidence_" + "0" * 32,
        "value": 3.14,
        "unit": "ha",
        "method": "projected_affine_determinant",
        "calculation_crs": "EPSG:32643",
        "positive_pixel_count": 314,
        "valid_pixel_count": 16384,
        "tool_id": "compute_mask_area_v1",
        "warnings": (),
        "provenance": _provenance("measurement"),
    }
    values.update(overrides)
    return MeasurementEvidence.model_validate(values)


def test_measurement_accepts_valid_area() -> None:
    evidence = _measurement()

    assert evidence.task == "measurement"
    assert evidence.unit == "ha"


def test_measurement_rejects_negative_or_nonfinite_area() -> None:
    with pytest.raises(ValidationError):
        _measurement(value=-1.0)

    with pytest.raises(ValidationError):
        _measurement(value=nan)


def test_measurement_rejects_positive_pixels_exceeding_valid() -> None:
    with pytest.raises(ValidationError):
        _measurement(positive_pixel_count=200, valid_pixel_count=100)


def test_measurement_rejects_unsupported_measurement_types() -> None:
    with pytest.raises(ValidationError):
        _measurement(measurement_type="distance")

    with pytest.raises(ValidationError):
        _measurement(measurement_type="count")


def test_measurement_requires_tool_and_crs() -> None:
    with pytest.raises(ValidationError):
        _measurement(tool_id="")

    with pytest.raises(ValidationError):
        _measurement(calculation_crs="")


# ---------------------------------------------------------------------------
# AgreementEvidence
# ---------------------------------------------------------------------------


def _agreement(**overrides) -> AgreementEvidence:
    values: dict = {
        "evidence_id": "evidence_" + "4" * 32,
        "first_evidence_id": "evidence_" + "0" * 32,
        "second_evidence_id": "evidence_" + "1" * 32,
        "metric": "mask_iou",
        "value": 0.75,
        "intersection_pixel_count": 75,
        "union_pixel_count": 100,
        "valid_pixel_count": 16384,
        "tool_id": "mask_agreement_v1",
        "warnings": (),
        "provenance": _provenance("mask_agreement"),
    }
    values.update(overrides)
    return AgreementEvidence.model_validate(values)


def test_agreement_accepts_valid_iou() -> None:
    evidence = _agreement()

    assert evidence.metric == "mask_iou"
    assert evidence.value == pytest.approx(0.75)


def test_agreement_requires_distinct_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        _agreement(second_evidence_id="evidence_" + "0" * 32)


def test_agreement_empty_union_requires_null_value() -> None:
    evidence = _agreement(
        value=None,
        intersection_pixel_count=0,
        union_pixel_count=0,
    )

    assert evidence.value is None

    with pytest.raises(ValidationError):
        _agreement(value=0.0, intersection_pixel_count=0, union_pixel_count=0)


def test_agreement_nonempty_union_requires_value_in_unit_range() -> None:
    with pytest.raises(ValidationError):
        _agreement(value=None)

    with pytest.raises(ValidationError):
        _agreement(value=1.5)


def test_agreement_rejects_invalid_count_relationships() -> None:
    with pytest.raises(ValidationError):
        _agreement(
            intersection_pixel_count=200,
            union_pixel_count=100,
        )

    with pytest.raises(ValidationError):
        _agreement(union_pixel_count=20000, valid_pixel_count=16384)


def test_agreement_interpretation_is_diagnostic_only() -> None:
    assert _agreement().interpretation == "agreement_not_accuracy"
    with pytest.raises(ValidationError):
        _agreement(interpretation="accuracy")


def test_agreement_forbids_decision_fields() -> None:
    for field, value in (
        ("confidence", 0.9),
        ("aggregate_confidence", 0.9),
        ("outcome", "ALLOW"),
        ("allow", True),
        ("abstain", False),
        ("threshold", 0.5),
        ("decision", "reject"),
    ):
        payload = _agreement().model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            AgreementEvidence.model_validate(payload)


# ---------------------------------------------------------------------------
# ID conventions and backward compatibility
# ---------------------------------------------------------------------------


def test_evidence_id_patterns_enforced() -> None:
    for factory in (_change_mask, _flood_mask, _caption, _measurement, _agreement):
        with pytest.raises(ValidationError):
            factory(evidence_id="not-an-evidence-id")


def test_existing_vqa_evidence_still_validates() -> None:
    evidence = VqaEvidence(
        evidence_id="evidence_" + "5" * 32,
        prediction=VqaPrediction(answer="Yes.", raw_score=0.8),
        source_observations=("obs_1",),
        source_modalities=(Modality.OPTICAL,),
        model=_model_provenance(),
        domain=_domain(),
        provenance=_provenance("single_image_vqa"),
    )

    assert evidence.task == "single_image_vqa"


def test_existing_grounding_evidence_still_validates() -> None:
    detection = GroundingDetection(
        detection_id="detection_" + "6" * 32,
        phrase="building",
        raw_score=0.9,
        model_input_box=PixelBoundingBox(
            coordinate_space="model_input",
            x_min=0,
            y_min=0,
            x_max=10,
            y_max=10,
            image_width=256,
            image_height=256,
        ),
        source_pixel_box=PixelBoundingBox(
            coordinate_space="source_image",
            x_min=0,
            y_min=0,
            x_max=10,
            y_max=10,
            image_width=1024,
            image_height=1024,
        ),
        normalized_box=NormalizedBoundingBox(x_min=0, y_min=0, x_max=0.1, y_max=0.1),
        world_polygon=WorldBoundingPolygon(
            crs="EPSG:4326",
            coordinates=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ),
    )

    evidence = GroundingEvidence(
        evidence_id="evidence_" + "7" * 32,
        query="building",
        detections=(detection,),
        source_observations=("obs_1",),
        source_modalities=(Modality.OPTICAL,),
        model=_model_provenance(),
        domain=_domain(),
        provenance=_provenance("text_guided_grounding"),
    )

    assert evidence.task == "text_guided_grounding"


def test_package_exports_contain_phase4_contracts() -> None:
    import satquery.evidence

    for name in (
        "MaskAsset",
        "TemporalPairEvidence",
        "ChangeMaskEvidence",
        "FloodMaskEvidence",
        "ChangeCaptionEvidence",
        "MeasurementEvidence",
        "AgreementEvidence",
        "DomainStatus",
        "VqaEvidence",
    ):
        assert hasattr(satquery.evidence, name), name
        assert name in satquery.evidence.__all__, name
