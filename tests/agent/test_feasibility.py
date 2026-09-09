from datetime import datetime, timezone

import pytest

from satquery.agent.feasibility import FeasibilityValidator
from satquery.agent.models import QueryIntent
from satquery.geo.models import (
    CrsCompatibility,
    GridCompatibility,
    ModalityCompatibility,
    ModalityPairType,
    OverlapCompatibility,
    PairCompatibility,
    PairResult,
    RegistrationStatus,
    CompatibilityStatus,
    TemporalCompatibility,
)
from satquery.ingestion.models import (
    AffineTransform,
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)
from satquery.registry.tools import CapabilityState, RuntimeCapability


_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def observation(
    *,
    observation_id: str = "obs_1",
    driver: str = "GTiff",
    crs: str | None = "EPSG:32643",
    modality: Modality = Modality.MULTISPECTRAL,
    descriptions: tuple[str, ...] = ("red", "green", "blue", "nir"),
    polarizations: tuple[str, ...] = (),
    sensor_name: str | None = "fixture",
    acquisition_time: datetime | None = _NOW,
) -> ObservationState:
    bands = tuple(
        BandMetadata(index=index, description=description, dtype="uint8")
        for index, description in enumerate(descriptions, start=1)
    )
    return ObservationState(
        observation_id=observation_id,
        source_asset=SourceAsset(
            asset_id=f"asset_{observation_id}",
            original_name=f"{observation_id}.tif",
            path=f"{observation_id}.tif",
            sha256="a" * 64,
        ),
        raster=RasterMetadata(
            driver=driver,
            width=10,
            height=10,
            band_count=len(bands),
            dtypes=tuple("uint8" for _ in bands),
            nodata=tuple(None for _ in bands),
        ),
        sensor=SensorMetadata(
            modality=modality,
            sensor_name=sensor_name,
            bands=bands,
            polarizations=polarizations,
        ),
        geo=GeoMetadata(
            crs=crs,
            transform=AffineTransform(a=1, b=0, c=0, d=0, e=-1, f=10),
            bounds=GeoBounds(left=0, bottom=0, right=10, top=10),
            native_gsd_x=1,
            native_gsd_y=1,
        ),
        temporal=TemporalMetadata(acquisition_time=acquisition_time),
        validity=ValidityMetadata(has_crs=crs is not None, has_transform=True, has_nodata=False),
        provenance=ObservationProvenance(created_at=_NOW, ingestion_version="test"),
    )


def pair(*, overlap: float = 1.0, aligned: bool | None = True, order_known: bool = True) -> PairCompatibility:
    return PairCompatibility(
        observation_a="obs_1",
        observation_b="obs_2",
        overlap=OverlapCompatibility(known=True, overlap_fraction=overlap, sufficient=overlap > 0),
        crs=CrsCompatibility(equal=True, transformable=True),
        grid=GridCompatibility(same_shape=True, same_resolution=True, aligned=aligned),
        temporal=TemporalCompatibility(
            order_known=order_known,
            first="obs_1" if order_known else None,
            second="obs_2" if order_known else None,
            time_delta_seconds=1 if order_known else None,
        ),
        modality=ModalityCompatibility(
            pair_type=ModalityPairType.TEMPORAL_SAME_MODALITY,
            compatible=True,
        ),
        registration=RegistrationStatus.VERIFIED if aligned else RegistrationStatus.UNKNOWN,
        result=PairResult(
            status=CompatibilityStatus.PASS if overlap > 0 else CompatibilityStatus.FAIL
        ),
    )


def capability(capability_id: str, status: CapabilityState = CapabilityState.AVAILABLE) -> RuntimeCapability:
    return RuntimeCapability(capability_id=capability_id, status=status)


def intent(**overrides: object) -> QueryIntent:
    values = {
        "task_family": "MEASURE",
        "requested_measurement": "area",
        "matched_rule": "test",
        "spatial_request": True,
    }
    values.update(overrides)
    return QueryIntent.model_validate(values)


validator = FeasibilityValidator()


def test_one_observation_change_requests_temporal_pair():
    result = validator.validate(
        intent(task_family="CHANGE_VQA", requested_measurement=None),
        [observation()],
        None,
        None,
        [],
    )
    assert result.outcome == "REQUEST_INPUT"
    assert "MISSING_TEMPORAL_PAIR" in {issue.code for issue in result.failures}


def test_rgb_without_nir_rejects_ndvi():
    result = validator.validate(
        intent(requested_measurement="ndvi"),
        [observation(descriptions=("red", "green", "blue"))],
        None,
        None,
        [capability("spectral_index_ndvi")],
    )
    assert result.outcome == "REJECT"
    assert "MISSING_REQUIRED_BAND" in {issue.code for issue in result.failures}


def test_area_without_crs_rejects():
    result = validator.validate(
        intent(),
        [observation(crs=None)],
        None,
        None,
        [capability("mask_area_measurement")],
    )
    assert result.outcome == "REJECT"
    assert "CRS_REQUIRED_FOR_MEASUREMENT" in {issue.code for issue in result.failures}


def test_unknown_sar_semantics_requests_input():
    sar = observation(
        modality=Modality.SAR,
        descriptions=("channel 1", "channel 2"),
        polarizations=(),
        sensor_name=None,
    )
    result = validator.validate(
        intent(task_family="CHANGE_VQA", target_semantic="sar", requested_measurement=None),
        [sar, observation(observation_id="obs_2")],
        pair(),
        None,
        [capability("temporal_difference")],
    )
    assert result.outcome == "REQUEST_INPUT"
    assert "UNKNOWN_SAR_POLARIZATION" in {issue.code for issue in result.failures}


def test_non_overlapping_temporal_pair_rejects():
    result = validator.validate(
        intent(task_family="CHANGE_LOCALIZE", requested_measurement=None),
        [observation(), observation(observation_id="obs_2")],
        pair(overlap=0),
        None,
        [capability("temporal_difference")],
    )
    assert result.outcome == "REJECT"
    assert "NO_SPATIAL_OVERLAP" in {issue.code for issue in result.failures}


def test_misaligned_pixelwise_pair_rejects():
    result = validator.validate(
        intent(task_family="CHANGE_MEASURE", requested_measurement="area"),
        [observation(), observation(observation_id="obs_2")],
        pair(aligned=False),
        None,
        [capability("mask_area_measurement")],
    )
    assert result.outcome == "REJECT"
    assert "PAIR_ALIGNMENT_INVALID" in {issue.code for issue in result.failures}


def test_jpeg_vqa_allows_with_warning():
    result = validator.validate(
        intent(task_family="SINGLE_VQA", requested_measurement=None),
        [observation(driver="JPEG")],
        None,
        None,
        [capability("single_image_vqa")],
    )
    assert result.outcome == "ALLOW_WITH_WARNING"
    assert result.warnings[0].code == "GEOREFERENCE_UNAVAILABLE"


def test_unavailable_model_abstains():
    result = validator.validate(
        intent(task_family="SINGLE_VQA", requested_measurement=None),
        [observation()],
        None,
        None,
        [capability("single_image_vqa", CapabilityState.UNAVAILABLE_RUNTIME)],
    )
    assert result.outcome == "ABSTAIN"
    assert result.failures[0].code == "MODEL_UNAVAILABLE"


def test_valid_deterministic_area_is_allowed_and_capability_is_exposed():
    result = validator.validate(
        intent(),
        [observation()],
        None,
        None,
        [capability("mask_area_measurement")],
    )
    assert result.outcome == "ALLOW"
    assert result.allowed_capability_ids == ("mask_area_measurement",)
    assert all(check.passed for check in result.checks)


def test_all_checks_are_preserved_before_precedence_is_applied():
    result = validator.validate(
        intent(task_family="CHANGE_MEASURE", requested_measurement="ndvi"),
        [observation(descriptions=("red", "green", "blue"))],
        None,
        None,
        [capability("temporal_difference", CapabilityState.UNAVAILABLE_RUNTIME)],
    )
    assert result.outcome == "REJECT"
    assert {check.check_id for check in result.checks} == {
        "temporal_pair",
        "temporal_order",
        "spatial_overlap",
        "pixel_grid",
        "required_bands",
        "measurement_crs",
        "sar_semantics",
        "image_quality",
        "capabilities",
    }
