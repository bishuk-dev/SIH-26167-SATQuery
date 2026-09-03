from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from pydantic import ValidationError
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import transform_bounds

from satquery.geo import CompatibilityStatus, PairValidator, RegistrationStatus
from satquery.geo.models import (
    ModalityPairType,
    OverlapCompatibility,
    TemporalCompatibility,
)
from satquery.ingestion import RasterInspector
from satquery.ingestion.models import ObservationState


def _observation(
    tmp_path: Path,
    observation_id: str,
    *,
    transform: Affine,
    crs: str = "EPSG:3857",
    width: int = 10,
    height: int = 10,
    modality: str = "optical",
    acquisition_time: str | None = "2026-01-01T00:00:00+00:00",
) -> ObservationState:
    path = tmp_path / f"{observation_id}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((1, height, width), dtype=np.uint8))
        tags = {"MODALITY": modality}
        if acquisition_time is not None:
            tags["ACQUISITION_TIME"] = acquisition_time
        dataset.update_tags(**tags)

    return RasterInspector().inspect(
        path,
        observation_id=observation_id,
        asset_id=f"asset-{observation_id}",
    )


def test_same_grid_pair_passes_and_has_known_temporal_order(tmp_path: Path) -> None:
    transform = from_origin(0, 100, 10, 10)
    earlier = _observation(tmp_path, "earlier", transform=transform)
    later = _observation(
        tmp_path,
        "later",
        transform=transform,
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(later, earlier)

    assert result.result.status is CompatibilityStatus.PASS
    assert result.result.reasons == ()
    assert result.crs.equal is True
    assert result.overlap.overlap_fraction == pytest.approx(1.0)
    assert result.grid.same_shape is True
    assert result.grid.same_resolution is True
    assert result.grid.aligned is True
    assert result.registration is RegistrationStatus.VERIFIED
    assert result.temporal.order_known is True
    assert result.temporal.first == "earlier"
    assert result.temporal.second == "later"
    assert result.temporal.time_delta_seconds == 86_400
    assert result.modality.pair_type is ModalityPairType.TEMPORAL_SAME_MODALITY


def test_different_transformable_crs_warns_without_rejecting_pair(
    tmp_path: Path,
) -> None:
    geographic = _observation(
        tmp_path,
        "geographic",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.1, 0.1),
    )
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", "EPSG:3857", 0, 0, 1, 1
    )
    projected = _observation(
        tmp_path,
        "projected",
        crs="EPSG:3857",
        transform=from_bounds(left, bottom, right, top, 10, 10),
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(geographic, projected)

    assert result.result.status is CompatibilityStatus.WARN
    assert result.crs.equal is False
    assert result.crs.transformable is True
    assert result.overlap.known is True
    assert result.overlap.overlap_fraction == pytest.approx(1.0)
    assert result.grid.same_resolution is None
    assert result.grid.aligned is None
    assert result.registration is RegistrationStatus.APPROXIMATE
    assert "CRS_TRANSFORM_REQUIRED" in result.result.reasons


def test_partial_overlap_uses_smaller_footprint_as_denominator(tmp_path: Path) -> None:
    first = _observation(tmp_path, "first", transform=from_origin(0, 100, 10, 10))
    second = _observation(
        tmp_path,
        "second",
        transform=from_origin(50, 100, 10, 10),
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(first, second)

    assert result.overlap.overlap_fraction == pytest.approx(0.5)
    assert result.overlap.sufficient is True
    assert result.result.status is CompatibilityStatus.PASS


def test_no_overlap_fails(tmp_path: Path) -> None:
    first = _observation(tmp_path, "first", transform=from_origin(0, 100, 10, 10))
    second = _observation(
        tmp_path,
        "second",
        transform=from_origin(200, 100, 10, 10),
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(first, second)

    assert result.result.status is CompatibilityStatus.FAIL
    assert result.overlap.overlap_fraction == 0
    assert result.registration is RegistrationStatus.INVALID
    assert "NO_SPATIAL_OVERLAP" in result.result.reasons


def test_different_resolution_warns_but_remains_analyzable(tmp_path: Path) -> None:
    coarse = _observation(tmp_path, "coarse", transform=from_origin(0, 100, 10, 10))
    fine = _observation(
        tmp_path,
        "fine",
        transform=from_origin(0, 100, 5, 5),
        width=20,
        height=20,
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(coarse, fine)

    assert result.result.status is CompatibilityStatus.WARN
    assert result.grid.same_shape is False
    assert result.grid.same_resolution is False
    assert result.grid.aligned is False
    assert result.registration is RegistrationStatus.APPROXIMATE
    assert "RESOLUTION_MISMATCH" in result.result.reasons


def test_integer_shift_is_aligned_but_fractional_shift_warns(tmp_path: Path) -> None:
    reference = _observation(
        tmp_path, "reference", transform=from_origin(0, 100, 10, 10)
    )
    aligned = _observation(
        tmp_path,
        "aligned",
        transform=from_origin(20, 100, 10, 10),
        acquisition_time="2026-01-02T00:00:00+00:00",
    )
    shifted = _observation(
        tmp_path,
        "shifted",
        transform=from_origin(5, 100, 10, 10),
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    aligned_result = PairValidator().validate(reference, aligned)
    shifted_result = PairValidator().validate(reference, shifted)

    assert aligned_result.grid.aligned is True
    assert aligned_result.result.status is CompatibilityStatus.PASS
    assert shifted_result.grid.aligned is False
    assert shifted_result.result.status is CompatibilityStatus.WARN
    assert "GRID_MISALIGNED" in shifted_result.result.reasons


def test_unknown_time_and_optical_sar_modality_are_preserved(tmp_path: Path) -> None:
    transform = from_origin(0, 100, 10, 10)
    optical = _observation(
        tmp_path,
        "optical",
        transform=transform,
        modality="optical",
        acquisition_time=None,
    )
    sar = _observation(
        tmp_path,
        "sar",
        transform=transform,
        modality="sar",
        acquisition_time="2026-01-02T00:00:00+00:00",
    )

    result = PairValidator().validate(optical, sar)

    assert result.temporal.order_known is False
    assert result.temporal.first is None
    assert result.modality.pair_type is ModalityPairType.OPTICAL_SAR
    assert result.modality.compatible is True
    assert result.result.status is CompatibilityStatus.WARN
    assert "TEMPORAL_ORDER_UNKNOWN" in result.result.reasons


def test_pair_component_schemas_reject_contradictory_known_states() -> None:
    with pytest.raises(ValidationError, match="known overlap requires"):
        OverlapCompatibility(known=True)

    with pytest.raises(ValidationError, match="unknown temporal order"):
        TemporalCompatibility(
            order_known=False,
            first="first",
            second="second",
            time_delta_seconds=1,
        )
