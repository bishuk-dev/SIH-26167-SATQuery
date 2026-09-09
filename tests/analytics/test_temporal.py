"""Deterministic temporal differencing and common-grid preparation tests."""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.analytics.exceptions import (
    GridPreparationError,
    InvalidRasterArrayError,
    NoSpatialOverlapError,
)
from satquery.analytics.temporal import (
    prepare_common_grid,
    temporal_difference,
    threshold_temporal_difference,
)
from satquery.ingestion.models import (
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    MetadataQuality,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_geotiff(
    path: Path,
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: str | None,
    values: np.ndarray | None = None,
    nodata: float | None = None,
    count: int = 1,
    dtype: str = "float32",
) -> None:
    if values is None:
        values = np.ones((count, height, width), dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=values.dtype if values is not None else dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(values)


def _observation(path: Path, observation_id: str) -> ObservationState:
    with rasterio.open(path) as dataset:
        crs = dataset.crs
        transform = dataset.transform
        width = dataset.width
        height = dataset.height
        count = dataset.count
        nodata = dataset.nodata
        bounds = dataset.bounds
    transform_dict = {
        "a": transform.a,
        "b": transform.b,
        "c": transform.c,
        "d": transform.d,
        "e": transform.e,
        "f": transform.f,
    }
    return ObservationState(
        observation_id=observation_id,
        source_asset=SourceAsset(
            asset_id=observation_id + "-asset",
            original_name=path.name,
            path=str(path),
            sha256=_sha256(path),
        ),
        raster=RasterMetadata(
            driver="GTiff",
            width=width,
            height=height,
            band_count=count,
            dtypes=tuple(dataset.dtypes),
            nodata=tuple(nodata) if isinstance(nodata, (list, tuple)) else (nodata,) * count,
        ),
        sensor=SensorMetadata(
            modality=Modality.MULTISPECTRAL,
            sensor_name="FixtureSat",
            bands=tuple(
                BandMetadata(index=i, description=f"band_{i}", dtype="float32")
                for i in range(1, count + 1)
            ),
        ),
        geo=GeoMetadata(
            crs=str(crs) if crs else None,
            transform=transform_dict,
            bounds=GeoBounds(
                left=bounds.left, bottom=bounds.bottom, right=bounds.right, top=bounds.top
            ),
            native_gsd_x=abs(transform.a) or None,
            native_gsd_y=abs(transform.e) or None,
            units="m" if crs and "326" in str(crs) else "degree",
        ),
        temporal=TemporalMetadata(
            acquisition_time=datetime(2020, 1, 1, tzinfo=timezone.utc)
        ),
        validity=ValidityMetadata(
            has_crs=crs is not None,
            has_transform=True,
            has_nodata=nodata is not None,
            metadata_quality=MetadataQuality.HIGH,
        ),
        provenance=ObservationProvenance(
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            ingestion_version="test",
        ),
    )


# ---------------------------------------------------------------------------
# temporal_difference / threshold_temporal_difference
# ---------------------------------------------------------------------------


def test_temporal_difference_is_signed_t2_minus_t1() -> None:
    t1 = np.array([[1.0, 2.0]])
    t2 = np.array([[4.0, 0.5]])
    valid = np.array([[True, True]])

    result = temporal_difference(t1, t2, valid)

    np.testing.assert_allclose(result.compressed(), [3.0, -1.5])


def test_temporal_difference_identity_is_exactly_zero_on_valid_pixels() -> None:
    values = np.array([[1.25, 2.5], [3.75, 4.0]])
    valid = np.array([[True, True], [True, False]])

    result = temporal_difference(values, values, valid)

    assert not result.mask[0, 0]
    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0
    assert result.mask[1, 1]


def test_temporal_difference_masks_invalid_and_nonfinite() -> None:
    t1 = np.array([[1.0, 2.0], [3.0, np.nan]])
    t2 = np.array([[np.inf, 2.0], [3.0, 4.0]])
    valid = np.array([[True, False], [True, True]])

    result = temporal_difference(t1, t2, valid)

    assert result.count() == 1
    np.testing.assert_allclose(result.compressed(), [0.0])


def test_temporal_difference_rejects_bad_shapes_and_dimensions() -> None:
    with pytest.raises(InvalidRasterArrayError):
        temporal_difference(np.ones((2, 2)), np.ones((3, 2)), np.ones((2, 2), dtype=bool))
    with pytest.raises(InvalidRasterArrayError):
        temporal_difference(np.ones((2, 2, 2)), np.ones((2, 2, 2)), np.ones((2, 2, 2), dtype=bool))


def test_temporal_difference_does_not_mutate_inputs() -> None:
    t1 = np.array([[1.0]])
    t2 = np.array([[2.0]])
    valid = np.array([[True]])
    t1_before, t2_before = t1.copy(), t2.copy()

    temporal_difference(t1, t2, valid)

    np.testing.assert_array_equal(t1, t1_before)
    np.testing.assert_array_equal(t2, t2_before)


def test_threshold_directions() -> None:
    t1 = np.array([[0.0, 5.0, 0.0, 1.0]])
    t2 = np.array([[2.0, 3.0, -3.0, 1.0]])
    valid = np.ones((1, 4), dtype=bool)

    absolute = threshold_temporal_difference(
        t1, t2, threshold=1.5, direction="absolute", valid=valid
    )
    increase = threshold_temporal_difference(
        t1, t2, threshold=1.5, direction="increase", valid=valid
    )
    decrease = threshold_temporal_difference(
        t1, t2, threshold=1.5, direction="decrease", valid=valid
    )

    np.testing.assert_array_equal(absolute, [[True, True, True, False]])
    np.testing.assert_array_equal(increase, [[True, False, False, False]])
    np.testing.assert_array_equal(decrease, [[False, True, True, False]])
    assert absolute.dtype == np.bool_


def test_threshold_invalid_pixels_are_always_false() -> None:
    t1 = np.array([[0.0, 1.0]])
    t2 = np.array([[5.0, 1.0]])
    valid = np.array([[False, True]])

    result = threshold_temporal_difference(
        t1, t2, threshold=1.0, direction="absolute", valid=valid
    )

    np.testing.assert_array_equal(result, [[False, False]])


def test_threshold_requires_explicit_finite_nonnegative_threshold() -> None:
    t1 = np.zeros((1, 2))
    t2 = np.ones((1, 2))
    valid = np.ones((1, 2), dtype=bool)

    with pytest.raises(TypeError):
        threshold_temporal_difference(t1, t2, direction="absolute", valid=valid)  # type: ignore[call-arg]

    for bad in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            threshold_temporal_difference(
                t1, t2, threshold=bad, direction="absolute", valid=valid
            )


def test_threshold_has_no_default_in_signature() -> None:
    signature = inspect.signature(threshold_temporal_difference)
    assert signature.parameters["threshold"].default is inspect.Parameter.empty
    assert signature.parameters["threshold"].kind is inspect.Parameter.KEYWORD_ONLY


def test_threshold_boundary_is_inclusive() -> None:
    t1 = np.array([[0.0]])
    t2 = np.array([[2.0]])
    valid = np.array([[True]])

    result = threshold_temporal_difference(
        t1, t2, threshold=2.0, direction="absolute", valid=valid
    )

    assert result[0, 0] is np.True_ or bool(result[0, 0]) is True


# ---------------------------------------------------------------------------
# prepare_common_grid
# ---------------------------------------------------------------------------

REF_TRANSFORM = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
CRS_METRIC = "EPSG:32643"


def _aligned_pair(tmp_path: Path, *, reference: str = "t1", **kwargs) -> object:
    tmp_path.mkdir(parents=True, exist_ok=True)
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    shifted = kwargs.pop("t2_transform", REF_TRANSFORM)
    t2_crs = kwargs.pop("t2_crs", CRS_METRIC)
    t2_values = kwargs.pop("t2_values", None)
    t2_nodata = kwargs.pop("t2_nodata", None)
    _write_geotiff(
        t2_path,
        width=kwargs.pop("t2_width", 4),
        height=kwargs.pop("t2_height", 4),
        transform=shifted,
        crs=t2_crs,
        values=t2_values,
        nodata=t2_nodata,
    )
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")
    return prepare_common_grid(
        t1, t2, tmp_path / "aligned", reference=reference, **kwargs
    )


def test_identical_grids_return_originals_without_derived_files(tmp_path: Path) -> None:
    result = _aligned_pair(tmp_path)

    assert result.reprojected is False
    assert result.reprojected_observation_ids == ()
    assert result.t1_path.name == "t1.tif"
    assert result.t2_path.name == "t2.tif"
    assert result.derived_hashes == {}
    assert result.valid_mask_paths == {"obs_t1": None, "obs_t2": None}
    assert not (tmp_path / "aligned").exists()


def test_equal_shape_shifted_affine_must_reproject(tmp_path: Path) -> None:
    shifted = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)
    result = _aligned_pair(tmp_path, t2_transform=shifted)

    assert result.reprojected is True
    assert result.reprojected_observation_ids == ("obs_t2",)
    assert result.t2_path.name != "t2.tif"
    with rasterio.open(result.t2_path) as derived:
        assert derived.crs.to_string() == CRS_METRIC
        assert derived.transform == REF_TRANSFORM
        assert (derived.width, derived.height) == (4, 4)


def test_different_resolution_reprojects_to_reference_grid(tmp_path: Path) -> None:
    coarse = Affine(20.0, 0.0, 1000.0, 0.0, -20.0, 2000.0)
    result = _aligned_pair(tmp_path, t2_transform=coarse, t2_width=2, t2_height=2)

    assert result.reprojected is True
    with rasterio.open(result.t2_path) as derived:
        assert (derived.width, derived.height) == (4, 4)
        assert derived.transform == REF_TRANSFORM


def test_transformable_crs_mismatch_reprojects(tmp_path: Path) -> None:
    # reference patch inside UTM zone 43N (~72.7°E, ~40.2°N)
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    utm_transform = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 4450000.0)
    _write_geotiff(t1_path, width=4, height=4, transform=utm_transform, crs=CRS_METRIC)
    # broad geographic raster covering 72.0–73.0°E / 40.0–40.6°N (WGS84),
    # containing the UTM reference patch at (72.651, 40.177)
    geo_transform = Affine(0.01, 0.0, 72.0, 0.0, -0.01, 40.6)
    _write_geotiff(
        t2_path, width=100, height=60, transform=geo_transform, crs="EPSG:4326"
    )
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")

    result = prepare_common_grid(
        t1, t2, tmp_path / "aligned", reference="t1", resampling="bilinear"
    )

    assert result.reprojected is True
    with rasterio.open(result.t2_path) as derived:
        assert derived.crs.to_string() == CRS_METRIC
        assert derived.transform == utm_transform


def test_zero_overlap_rejects(tmp_path: Path) -> None:
    far_away = Affine(10.0, 0.0, 1_000_000.0, 0.0, -10.0, 2_000_000.0)

    with pytest.raises(NoSpatialOverlapError):
        _aligned_pair(tmp_path, t2_transform=far_away)


def test_missing_reference_crs_rejects(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=None)
    _write_geotiff(t2_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")

    with pytest.raises(GridPreparationError, match="reference"):
        prepare_common_grid(t1, t2, tmp_path / "aligned")


def test_missing_source_crs_rejects(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    _write_geotiff(t2_path, width=4, height=4, transform=REF_TRANSFORM, crs=None)
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")

    with pytest.raises(GridPreparationError, match="source"):
        prepare_common_grid(t1, t2, tmp_path / "aligned")


def test_singular_transform_rejects(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    _write_geotiff(t2_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")
    from satquery.ingestion.models import AffineTransform as TransformModel

    degenerate = t1.model_copy(
        update={
            "geo": t1.geo.model_copy(
                update={
                    "transform": TransformModel(
                        a=0.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0
                    )
                }
            )
        }
    )

    with pytest.raises(GridPreparationError, match="singular"):
        prepare_common_grid(degenerate, t2, tmp_path / "aligned")


def test_resampling_is_recorded_explicitly(tmp_path: Path) -> None:
    shifted = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)

    bilinear = _aligned_pair(tmp_path / "b", t2_transform=shifted, resampling="bilinear")
    nearest = _aligned_pair(tmp_path / "n", t2_transform=shifted, resampling="nearest")

    assert bilinear.resampling == "bilinear"
    assert nearest.resampling == "nearest"


def test_originals_remain_byte_identical(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    shifted = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)
    _write_geotiff(t2_path, width=4, height=4, transform=shifted, crs=CRS_METRIC)
    before = (_sha256(t1_path), _sha256(t2_path))

    _aligned_pair(tmp_path, t2_transform=shifted)

    assert (_sha256(t1_path), _sha256(t2_path)) == before


def test_reference_t2_uses_exact_t2_grid(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    coarse = Affine(20.0, 0.0, 1000.0, 0.0, -20.0, 2000.0)
    _write_geotiff(t2_path, width=2, height=2, transform=coarse, crs=CRS_METRIC)
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")

    result = prepare_common_grid(
        t1, t2, tmp_path / "aligned", reference="t2", resampling="bilinear"
    )

    assert result.reprojected is True
    assert result.reprojected_observation_ids == ("obs_t1",)
    assert result.reference_observation_id == "obs_t2"
    assert result.t2_path.name == "t2.tif"
    with rasterio.open(result.t1_path) as derived:
        assert derived.transform == coarse
        assert (derived.width, derived.height) == (2, 2)


def test_nodata_and_out_of_footprint_remain_invalid(tmp_path: Path) -> None:
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=REF_TRANSFORM, crs=CRS_METRIC)
    t2_values = np.ones((1, 4, 4), dtype="float32")
    t2_values[0, :, 3] = 0.0
    shifted = Affine(10.0, 0.0, 1030.0, 0.0, -10.0, 2000.0)
    _write_geotiff(
        t2_path,
        width=4,
        height=4,
        transform=shifted,
        crs=CRS_METRIC,
        values=t2_values,
        nodata=0.0,
    )
    t1 = _observation(t1_path, "obs_t1")
    t2 = _observation(t2_path, "obs_t2")

    result = prepare_common_grid(t1, t2, tmp_path / "aligned", resampling="bilinear")

    valid_path = result.valid_mask_paths["obs_t2"]
    assert valid_path is not None
    with rasterio.open(valid_path) as valid_raster:
        valid = valid_raster.read(1)
    assert valid.shape == (4, 4)
    assert 0 in valid
    assert valid.max() == 1
    # every invalid cell maps to the source NoData column, not invented data
    with rasterio.open(result.t2_path) as derived:
        data = derived.read(1)
        assert data[valid == 0].min() == 0.0


def test_result_carries_hashes_and_reference_identity(tmp_path: Path) -> None:
    result = _aligned_pair(tmp_path)

    assert result.reference_observation_id == "obs_t1"
    assert result.reference_crs == CRS_METRIC
    assert result.source_hashes["obs_t1"] == _sha256(tmp_path / "t1.tif")
    assert result.source_hashes["obs_t2"] == _sha256(tmp_path / "t2.tif")
    assert result.warnings is not None
