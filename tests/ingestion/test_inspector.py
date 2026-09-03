from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning

from satquery.ingestion import RasterInspector, RasterSafetyLimits
from satquery.ingestion.exceptions import (
    InvalidRasterError,
    RasterDriverMismatchError,
    RasterResourceLimitError,
    UnsupportedRasterDriverError,
)
from satquery.ingestion.models import Modality


@pytest.fixture
def georeferenced_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "sentinel_sample.tif"
    transform = Affine(10, 0, 500_000, 0, -20, 2_000_000)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=2,
        dtype="uint16",
        crs="EPSG:32643",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((2, 3, 4), dtype=np.uint16))
        dataset.set_band_description(1, "VV backscatter")
        dataset.set_band_description(2, "VH backscatter")
        dataset.update_tags(1, POLARIZATION="VV", UNIT="sigma0")
        dataset.update_tags(2, POLARIZATION="VH", UNIT="sigma0")
        dataset.update_tags(
            MODALITY="sar",
            SENSOR_NAME="Sentinel-1 C-SAR",
            PLATFORM="Sentinel-1A",
            PRODUCT_LEVEL="GRD",
            POLARIZATIONS="VV,VH",
            ACQUISITION_TIME="2026-08-14T10:30:00+00:00",
        )
    return path


def _inspect(path: Path, limits: RasterSafetyLimits | None = None):
    return RasterInspector(limits).inspect(
        path,
        observation_id="obs-1",
        asset_id="asset-1",
    )


def test_inspector_extracts_geospatial_sensor_and_band_metadata(
    georeferenced_tiff: Path,
) -> None:
    observation = _inspect(georeferenced_tiff)

    assert observation.raster.driver == "GTiff"
    assert (observation.raster.width, observation.raster.height) == (4, 3)
    assert observation.raster.band_count == 2
    assert observation.raster.dtypes == ("uint16", "uint16")
    assert observation.raster.nodata == (0.0, 0.0)
    assert observation.geo.crs == "EPSG:32643"
    assert observation.geo.transform is not None
    assert observation.geo.transform.model_dump() == {
        "a": 10.0,
        "b": 0.0,
        "c": 500_000.0,
        "d": 0.0,
        "e": -20.0,
        "f": 2_000_000.0,
    }
    assert observation.geo.bounds is not None
    assert observation.geo.bounds.model_dump() == {
        "left": 500_000.0,
        "bottom": 1_999_940.0,
        "right": 500_040.0,
        "top": 2_000_000.0,
    }
    assert observation.geo.native_gsd_x == 10.0
    assert observation.geo.native_gsd_y == 20.0
    assert observation.geo.units == "metre"
    assert observation.sensor.modality is Modality.SAR
    assert observation.sensor.sensor_name == "Sentinel-1 C-SAR"
    assert observation.sensor.platform == "Sentinel-1A"
    assert observation.sensor.product_level == "GRD"
    assert observation.sensor.polarizations == ("VV", "VH")
    assert observation.sensor.bands[0].description == "VV backscatter"
    assert observation.sensor.bands[0].tags["UNIT"] == "sigma0"
    assert observation.temporal.acquisition_time is not None
    assert observation.temporal.acquisition_time.isoformat() == "2026-08-14T10:30:00+00:00"
    assert observation.validity.has_crs is True
    assert observation.validity.has_transform is True
    assert observation.validity.has_nodata is True
    assert observation.source_asset.sha256 == hashlib.sha256(
        georeferenced_tiff.read_bytes()
    ).hexdigest()


def test_inspector_keeps_unavailable_metadata_unknown(tmp_path: Path) -> None:
    path = tmp_path / "plain.tiff"
    with pytest.warns(NotGeoreferencedWarning):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
        ) as dataset:
            dataset.write(np.zeros((1, 2, 2), dtype=np.uint8))

    with pytest.warns(NotGeoreferencedWarning):
        observation = _inspect(path)

    assert observation.geo.crs is None
    assert observation.geo.transform is None
    assert observation.geo.bounds is None
    assert observation.geo.native_gsd_x is None
    assert observation.sensor.modality is Modality.UNKNOWN
    assert observation.sensor.sensor_name is None
    assert observation.sensor.polarizations == ()
    assert observation.temporal.acquisition_time is None
    assert observation.validity.warnings == (
        "CRS_UNAVAILABLE",
        "TRANSFORM_UNAVAILABLE",
    )


@pytest.mark.parametrize(
    ("limits", "expected_field"),
    [
        (RasterSafetyLimits(max_width=3), "width"),
        (RasterSafetyLimits(max_height=2), "height"),
        (RasterSafetyLimits(max_band_count=1), "band count"),
        (RasterSafetyLimits(max_pixel_count=11), "pixel count"),
    ],
)
def test_inspector_rejects_header_values_over_limits(
    georeferenced_tiff: Path,
    limits: RasterSafetyLimits,
    expected_field: str,
) -> None:
    with pytest.raises(RasterResourceLimitError, match=expected_field):
        _inspect(georeferenced_tiff, limits)


def test_inspector_rejects_file_over_size_limit(georeferenced_tiff: Path) -> None:
    limits = RasterSafetyLimits(max_file_size_bytes=1)

    with pytest.raises(RasterResourceLimitError, match="file size"):
        _inspect(georeferenced_tiff, limits)


def test_inspector_rejects_non_tiff_driver(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    with pytest.warns(NotGeoreferencedWarning):
        with rasterio.open(
            path,
            "w",
            driver="PNG",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
        ) as dataset:
            dataset.write(np.zeros((1, 2, 2), dtype=np.uint8))

    with pytest.warns(NotGeoreferencedWarning):
        with pytest.raises(UnsupportedRasterDriverError):
            _inspect(path)


def test_inspector_rejects_tiff_with_mismatched_extension(
    georeferenced_tiff: Path,
) -> None:
    disguised_path = georeferenced_tiff.with_suffix(".bin")
    shutil.copyfile(georeferenced_tiff, disguised_path)

    with pytest.raises(RasterDriverMismatchError):
        _inspect(disguised_path)


def test_inspector_rejects_corrupt_tiff(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.tif"
    path.write_bytes(b"not a raster")

    with pytest.raises(InvalidRasterError):
        _inspect(path)


def test_safety_limits_load_from_environment() -> None:
    limits = RasterSafetyLimits.from_env(
        {
            "MAX_UPLOAD_SIZE_MB": "64",
            "MAX_RASTER_WIDTH": "1000",
            "MAX_RASTER_HEIGHT": "2000",
            "MAX_RASTER_PIXELS": "3000000",
            "MAX_RASTER_BANDS": "8",
        }
    )

    assert limits.max_file_size_bytes == 64 * 1024 * 1024
    assert limits.max_width == 1000
    assert limits.max_height == 2000
    assert limits.max_pixel_count == 3_000_000
    assert limits.max_band_count == 8
