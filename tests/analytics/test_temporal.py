from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from satquery.analytics.temporal import (
    prepare_common_grid,
    threshold_temporal_difference,
)


def _write_raster(path: Path, data: np.ndarray, transform: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
    ) as dataset:
        dataset.write(data.astype("float32"))


def test_equal_shape_misaligned_grids_require_reprojection(tmp_path: Path) -> None:
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    _write_raster(t1, np.ones((1, 2, 2)), Affine(10, 0, 0, 0, -10, 20))
    _write_raster(t2, np.ones((1, 2, 2)), Affine(10, 0, 10, 0, -10, 20))

    result = prepare_common_grid(t1, t2, tmp_path / "aligned", resampling="bilinear")

    assert result.reprojected is True
    assert result.t1_transform == result.t2_transform
    assert result.t1_path.is_file()
    assert result.t2_path.is_file()


def test_identity_pair_has_zero_deterministic_change() -> None:
    image = np.ones((2, 2), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)

    mask = threshold_temporal_difference(image, image, threshold=0.2, valid=valid)

    assert not mask.any()
