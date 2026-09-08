from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.inference.temporal_inputs import read_aligned_rgb_pair
from satquery.ingestion.models import Modality, TemporalMetadata
from tests.inference.test_change_detection import _observation, _write_image


def test_read_aligned_rgb_pair_success(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")

    first, second = read_aligned_rgb_pair(t1, t2)

    assert first.shape == (3, 2, 2)
    assert second.shape == (3, 2, 2)
    assert first.dtype == np.float32
    assert second.dtype == np.float32
    np.testing.assert_allclose(first, 1.0 / 255.0)


def test_read_aligned_rgb_pair_rejects_missing_crs_or_transform(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")

    no_crs = t1.model_copy(update={"geo": t1.geo.model_copy(update={"crs": None})})
    with pytest.raises(ModelInputUnsupportedError, match="georeferencing"):
        read_aligned_rgb_pair(no_crs, t2)

    no_tf = t2.model_copy(update={"geo": t2.geo.model_copy(update={"transform": None})})
    with pytest.raises(ModelInputUnsupportedError, match="georeferencing"):
        read_aligned_rgb_pair(t1, no_tf)


def test_read_aligned_rgb_pair_rejects_misalignment(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")

    shifted = t2.model_copy(
        update={"geo": t2.geo.model_copy(update={"transform": t2.geo.transform.model_copy(update={"c": 5.0})})}
    )
    with pytest.raises(ModelInputUnsupportedError, match="aligned temporal pair"):
        read_aligned_rgb_pair(t1, shifted)


def test_read_aligned_rgb_pair_rejects_non_rgb_bands(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2")

    wrong_bands = t1.model_copy(
        update={
            "sensor": t1.sensor.model_copy(
                update={"bands": (t1.sensor.bands[0].model_copy(update={"description": "NIR"}),) + t1.sensor.bands[1:]}
            )
        }
    )
    with pytest.raises(ModelInputUnsupportedError, match="semantic R/G/B bands"):
        read_aligned_rgb_pair(wrong_bands, t2)
