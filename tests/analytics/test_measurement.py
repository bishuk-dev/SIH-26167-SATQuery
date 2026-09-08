from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from satquery.analytics.measurement import measure_mask_area


def test_projected_rotated_pixel_area_uses_affine_determinant() -> None:
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    transform = Affine(10, 2, 0, 1, -10, 0)

    result = measure_mask_area(mask, transform, "EPSG:32643", unit="m2")

    assert result.value == pytest.approx(306.0)
    assert result.positive_pixel_count == 3


def test_geographic_crs_never_squares_degrees() -> None:
    mask = np.ones((2, 2), dtype=bool)
    transform = Affine(0.01, 0, 70, 0, -0.01, 20)

    result = measure_mask_area(mask, transform, "EPSG:4326", unit="ha")

    assert result.method == "equal_area_reprojection_epsg_6933"
    assert result.value > 0
