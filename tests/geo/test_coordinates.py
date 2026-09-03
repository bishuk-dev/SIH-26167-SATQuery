from __future__ import annotations

import pytest
from affine import Affine

from satquery.geo import (
    PixelWindow,
    crop_to_source,
    pixel_to_world,
    transform_bounds,
    world_to_pixel,
)
from satquery.geo.exceptions import CoordinateTransformError
from satquery.ingestion.models import AffineTransform, GeoBounds


def test_pixel_world_round_trip_respects_center_and_rotation() -> None:
    affine = Affine(10, 2, 100, 1, -10, 200)
    transform = AffineTransform(
        a=affine.a,
        b=affine.b,
        c=affine.c,
        d=affine.d,
        e=affine.e,
        f=affine.f,
    )

    world = pixel_to_world(transform, 2, 3)

    assert world_to_pixel(transform, *world) == pytest.approx((2, 3))
    assert pixel_to_world(transform, 0, 0, offset="upper_left") == (100, 200)


def test_crop_to_source_reverses_resize_then_applies_window_offset() -> None:
    window = PixelWindow(
        column_offset=100,
        row_offset=200,
        width=400,
        height=200,
    )

    source = crop_to_source(10, 20, window, crop_width=200, crop_height=100)

    assert source == pytest.approx((120, 240))


def test_world_to_pixel_rejects_noninvertible_affine() -> None:
    transform = AffineTransform(a=1, b=2, c=0, d=2, e=4, f=0)

    with pytest.raises(CoordinateTransformError, match="not invertible"):
        world_to_pixel(transform, 1, 1)


def test_transform_bounds_between_supported_crs() -> None:
    transformed = transform_bounds(
        GeoBounds(left=0, bottom=0, right=1, top=1),
        "EPSG:4326",
        "EPSG:3857",
    )

    assert transformed.left == pytest.approx(0)
    assert transformed.bottom == pytest.approx(0)
    assert transformed.right == pytest.approx(111_319.49, rel=1e-5)
    assert transformed.top == pytest.approx(111_325.14, rel=1e-5)

