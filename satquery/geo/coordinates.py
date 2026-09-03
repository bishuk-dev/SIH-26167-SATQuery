"""Coordinate conversions used by crops, evidence, and pair validation."""

from __future__ import annotations

import math
from typing import Literal

from affine import Affine, TransformNotInvertibleError
from rasterio.crs import CRS
from rasterio.errors import RasterioError
from rasterio.warp import transform_bounds as rasterio_transform_bounds

from satquery.geo.exceptions import CoordinateTransformError
from satquery.geo.models import PixelWindow
from satquery.ingestion.models import AffineTransform, GeoBounds

PixelOffset = Literal["center", "upper_left"]


def pixel_to_world(
    transform: AffineTransform | Affine,
    column: float,
    row: float,
    *,
    offset: PixelOffset = "center",
) -> tuple[float, float]:
    """Map a pixel coordinate to its world coordinate through an affine."""

    _require_finite(column, row, label="pixel coordinates")
    affine = _as_affine(transform)
    adjustment = _pixel_adjustment(offset)
    x, y = affine @ (column + adjustment, row + adjustment)
    return float(x), float(y)


def world_to_pixel(
    transform: AffineTransform | Affine,
    x: float,
    y: float,
    *,
    offset: PixelOffset = "center",
) -> tuple[float, float]:
    """Map a world coordinate to continuous pixel coordinates.

    No rounding is performed, so callers retain control over containment rules.
    Using the same ``offset`` as :func:`pixel_to_world` makes the operations
    inverses of one another.
    """

    _require_finite(x, y, label="world coordinates")
    affine = _as_affine(transform)
    try:
        inverse = ~affine
    except (TransformNotInvertibleError, ZeroDivisionError) as exc:
        raise CoordinateTransformError("affine transform is not invertible") from exc

    column, row = inverse @ (x, y)
    adjustment = _pixel_adjustment(offset)
    return float(column - adjustment), float(row - adjustment)


def crop_to_source(
    column: float,
    row: float,
    window: PixelWindow,
    *,
    crop_width: float | None = None,
    crop_height: float | None = None,
) -> tuple[float, float]:
    """Map crop coordinates back to source pixels, including resize reversal."""

    rendered_width = window.width if crop_width is None else crop_width
    rendered_height = window.height if crop_height is None else crop_height
    if rendered_width <= 0 or rendered_height <= 0:
        raise CoordinateTransformError("crop dimensions must be positive")
    _require_finite(column, row, label="crop coordinates")

    source_column = window.column_offset + column * window.width / rendered_width
    source_row = window.row_offset + row * window.height / rendered_height
    return source_column, source_row


def transform_bounds(
    bounds: GeoBounds,
    source_crs: str,
    target_crs: str,
    *,
    densify_points: int = 21,
) -> GeoBounds:
    """Transform axis-aligned bounds, densifying edges for nonlinear CRS changes."""

    if densify_points < 0:
        raise CoordinateTransformError("densify_points cannot be negative")

    try:
        source = CRS.from_user_input(source_crs)
        target = CRS.from_user_input(target_crs)
        left, bottom, right, top = rasterio_transform_bounds(
            source,
            target,
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            densify_pts=densify_points,
        )
    except (RasterioError, ValueError, OverflowError) as exc:
        raise CoordinateTransformError(
            f"cannot transform bounds from {source_crs!r} to {target_crs!r}"
        ) from exc

    if not all(math.isfinite(value) for value in (left, bottom, right, top)):
        raise CoordinateTransformError("transformed bounds are not finite")
    return GeoBounds(left=left, bottom=bottom, right=right, top=top)


def _as_affine(transform: AffineTransform | Affine) -> Affine:
    if isinstance(transform, Affine):
        return transform
    return Affine(
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def _pixel_adjustment(offset: PixelOffset) -> float:
    if offset == "center":
        return 0.5
    if offset == "upper_left":
        return 0.0
    raise CoordinateTransformError(f"unsupported pixel offset: {offset!r}")


def _require_finite(first: float, second: float, *, label: str) -> None:
    if not math.isfinite(first) or not math.isfinite(second):
        raise CoordinateTransformError(f"{label} must be finite")
