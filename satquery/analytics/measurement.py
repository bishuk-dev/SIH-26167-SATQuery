"""CRS-safe deterministic mask measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


class CrsRequiredForMeasurementError(ValueError):
    """Raised when an area cannot be computed without spatial reference."""


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    value: float
    unit: str
    method: str
    positive_pixel_count: int


def measure_mask_area(
    mask: np.ndarray,
    transform: Affine,
    crs: str | CRS,
    *,
    unit: str,
) -> MeasurementResult:
    """Measure positive mask area in square metres, hectares, or square kilometres."""

    if unit not in {"m2", "ha", "km2"}:
        raise ValueError(f"Unsupported area unit: {unit}")
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.size:
        raise ValueError("mask must be a non-empty two-dimensional array")
    try:
        source_crs = CRS.from_user_input(crs)
    except (rasterio.errors.CRSError, ValueError) as exc:
        raise CrsRequiredForMeasurementError("measurement requires a valid CRS") from exc

    if source_crs.is_projected:
        factor = float(source_crs.linear_units_factor[1])
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        area_m2 = int(values.sum()) * pixel_area_m2 * factor**2
        method = "projected_affine_determinant"
        count = int(values.sum())
    else:
        projected_mask, projected_transform = _to_equal_area(values, transform, source_crs)
        pixel_area_m2 = abs(
            projected_transform.a * projected_transform.e
            - projected_transform.b * projected_transform.d
        )
        area_m2 = int(projected_mask.sum()) * pixel_area_m2
        method = "equal_area_reprojection_epsg_6933"
        count = int(projected_mask.sum())

    divisors = {"m2": 1.0, "ha": 10_000.0, "km2": 1_000_000.0}
    return MeasurementResult(area_m2 / divisors[unit], unit, method, count)


def _to_equal_area(
    mask: np.ndarray,
    transform: Affine,
    source_crs: CRS,
) -> tuple[np.ndarray, Affine]:
    bounds = array_bounds(mask.shape[0], mask.shape[1], transform)
    target_crs = CRS.from_epsg(6933)
    target_transform, width, height = calculate_default_transform(
        source_crs,
        target_crs,
        mask.shape[1],
        mask.shape[0],
        *bounds,
    )
    destination = np.zeros((height, width), dtype="uint8")
    reproject(
        source=mask.astype("uint8"),
        destination=destination,
        src_transform=transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return destination.astype(bool), target_transform
