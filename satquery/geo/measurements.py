"""Deterministic GIS measurement tools for area and pixel counting."""

from __future__ import annotations

import math
from typing import Any

from affine import Affine
import numpy as np
from rasterio.crs import CRS
from rasterio.errors import CRSError

from satquery.geo.coordinates import _as_affine
from satquery.geo.exceptions import (
    CrsMeasurementError,
    MeasurementError,
)
from satquery.geo.models import MeasurementResult, MeasurementUnit
from satquery.ingestion.models import AffineTransform

AREA_UNIT_FACTORS: dict[MeasurementUnit, float] = {
    MeasurementUnit.M2: 1.0,
    MeasurementUnit.HA: 1e-4,       # 1 ha = 10,000 m2
    MeasurementUnit.KM2: 1e-6,      # 1 km2 = 1,000,000 m2
}

# WGS84 Ellipsoidal constants (meters)
WGS84_A = 6378137.0             # Semi-major axis
WGS84_B = 6356752.314245        # Semi-minor axis
WGS84_E2 = 1.0 - (WGS84_B / WGS84_A) ** 2  # First eccentricity squared (~0.00669438)


def convert_area(area_m2: float, target_unit: MeasurementUnit | str) -> float:
    """Convert an area in square meters to the target unit."""
    if not math.isfinite(area_m2) or area_m2 < 0.0:
        raise MeasurementError(f"Area in m2 must be finite and non-negative, got {area_m2}")

    try:
        unit = MeasurementUnit(target_unit)
    except ValueError as exc:
        raise MeasurementError(f"Unsupported measurement unit: {target_unit!r}") from exc

    if unit == MeasurementUnit.COUNT:
        raise MeasurementError("Cannot convert square meters to pixel count without pixel geometry")

    factor = AREA_UNIT_FACTORS[unit]
    return float(area_m2 * factor)


def count_pixels(
    mask: np.ndarray | Any,
    *,
    threshold: float | None = None,
) -> int:
    """Count positive pixels in a boolean or thresholded binary mask.

    Non-finite pixels (NaN, inf) are safely excluded and never counted as positive.
    """
    arr = np.asarray(mask)
    if arr.size == 0:
        return 0

    if threshold is not None:
        if not math.isfinite(threshold):
            raise MeasurementError(f"Threshold must be finite, got {threshold}")
        finite_mask = np.isfinite(arr)
        positive = finite_mask & (arr > threshold)
    else:
        if arr.dtype == bool:
            positive = arr
        else:
            finite_mask = np.isfinite(arr)
            positive = finite_mask & (arr != 0)

    return int(np.count_nonzero(positive))


def calculate_pixel_area_m2(
    transform: AffineTransform | Affine,
    crs: str | None,
    *,
    center_coord: tuple[float, float] | None = None,
) -> tuple[float, str]:
    """Calculate the ground area in square meters of a single pixel.

    Returns (pixel_area_m2, calculation_path).

    Enforces R-GEO-005:
    - Projected CRS with linear units uses the determinant of the affine transform.
    - Geographic CRS (degrees) uses geodesic surface element integration on the
      WGS84 ellipsoid at the specified (or origin) latitude. Degrees are NEVER
      treated as meters.
    - Missing or unparseable CRS raises CrsMeasurementError.
    """
    if crs is None:
        raise CrsMeasurementError("Metric area calculation requires a valid CRS; CRS is None")

    try:
        parsed_crs = CRS.from_user_input(crs)
    except CRSError as exc:
        raise CrsMeasurementError(f"Invalid or unparseable CRS: {crs!r}") from exc

    affine = _as_affine(transform)
    det = abs(affine.a * affine.e - affine.b * affine.d)
    if not math.isfinite(det) or det <= 0.0:
        raise MeasurementError(f"Affine transform has non-positive determinant: {det}")

    if parsed_crs.is_projected:
        # Check linear unit conversion factor to meters
        try:
            linear_factor = parsed_crs.linear_units_factor[1]
        except (AttributeError, IndexError, TypeError):
            linear_factor = 1.0

        pixel_area_m2 = det * (linear_factor ** 2)
        return float(pixel_area_m2), "projected_planar"

    if parsed_crs.is_geographic:
        # Angular CRS (e.g. EPSG:4326): Compute geodesic cell area on WGS84 ellipsoid
        # Latitude phi determines ground meters per degree of longitude and latitude
        if center_coord is not None:
            lon_deg, lat_deg = center_coord
        else:
            # Fall back to affine origin
            lon_deg, lat_deg = affine.c, affine.f

        if not math.isfinite(lat_deg) or abs(lat_deg) > 90.0:
            raise MeasurementError(f"Invalid latitude for geodesic calculation: {lat_deg}")

        phi = math.radians(lat_deg)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        # Radii of curvature on WGS84
        denom = math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
        r_n = WGS84_A / denom  # Prime vertical radius
        r_m = (WGS84_A * (1.0 - WGS84_E2)) / (denom ** 3)  # Meridional radius

        # Angular step per pixel in radians
        d_lon_rad = math.radians(abs(affine.a))
        d_lat_rad = math.radians(abs(affine.e))

        dx_m = r_n * cos_phi * d_lon_rad
        dy_m = r_m * d_lat_rad
        pixel_area_m2 = dx_m * dy_m

        if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0.0:
            raise MeasurementError(f"Geodesic pixel area calculation produced invalid value: {pixel_area_m2}")

        return float(pixel_area_m2), "geodesic_wgs84"

    raise CrsMeasurementError(f"Unsupported CRS type for metric area calculation: {crs!r}")


def calculate_area(
    mask: np.ndarray | Any,
    transform: AffineTransform | Affine,
    crs: str | None,
    *,
    target_unit: MeasurementUnit | str = MeasurementUnit.M2,
    threshold: float | None = None,
    center_coord: tuple[float, float] | None = None,
) -> MeasurementResult:
    """Calculate deterministic area from a raster mask, affine transform, and CRS.

    Adheres strictly to R-GEO-004 and R-GEO-005.
    """
    unit = MeasurementUnit(target_unit) if isinstance(target_unit, str) else target_unit
    pixel_count = count_pixels(mask, threshold=threshold)

    if unit == MeasurementUnit.COUNT:
        return MeasurementResult(
            pixel_count=pixel_count,
            area=float(pixel_count),
            unit=MeasurementUnit.COUNT,
            crs=crs,
            calculation_path="pixel_counting",
            pixel_area_m2=0.0,
        )

    pixel_area_m2, calc_path = calculate_pixel_area_m2(
        transform, crs, center_coord=center_coord
    )
    total_area_m2 = pixel_count * pixel_area_m2
    converted_area = convert_area(total_area_m2, unit)

    return MeasurementResult(
        pixel_count=pixel_count,
        area=converted_area,
        unit=unit,
        crs=crs,
        calculation_path=calc_path,
        pixel_area_m2=pixel_area_m2,
    )
