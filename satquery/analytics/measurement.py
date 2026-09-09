"""CRS-safe deterministic mask-area measurement.

Pure computation only: no raster I/O, no evidence construction. Degrees are
never squared — geographic areas come from exact ellipsoidal integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from satquery.analytics.exceptions import (
    CrsRequiredForMeasurementError,
    InvalidRasterArrayError,
    UnsupportedCrsMeasurementError,
)
from satquery.ingestion.models import AffineTransform

AreaUnit = Literal["m2", "ha", "km2"]

_UNIT_DIVISORS_M2 = {"m2": 1.0, "ha": 10_000.0, "km2": 1_000_000.0}


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    value: float
    unit: str
    positive_pixel_count: int
    valid_pixel_count: int
    method: str
    calculation_crs: str


def _as_affine(transform: Affine | AffineTransform) -> Affine:
    if isinstance(transform, AffineTransform):
        return Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    return transform


def _prepare_mask(
    mask: np.ndarray, valid: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size:
        raise InvalidRasterArrayError("mask must be a non-empty two-dimensional array")
    if values.dtype != bool:
        finite = np.isfinite(values) if values.dtype.kind in "fc" else np.ones_like(values, dtype=bool)
        if not np.all(finite) or not np.isin(values, (0, 1)).all():
            raise InvalidRasterArrayError("mask must be boolean or strictly binary 0/1")
        values = values.astype(bool)
    valid_array = (
        np.ones(values.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid_array.shape != values.shape:
        raise InvalidRasterArrayError("valid mask must match the mask shape")
    return values & valid_array, valid_array


def _parallel_band_area_factor(phi: np.ndarray, semi_major: float, eccentricity: float) -> np.ndarray:
    """Exact antiderivative of the ellipsoidal surface element.

    ``integral of M(phi) * N(phi) * cos(phi) dphi`` between parallels equals
    the difference of this closed form, so north-up geographic cells are
    measured exactly — no degree-squaring, no scale approximation.
    """

    u = np.sin(phi)
    one_minus = 1.0 - eccentricity**2 * u**2
    return semi_major**2 * (1.0 - eccentricity**2) * (
        u / (2.0 * one_minus) + np.arctanh(eccentricity * u) / (2.0 * eccentricity)
    )


# rasterio's CRS cannot report ellipsoid parameters here and pyproj is not a
# declared dependency, so geographic measurements are supported for the common
# datums below and fail closed for anything else (extend when pyproj lands).
_DATUM_ELLIPSOIDS = {
    "WGS84": (6378137.0, 298.257223563),
    "NAD83": (6378137.0, 298.257222101),
}


def _ellipsoid_parameters(crs_object: CRS) -> tuple[float, float, float]:
    """Return ``(semi_major_metres, flattening, eccentricity)`` for the datum.

    The eccentricity is derived explicitly from the flattening via
    ``e = sqrt(f * (2 - f))``; the flattening itself is never used where an
    eccentricity is required.
    """

    datum = crs_object.to_dict().get("datum")
    if datum not in _DATUM_ELLIPSOIDS:
        raise UnsupportedCrsMeasurementError(
            f"geographic datum {datum!r} has no pinned ellipsoid parameters; "
            "cannot measure area without guessing"
        )
    semi_major, inverse_flattening = _DATUM_ELLIPSOIDS[datum]
    flattening = 1.0 / inverse_flattening
    eccentricity_squared = flattening * (2.0 - flattening)
    return semi_major, flattening, math.sqrt(eccentricity_squared)


def measure_mask_area(
    mask: np.ndarray,
    transform: Affine | AffineTransform,
    crs: str,
    *,
    unit: AreaUnit,
    valid: np.ndarray | None = None,
) -> MeasurementResult:
    """Measure positive mask area in square metres, hectares, or square kilometres.

    Projected CRS: exact affine-determinant pixel area times the CRS-declared
    linear-unit conversion. Geographic CRS: exact ellipsoidal parallel-band
    integration for north-up grids; rotated geographic grids are rejected
    rather than approximated.
    """

    if unit not in _UNIT_DIVISORS_M2:
        raise ValueError(f"unsupported area unit: {unit!r}")
    positives, valid_array = _prepare_mask(mask, valid)
    affine = _as_affine(transform)
    determinant = abs(affine.a * affine.e - affine.b * affine.d)
    if determinant == 0.0 or not math.isfinite(determinant):
        raise InvalidRasterArrayError("transform is singular or has zero area")
    if crs is None:
        raise CrsRequiredForMeasurementError("measurement requires a valid CRS")
    try:
        crs_object = CRS.from_user_input(crs)
    except Exception as exc:  # rasterio raises several types for malformed CRS
        raise CrsRequiredForMeasurementError(
            f"measurement requires a valid CRS: {crs!r}"
        ) from exc

    positive_pixel_count = int(positives.sum())
    valid_pixel_count = int(valid_array.sum())

    if crs_object.is_projected:
        try:
            _, linear_units_per_metre = crs_object.linear_units_factor
        except Exception as exc:
            raise UnsupportedCrsMeasurementError(
                f"projected CRS {crs!r} does not declare a convertible linear unit"
            ) from exc
        # linear_units_factor is metres per CRS unit; pixel area in CRS units²
        # becomes m² after multiplying by factor²
        area_m2 = positive_pixel_count * determinant * linear_units_per_metre**2
        method = "projected_affine_determinant"
    elif crs_object.is_geographic:
        if affine.b != 0.0 or affine.d != 0.0:
            raise UnsupportedCrsMeasurementError(
                "rotated geographic grids require a geodesic library; "
                "only north-up geographic grids are supported exactly"
            )
        semi_major, _flattening, eccentricity = _ellipsoid_parameters(crs_object)
        rows = positives.shape[0]
        area_m2 = 0.0
        for row in range(rows):
            phi_top = math.radians(affine.f + row * affine.e)
            phi_bottom = math.radians(affine.f + (row + 1) * affine.e)
            band = abs(
                _parallel_band_area_factor(np.array([phi_top]), semi_major, eccentricity)[0]
                - _parallel_band_area_factor(np.array([phi_bottom]), semi_major, eccentricity)[0]
            )
            delta_lambda = math.radians(abs(affine.a))
            area_m2 += int(positives[row].sum()) * band * delta_lambda
        method = "ellipsoidal_parallel_band_sum"
    else:
        raise UnsupportedCrsMeasurementError(
            f"CRS {crs!r} is neither projected nor geographic; cannot measure area"
        )

    return MeasurementResult(
        value=area_m2 / _UNIT_DIVISORS_M2[unit],
        unit=unit,
        positive_pixel_count=positive_pixel_count,
        valid_pixel_count=valid_pixel_count,
        method=method,
        calculation_crs=str(crs),
    )
