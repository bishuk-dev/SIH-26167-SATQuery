"""Geospatial measurement engine converting raster masks to physical area and evidence."""

from __future__ import annotations

import math
from typing import Any
import uuid

from affine import Affine
import numpy as np

from satquery.core.contracts.evidence import MeasurementEvidence
from satquery.geo.coordinates import _as_affine
from satquery.geo.exceptions import MeasurementError
from satquery.geo.measurements import (
    calculate_area,
    calculate_pixel_area_m2,
    count_pixels,
)
from satquery.geo.models import MeasurementResult, MeasurementUnit
from satquery.ingestion.models import AffineTransform


class MeasurementEngine:
    """Deterministic geospatial measurement engine for masks and features."""

    @classmethod
    def measure_mask(
        cls,
        mask: np.ndarray | Any,
        transform: AffineTransform | Affine,
        crs: str | None,
        *,
        measurement_type: str = "area_measurement",
        target_unit: MeasurementUnit | str = MeasurementUnit.M2,
        threshold: float | None = None,
        center_coord: tuple[float, float] | None = None,
    ) -> MeasurementEvidence:
        """Measure physical area of positive mask pixels and generate MeasurementEvidence.

        Enforces R-GEO-004 and R-GEO-005.
        """
        meas_result = calculate_area(
            mask=mask,
            transform=transform,
            crs=crs,
            target_unit=target_unit,
            threshold=threshold,
            center_coord=center_coord,
        )

        return MeasurementEvidence(
            evidence_id=f"meas_{uuid.uuid4().hex[:16]}",
            measurement_type=measurement_type,
            area_value=meas_result.area,
            unit=meas_result.unit.value,
            pixel_count=meas_result.pixel_count,
            pixel_area_m2=meas_result.pixel_area_m2,
            crs=crs or "UNKNOWN",
            calculation_path=meas_result.calculation_path,
        )

    @classmethod
    def compute_mask_coverage(
        cls,
        mask: np.ndarray | Any,
        total_pixels: int | None = None,
    ) -> tuple[int, int, float]:
        """Compute (positive_pixels, total_pixels, percentage_coverage)."""
        arr = np.asarray(mask)
        pos = count_pixels(arr)
        total = total_pixels if total_pixels is not None else int(arr.size)
        if total <= 0:
            raise MeasurementError("Total pixels must be positive")
        pct = float((pos / total) * 100.0)
        return pos, total, pct

    @classmethod
    def compute_mask_bounds(
        cls,
        mask: np.ndarray,
        transform: AffineTransform | Affine,
    ) -> tuple[float, float, float, float] | None:
        """Compute (min_x, min_y, max_x, max_y) in coordinate space of positive pixels.

        Returns None if no positive pixels exist.
        """
        arr = np.asarray(mask, dtype=bool)
        if not np.any(arr):
            return None

        rows, cols = np.where(arr)
        min_r, max_r = int(np.min(rows)), int(np.max(rows))
        min_c, max_c = int(np.min(cols)), int(np.max(cols))

        affine = _as_affine(transform)
        # Transform pixel corners to world coordinates
        x0, y0 = affine * (min_c, min_r)
        x1, y1 = affine * (max_c + 1, max_r + 1)

        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)

        return float(min_x), float(min_y), float(max_x), float(max_y)


def measure_mask_area(
    mask: np.ndarray | Any,
    transform: AffineTransform | Affine,
    crs: str | None,
    *,
    target_unit: MeasurementUnit | str = MeasurementUnit.M2,
    threshold: float | None = None,
    center_coord: tuple[float, float] | None = None,
) -> MeasurementEvidence:
    """Convenience wrapper for MeasurementEngine.measure_mask."""
    return MeasurementEngine.measure_mask(
        mask=mask,
        transform=transform,
        crs=crs,
        target_unit=target_unit,
        threshold=threshold,
        center_coord=center_coord,
    )
