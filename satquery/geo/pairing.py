"""Deterministic compatibility checks for registered observation pairs."""

from __future__ import annotations

import math

from affine import Affine, TransformNotInvertibleError
from rasterio.crs import CRS
from rasterio.errors import CRSError

from satquery.geo.coordinates import transform_bounds
from satquery.geo.exceptions import CoordinateTransformError
from satquery.geo.models import (
    CompatibilityStatus,
    CrsCompatibility,
    GridCompatibility,
    ModalityCompatibility,
    ModalityPairType,
    OverlapCompatibility,
    PairCompatibility,
    PairResult,
    RegistrationStatus,
    TemporalCompatibility,
)
from satquery.ingestion.models import (
    AffineTransform,
    GeoBounds,
    Modality,
    ObservationState,
)


class PairValidator:
    """Compare observation metadata without modifying either raster.

    Overlap is intersection area divided by the smaller footprint area in a
    common CRS. A cross-CRS result uses transformed bounding envelopes and is
    therefore an approximation, reported alongside a reprojection warning.
    """

    def __init__(
        self,
        *,
        minimum_overlap_fraction: float = 0.01,
        resolution_relative_tolerance: float = 1e-6,
        alignment_tolerance: float = 1e-6,
    ) -> None:
        if not 0.0 <= minimum_overlap_fraction <= 1.0:
            raise ValueError("minimum_overlap_fraction must be between 0 and 1")
        if resolution_relative_tolerance < 0 or alignment_tolerance < 0:
            raise ValueError("comparison tolerances cannot be negative")
        self.minimum_overlap_fraction = minimum_overlap_fraction
        self.resolution_relative_tolerance = resolution_relative_tolerance
        self.alignment_tolerance = alignment_tolerance

    def validate(
        self, observation_a: ObservationState, observation_b: ObservationState
    ) -> PairCompatibility:
        reasons: list[str] = []
        crs = self._compare_crs(observation_a, observation_b, reasons)
        overlap = self._compare_overlap(observation_a, observation_b, crs, reasons)
        grid = self._compare_grid(observation_a, observation_b, crs, reasons)
        temporal = self._compare_temporal(observation_a, observation_b, reasons)
        modality = self._compare_modality(observation_a, observation_b, reasons)

        if observation_a.source_asset.sha256 == observation_b.source_asset.sha256:
            reasons.append("DUPLICATE_SOURCE_ASSET")

        registration = self._registration_status(crs, overlap, grid)
        status = self._overall_status(crs, overlap, reasons)
        return PairCompatibility(
            observation_a=observation_a.observation_id,
            observation_b=observation_b.observation_id,
            overlap=overlap,
            crs=crs,
            grid=grid,
            temporal=temporal,
            modality=modality,
            registration=registration,
            result=PairResult(status=status, reasons=tuple(dict.fromkeys(reasons))),
        )

    def _compare_crs(
        self,
        observation_a: ObservationState,
        observation_b: ObservationState,
        reasons: list[str],
    ) -> CrsCompatibility:
        crs_a = observation_a.geo.crs
        crs_b = observation_b.geo.crs
        if crs_a is None or crs_b is None:
            reasons.append("CRS_UNAVAILABLE")
            return CrsCompatibility(equal=None, transformable=None)

        try:
            parsed_a = CRS.from_user_input(crs_a)
            parsed_b = CRS.from_user_input(crs_b)
        except CRSError:
            reasons.append("CRS_NOT_TRANSFORMABLE")
            return CrsCompatibility(equal=False, transformable=False)

        if parsed_a == parsed_b:
            return CrsCompatibility(equal=True, transformable=True)

        try:
            if observation_b.geo.bounds is not None:
                transform_bounds(observation_b.geo.bounds, crs_b, crs_a)
            transformable = True
        except CoordinateTransformError:
            transformable = False

        reasons.append(
            "CRS_TRANSFORM_REQUIRED" if transformable else "CRS_NOT_TRANSFORMABLE"
        )
        return CrsCompatibility(equal=False, transformable=transformable)

    def _compare_overlap(
        self,
        observation_a: ObservationState,
        observation_b: ObservationState,
        crs: CrsCompatibility,
        reasons: list[str],
    ) -> OverlapCompatibility:
        bounds_a = observation_a.geo.bounds
        bounds_b = observation_b.geo.bounds
        crs_a = observation_a.geo.crs
        crs_b = observation_b.geo.crs
        if (
            bounds_a is None
            or bounds_b is None
            or crs_a is None
            or crs_b is None
            or crs.transformable is not True
        ):
            reasons.append("OVERLAP_UNKNOWN")
            return OverlapCompatibility(known=False)

        if crs.equal is not True:
            try:
                bounds_b = transform_bounds(bounds_b, crs_b, crs_a)
            except CoordinateTransformError:
                reasons.append("OVERLAP_UNKNOWN")
                return OverlapCompatibility(known=False)

        fraction = _overlap_fraction(bounds_a, bounds_b)
        if fraction is None:
            reasons.append("OVERLAP_UNKNOWN")
            return OverlapCompatibility(known=False)
        sufficient = fraction >= self.minimum_overlap_fraction and fraction > 0.0
        if fraction == 0.0:
            reasons.append("NO_SPATIAL_OVERLAP")
        elif not sufficient:
            reasons.append("LOW_SPATIAL_OVERLAP")
        return OverlapCompatibility(
            known=True,
            overlap_fraction=fraction,
            sufficient=sufficient,
        )

    def _compare_grid(
        self,
        observation_a: ObservationState,
        observation_b: ObservationState,
        crs: CrsCompatibility,
        reasons: list[str],
    ) -> GridCompatibility:
        raster_a = observation_a.raster
        raster_b = observation_b.raster
        same_shape = (raster_a.width, raster_a.height) == (
            raster_b.width,
            raster_b.height,
        )

        if crs.equal is not True:
            reasons.extend(
                (
                    "RESOLUTION_COMPARISON_REQUIRES_COMMON_CRS",
                    "GRID_ALIGNMENT_UNKNOWN",
                )
            )
            return GridCompatibility(
                same_shape=same_shape,
                same_resolution=None,
                aligned=None,
            )

        resolution_a = (
            observation_a.geo.native_gsd_x,
            observation_a.geo.native_gsd_y,
        )
        resolution_b = (
            observation_b.geo.native_gsd_x,
            observation_b.geo.native_gsd_y,
        )
        if None in resolution_a or None in resolution_b:
            same_resolution = None
            reasons.append("RESOLUTION_UNKNOWN")
        else:
            same_resolution = all(
                math.isclose(
                    value_a,
                    value_b,
                    rel_tol=self.resolution_relative_tolerance,
                    abs_tol=self.resolution_relative_tolerance,
                )
                for value_a, value_b in zip(resolution_a, resolution_b, strict=True)
            )
            if not same_resolution:
                reasons.append("RESOLUTION_MISMATCH")

        transform_a = observation_a.geo.transform
        transform_b = observation_b.geo.transform
        if transform_a is None or transform_b is None or same_resolution is None:
            aligned = None
            reasons.append("GRID_ALIGNMENT_UNKNOWN")
        elif not same_resolution:
            aligned = False
            reasons.append("GRID_MISALIGNED")
        else:
            affine_a = _as_affine(transform_a)
            affine_b = _as_affine(transform_b)
            vectors_equal = all(
                math.isclose(
                    value_a,
                    value_b,
                    rel_tol=self.resolution_relative_tolerance,
                    abs_tol=self.resolution_relative_tolerance,
                )
                for value_a, value_b in zip(
                    (affine_a.a, affine_a.b, affine_a.d, affine_a.e),
                    (affine_b.a, affine_b.b, affine_b.d, affine_b.e),
                    strict=True,
                )
            )
            aligned = vectors_equal and _origins_share_grid(
                affine_a, affine_b, self.alignment_tolerance
            )
            if not aligned:
                reasons.append("GRID_MISALIGNED")

        return GridCompatibility(
            same_shape=same_shape,
            same_resolution=same_resolution,
            aligned=aligned,
        )

    @staticmethod
    def _compare_temporal(
        observation_a: ObservationState,
        observation_b: ObservationState,
        reasons: list[str],
    ) -> TemporalCompatibility:
        timestamp_a = observation_a.temporal.acquisition_time
        timestamp_b = observation_b.temporal.acquisition_time
        if timestamp_a is None or timestamp_b is None or timestamp_a == timestamp_b:
            reasons.append("TEMPORAL_ORDER_UNKNOWN")
            return TemporalCompatibility(order_known=False)

        first, second = (
            (observation_a, observation_b)
            if timestamp_a < timestamp_b
            else (observation_b, observation_a)
        )
        delta = abs((timestamp_b - timestamp_a).total_seconds())
        return TemporalCompatibility(
            order_known=True,
            first=first.observation_id,
            second=second.observation_id,
            time_delta_seconds=delta,
        )

    @staticmethod
    def _compare_modality(
        observation_a: ObservationState,
        observation_b: ObservationState,
        reasons: list[str],
    ) -> ModalityCompatibility:
        modality_a = observation_a.sensor.modality
        modality_b = observation_b.sensor.modality
        if Modality.UNKNOWN in (modality_a, modality_b):
            reasons.append("MODALITY_UNKNOWN")
            return ModalityCompatibility(
                pair_type=ModalityPairType.UNKNOWN,
                compatible=None,
            )

        optical_family = {Modality.OPTICAL, Modality.MULTISPECTRAL}
        modalities = {modality_a, modality_b}
        if Modality.SAR in modalities and modalities & optical_family:
            pair_type = ModalityPairType.OPTICAL_SAR
        elif modality_a == modality_b:
            pair_type = ModalityPairType.TEMPORAL_SAME_MODALITY
        else:
            pair_type = ModalityPairType.TEMPORAL_CROSS_MODAL
        return ModalityCompatibility(pair_type=pair_type, compatible=True)

    @staticmethod
    def _registration_status(
        crs: CrsCompatibility,
        overlap: OverlapCompatibility,
        grid: GridCompatibility,
    ) -> RegistrationStatus:
        if crs.transformable is False or (
            overlap.known and overlap.overlap_fraction == 0.0
        ):
            return RegistrationStatus.INVALID
        if crs.equal and grid.same_resolution and grid.aligned:
            return RegistrationStatus.VERIFIED
        if (
            crs.transformable is True
            and overlap.known
            and overlap.overlap_fraction is not None
            and overlap.overlap_fraction > 0.0
        ):
            return RegistrationStatus.APPROXIMATE
        return RegistrationStatus.UNKNOWN

    @staticmethod
    def _overall_status(
        crs: CrsCompatibility,
        overlap: OverlapCompatibility,
        reasons: list[str],
    ) -> CompatibilityStatus:
        if crs.transformable is False or (
            overlap.known and overlap.overlap_fraction == 0.0
        ):
            return CompatibilityStatus.FAIL
        if reasons:
            return CompatibilityStatus.WARN
        return CompatibilityStatus.PASS


def _overlap_fraction(bounds_a: GeoBounds, bounds_b: GeoBounds) -> float | None:
    intersection_width = max(
        0.0, min(bounds_a.right, bounds_b.right) - max(bounds_a.left, bounds_b.left)
    )
    intersection_height = max(
        0.0, min(bounds_a.top, bounds_b.top) - max(bounds_a.bottom, bounds_b.bottom)
    )
    intersection_area = intersection_width * intersection_height
    area_a = (bounds_a.right - bounds_a.left) * (bounds_a.top - bounds_a.bottom)
    area_b = (bounds_b.right - bounds_b.left) * (bounds_b.top - bounds_b.bottom)
    if area_a <= 0.0 or area_b <= 0.0:
        return None
    return min(1.0, max(0.0, intersection_area / min(area_a, area_b)))


def _as_affine(transform: AffineTransform) -> Affine:
    return Affine(
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def _origins_share_grid(
    affine_a: Affine, affine_b: Affine, tolerance: float
) -> bool:
    try:
        column_offset, row_offset = (~affine_a) @ (affine_b.c, affine_b.f)
    except (TransformNotInvertibleError, ZeroDivisionError):
        return False
    return math.isclose(
        column_offset, round(column_offset), abs_tol=tolerance
    ) and math.isclose(row_offset, round(row_offset), abs_tol=tolerance)
