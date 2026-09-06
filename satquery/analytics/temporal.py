"""Deterministic bi-temporal change detection and evidence generation."""

from __future__ import annotations

import math
from typing import Any, Mapping
import uuid

from affine import Affine
import numpy as np

from satquery.analytics.spectral import SpectralAnalytics
from satquery.core.contracts.evidence import ChangeMaskEvidence
from satquery.core.contracts.temporal import (
    TemporalChangeResult,
    TemporalObservationPair,
)
from satquery.geo.exceptions import MeasurementError
from satquery.geo.measurements import calculate_area, count_pixels
from satquery.geo.models import MeasurementUnit
from satquery.geo.pairing import PairValidator
from satquery.ingestion.models import AffineTransform, ObservationState
from satquery.verification.models import VerificationReport
from satquery.verification.verifiers import VerificationEngine


class TemporalAnalytics:
    """Deterministic temporal analytics engine for observation pairs."""

    @classmethod
    def build_temporal_pair(
        cls,
        obs_1: ObservationState,
        obs_2: ObservationState,
    ) -> tuple[TemporalObservationPair, VerificationReport]:
        """Validate and construct a TemporalObservationPair with verification report."""
        validator = PairValidator()
        compat = validator.validate(obs_1, obs_2)

        t1 = obs_1.temporal.acquisition_time
        t2 = obs_2.temporal.acquisition_time

        delta_sec: float | None = None
        is_ordered = True
        if t1 is not None and t2 is not None:
            delta_sec = (t2 - t1).total_seconds()
            if delta_sec <= 0:
                is_ordered = False

        overlap_frac = (
            compat.overlap.overlap_fraction
            if compat.overlap.overlap_fraction is not None
            else 0.0
        )
        is_grid_aligned = bool(compat.grid.aligned)

        crs_str = obs_1.geo.crs or obs_2.geo.crs or "UNKNOWN"
        modality_str = compat.modality.pair_type.value

        pair = TemporalObservationPair(
            pair_id=f"pair_{obs_1.observation_id}_{obs_2.observation_id}",
            pre_observation_id=obs_1.observation_id,
            post_observation_id=obs_2.observation_id,
            pre_acquisition_time=t1,
            post_acquisition_time=t2,
            delta_seconds=delta_sec,
            overlap_fraction=overlap_frac,
            grid_aligned=is_grid_aligned,
            crs=crs_str,
            modality_pair=modality_str,
            is_valid_temporal_order=is_ordered,
        )

        verification = VerificationEngine.verify_temporal_pair(
            obs_1=obs_1,
            obs_2=obs_2,
            overlap_fraction=overlap_frac,
            grid_aligned=is_grid_aligned,
            tool_name="satquery.analytics.temporal.TemporalAnalytics.build_temporal_pair",
        )

        return pair, verification

    @classmethod
    def compute_vegetation_change(
        cls,
        bands_t1: Mapping[str, np.ndarray],
        bands_t2: Mapping[str, np.ndarray],
        transform: AffineTransform | Affine,
        crs: str,
        *,
        loss_threshold: float = 0.20,
        gain_threshold: float = 0.20,
        detect_mode: str = "loss",  # "loss", "gain", or "both"
        sensor_t1: str | None = None,
        sensor_t2: str | None = None,
        pair_id: str = "temporal_pair",
        pre_obs_id: str = "obs_t1",
        post_obs_id: str = "obs_t2",
    ) -> tuple[np.ndarray, TemporalChangeResult, ChangeMaskEvidence]:
        """Compute vegetation change between two optical observations using NDVI differencing.

        Delta NDVI = NDVI(T2) - NDVI(T1).
        - Vegetation Loss: Delta NDVI <= -loss_threshold and NDVI(T1) >= 0.20.
        - Vegetation Gain: Delta NDVI >= gain_threshold and NDVI(T2) >= 0.20.
        """
        ndvi_t1, _ = SpectralAnalytics.compute_index(bands_t1, "ndvi", sensor=sensor_t1)
        ndvi_t2, _ = SpectralAnalytics.compute_index(bands_t2, "ndvi", sensor=sensor_t2)

        if ndvi_t1.shape != ndvi_t2.shape:
            raise MeasurementError(f"Raster dimensions do not match for temporal change: {ndvi_t1.shape} vs {ndvi_t2.shape}")

        valid_both = np.isfinite(ndvi_t1) & np.isfinite(ndvi_t2)
        delta_ndvi = np.full(ndvi_t1.shape, np.nan, dtype=np.float32)
        delta_ndvi[valid_both] = ndvi_t2[valid_both] - ndvi_t1[valid_both]

        if detect_mode == "loss":
            change_mask = valid_both & (delta_ndvi <= -loss_threshold) & (ndvi_t1 >= 0.20)
            threshold_val = -loss_threshold
            change_type_str = "vegetation_loss"
        elif detect_mode == "gain":
            change_mask = valid_both & (delta_ndvi >= gain_threshold) & (ndvi_t2 >= 0.20)
            threshold_val = gain_threshold
            change_type_str = "vegetation_gain"
        else:
            change_mask = valid_both & (np.abs(delta_ndvi) >= min(loss_threshold, gain_threshold))
            threshold_val = min(loss_threshold, gain_threshold)
            change_type_str = "vegetation_change"

        # Deterministic area calculation
        area_res = calculate_area(change_mask, transform, crs, target_unit=MeasurementUnit.M2)

        # Baseline vegetation area (in T1)
        base_mask = valid_both & (ndvi_t1 >= 0.20)
        base_area_res = calculate_area(base_mask, transform, crs, target_unit=MeasurementUnit.M2)
        pct_change = (
            float((area_res.area / base_area_res.area) * 100.0)
            if base_area_res.area > 0.0
            else None
        )

        valid_count = int(np.count_nonzero(valid_both))
        if valid_count > 0:
            deltas = delta_ndvi[valid_both]
            d_stats = {
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "min_delta": float(np.min(deltas)),
                "std_delta": float(np.std(deltas)),
            }
        else:
            d_stats = {"mean_delta": 0.0, "max_delta": 0.0, "min_delta": 0.0, "std_delta": 0.0}

        evidence_mask_id = f"cmask_{uuid.uuid4().hex[:16]}"
        mask_evidence = ChangeMaskEvidence(
            evidence_id=evidence_mask_id,
            change_type=change_type_str,
            pre_observation_id=pre_obs_id,
            post_observation_id=post_obs_id,
            changed_pixels=area_res.pixel_count,
            total_pixels=ndvi_t1.size,
            change_fraction=float(area_res.pixel_count / ndvi_t1.size),
            threshold=threshold_val,
            provenance={"method": "ndvi_differencing", "detect_mode": detect_mode},
        )

        change_result = TemporalChangeResult(
            change_id=f"change_{uuid.uuid4().hex[:16]}",
            pair_id=pair_id,
            change_type=change_type_str,
            delta_stats=d_stats,
            change_area_m2=area_res.area,
            change_pixel_count=area_res.pixel_count,
            baseline_area_m2=base_area_res.area,
            percent_change=pct_change,
            evidence_mask_id=evidence_mask_id,
            provenance={
                "tool": "satquery.analytics.temporal.TemporalAnalytics.compute_vegetation_change",
                "loss_threshold": loss_threshold,
                "gain_threshold": gain_threshold,
                "detect_mode": detect_mode,
            },
        )

        return delta_ndvi, change_result, mask_evidence

    @classmethod
    def compute_water_change(
        cls,
        bands_t1: Mapping[str, np.ndarray],
        bands_t2: Mapping[str, np.ndarray],
        transform: AffineTransform | Affine,
        crs: str,
        *,
        appearance_threshold: float = 0.20,
        disappearance_threshold: float = 0.20,
        detect_mode: str = "appearance",  # "appearance" (flood) or "disappearance" (drought)
        index_name: str = "ndwi",
        sensor_t1: str | None = None,
        sensor_t2: str | None = None,
        pair_id: str = "temporal_pair",
        pre_obs_id: str = "obs_t1",
        post_obs_id: str = "obs_t2",
    ) -> tuple[np.ndarray, TemporalChangeResult, ChangeMaskEvidence]:
        """Compute surface water change (flood/drought) using NDWI/MNDWI differencing."""
        idx_t1, _ = SpectralAnalytics.compute_index(bands_t1, index_name, sensor=sensor_t1)
        idx_t2, _ = SpectralAnalytics.compute_index(bands_t2, index_name, sensor=sensor_t2)

        if idx_t1.shape != idx_t2.shape:
            raise MeasurementError(f"Raster dimensions do not match for temporal change: {idx_t1.shape} vs {idx_t2.shape}")

        valid_both = np.isfinite(idx_t1) & np.isfinite(idx_t2)
        delta_idx = np.full(idx_t1.shape, np.nan, dtype=np.float32)
        delta_idx[valid_both] = idx_t2[valid_both] - idx_t1[valid_both]

        if detect_mode == "appearance":
            # Water appearance: significant increase in index, and post-event is positive water
            change_mask = valid_both & (delta_idx >= appearance_threshold) & (idx_t2 > 0.0)
            threshold_val = appearance_threshold
            change_type_str = "water_appearance"
        elif detect_mode == "disappearance":
            # Water recession: significant decrease in index, and pre-event was positive water
            change_mask = valid_both & (delta_idx <= -disappearance_threshold) & (idx_t1 > 0.0)
            threshold_val = -disappearance_threshold
            change_type_str = "water_disappearance"
        else:
            change_mask = valid_both & (np.abs(delta_idx) >= min(appearance_threshold, disappearance_threshold))
            threshold_val = min(appearance_threshold, disappearance_threshold)
            change_type_str = "water_change"

        area_res = calculate_area(change_mask, transform, crs, target_unit=MeasurementUnit.M2)

        base_mask = valid_both & (idx_t1 > 0.0)
        base_area_res = calculate_area(base_mask, transform, crs, target_unit=MeasurementUnit.M2)
        pct_change = (
            float((area_res.area / base_area_res.area) * 100.0)
            if base_area_res.area > 0.0
            else None
        )

        valid_count = int(np.count_nonzero(valid_both))
        if valid_count > 0:
            deltas = delta_idx[valid_both]
            d_stats = {
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "min_delta": float(np.min(deltas)),
                "std_delta": float(np.std(deltas)),
            }
        else:
            d_stats = {"mean_delta": 0.0, "max_delta": 0.0, "min_delta": 0.0, "std_delta": 0.0}

        evidence_mask_id = f"cmask_{uuid.uuid4().hex[:16]}"
        mask_evidence = ChangeMaskEvidence(
            evidence_id=evidence_mask_id,
            change_type=change_type_str,
            pre_observation_id=pre_obs_id,
            post_observation_id=post_obs_id,
            changed_pixels=area_res.pixel_count,
            total_pixels=idx_t1.size,
            change_fraction=float(area_res.pixel_count / idx_t1.size),
            threshold=threshold_val,
            provenance={"method": f"{index_name}_differencing", "detect_mode": detect_mode},
        )

        change_result = TemporalChangeResult(
            change_id=f"change_{uuid.uuid4().hex[:16]}",
            pair_id=pair_id,
            change_type=change_type_str,
            delta_stats=d_stats,
            change_area_m2=area_res.area,
            change_pixel_count=area_res.pixel_count,
            baseline_area_m2=base_area_res.area,
            percent_change=pct_change,
            evidence_mask_id=evidence_mask_id,
            provenance={
                "tool": "satquery.analytics.temporal.TemporalAnalytics.compute_water_change",
                "index_name": index_name,
                "detect_mode": detect_mode,
            },
        )

        return delta_idx, change_result, mask_evidence


def compute_bitemporal_change(
    bands_t1: Mapping[str, np.ndarray],
    bands_t2: Mapping[str, np.ndarray],
    transform: AffineTransform | Affine,
    crs: str,
    *,
    change_category: str = "vegetation",  # "vegetation" or "water"
    threshold: float = 0.20,
    detect_mode: str = "loss",
    sensor_t1: str | None = None,
    sensor_t2: str | None = None,
    pair_id: str = "temporal_pair",
) -> tuple[np.ndarray, TemporalChangeResult, ChangeMaskEvidence]:
    """Functional convenience wrapper for bi-temporal change computation."""
    if change_category == "vegetation":
        return TemporalAnalytics.compute_vegetation_change(
            bands_t1=bands_t1,
            bands_t2=bands_t2,
            transform=transform,
            crs=crs,
            loss_threshold=threshold,
            gain_threshold=threshold,
            detect_mode=detect_mode,
            sensor_t1=sensor_t1,
            sensor_t2=sensor_t2,
            pair_id=pair_id,
        )
    if change_category == "water":
        return TemporalAnalytics.compute_water_change(
            bands_t1=bands_t1,
            bands_t2=bands_t2,
            transform=transform,
            crs=crs,
            appearance_threshold=threshold,
            disappearance_threshold=threshold,
            detect_mode=detect_mode,
            sensor_t1=sensor_t1,
            sensor_t2=sensor_t2,
            pair_id=pair_id,
        )
    raise ValueError(f"Unsupported change category: {change_category!r}")
