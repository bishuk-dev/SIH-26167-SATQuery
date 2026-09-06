"""Deterministic SAR temporal backscatter analytics and flood detection."""

from __future__ import annotations

import math
from typing import Any, Mapping
import uuid

from affine import Affine
import numpy as np

from satquery.core.contracts.evidence import ChangeMaskEvidence, SarChangeResult
from satquery.geo.exceptions import MeasurementError
from satquery.geo.measurements import calculate_area, count_pixels
from satquery.geo.models import MeasurementUnit
from satquery.ingestion.models import AffineTransform
from satquery.sensors.semantics import SemanticBandRole, get_semantic_band


# Default calibrated decibel thresholds for Sentinel-1 / generic C-band SAR
DEFAULT_SAR_THRESHOLDS: dict[str, dict[str, float]] = {
    "vv": {
        "water_max_db": -16.0,          # Water bodies typically exhibit <= -16 dB in VV
        "flood_decrease_db": 3.0,       # Minimum drop from pre to post to flag flood (dB)
    },
    "vh": {
        "water_max_db": -23.0,          # Water bodies typically exhibit <= -23 dB in VH
        "flood_decrease_db": 3.0,
    },
    "hh": {
        "water_max_db": -16.0,
        "flood_decrease_db": 3.0,
    },
}


class SarTemporalAnalytics:
    """Deterministic SAR temporal processing and flood mapping engine."""

    @classmethod
    def linear_to_db(cls, linear_arr: np.ndarray) -> np.ndarray:
        """Convert linear amplitude or intensity backscatter to decibels (dB).

        Applies numerical protection against non-positive and non-finite values.
        """
        arr = np.asarray(linear_arr, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 1e-10)
        db_arr = np.full(arr.shape, np.nan, dtype=np.float32)
        db_arr[valid] = 10.0 * np.log10(arr[valid])
        return db_arr

    @classmethod
    def db_to_linear(cls, db_arr: np.ndarray) -> np.ndarray:
        """Convert decibel backscatter (dB) to linear intensity."""
        arr = np.asarray(db_arr, dtype=np.float32)
        valid = np.isfinite(arr)
        linear_arr = np.full(arr.shape, np.nan, dtype=np.float32)
        linear_arr[valid] = np.power(10.0, arr[valid] / 10.0)
        return linear_arr

    @classmethod
    def compute_backscatter_delta(
        cls,
        pre_sar_db: np.ndarray,
        post_sar_db: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Compute decibel backscatter difference: Delta = Post - Pre.

        Negative delta indicates backscatter decrease (attenuation/specular loss).
        Positive delta indicates backscatter increase.
        """
        pre_arr = np.asarray(pre_sar_db, dtype=np.float32)
        post_arr = np.asarray(post_sar_db, dtype=np.float32)

        if pre_arr.shape != post_arr.shape:
            raise MeasurementError(f"SAR raster shape mismatch: {pre_arr.shape} vs {post_arr.shape}")

        valid = np.isfinite(pre_arr) & np.isfinite(post_arr)
        delta_db = np.full(pre_arr.shape, np.nan, dtype=np.float32)
        delta_db[valid] = post_arr[valid] - pre_arr[valid]

        valid_count = int(np.count_nonzero(valid))
        if valid_count > 0:
            deltas = delta_db[valid]
            stats = {
                "mean_delta_db": float(np.mean(deltas)),
                "max_delta_db": float(np.max(deltas)),
                "min_delta_db": float(np.min(deltas)),
                "std_delta_db": float(np.std(deltas)),
                "valid_pixels": valid_count,
            }
        else:
            stats = {
                "mean_delta_db": 0.0,
                "max_delta_db": 0.0,
                "min_delta_db": 0.0,
                "std_delta_db": 0.0,
                "valid_pixels": 0,
            }

        return delta_db, stats

    @classmethod
    def detect_flood(
        cls,
        bands_pre: Mapping[str, np.ndarray],
        bands_post: Mapping[str, np.ndarray],
        transform: AffineTransform | Affine,
        crs: str,
        *,
        polarization_role: SemanticBandRole | str = SemanticBandRole.SAR_CO_POL,
        decrease_threshold_db: float | None = None,
        water_max_threshold_db: float | None = None,
        pre_observation_id: str = "sar_pre",
        post_observation_id: str = "sar_post",
    ) -> tuple[np.ndarray, SarChangeResult, ChangeMaskEvidence]:
        """Detect probable flood inundation from bi-temporal SAR backscatter.

        Physical principle: Specular reflection over smooth floodwater causes a sharp
        decrease in SAR backscatter relative to dry baseline conditions.
        """
        pol_name_pre, arr_pre = get_semantic_band(bands_pre, polarization_role, sensor="sentinel1")
        pol_name_post, arr_post = get_semantic_band(bands_post, polarization_role, sensor="sentinel1")

        data_pre = np.asarray(arr_pre, dtype=np.float32)
        data_post = np.asarray(arr_post, dtype=np.float32)

        # Check if already in dB (typical values between -40 and +10 dB) or linear amplitude/intensity
        # If mean is positive and > 50, it is likely raw amplitude or linear DN
        if np.nanmean(data_pre) > 50.0:
            pre_db = cls.linear_to_db(data_pre)
        else:
            pre_db = data_pre

        if np.nanmean(data_post) > 50.0:
            post_db = cls.linear_to_db(data_post)
        else:
            post_db = data_post

        # Resolve thresholds
        pol_clean = pol_name_pre.lower().replace("sar_", "").replace("sigma0_", "")[:2]
        threshold_config = DEFAULT_SAR_THRESHOLDS.get(pol_clean, DEFAULT_SAR_THRESHOLDS["vv"])
        dec_thresh = (
            decrease_threshold_db
            if decrease_threshold_db is not None
            else threshold_config["flood_decrease_db"]
        )
        water_thresh = (
            water_max_threshold_db
            if water_max_threshold_db is not None
            else threshold_config["water_max_db"]
        )

        delta_db, stats = cls.compute_backscatter_delta(pre_db, post_db)

        # Flood condition:
        # 1. Significant decrease in backscatter (delta_db <= -decrease_threshold)
        # 2. Post-event backscatter is characteristic of water (post_db <= water_max_threshold)
        valid = np.isfinite(delta_db) & np.isfinite(post_db)
        flood_mask = valid & (delta_db <= -abs(dec_thresh)) & (post_db <= water_thresh)

        # Measure flood area
        area_res = calculate_area(flood_mask, transform, crs, target_unit=MeasurementUnit.M2)

        flood_detected = area_res.pixel_count > 0
        mean_delta = stats["mean_delta_db"]

        evidence_mask_id = f"sar_mask_{uuid.uuid4().hex[:16]}"
        mask_evidence = ChangeMaskEvidence(
            evidence_id=evidence_mask_id,
            change_type="sar_flood_inundation",
            pre_observation_id=pre_observation_id,
            post_observation_id=post_observation_id,
            changed_pixels=area_res.pixel_count,
            total_pixels=pre_db.size,
            change_fraction=float(area_res.pixel_count / pre_db.size),
            threshold=float(-abs(dec_thresh)),
            provenance={
                "method": "sar_backscatter_differencing",
                "polarization": pol_name_pre,
                "decrease_threshold_db": dec_thresh,
                "water_max_threshold_db": water_thresh,
            },
        )

        sar_result = SarChangeResult(
            analysis_id=f"sar_change_{uuid.uuid4().hex[:16]}",
            pre_observation_id=pre_observation_id,
            post_observation_id=post_observation_id,
            polarization=pol_name_pre,
            flood_detected=flood_detected,
            flood_area_m2=area_res.area,
            flood_pixel_count=area_res.pixel_count,
            backscatter_decrease_db_threshold=dec_thresh,
            mean_backscatter_delta_db=mean_delta,
            provenance={
                "tool": "satquery.analytics.sar.SarTemporalAnalytics.detect_flood",
                "polarization": pol_name_pre,
                "water_max_threshold_db": water_thresh,
                "backscatter_stats": stats,
            },
        )

        return flood_mask, sar_result, mask_evidence


def detect_sar_flood(
    bands_pre: Mapping[str, np.ndarray],
    bands_post: Mapping[str, np.ndarray],
    transform: AffineTransform | Affine,
    crs: str,
    *,
    polarization_role: SemanticBandRole | str = SemanticBandRole.SAR_CO_POL,
    decrease_threshold_db: float | None = None,
    water_max_threshold_db: float | None = None,
    pre_observation_id: str = "sar_pre",
    post_observation_id: str = "sar_post",
) -> tuple[np.ndarray, SarChangeResult, ChangeMaskEvidence]:
    """Convenience wrapper for SarTemporalAnalytics.detect_flood."""
    return SarTemporalAnalytics.detect_flood(
        bands_pre=bands_pre,
        bands_post=bands_post,
        transform=transform,
        crs=crs,
        polarization_role=polarization_role,
        decrease_threshold_db=decrease_threshold_db,
        water_max_threshold_db=water_max_threshold_db,
        pre_observation_id=pre_observation_id,
        post_observation_id=post_observation_id,
    )
