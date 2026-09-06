"""Deterministic spectral index analytics and evidence generation."""

from __future__ import annotations

import math
from typing import Any, Mapping
import uuid

import numpy as np

from satquery.core.contracts.evidence import (
    IndexRasterEvidence,
    MaskEvidence,
    RasterStatistic,
    SpectralAnalysisResult,
)
from satquery.core.contracts.temporal import AnalysisROI
from satquery.geo.exceptions import MissingBandError, SpectralIndexError
from satquery.sensors.semantics import SemanticBandRole, get_semantic_band


# Normalized difference indices and their required semantic band roles (Band A, Band B)
INDEX_ROLE_MAPPING: dict[str, tuple[SemanticBandRole, SemanticBandRole]] = {
    "ndvi": (SemanticBandRole.NIR, SemanticBandRole.RED),
    "ndwi": (SemanticBandRole.GREEN, SemanticBandRole.NIR),
    "mndwi": (SemanticBandRole.GREEN, SemanticBandRole.SWIR),
    "ndwi_gao": (SemanticBandRole.NIR, SemanticBandRole.SWIR),
    "ndbi": (SemanticBandRole.SWIR, SemanticBandRole.NIR),
    "nbr": (SemanticBandRole.NIR, SemanticBandRole.SWIR),
}

# Standard thresholds for physical classification
STANDARD_THRESHOLDS: dict[str, dict[str, float]] = {
    "ndvi": {"vegetation_dense": 0.50, "vegetation_moderate": 0.30, "sparse": 0.10},
    "ndwi": {"water_body": 0.0},
    "mndwi": {"water_body": 0.0},
    "ndbi": {"built_up": 0.0},
    "nbr": {"high_severity_burn": -0.25, "moderate_burn": -0.10},
}


class SpectralAnalytics:
    """Deterministic spectral index computer and evidence generator."""

    @classmethod
    def compute_index(
        cls,
        bands: Mapping[str, np.ndarray],
        index_name: str,
        *,
        sensor: str | None = None,
        roi: AnalysisROI | None = None,
        nodata_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, RasterStatistic]:
        """Compute a normalized difference spectral index using semantic band roles.

        Args:
            bands: Mapping of band names to 2D numpy arrays.
            index_name: One of 'ndvi', 'ndwi', 'mndwi', 'ndwi_gao', 'ndbi', 'nbr'.
            sensor: Optional sensor name (e.g. 'sentinel2', 'landsat') to assist role resolution.
            roi: Optional AnalysisROI bounding box / polygon to restrict computation.
            nodata_mask: Optional boolean mask where True indicates NoData pixels.

        Returns:
            (index_array_float32, RasterStatistic)
        """
        norm_index = index_name.lower().strip()
        if norm_index not in INDEX_ROLE_MAPPING:
            raise SpectralIndexError(
                f"Unsupported spectral index: {index_name!r}. Supported: {list(INDEX_ROLE_MAPPING.keys())}"
            )

        role_a, role_b = INDEX_ROLE_MAPPING[norm_index]
        _, arr_a = get_semantic_band(bands, role_a, sensor=sensor)
        _, arr_b = get_semantic_band(bands, role_b, sensor=sensor)

        data_a = np.asarray(arr_a, dtype=np.float32)
        data_b = np.asarray(arr_b, dtype=np.float32)

        if data_a.ndim != 2 or data_b.ndim != 2:
            raise SpectralIndexError(f"Spectral bands must be 2D arrays, got {data_a.ndim}D and {data_b.ndim}D")
        if data_a.shape != data_b.shape:
            raise SpectralIndexError(f"Spectral band shape mismatch: {data_a.shape} vs {data_b.shape}")

        height, width = data_a.shape
        invalid = ~np.isfinite(data_a) | ~np.isfinite(data_b)

        if nodata_mask is not None:
            invalid |= np.asarray(nodata_mask, dtype=bool)

        # Apply ROI pixel bounds if provided
        if roi is not None and roi.pixel_bounds is not None:
            min_x, min_y, max_x, max_y = roi.pixel_bounds
            roi_mask = np.zeros((height, width), dtype=bool)
            # Clip bounds safely within raster extents
            c_min_x = max(0, min_x)
            c_min_y = max(0, min_y)
            c_max_x = min(width, max_x)
            c_max_y = min(height, max_y)
            roi_mask[c_min_y:c_max_y, c_min_x:c_max_x] = True
            invalid |= ~roi_mask

        num = data_a - data_b
        den = data_a + data_b

        # Division by zero or near-zero protection
        near_zero = np.abs(den) < 1e-7
        invalid |= near_zero

        result = np.full(data_a.shape, np.nan, dtype=np.float32)
        valid_idx = ~invalid
        np.divide(num, den, out=result, where=valid_idx)

        # Clamp strictly to physical range [-1.0, 1.0]
        np.clip(result, -1.0, 1.0, out=result, where=valid_idx)

        valid_count = int(np.count_nonzero(valid_idx))
        total_pixels = data_a.size
        nodata_count = int(total_pixels - valid_count)

        if valid_count > 0:
            valid_vals = result[valid_idx]
            min_v = float(np.min(valid_vals))
            max_v = float(np.max(valid_vals))
            mean_v = float(np.mean(valid_vals))
            std_v = float(np.std(valid_vals))
        else:
            min_v, max_v, mean_v, std_v = 0.0, 0.0, 0.0, 0.0

        stats = RasterStatistic(
            min_value=min_v,
            max_value=max_v,
            mean_value=mean_v,
            std_value=std_v,
            valid_pixels=valid_count,
            nodata_pixels=nodata_count,
        )

        return result, stats

    @classmethod
    def create_threshold_mask(
        cls,
        index_data: np.ndarray,
        *,
        min_threshold: float | None = None,
        max_threshold: float | None = None,
        source_observation_id: str = "unknown",
        mask_type: str = "custom_threshold",
    ) -> tuple[np.ndarray, MaskEvidence]:
        """Generate a boolean mask and corresponding MaskEvidence from an index raster."""
        data = np.asarray(index_data)
        valid = np.isfinite(data)
        mask = valid.copy()

        criteria: dict[str, float] = {}
        if min_threshold is not None:
            mask &= (data >= min_threshold)
            criteria["min_threshold"] = float(min_threshold)
        if max_threshold is not None:
            mask &= (data <= max_threshold)
            criteria["max_threshold"] = float(max_threshold)

        pos_count = int(np.count_nonzero(mask))
        tot_count = int(data.size)

        evidence = MaskEvidence(
            evidence_id=f"mask_{uuid.uuid4().hex[:16]}",
            mask_type=mask_type,
            source_observation_id=source_observation_id,
            positive_pixels=pos_count,
            total_pixels=tot_count,
            threshold_criteria=criteria,
        )
        return mask, evidence

    @classmethod
    def generate_index_evidence(
        cls,
        index_data: np.ndarray,
        stats: RasterStatistic,
        *,
        index_name: str,
        observation_id: str,
        crs: str,
        transform: tuple[float, float, float, float, float, float],
    ) -> IndexRasterEvidence:
        """Create structured IndexRasterEvidence for the computed index."""
        return IndexRasterEvidence(
            evidence_id=f"idx_{uuid.uuid4().hex[:16]}",
            index_name=index_name,
            observation_id=observation_id,
            stats=stats,
            crs=crs,
            transform=transform,
            shape=(int(index_data.shape[0]), int(index_data.shape[1])),
            provenance={
                "index_name": index_name,
                "tool": "satquery.analytics.spectral.SpectralAnalytics",
                "version": "1.0.0",
            },
        )


def compute_index(
    bands: Mapping[str, np.ndarray],
    index_name: str,
    *,
    sensor: str | None = None,
    roi: AnalysisROI | None = None,
    nodata_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, RasterStatistic]:
    """Functional convenience wrapper for SpectralAnalytics.compute_index."""
    return SpectralAnalytics.compute_index(
        bands=bands,
        index_name=index_name,
        sensor=sensor,
        roi=roi,
        nodata_mask=nodata_mask,
    )
