"""Verified common-grid preparation and deterministic temporal differences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass(frozen=True, slots=True)
class AlignedPair:
    t1_path: Path
    t2_path: Path
    crs: str
    t1_transform: Affine
    t2_transform: Affine
    reprojected: bool


def prepare_common_grid(
    t1_path: str | Path,
    t2_path: str | Path,
    output_dir: str | Path,
    *,
    resampling: str,
) -> AlignedPair:
    """Write both rasters on T1's verified grid and return derived paths."""

    try:
        method = Resampling[resampling]
    except KeyError as exc:
        raise ValueError(f"Unsupported resampling method: {resampling}") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    first_output = output / "t1_aligned.tif"
    second_output = output / "t2_aligned.tif"
    with rasterio.open(t1_path, "r", sharing=False) as first, rasterio.open(
        t2_path, "r", sharing=False
    ) as second:
        if first.crs is None or second.crs is None:
            raise ValueError("common-grid preparation requires CRS on both rasters")
        if first.count != second.count:
            raise ValueError("temporal rasters must have equal band counts")
        reprojected = not (
            first.crs == second.crs
            and first.transform == second.transform
            and first.width == second.width
            and first.height == second.height
        )
        profile = first.profile.copy()
        profile.update(
            driver="GTiff",
            width=first.width,
            height=first.height,
            count=first.count,
            crs=first.crs,
            transform=first.transform,
        )
        _write_on_grid(
            first,
            first_output,
            profile,
            first.crs,
            first.transform,
            method,
        )
        _write_on_grid(
            second,
            second_output,
            profile,
            first.crs,
            first.transform,
            method,
        )
        return AlignedPair(
            t1_path=first_output,
            t2_path=second_output,
            crs=first.crs.to_string(),
            t1_transform=first.transform,
            t2_transform=first.transform,
            reprojected=reprojected,
        )


def threshold_temporal_difference(
    t1: np.ndarray,
    t2: np.ndarray,
    *,
    threshold: float,
    valid: np.ndarray,
) -> np.ndarray:
    """Return pixels whose absolute temporal difference exceeds threshold."""

    if threshold < 0:
        raise ValueError("temporal threshold cannot be negative")
    first, second, valid_array = (
        np.asarray(t1, dtype="float64"),
        np.asarray(t2, dtype="float64"),
        np.asarray(valid, dtype=bool),
    )
    if first.shape != second.shape or first.shape != valid_array.shape:
        raise ValueError("temporal arrays and valid mask must have equal shapes")
    finite = np.isfinite(first) & np.isfinite(second)
    return valid_array & finite & (np.abs(second - first) > threshold)


def _write_on_grid(
    source: rasterio.DatasetReader,
    destination: Path,
    profile: dict[str, object],
    crs: object,
    transform: Affine,
    resampling: Resampling,
) -> None:
    with rasterio.open(destination, "w", **profile) as target:
        for band in range(1, source.count + 1):
            values = np.zeros((target.height, target.width), dtype=source.dtypes[band - 1])
            reproject(
                source=rasterio.band(source, band),
                destination=values,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling,
                src_nodata=source.nodatavals[band - 1],
                dst_nodata=target.nodatavals[band - 1],
            )
            target.write(values, band)
