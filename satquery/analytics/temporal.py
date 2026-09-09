"""Common-grid preparation and deterministic temporal differencing.

Only this module may write derived alignment rasters; originals stay
byte-identical. The destination grid is always exactly one declared
observation grid (T1 or T2) so evidence grid identity stays truthful.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

from satquery.analytics.exceptions import (
    GridPreparationError,
    InvalidRasterArrayError,
    NoSpatialOverlapError,
    SourceAssetIntegrityError,
)
from satquery.geo.pairing import PairValidator
from satquery.ingestion.models import AffineTransform, ObservationState

ResamplingMethod = Literal["bilinear", "nearest"]
ReferenceSide = Literal["t1", "t2"]

_RESAMPLING = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


@dataclass(frozen=True, slots=True)
class AlignedPair:
    t1_path: Path
    t2_path: Path
    reference_observation_id: str
    reference_crs: str
    reference_transform: AffineTransform
    reference_width: int
    reference_height: int
    reprojected: bool
    reprojected_observation_ids: tuple[str, ...]
    resampling: ResamplingMethod
    source_hashes: dict[str, str]
    derived_hashes: dict[str, str]
    valid_mask_paths: dict[str, Path | None]
    warnings: tuple[str, ...]


def temporal_difference(
    t1: np.ndarray,
    t2: np.ndarray,
    valid: np.ndarray,
) -> np.ma.MaskedArray:
    """Return the signed temporal difference ``T2 - T1``.

    Direction matters: gain and loss are not interchangeable. Invalid and
    non-finite cells stay masked; inputs are never mutated.
    """

    first = np.asarray(t1, dtype="float64")
    second = np.asarray(t2, dtype="float64")
    valid_array = np.asarray(valid, dtype=bool)
    if first.ndim != 2 or second.ndim != 2 or valid_array.ndim != 2:
        raise InvalidRasterArrayError("temporal arrays must be two-dimensional")
    if first.shape != second.shape or first.shape != valid_array.shape:
        raise InvalidRasterArrayError("temporal arrays and valid mask must have equal shapes")

    invalid = (
        ~valid_array
        | ~np.isfinite(first)
        | ~np.isfinite(second)
    )
    values = np.zeros(first.shape, dtype="float64")
    np.subtract(second, first, out=values, where=~invalid)
    return np.ma.array(values, mask=invalid)


def threshold_temporal_difference(
    t1: np.ndarray,
    t2: np.ndarray,
    *,
    threshold: float,
    direction: Literal["absolute", "increase", "decrease"],
    valid: np.ndarray,
) -> np.ndarray:
    """Deterministically binarize the signed temporal difference.

    The threshold must be supplied explicitly; there is no default. Invalid
    pixels are always ``False``. This is a generic mathematical primitive —
    it is not a change-detection method and must not be compared against
    structural-change labels without a separately audited experiment.
    """

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if direction not in ("absolute", "increase", "decrease"):
        raise ValueError(f"unsupported threshold direction: {direction!r}")

    difference = temporal_difference(t1, t2, valid)
    delta = difference.filled(np.nan)
    valid_array = np.asarray(valid, dtype=bool)
    if direction == "absolute":
        condition = np.abs(delta) >= threshold
    elif direction == "increase":
        condition = delta >= threshold
    else:
        condition = delta <= -threshold
    return valid_array & np.isfinite(delta) & condition


def _affine_from_model(transform: AffineTransform) -> Affine:
    return Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_before_alignment(
    t1: ObservationState, t2: ObservationState, reference: ReferenceSide
) -> tuple[ObservationState, ObservationState, tuple[str, ...]]:
    reference_observation, source_observation = (t1, t2) if reference == "t1" else (t2, t1)
    for role, observation in (("reference", reference_observation), ("source", source_observation)):
        if observation.geo.crs is None:
            raise GridPreparationError(f"{role} observation lacks CRS")
        if observation.geo.transform is None:
            raise GridPreparationError(f"{role} observation lacks transform")

    reference_affine = _affine_from_model(reference_observation.geo.transform)
    source_affine = _affine_from_model(source_observation.geo.transform)
    for role, affine in (("reference", reference_affine), ("source", source_affine)):
        if math.isclose(abs(affine.a * affine.e - affine.b * affine.d), 0.0, abs_tol=1e-12):
            raise GridPreparationError(
                f"{role} observation transform is singular or has zero area"
            )

    compatibility = PairValidator().validate(t1, t2)
    if compatibility.overlap.known and compatibility.overlap.overlap_fraction == 0.0:
        raise NoSpatialOverlapError(
            f"temporal pair has zero spatial overlap: {compatibility.result.reasons}"
        )
    if compatibility.crs.transformable is False:
        raise GridPreparationError(
            f"CRS cannot be transformed between observations: {compatibility.result.reasons}"
        )
    if compatibility.result.status.value == "FAIL":
        raise GridPreparationError(
            f"pair validation failed: {compatibility.result.reasons}"
        )
    return reference_observation, source_observation, tuple(compatibility.result.reasons)


def _grids_match(
    reference: ObservationState, source: ObservationState
) -> bool:
    if (reference.raster.width, reference.raster.height) != (
        source.raster.width,
        source.raster.height,
    ):
        return False
    if reference.geo.crs != source.geo.crs:
        return False
    assert reference.geo.transform is not None and source.geo.transform is not None
    return reference.geo.transform == source.geo.transform


def _reproject_onto_reference(
    reference: ObservationState,
    source: ObservationState,
    output_dir: Path,
    resampling: ResamplingMethod,
) -> tuple[Path, Path]:
    assert reference.geo.transform is not None
    reference_affine = _affine_from_model(reference.geo.transform)
    source_path = Path(source.source_asset.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_path = output_dir / (
        f"{source.observation_id}_{resampling}_on_{reference.observation_id}_grid.tif"
    )
    valid_path = output_dir / (
        f"{source.observation_id}_{resampling}_on_{reference.observation_id}_grid_valid.tif"
    )

    destination_grid = (
        reference.raster.height,
        reference.raster.width,
    )
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise GridPreparationError(
                f"source raster file lacks CRS metadata: {source_path}"
            )
        warp_grid = {
            "src_transform": src.transform,
            "src_crs": src.crs,
            "dst_transform": reference_affine,
            "dst_crs": rasterio.crs.CRS.from_string(reference.geo.crs),
        }
        out_dtype = np.dtype(src.dtypes[0])
        destination = np.zeros((src.count, *destination_grid), dtype=out_dtype)
        # start optimistic; every band must declare support for a pixel to be valid
        destination_validity = np.ones(destination_grid, dtype=bool)

        for band in range(1, src.count + 1):
            band_data = src.read(band)
            band_valid = src.read_masks(band) > 0
            nodata = src.nodatavals[band - 1]
            if nodata is not None:
                with np.errstate(invalid="ignore"):
                    band_valid &= band_data != nodata

            if resampling == "bilinear":
                # mask-aware normalized bilinear: invalid source pixels must
                # never contaminate interpolated neighbours. Zero the invalid
                # contributions, reproject data and support weights with the
                # same kernel, then renormalize where support exists.
                weight_source = band_valid.astype("float64")
                data_source = np.where(band_valid, band_data.astype("float64"), 0.0)
                numerator = np.zeros(destination_grid, dtype="float64")
                weight = np.zeros(destination_grid, dtype="float64")
                rasterio.warp.reproject(
                    source=data_source,
                    destination=numerator,
                    resampling=Resampling.bilinear,
                    **warp_grid,
                )
                rasterio.warp.reproject(
                    source=weight_source,
                    destination=weight,
                    resampling=Resampling.bilinear,
                    **warp_grid,
                )
                supported = weight > 0.0
                values = np.zeros(destination_grid, dtype="float64")
                np.divide(numerator, weight, out=values, where=supported)
                destination[band - 1] = values.astype(out_dtype)
                destination_validity &= supported
            else:
                band_out = np.zeros(destination_grid, dtype=out_dtype)
                rasterio.warp.reproject(
                    source=rasterio.band(src, band),
                    destination=band_out,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                    resampling=Resampling.nearest,
                    **warp_grid,
                )
                destination[band - 1] = band_out
                validity_out = np.zeros(destination_grid, dtype="uint8")
                rasterio.warp.reproject(
                    source=band_valid.astype("uint8"),
                    destination=validity_out,
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                    **warp_grid,
                )
                destination_validity &= validity_out > 0

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=reference.raster.width,
            height=reference.raster.height,
            crs=reference.geo.crs,
            transform=reference_affine,
        )

    with rasterio.open(derived_path, "w", **profile) as target:
        target.write(destination)

    valid_profile = profile.copy()
    valid_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(valid_path, "w", **valid_profile) as target:
        target.write(destination_validity.astype("uint8"), 1)

    return derived_path, valid_path


def prepare_common_grid(
    t1: ObservationState,
    t2: ObservationState,
    output_dir: Path,
    *,
    reference: ReferenceSide = "t1",
    resampling: ResamplingMethod = "bilinear",
) -> AlignedPair:
    """Place a temporal pair on exactly one declared observation grid.

    Source files are hash-verified against their registered SHA-256 before
    anything else runs. Identical grids return the original paths untouched;
    otherwise only the non-reference observation is reprojected (with an
    explicit valid-mask raster) and originals stay byte-identical.
    """

    if resampling not in _RESAMPLING:
        raise ValueError(f"unsupported resampling method: {resampling!r}")
    if reference not in ("t1", "t2"):
        raise ValueError(f"reference must be 't1' or 't2', got {reference!r}")

    for observation in (t1, t2):
        actual = _file_sha256(Path(observation.source_asset.path))
        if actual != observation.source_asset.sha256:
            raise SourceAssetIntegrityError(
                f"source asset hash mismatch for {observation.observation_id}: "
                f"{observation.source_asset.path}"
            )

    reference_observation, source_observation, warnings = _validate_before_alignment(
        t1, t2, reference
    )
    reference_id, source_id = (
        (reference_observation.observation_id, source_observation.observation_id)
    )
    paths: dict[str, Path] = {
        t1.observation_id: Path(t1.source_asset.path),
        t2.observation_id: Path(t2.source_asset.path),
    }

    if _grids_match(reference_observation, source_observation):
        return AlignedPair(
            t1_path=paths[t1.observation_id],
            t2_path=paths[t2.observation_id],
            reference_observation_id=reference_id,
            reference_crs=reference_observation.geo.crs,
            reference_transform=reference_observation.geo.transform,
            reference_width=reference_observation.raster.width,
            reference_height=reference_observation.raster.height,
            reprojected=False,
            reprojected_observation_ids=(),
            resampling=resampling,
            source_hashes={
                observation_id: _file_sha256(path) for observation_id, path in paths.items()
            },
            derived_hashes={},
            valid_mask_paths={t1.observation_id: None, t2.observation_id: None},
            warnings=warnings,
        )

    derived_path, valid_path = _reproject_onto_reference(
        reference_observation, source_observation, Path(output_dir), resampling
    )
    return AlignedPair(
        t1_path=(
            paths[t1.observation_id]
            if reference == "t1"
            else derived_path
        ),
        t2_path=(
            paths[t2.observation_id]
            if reference == "t2"
            else derived_path
        ),
        reference_observation_id=reference_id,
        reference_crs=reference_observation.geo.crs,
        reference_transform=reference_observation.geo.transform,
        reference_width=reference_observation.raster.width,
        reference_height=reference_observation.raster.height,
        reprojected=True,
        reprojected_observation_ids=(source_id,),
        resampling=resampling,
        source_hashes={
            observation_id: _file_sha256(path) for observation_id, path in paths.items()
        },
        derived_hashes={source_id: _file_sha256(derived_path)},
        valid_mask_paths={source_id: valid_path, reference_id: None},
        warnings=warnings,
    )
