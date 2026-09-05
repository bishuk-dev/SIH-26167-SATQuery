"""Deterministic BigEarthNet v2 inputs for the frozen BIFOLD baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from satquery.registry.models import BifoldPreprocessingProfile


class MultisensorPreprocessingError(ValueError):
    """Raised when native bands cannot safely satisfy a frozen input contract."""


class SealedTestAccessError(PermissionError):
    """Raised when ordinary code attempts to open Phase 4 test imagery."""


def preprocess_bifold_bands(
    bands: Mapping[str, np.ndarray],
    profile: BifoldPreprocessingProfile,
    *,
    validity_masks: Mapping[str, np.ndarray] | None = None,
    nodata_values: Mapping[str, float | int | None] | None = None,
) -> torch.Tensor:
    """Map semantic native bands to a normalized ``C x 120 x 120`` tensor.

    Channel position comes only from the frozen profile. Masked, non-finite, or
    explicit NoData pixels fail closed because BIFOLD's published contract does
    not define an imputation value.
    """

    missing = [name for name in profile.band_order if name not in bands]
    if missing:
        raise MultisensorPreprocessingError(
            f"Missing required semantic bands: {', '.join(missing)}"
        )

    resized: list[torch.Tensor] = []
    for name in profile.band_order:
        raw = bands[name]
        masked: Any = np.ma.asarray(raw)
        values = np.asarray(masked.data)
        if values.ndim != 2 or 0 in values.shape:
            raise MultisensorPreprocessingError(
                f"Band {name} must be a non-empty two-dimensional raster"
            )

        invalid: Any = np.ma.getmaskarray(masked).astype(bool, copy=True)
        invalid |= ~np.isfinite(values)
        if validity_masks is not None and name in validity_masks:
            valid = np.asarray(validity_masks[name], dtype=bool)
            if valid.shape != values.shape:
                raise MultisensorPreprocessingError(
                    f"Validity mask shape does not match band {name}"
                )
            invalid |= ~valid
        nodata = None if nodata_values is None else nodata_values.get(name)
        if nodata is not None:
            invalid |= np.isnan(values) if np.isnan(nodata) else values == nodata
        if invalid.any():
            raise MultisensorPreprocessingError(
                f"Band {name} contains masked, non-finite, or NoData pixels"
            )

        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        if tensor.shape != (profile.height, profile.width):
            tensor = functional.interpolate(
                tensor[None, None],
                size=(profile.height, profile.width),
                mode=profile.continuous_resampling,
            )[0, 0]
        resized.append(tensor)

    stacked = torch.stack(resized)
    means = torch.tensor(profile.means, dtype=torch.float32)[:, None, None]
    stds = torch.tensor(profile.stds, dtype=torch.float32)[:, None, None]
    return (stacked - means) / stds


def materialized_band_paths(
    dataset_root: Path,
    sample: Mapping[str, object],
    *,
    split: str,
    allow_sealed_test: bool = False,
) -> dict[str, Path]:
    """Resolve a manifest sample without exposing sealed test pixels by default."""

    official_split = sample.get("official_split")
    if split != official_split:
        raise MultisensorPreprocessingError(
            f"Requested split {split!r} does not match manifest split {official_split!r}"
        )
    if split == "test" and not allow_sealed_test:
        raise SealedTestAccessError(
            "Phase 4 test imagery is sealed; an explicit final-evaluation action is required"
        )
    storage_split = "sealed_test" if split == "test" else split
    s1_name = _required_manifest_text(sample, "s1_name")
    patch_id = _required_manifest_text(sample, "patch_id")
    paths = {
        band: dataset_root / "s1" / storage_split / s1_name / f"{s1_name}_{band}.tif"
        for band in ("VV", "VH")
    }
    paths.update(
        {
            band: dataset_root
            / "s2"
            / storage_split
            / patch_id
            / f"{patch_id}_{band}.tif"
            for band in (
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            )
        }
    )
    return paths


def _required_manifest_text(sample: Mapping[str, object], field: str) -> str:
    value = sample.get(field)
    if not isinstance(value, str) or not value:
        raise MultisensorPreprocessingError(f"Manifest sample has invalid {field}")
    return value
