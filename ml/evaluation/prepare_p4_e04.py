"""Prepare a local Sentinel-1 flood segmentation manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import rasterio


def prepare_flood_manifest(
    source_root: Path,
    *,
    dataset: Literal["sturm_flood", "sen1floods11"],
) -> dict[str, Any]:
    samples = []
    for scene_dir in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        paths = {path.name.lower(): path for path in scene_dir.iterdir() if path.is_file()}
        if not {"image.tif", "label.tif"}.issubset(paths):
            continue
        with rasterio.open(paths["image.tif"]) as image, rasterio.open(paths["label.tif"]) as label:
            if (image.count, image.width, image.height) != (2, 128, 128):
                raise ValueError(f"{scene_dir}: STURM requires 2-band 128x128 input")
            if label.count != 1 or (label.width, label.height) != (128, 128):
                raise ValueError(f"{scene_dir}: invalid flood label dimensions")
            if image.crs is None or label.crs != image.crs or label.transform != image.transform:
                raise ValueError(f"{scene_dir}: image and label grids are not verified")
        relative = scene_dir.relative_to(source_root).as_posix()
        samples.append(
            {
                "sample_id": relative,
                "split": "external_holdout" if dataset == "sen1floods11" else "validation",
                "image": f"{relative}/image.tif",
                "label": f"{relative}/label.tif",
                "image_sha256": _sha256(paths["image.tif"]),
                "label_sha256": _sha256(paths["label.tif"]),
                "polarizations": ["VV", "VH"],
                "radiometric_domain": "backscatter_db",
                "gsd_m": 10.0,
            }
        )
    if not samples:
        raise ValueError(f"No {dataset} flood scenes found")
    return {
        "schema_version": 1,
        "dataset": dataset,
        "profile": "sturm_s1_audited_v1",
        "threshold": 0.5,
        "samples": samples,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()
