"""Prepare a local LEVIR-CD/S2Looking-style RGB pair manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import rasterio


def prepare_change_manifest(
    source_root: Path,
    *,
    dataset: Literal["levir_cd", "s2looking"],
) -> dict[str, Any]:
    samples = []
    for pair_dir in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        paths = {path.name.lower(): path for path in pair_dir.iterdir() if path.is_file()}
        if not {"a.tif", "b.tif", "label.tif"}.issubset(paths):
            continue
        with rasterio.open(paths["a.tif"]) as first, rasterio.open(paths["b.tif"]) as second, rasterio.open(paths["label.tif"]) as label:
            if first.count != 3 or second.count != 3:
                raise ValueError(f"{pair_dir}: expected RGB pairs")
            if (first.width, first.height, first.crs, first.transform) != (
                second.width,
                second.height,
                second.crs,
                second.transform,
            ) or label.crs != first.crs or label.transform != first.transform:
                raise ValueError(f"{pair_dir}: pair and label grids are not verified")
            if (label.width, label.height) != (first.width, first.height) or label.count != 1:
                raise ValueError(f"{pair_dir}: invalid label dimensions")
        relative = pair_dir.relative_to(source_root).as_posix()
        samples.append(
            {
                "pair_id": relative,
                "split": "robustness" if dataset == "s2looking" else "validation",
                "t1": f"{relative}/A.tif",
                "t2": f"{relative}/B.tif",
                "label": f"{relative}/label.tif",
                "t1_sha256": _sha256(paths["a.tif"]),
                "t2_sha256": _sha256(paths["b.tif"]),
                "label_sha256": _sha256(paths["label.tif"]),
            }
        )
    if not samples:
        raise ValueError(f"No {dataset} change pairs found")
    return {
        "schema_version": 1,
        "dataset": dataset,
        "profile": "changerex_audited_v1",
        "threshold": 0.5,
        "samples": samples,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()
