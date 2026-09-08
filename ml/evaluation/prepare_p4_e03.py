"""Prepare a local LEVIR-CC-style caption manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import rasterio


def prepare_levir_cc(source_root: Path, contract: object | None = None) -> dict[str, Any]:
    samples = []
    for pair_dir in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        paths = {path.name.lower(): path for path in pair_dir.iterdir() if path.is_file()}
        if not {"a.tif", "b.tif", "references.json"}.issubset(paths):
            continue
        with rasterio.open(paths["a.tif"]) as first, rasterio.open(paths["b.tif"]) as second:
            if first.count != 3 or second.count != 3:
                raise ValueError(f"{pair_dir}: LEVIR-CC requires RGB pairs")
            if (first.width, first.height, first.crs, first.transform) != (
                second.width,
                second.height,
                second.crs,
                second.transform,
            ):
                raise ValueError(f"{pair_dir}: pair grid is not verified")
        references = json.loads(paths["references.json"].read_text(encoding="utf-8"))
        if not isinstance(references, list) or not references or not all(isinstance(item, str) and item.strip() for item in references):
            raise ValueError(f"{pair_dir}: references must be a non-empty string list")
        relative = pair_dir.relative_to(source_root).as_posix()
        samples.append(
            {
                "pair_id": relative,
                "split": "validation",
                "t1": f"{relative}/A.tif",
                "t2": f"{relative}/B.tif",
                "t1_sha256": _sha256(paths["a.tif"]),
                "t2_sha256": _sha256(paths["b.tif"]),
                "references": [item.strip() for item in references],
            }
        )
    if not samples:
        raise ValueError("No LEVIR-CC caption pairs found")
    return {
        "schema_version": 1,
        "dataset": "levir_cc",
        "profile": "chg2cap_audited_v1",
        "samples": samples,
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()
