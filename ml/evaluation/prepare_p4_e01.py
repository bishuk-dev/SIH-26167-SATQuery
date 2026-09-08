"""Prepare a validated local OSCD-style pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import rasterio

from ml.evaluation.phase4_contracts import DatasetContract, DatasetSemantics

EXPECTED_BAND_COUNT = 13


def prepare_oscd_manifest(source_root: Path, contract: DatasetContract) -> dict[str, Any]:
    """Discover pairs, validate native raster contracts, and write no derived data."""

    if contract.contract is None:
        raise ValueError("OSCD dataset contract is missing band semantics")
    band_order = contract.contract.bands_or_polarizations
    if len(band_order) != EXPECTED_BAND_COUNT:
        raise ValueError("OSCD contract must identify all 13 semantic bands")
    samples = []
    for pair_dir in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        paths = {path.name.lower(): path for path in pair_dir.iterdir() if path.is_file()}
        if not {"t1.tif", "t2.tif", "label.tif"}.issubset(paths):
            continue
        sample = _validate_pair(source_root, pair_dir, paths)
        samples.append(sample)
    if not samples:
        raise ValueError("No OSCD temporal pairs found")
    return {
        "schema_version": 1,
        "dataset_id": "oscd",
        "band_order": list(band_order),
        "threshold": 0.05,
        "samples": samples,
    }


def _validate_pair(source_root: Path, pair_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    with rasterio.open(paths["t1.tif"]) as t1, rasterio.open(paths["t2.tif"]) as t2, rasterio.open(
        paths["label.tif"]
    ) as label:
        if t1.count != EXPECTED_BAND_COUNT or t2.count != EXPECTED_BAND_COUNT:
            raise ValueError(f"{pair_dir}: expected 13 bands in T1 and T2")
        if (t1.width, t1.height, t1.crs, t1.transform) != (
            t2.width,
            t2.height,
            t2.crs,
            t2.transform,
        ):
            raise ValueError(f"{pair_dir}: temporal pair is not on a common grid")
        if t1.crs is None or label.crs != t1.crs or label.transform != t1.transform:
            raise ValueError(f"{pair_dir}: label grid or CRS is not verified")
        if (label.width, label.height) != (t1.width, t1.height) or label.count != 1:
            raise ValueError(f"{pair_dir}: label dimensions do not match imagery")
    split_file = pair_dir / "split.txt"
    split = split_file.read_text(encoding="utf-8").strip() if split_file.exists() else "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"{pair_dir}: unknown split {split!r}")
    relative = pair_dir.relative_to(source_root).as_posix()
    return {
        "pair_id": relative,
        "split": split,
        "t1": f"{relative}/T1.tif",
        "t2": f"{relative}/T2.tif",
        "label": f"{relative}/label.tif",
        "t1_sha256": _sha256(paths["t1.tif"]),
        "t2_sha256": _sha256(paths["t2.tif"]),
        "label_sha256": _sha256(paths["label.tif"]),
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A blocked audit contract may prepare a local manifest, but never authorizes a benchmark claim.
    contract = DatasetContract(
        status="BLOCKED",
        role="primary_benchmark",
        blockers=("external source audit required",),
        contract=DatasetSemantics(
            modalities=("multispectral_optical",),
            sensors=("Sentinel-2",),
            bands_or_polarizations=(
                "B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", "B10", "B11", "B12",
            ),
            radiometric_domain="audit_required",
            pair_order="T1_then_T2",
            labels="pixel_level_binary_change",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prepare_oscd_manifest(args.source_root, contract), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
