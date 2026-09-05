"""Inspect only the three Phase 4C-frozen training pairs after materialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from ml.evaluation.materialize_phase4_bigearthnet import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    EXPERIMENT_DIR,
    MaterializationError,
    load_frozen_manifest,
)
from satquery.inference.multisensor_preprocessing import materialized_band_paths

DEFAULT_OUTPUT = EXPERIMENT_DIR / "representative_raster_audit.json"
FROZEN_SAMPLE_IDS = (
    "ben2_pair_54d6cd2db80bec3b88bd1bde",
    "ben2_pair_ceb2b2ff7000d820e6d70028",
    "ben2_pair_90d4a6de879ee97d50bf2c2d",
)
S1_BANDS = ("VV", "VH")
S2_RESOLUTION = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
}


class NativeRasterAuditError(RuntimeError):
    """Raised when frozen native raster expectations are not observed."""


def inspect_frozen_train_pairs(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Measure all native bands for exactly the predeclared train candidates."""

    manifest = load_frozen_manifest(manifest_path)
    _require_completed_modalities(dataset_root)
    samples = manifest["samples"]
    assert isinstance(samples, list)
    by_id = {sample["sample_id"]: sample for sample in samples}
    observations = []
    for sample_id in FROZEN_SAMPLE_IDS:
        sample = by_id.get(sample_id)
        if not isinstance(sample, dict) or sample.get("official_split") != "train":
            raise NativeRasterAuditError(
                f"Frozen representative {sample_id} is missing or is not train"
            )
        paths = materialized_band_paths(dataset_root, sample, split="train")
        band_records = {name: _inspect_band(name, path) for name, path in paths.items()}
        checks = _spatial_checks(band_records)
        observations.append(
            {
                "sample_id": sample_id,
                "country": sample["country"],
                "geographic_group_id": sample["geographic_group_id"],
                "s1_name": sample["s1_name"],
                "s2_patch_id": sample["patch_id"],
                "bands": [band_records[name] for name in (*S1_BANDS, *S2_RESOLUTION)],
                "checks": checks,
            }
        )
    return {
        "schema_version": 1,
        "status": "measured_from_three_predeclared_train_pairs",
        "manifest_sha256": (
            "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
        ),
        "sample_ids": list(FROZEN_SAMPLE_IDS),
        "test_pixels_opened": False,
        "native_geotiff_observations": observations,
    }


def _inspect_band(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise NativeRasterAuditError(f"Materialized band is missing: {path}")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(path, sharing=False) as dataset:
            if dataset.count != 1:
                raise NativeRasterAuditError(f"Expected one band in {path.name}")
            data = dataset.read(1, masked=True)
            raw = np.asarray(data.data)
            mask = np.ma.getmaskarray(data)
            finite = np.isfinite(raw)
            valid = ~mask & finite
            values = raw[valid]
            if values.size == 0:
                raise NativeRasterAuditError(f"No valid pixels in {path.name}")
            percentiles = np.percentile(values.astype(np.float64), [1, 5, 50, 95, 99])
            tags = dataset.tags(1)
            return {
                "semantic_band": name,
                "filename": path.name,
                "dtype": dataset.dtypes[0],
                "width": dataset.width,
                "height": dataset.height,
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "transform": list(dataset.transform)[:6],
                "bounds": [
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                ],
                "pixel_size": [
                    math.hypot(dataset.transform.a, dataset.transform.d),
                    math.hypot(dataset.transform.b, dataset.transform.e),
                ],
                "nodata": dataset.nodata,
                "scale": dataset.scales[0],
                "offset": dataset.offsets[0],
                "unit": dataset.units[0],
                "band_tags": dict(sorted(tags.items())),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "percentiles": {
                    key: float(value)
                    for key, value in zip(
                        ("p01", "p05", "p50", "p95", "p99"), percentiles
                    )
                },
                "nan_count": int(np.isnan(raw).sum()),
                "inf_count": int(np.isinf(raw).sum()),
                "invalid_value_count": int((~valid).sum()),
            }


def _spatial_checks(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vv = records["VV"]
    s1_aligned = all(
        records[name][field] == vv[field]
        for name in S1_BANDS
        for field in ("width", "height", "crs", "transform", "bounds")
    )
    if not s1_aligned:
        raise NativeRasterAuditError("S1 VV/VH grids are not aligned")

    s2_expected = all(
        records[name]["width"] == 1200 // resolution
        and records[name]["height"] == 1200 // resolution
        and all(
            abs(float(pixel) - resolution) <= 0.01
            for pixel in records[name]["pixel_size"]
        )
        for name, resolution in S2_RESOLUTION.items()
    )
    if not s2_expected:
        raise NativeRasterAuditError("S2 native 10/20/60 m grid contract failed")

    reference = records["B02"]
    s2_footprints = all(
        record["crs"] == reference["crs"]
        and np.allclose(record["bounds"], reference["bounds"], atol=0.01)
        for name, record in records.items()
        if name in S2_RESOLUTION
    )
    if not s2_footprints:
        raise NativeRasterAuditError("S2 native band footprints are inconsistent")

    if vv["crs"] is None or reference["crs"] is None:
        colocated = False
    else:
        vv_bounds = transform_bounds(
            str(vv["crs"]), str(reference["crs"]), *vv["bounds"]
        )
        colocated = bool(np.allclose(vv_bounds, reference["bounds"], atol=10.0))
    if not colocated:
        raise NativeRasterAuditError(
            "S1/S2 georeferenced footprints are not co-located"
        )

    s1_db_consistent = all(
        -60.0 <= float(records[name]["percentiles"]["p50"]) <= 20.0 for name in S1_BANDS
    )
    if not s1_db_consistent:
        raise NativeRasterAuditError(
            "S1 values are inconsistent with the documented decibel domain"
        )
    return {
        "s1_vv_vh_grid_aligned": s1_aligned,
        "s1_values_consistent_with_documented_db_domain": s1_db_consistent,
        "s2_native_10_20_60m_grids_verified": s2_expected,
        "s2_cross_band_footprints_consistent": s2_footprints,
        "s1_s2_footprints_colocated": colocated,
    }


def _require_completed_modalities(root: Path) -> None:
    for modality in ("s1", "s2"):
        marker = root / modality / "_materialization.json"
        try:
            payload: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MaterializationError(
                f"Cannot inspect rasters before {modality} materialization completes"
            ) from exc
        if payload.get("manifest_sha256") != (
            "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
        ):
            raise MaterializationError(f"{modality} marker has the wrong manifest SHA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = inspect_frozen_train_pairs(
        manifest_path=args.manifest, dataset_root=args.dataset_root
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
