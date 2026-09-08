"""Evaluate deterministic OSCD change masks without touching sealed test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio

from satquery.analytics.spectral import compute_index
from satquery.analytics.temporal import prepare_common_grid, threshold_temporal_difference


class SealedTestAccessError(ValueError):
    """Raised when an evaluator is asked to inspect the sealed test split."""


def run_p4_e01(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    *,
    split: Literal["validation"],
) -> dict[str, Any]:
    if split != "validation":
        raise SealedTestAccessError("P4-E01 only permits the validation split")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported P4-E01 manifest")
    band_order = tuple(payload.get("band_order", ()))
    required = {"B04", "B08"}
    if not required.issubset(band_order):
        raise ValueError("P4-E01 requires semantic B04 and B08 bands")
    samples = [row for row in payload.get("samples", ()) if row.get("split") == split]
    if not samples:
        raise ValueError("P4-E01 validation has no samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold = float(payload.get("threshold", 0.05))
    for sample in samples:
        pair_id = str(sample["pair_id"])
        t1 = _resolve(data_root, sample["t1"])
        t2 = _resolve(data_root, sample["t2"])
        label_path = _resolve(data_root, sample["label"])
        _verify_hash(t1, sample["t1_sha256"])
        _verify_hash(t2, sample["t2_sha256"])
        _verify_hash(label_path, sample["label_sha256"])
        aligned = prepare_common_grid(
            t1,
            t2,
            output_dir / "aligned" / _safe_id(pair_id),
            resampling="bilinear",
        )
        with rasterio.open(aligned.t1_path) as first, rasterio.open(
            aligned.t2_path
        ) as second, rasterio.open(label_path) as label_file:
            valid = np.ones((first.height, first.width), dtype=bool)
            bands_first = {
                "RED": first.read(band_order.index("B04") + 1),
                "NIR": first.read(band_order.index("B08") + 1),
            }
            bands_second = {
                "RED": second.read(band_order.index("B04") + 1),
                "NIR": second.read(band_order.index("B08") + 1),
            }
            valid &= np.isfinite(bands_first["RED"]) & np.isfinite(bands_first["NIR"])
            valid &= np.isfinite(bands_second["RED"]) & np.isfinite(bands_second["NIR"])
            if label_file.nodata is not None:
                valid &= label_file.read(1) != label_file.nodata
            truth = label_file.read(1) > 0
            first_index = compute_index("ndvi", bands_first, valid).filled(np.nan)
            second_index = compute_index("ndvi", bands_second, valid).filled(np.nan)
            prediction = threshold_temporal_difference(
                first_index,
                second_index,
                threshold=threshold,
                valid=valid,
            )
            counts = _confusion(prediction, truth, valid)
            mask_path = output_dir / "masks" / f"{_safe_id(pair_id)}.tif"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            profile = first.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=0)
            with rasterio.open(mask_path, "w", **profile) as mask_file:
                mask_file.write(prediction.astype("uint8"), 1)
        rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "t1_sha256": sample["t1_sha256"],
                "t2_sha256": sample["t2_sha256"],
                "selected_bands": ["B04", "B08"],
                "grid_operation": "verified_common_grid_bilinear",
                "threshold": threshold,
                "positive_pixel_count": int(prediction.sum()),
                **counts,
                "mask_path": str(mask_path.relative_to(output_dir)),
            }
        )
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    result = {
        "schema_version": 1,
        "split": split,
        "sample_count": len(rows),
        "predictions_path": str(predictions_path),
        "confusion": totals,
        "metrics": _metrics(totals),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _confusion(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero(prediction & truth & valid)),
        "fp": int(np.count_nonzero(prediction & ~truth & valid)),
        "fn": int(np.count_nonzero(~prediction & truth & valid)),
        "tn": int(np.count_nonzero(~prediction & ~truth & valid)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _resolve(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("manifest path escapes data root")
    return path


def _verify_hash(path: Path, expected: str) -> None:
    with path.open("rb") as file_handle:
        actual = hashlib.file_digest(file_handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"manifest hash mismatch: {path}")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args()
    result = run_p4_e01(
        args.manifest,
        args.data_root,
        args.output_dir,
        split=args.split,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
