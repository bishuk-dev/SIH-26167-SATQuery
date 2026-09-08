"""Run frozen Sentinel-1 flood segmentation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import rasterio

from satquery.inference.exceptions import ModelUnavailableError


class FloodEvaluationBackend(Protocol):
    def segment(self, image: np.ndarray) -> np.ndarray: ...


def run_p4_e04(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    *,
    dataset: Literal["sturm_flood", "sen1floods11"],
    split: Literal["validation", "external_holdout"],
    backend: FloodEvaluationBackend | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if dataset == "sen1floods11" and threshold is not None and threshold != 0.5:
        raise ValueError("Sen1Floods11 must use the frozen STURM threshold")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("profile") != "sturm_s1_audited_v1":
        raise ValueError("manifest does not match the frozen P4-E04 contract")
    if backend is None:
        raise ModelUnavailableError("P4-E04 requires an explicitly supplied STURM backend")
    samples = [row for row in payload.get("samples", ()) if row.get("split") == split]
    if not samples:
        raise ValueError("P4-E04 split has no samples")
    threshold_value = float(payload.get("threshold", 0.5) if threshold is None else threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        image_path = _resolve(data_root, sample["image"])
        label_path = _resolve(data_root, sample["label"])
        _verify_hash(image_path, sample["image_sha256"])
        _verify_hash(label_path, sample["label_sha256"])
        with rasterio.open(image_path) as image_file, rasterio.open(label_path) as label_file:
            scores = np.asarray(backend.segment(image_file.read((1, 2))), dtype="float32")
            if scores.shape != (128, 128) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError("STURM backend returned invalid scores")
            counts = _confusion(scores >= threshold_value, label_file.read(1) > 0)
        rows.append({"sample_id": sample["sample_id"], "split": split, "threshold": threshold_value, **counts})
    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    result = {"schema_version": 1, "dataset": dataset, "split": split, "sample_count": len(rows), "prediction_path": str(prediction_path), "confusion": totals, "metrics": _metrics(totals)}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.count_nonzero(prediction & truth)),
        "fp": int(np.count_nonzero(prediction & ~truth)),
        "fn": int(np.count_nonzero(~prediction & truth)),
        "tn": int(np.count_nonzero(~prediction & ~truth)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sturm_flood", "sen1floods11"), required=True)
    parser.add_argument("--split", choices=("validation", "external_holdout"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_p4_e04(args.manifest, args.data_root, args.output_dir, dataset=args.dataset, split=args.split), sort_keys=True))


if __name__ == "__main__":
    main()
