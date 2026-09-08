"""Run the frozen ChangerEx evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import rasterio

from ml.evaluation.common import (
    binary_confusion,
    binary_metrics,
    resolve_under_root,
    verify_sha256,
)
from satquery.inference.exceptions import ModelUnavailableError


class BinaryChangeBackend(Protocol):
    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray: ...


def run_p4_e02(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    *,
    dataset: Literal["levir_cd", "s2looking"],
    split: Literal["validation", "robustness"],
    backend: BinaryChangeBackend | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is not None and dataset == "s2looking" and threshold != 0.5:
        raise ValueError("S2Looking must use the frozen ChangerEx threshold")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("profile") != "changerex_audited_v1":
        raise ValueError("manifest does not match the frozen P4-E02 contract")
    if split not in {"validation", "robustness"}:
        raise ValueError("unsupported P4-E02 split")
    samples = [row for row in payload.get("samples", ()) if row.get("split") == split]
    if not samples:
        raise ValueError("P4-E02 split has no samples")
    if backend is None:
        raise ModelUnavailableError("P4-E02 requires an explicitly supplied ChangerEx backend")
    threshold_value = float(payload.get("threshold", 0.5) if threshold is None else threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        t1 = resolve_under_root(data_root, sample["t1"])
        t2 = resolve_under_root(data_root, sample["t2"])
        label_path = resolve_under_root(data_root, sample["label"])
        for path, key in ((t1, "t1_sha256"), (t2, "t2_sha256"), (label_path, "label_sha256")):
            verify_sha256(path, sample[key])
        with rasterio.open(t1) as first, rasterio.open(t2) as second, rasterio.open(label_path) as label_file:
            first_rgb = first.read((1, 2, 3)).astype("float32") / 255.0
            second_rgb = second.read((1, 2, 3)).astype("float32") / 255.0
            scores = np.asarray(backend.predict(first_rgb, second_rgb), dtype="float32")
            if scores.shape != (first.height, first.width) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError("ChangerEx backend returned invalid scores")
            truth = label_file.read(1) > 0
            prediction = scores >= threshold_value
            counts = binary_confusion(prediction, truth)
        rows.append({"pair_id": sample["pair_id"], "split": split, "threshold": threshold_value, **counts})
    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    result = {"schema_version": 1, "dataset": dataset, "split": split, "sample_count": len(rows), "prediction_path": str(prediction_path), "confusion": totals, "metrics": binary_metrics(totals)}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("levir_cd", "s2looking"), required=True)
    parser.add_argument("--split", choices=("validation", "robustness"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_p4_e02(args.manifest, args.data_root, args.output_dir, dataset=args.dataset, split=args.split), sort_keys=True))


if __name__ == "__main__":
    main()
