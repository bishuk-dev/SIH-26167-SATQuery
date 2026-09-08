"""Run caption generation while preserving every reference caption."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import rasterio

from ml.evaluation.common import resolve_under_root, verify_sha256


class CaptionBackend(Protocol):
    def caption(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> str: ...


@dataclass(frozen=True, slots=True)
class CaptionEvaluationResult:
    prediction_path: Path
    prediction_count: int
    reference_count: int
    metrics: dict[str, float | None]


def run_p4_e03(
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    *,
    split: Literal["validation"],
    backend: CaptionBackend | None = None,
) -> CaptionEvaluationResult:
    if split != "validation":
        raise ValueError("P4-E03 only permits the validation split")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("dataset") != "levir_cc" or payload.get("profile") != "chg2cap_audited_v1":
        raise ValueError("manifest does not match the frozen P4-E03 contract")
    if backend is None:
        raise ValueError("P4-E03 requires an explicitly supplied Chg2Cap backend")
    samples = [row for row in payload.get("samples", ()) if row.get("split") == split]
    if not samples:
        raise ValueError("P4-E03 validation has no samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        t1 = resolve_under_root(data_root, sample["t1"])
        t2 = resolve_under_root(data_root, sample["t2"])
        verify_sha256(t1, sample["t1_sha256"])
        verify_sha256(t2, sample["t2_sha256"])
        with rasterio.open(t1) as first, rasterio.open(t2) as second:
            caption = backend.caption(
                first.read((1, 2, 3)).astype("float32") / 255.0,
                second.read((1, 2, 3)).astype("float32") / 255.0,
            ).strip()
        if not caption:
            raise ValueError("caption backend returned an empty caption")
        rows.append({"pair_id": sample["pair_id"], "caption": caption, "references": sample["references"]})
    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    result = CaptionEvaluationResult(
        prediction_path=prediction_path,
        prediction_count=len(rows),
        reference_count=sum(len(row["references"]) for row in rows),
        metrics={"BLEU-4": None, "METEOR": None, "ROUGE-L": None, "CIDEr": None},
    )
    (output_dir / "metrics.json").write_text(json.dumps(asdict(result), default=str, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args()
    result = run_p4_e03(args.manifest, args.data_root, args.output_dir, split=args.split)
    print(json.dumps(asdict(result), default=str, sort_keys=True))


if __name__ == "__main__":
    main()
