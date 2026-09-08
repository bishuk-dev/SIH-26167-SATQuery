from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from ml.evaluation.prepare_p4_e03 import prepare_levir_cc
from ml.evaluation.run_p4_e03 import run_p4_e03


class FakeCaptionBackend:
    def caption(self, t1: np.ndarray, t2: np.ndarray) -> str:
        return "A building appears."


def _fixture(root: Path) -> None:
    pair = root / "pair_01"
    pair.mkdir()
    for name in ("A.tif", "B.tif"):
        with rasterio.open(
            pair / name,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=3,
            dtype="uint8",
            crs="EPSG:32643",
            transform=(1, 0, 0, 0, -1, 2),
        ) as dataset:
            dataset.write(np.ones((3, 2, 2), dtype="uint8"))
    (pair / "references.json").write_text(
        json.dumps(["A building appears.", "A new structure appears."]),
        encoding="utf-8",
    )


def test_levir_cc_pair_and_all_captions_stay_in_one_split(tmp_path: Path) -> None:
    _fixture(tmp_path)

    manifest = prepare_levir_cc(tmp_path)

    assert len(manifest["samples"]) == 1
    assert {row["split"] for row in manifest["samples"]} == {"validation"}
    assert len(manifest["samples"][0]["references"]) == 2


def test_caption_metric_input_counts_match_declared_samples(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(prepare_levir_cc(tmp_path)), encoding="utf-8")

    result = run_p4_e03(
        manifest_path,
        tmp_path,
        tmp_path / "out",
        split="validation",
        backend=FakeCaptionBackend(),
    )

    assert result.prediction_count == 1
    assert result.reference_count == 2
