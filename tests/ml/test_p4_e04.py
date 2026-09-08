from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from ml.evaluation.prepare_p4_e04 import prepare_flood_manifest
from ml.evaluation.run_p4_e04 import run_p4_e04


class FakeBackend:
    def segment(self, image: np.ndarray) -> np.ndarray:
        return np.full((128, 128), 0.75, dtype=np.float32)


def _fixture(root: Path) -> None:
    pair = root / "scene_01"
    pair.mkdir()
    for name, count in (("image.tif", 2), ("label.tif", 1)):
        with rasterio.open(
            pair / name,
            "w",
            driver="GTiff",
            width=128,
            height=128,
            count=count,
            dtype="float32" if count == 2 else "uint8",
            crs="EPSG:32643",
            transform=(10, 0, 0, 0, -10, 1280),
        ) as dataset:
            dataset.write(np.ones((count, 128, 128), dtype="float32" if count == 2 else "uint8"))


def test_flood_metrics_reconstruct_from_saved_confusion_counts(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(prepare_flood_manifest(tmp_path, dataset="sturm_flood")), encoding="utf-8")

    result = run_p4_e04(
        manifest_path,
        tmp_path,
        tmp_path / "out",
        dataset="sturm_flood",
        split="validation",
        backend=FakeBackend(),
    )

    assert result["sample_count"] == 1
    assert result["metrics"]["iou"] == pytest.approx(1.0)


def test_sen1floods11_cannot_tune_sturm_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_p4_e04(
            tmp_path / "missing.json",
            tmp_path,
            tmp_path / "out",
            dataset="sen1floods11",
            split="external_holdout",
            threshold=0.6,
            backend=FakeBackend(),
        )
