from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from ml.evaluation.prepare_p4_e02 import prepare_change_manifest
from ml.evaluation.run_p4_e02 import run_p4_e02


class FakeBackend:
    def predict(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        return np.array([[0.9, 0.1], [0.1, 0.8]], dtype=np.float32)


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
            transform=Affine(1, 0, 0, 0, -1, 2),
        ) as dataset:
            dataset.write(np.ones((3, 2, 2), dtype="uint8"))
    with rasterio.open(
        pair / "label.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=Affine(1, 0, 0, 0, -1, 2),
    ) as dataset:
        dataset.write(np.array([[1, 0], [0, 1]], dtype="uint8"), 1)


def test_p4_e02_metrics_reconstruct_from_prediction_rows(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest = prepare_change_manifest(tmp_path, dataset="levir_cd")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_p4_e02(
        manifest_path,
        tmp_path,
        tmp_path / "out",
        dataset="levir_cd",
        split="validation",
        backend=FakeBackend(),
    )

    assert result["sample_count"] == 1
    assert result["confusion"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert result["metrics"]["precision"] == pytest.approx(1.0)
    assert result["metrics"]["recall"] == pytest.approx(1.0)
    assert result["metrics"]["f1"] == pytest.approx(1.0)
    assert result["metrics"]["iou"] == pytest.approx(1.0)


def test_s2looking_cannot_change_threshold_or_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen"):
        run_p4_e02(
            tmp_path / "missing.json",
            tmp_path,
            tmp_path / "out",
            dataset="s2looking",
            split="robustness",
            threshold=0.4,
            backend=FakeBackend(),
        )
