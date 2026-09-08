from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from ml.evaluation.prepare_p4_e01 import prepare_oscd_manifest
from ml.evaluation.run_p4_e01 import SealedTestAccessError, run_p4_e01
from ml.evaluation.phase4_contracts import DatasetContract, DatasetSemantics


BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def _contract() -> DatasetContract:
    return DatasetContract(
        status="BLOCKED",
        role="primary_benchmark",
        blockers=("fixture only",),
        contract=DatasetSemantics(
            modalities=("multispectral_optical",),
            sensors=("Sentinel-2",),
            bands_or_polarizations=BANDS,
            radiometric_domain="fixture",
            pair_order="T1_then_T2",
            labels="pixel_level_binary_change",
        ),
    )


def _write_pair(root: Path) -> None:
    pair = root / "pair_01"
    pair.mkdir(parents=True)
    transform = Affine(10, 0, 0, 0, -10, 20)
    for name, value in (("T1.tif", 1.0), ("T2.tif", 1.0)):
        data = np.full((13, 2, 2), value, dtype="float32")
        data[7, 0, 0] = 2.0 if name == "T2.tif" else 1.0
        with rasterio.open(
            pair / name,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=13,
            dtype="float32",
            crs="EPSG:32643",
            transform=transform,
        ) as dataset:
            dataset.write(data)
    with rasterio.open(
        pair / "label.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=transform,
    ) as dataset:
        dataset.write(np.array([[1, 0], [0, 0]], dtype="uint8"), 1)


def test_oscd_manifest_preserves_all_13_semantic_bands_and_pairs(tmp_path: Path) -> None:
    _write_pair(tmp_path)

    manifest = prepare_oscd_manifest(tmp_path, _contract())

    assert manifest["band_order"] == list(BANDS)
    assert all(row["pair_id"] and row["t1"] != row["t2"] for row in manifest["samples"])


def test_runner_refuses_test_split(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(prepare_oscd_manifest(tmp_path, _contract())),
        encoding="utf-8",
    )

    with pytest.raises(SealedTestAccessError):
        run_p4_e01(manifest_path, tmp_path, tmp_path / "out", split="test")


def test_runner_writes_reconstructible_fixture_metrics(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(prepare_oscd_manifest(tmp_path, _contract())),
        encoding="utf-8",
    )

    result = run_p4_e01(manifest_path, tmp_path, tmp_path / "out", split="validation")

    assert result["sample_count"] == 1
    assert result["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 3}
    assert result["metrics"]["precision"] == pytest.approx(1.0)
    assert result["metrics"]["recall"] == pytest.approx(1.0)
    assert result["metrics"]["f1"] == pytest.approx(1.0)
    assert result["metrics"]["iou"] == pytest.approx(1.0)
