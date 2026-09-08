from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.common import (
    binary_confusion,
    binary_metrics,
    resolve_under_root,
    verify_sha256,
)


def test_resolve_under_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    child = data_dir / "sample.tif"
    child.write_bytes(b"hello")

    assert resolve_under_root(data_dir, "sample.tif") == child

    with pytest.raises(ValueError, match="escapes data root"):
        resolve_under_root(data_dir, "../other.tif")


def test_verify_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"satquery")
    expected_hash = hashlib.sha256(b"satquery").hexdigest()

    verify_sha256(test_file, expected_hash)

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        verify_sha256(test_file, "0" * 64)


def test_binary_confusion_and_metrics() -> None:
    pred = np.array([1, 1, 0, 0], dtype=bool)
    truth = np.array([1, 0, 1, 0], dtype=bool)

    counts = binary_confusion(pred, truth)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    metrics = binary_metrics(counts)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)


def test_binary_confusion_with_valid_mask() -> None:
    pred = np.array([1, 1, 0, 0], dtype=bool)
    truth = np.array([1, 0, 1, 0], dtype=bool)
    valid = np.array([True, True, False, True], dtype=bool)

    counts = binary_confusion(pred, truth, valid=valid)
    assert counts == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}


def test_binary_metrics_zeros() -> None:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 5}
    metrics = binary_metrics(counts)
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["iou"] is None
