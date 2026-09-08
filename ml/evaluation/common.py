"""Shared evaluation utilities for Phase 4 benchmark runs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def resolve_under_root(root: Path, relative: str) -> Path:
    """Resolve a relative path ensuring it does not escape root."""
    root_resolved = root.resolve()
    target = (root_resolved / relative).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError("manifest path escapes data root")
    return target


def verify_sha256(path: Path, expected: str) -> None:
    """Verify SHA-256 digest of a file against expected value."""
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"manifest hash mismatch: {path}")


def binary_confusion(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, int]:
    """Calculate binary confusion matrix counts."""
    pred_bool = np.asarray(prediction, dtype=bool)
    truth_bool = np.asarray(truth, dtype=bool)
    if valid is not None:
        valid_bool = np.asarray(valid, dtype=bool)
        pred_bool = pred_bool & valid_bool
        truth_bool = truth_bool & valid_bool
        return {
            "tp": int(np.count_nonzero(pred_bool & truth_bool & valid_bool)),
            "fp": int(np.count_nonzero(pred_bool & ~truth_bool & valid_bool)),
            "fn": int(np.count_nonzero(~pred_bool & truth_bool & valid_bool)),
            "tn": int(np.count_nonzero(~pred_bool & ~truth_bool & valid_bool)),
        }
    return {
        "tp": int(np.count_nonzero(pred_bool & truth_bool)),
        "fp": int(np.count_nonzero(pred_bool & ~truth_bool)),
        "fn": int(np.count_nonzero(~pred_bool & truth_bool)),
        "tn": int(np.count_nonzero(~pred_bool & ~truth_bool)),
    }


def binary_metrics(counts: Mapping[str, int]) -> dict[str, float | None]:
    """Calculate standard precision, recall, F1, and IoU from confusion counts."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
