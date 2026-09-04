from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.evaluation.prepare_vrsbench_grounding import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    parse_vrsbench_box,
    prepare_subset,
)
from ml.evaluation.run_phase3a_grounding import intersection_over_union


def test_committed_vrsbench_manifest_is_scene_safe_and_reproducible() -> None:
    before = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    manifest = prepare_subset(DEFAULT_MANIFEST, DEFAULT_DATA_ROOT, allow_download=False)
    after = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()

    validation_scenes = {
        sample["source_image_id"]
        for sample in manifest["samples"]
        if sample["split"] == "validation"
    }
    test_scenes = {
        sample["source_image_id"]
        for sample in manifest["samples"]
        if sample["split"] == "test"
    }
    assert manifest["counts"] == {
        "validation": {"scenes": 12, "references": 24},
        "test": {"scenes": 8, "references": 16},
    }
    assert validation_scenes.isdisjoint(test_scenes)
    assert before == after


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{<25><40><33><60>}", [0.25, 0.4, 0.33, 0.6]),
        ("{<0><0><100><100>}", [0.0, 0.0, 1.0, 1.0]),
        ("{<25><40><25><60>}", None),
        ("not-a-box", None),
    ],
)
def test_vrsbench_box_parser(raw: str, expected: list[float] | None) -> None:
    assert parse_vrsbench_box(raw) == expected


def test_grounding_metrics_use_actual_top_prediction() -> None:
    expected = [0.25, 0.25, 0.75, 0.75]
    assert intersection_over_union(expected, expected) == 1.0
    assert intersection_over_union(None, expected) == 0.0
    assert intersection_over_union([0.0, 0.0, 0.5, 0.5], expected) == pytest.approx(
        1 / 7
    )
