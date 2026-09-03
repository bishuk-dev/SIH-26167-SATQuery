from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "phase2a_smolvlm_rsvqa_lr"
    / "split_manifest.json"
)


def test_phase2a_manifest_is_scene_grouped_and_has_expected_split_sizes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        scene_splits[sample["scene_id"]].add(sample["split"])

    assert len(samples) == 24
    assert len(scene_splits) == 12
    assert all(len(splits) == 1 for splits in scene_splits.values())
    assert Counter(sample["split"] for sample in samples) == {
        "train": 14,
        "validation": 4,
        "test": 6,
    }
    assert all(
        manifest["scene_assignments"][scene_id] == next(iter(splits))
        for scene_id, splits in scene_splits.items()
    )
