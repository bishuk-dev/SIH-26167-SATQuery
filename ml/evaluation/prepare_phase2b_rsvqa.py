"""Prepare the scene-grouped RSVQA-LR adaptation manifest for Phase 2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from ml.evaluation.rsvqa_lr_subset import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SERVER,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2A_MANIFEST = (
    PROJECT_ROOT / "experiments/phase2a_smolvlm_rsvqa_lr/split_manifest.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments/phase2b_smolvlm_rsvqa_lr/split_manifest.json"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/benchmarks/rsvqa_lr_phase2b"
SOURCE_ROW_LIMIT = 2000
SPLIT_SEED = 42


def build_manifest(
    manifest_path: Path,
    data_root: Path,
    phase2a_manifest_path: Path,
    *,
    source_row_limit: int = SOURCE_ROW_LIMIT,
) -> dict[str, object]:
    rows = _fetch_rows(source_row_limit)
    rows_by_url: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_url.setdefault(str(row["image_url"]), []).append(row)

    urls = list(rows_by_url)
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(_download_image, urls))

    scenes: dict[str, dict[str, object]] = {}
    for image_url, image_bytes in zip(urls, images, strict=True):
        scene_id = hashlib.sha256(image_bytes).hexdigest()
        scene = scenes.setdefault(scene_id, {"image": image_bytes, "rows": []})
        scene["rows"].extend(rows_by_url[image_url])

    phase2a = json.loads(phase2a_manifest_path.read_text(encoding="utf-8"))
    excluded_scenes = set(phase2a["scene_assignments"])
    selected_scene_ids = sorted(set(scenes) - excluded_scenes)
    if len(selected_scene_ids) < 20:
        raise RuntimeError("Too few scenes remain after excluding Phase 2A")
    assignments = _assign_splits(selected_scene_ids, seed=SPLIT_SEED)

    data_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for scene_id in selected_scene_ids:
        scene = scenes[scene_id]
        image_path = data_root / f"{scene_id}.jpg"
        image_path.write_bytes(scene["image"])
        for row in scene["rows"]:
            samples.append(
                {
                    "sample_id": f"rsvqa_lr_2k_row_{row['row_idx']}",
                    "scene_id": scene_id,
                    "source_row": row["row_idx"],
                    "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "question": row["question"],
                    "answer": row["answer"],
                    "split": assignments[scene_id],
                }
            )
    samples.sort(key=lambda item: int(item["source_row"]))
    _assert_integrity(samples, excluded_scenes)

    counts = {
        split: {
            "scenes": sum(1 for value in assignments.values() if value == split),
            "questions": sum(1 for item in samples if item["split"] == split),
        }
        for split in ("train", "validation", "test")
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
            "source_split": "validation",
            "source_row_limit": source_row_limit,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_group_key": "sha256_image_bytes",
        "split_policy": {
            "seed": SPLIT_SEED,
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "method": "seeded_scene_shuffle_then_80_10_10",
        },
        "excluded_phase2a_scene_ids": sorted(excluded_scenes),
        "counts": counts,
        "scene_assignments": assignments,
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _assign_splits(scene_ids: list[str], *, seed: int) -> dict[str, str]:
    ordered = sorted(scene_ids)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    validation_count = max(1, round(count * 0.1))
    test_count = max(1, round(count * 0.1))
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError("At least three scenes are required")
    assignments: dict[str, str] = {}
    for index, scene_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments[scene_id] = split
    return assignments


def _fetch_rows(limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(0, limit, 100):
        length = min(100, limit - offset)
        query = urlencode(
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": "validation",
                "offset": offset,
                "length": length,
            }
        )
        with urlopen(f"{DATASET_SERVER}?{query}", timeout=60) as response:
            payload = json.load(response)
        page = payload.get("rows", [])
        rows.extend(
            {
                "row_idx": item["row_idx"],
                "question": item["row"]["question"],
                "answer": item["row"]["answer"],
                "image_url": item["row"]["image"]["src"],
            }
            for item in page
        )
        if len(page) < length:
            break
    if not rows:
        raise RuntimeError("RSVQA-LR source returned no rows")
    return rows


def _download_image(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _assert_integrity(
    samples: list[dict[str, object]], excluded_scenes: set[str]
) -> None:
    splits_by_scene: dict[str, set[str]] = {}
    for sample in samples:
        scene_id = str(sample["scene_id"])
        splits_by_scene.setdefault(scene_id, set()).add(str(sample["split"]))
    if excluded_scenes & set(splits_by_scene):
        raise RuntimeError("Phase 2A scenes leaked into Phase 2B")
    if any(len(splits) != 1 for splits in splits_by_scene.values()):
        raise RuntimeError("A scene crosses Phase 2B split boundaries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--phase2a-manifest", type=Path, default=DEFAULT_PHASE2A_MANIFEST
    )
    parser.add_argument("--source-row-limit", type=int, default=SOURCE_ROW_LIMIT)
    args = parser.parse_args()
    manifest = build_manifest(
        args.manifest.resolve(),
        args.data_root.resolve(),
        args.phase2a_manifest.resolve(),
        source_row_limit=args.source_row_limit,
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
