"""Build the small, scene-grouped RSVQA-LR subset used by Phase 2A."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET_ID = "dmarsili/RSVQA-LR-2k"
DATASET_REVISION = "35de41f26170edda2ccc4f88c0f62f641bb9e1f1"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"
DEFAULT_MANIFEST = Path(
    "experiments/phase2a_smolvlm_rsvqa_lr/split_manifest.json"
)
DEFAULT_DATA_ROOT = Path("data/benchmarks/rsvqa_lr_2k")
SCENE_COUNT = 12
QUESTIONS_PER_SCENE = 2
SCAN_ROWS = 260


def build_subset(manifest_path: Path, data_root: Path) -> dict[str, object]:
    rows = _fetch_rows(SCAN_ROWS)
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(_download_image, rows))

    scenes: dict[str, dict[str, object]] = {}
    for row, image_bytes in zip(rows, images, strict=True):
        scene_id = hashlib.sha256(image_bytes).hexdigest()
        scene = scenes.setdefault(
            scene_id,
            {"image": image_bytes, "rows": []},
        )
        scene["rows"].append(row)
        first_scenes = list(scenes.values())[:SCENE_COUNT]
        if len(first_scenes) == SCENE_COUNT and all(
            len(item["rows"]) >= QUESTIONS_PER_SCENE for item in first_scenes
        ):
            break

    if len(scenes) < SCENE_COUNT:
        raise RuntimeError("Dataset scan did not contain enough unique scenes")
    selected_scene_ids = list(scenes)[:SCENE_COUNT]
    assignments = _split_scenes(selected_scene_ids)
    data_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for scene_id in selected_scene_ids:
        scene = scenes[scene_id]
        image_path = data_root / f"{scene_id}.jpg"
        image_path.write_bytes(scene["image"])
        for row in scene["rows"][:QUESTIONS_PER_SCENE]:
            samples.append(
                {
                    "sample_id": f"rsvqa_lr_2k_row_{row['row_idx']}",
                    "scene_id": scene_id,
                    "source_row": row["row_idx"],
                    "image_path": image_path.as_posix(),
                    "question": row["question"],
                    "answer": row["answer"],
                    "split": assignments[scene_id],
                }
            )

    _assert_scene_integrity(samples)
    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": "CC-BY-4.0",
            "upstream": "RSVQA-LR validation data",
        },
        "subset": {
            "selection": "first_12_unique_scenes_in_source_order",
            "scene_count": SCENE_COUNT,
            "questions_per_scene": QUESTIONS_PER_SCENE,
            "scene_group_key": "sha256_image_bytes",
        },
        "split_policy": {
            "method": "sort_scene_sha256_then_7_train_2_validation_3_test",
            "grouped_by_scene": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_assignments": assignments,
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


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
        rows.extend(
            {
                "row_idx": item["row_idx"],
                "question": item["row"]["question"],
                "answer": item["row"]["answer"],
                "image_url": item["row"]["image"]["src"],
            }
            for item in payload["rows"]
        )
    return rows


def _download_image(row: dict[str, object]) -> bytes:
    with urlopen(str(row["image_url"]), timeout=60) as response:
        return response.read()


def _split_scenes(scene_ids: list[str]) -> dict[str, str]:
    ordered = sorted(scene_ids)
    assignments = {}
    for index, scene_id in enumerate(ordered):
        if index < 7:
            split = "train"
        elif index < 9:
            split = "validation"
        else:
            split = "test"
        assignments[scene_id] = split
    return assignments


def _assert_scene_integrity(samples: list[dict[str, object]]) -> None:
    seen: dict[str, str] = {}
    for sample in samples:
        scene_id = str(sample["scene_id"])
        split = str(sample["split"])
        previous = seen.setdefault(scene_id, split)
        if previous != split:
            raise RuntimeError(f"Scene {scene_id} crosses split boundaries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    manifest = build_subset(args.manifest, args.data_root)
    counts = {
        split: sum(1 for sample in manifest["samples"] if sample["split"] == split)
        for split in ("train", "validation", "test")
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
