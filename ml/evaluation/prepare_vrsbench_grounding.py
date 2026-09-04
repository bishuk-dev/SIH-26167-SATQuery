"""Prepare a small scene-safe VRSBench grounding benchmark subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "xiang709/VRSBench"
DATASET_REVISION = "6cee2968fd752a6d51c6cb2d18dded2bc0baa218"
ANNOTATION_FILE = "VRSBench_EVAL_referring.json"
ANNOTATION_SHA256 = "fd63f7c6b77a158f4cc933a1ead88fa63aa23ebcf30f2ec9be111f3567ff1b44"
IMAGE_ARCHIVE = "Images_val.zip"
IMAGE_ARCHIVE_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    f"{DATASET_REVISION}/{IMAGE_ARCHIVE}"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments/phase3a_grounding_dino_vrsbench/split_manifest.json"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/benchmarks/vrsbench_grounding_phase3a"
SELECTION_SEED = 43
SCENE_COUNT = 20
REFERENCES_PER_SCENE = 2
VALIDATION_SCENES = 12


def prepare_subset(
    manifest_path: Path,
    data_root: Path,
    *,
    allow_download: bool,
) -> dict[str, Any]:
    if manifest_path.is_file():
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        _validate_manifest(manifest)
        _materialize_images(manifest, data_root, allow_download=allow_download)
        if manifest_path.read_bytes() != original:
            raise RuntimeError("Frozen VRSBench subset manifest was modified")
        return manifest

    annotations = _load_annotations(allow_download=allow_download)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        box = parse_vrsbench_box(str(annotation["ground_truth"]))
        if box is not None:
            grouped[str(annotation["image_id"])].append(annotation)
    eligible = sorted(
        image_id
        for image_id, values in grouped.items()
        if len(values) >= REFERENCES_PER_SCENE
    )
    random.Random(SELECTION_SEED).shuffle(eligible)
    selected_scenes = eligible[:SCENE_COUNT]
    if len(selected_scenes) != SCENE_COUNT:
        raise RuntimeError("VRSBench has too few eligible grounding scenes")
    assignments = {
        image_id: "validation" if index < VALIDATION_SCENES else "test"
        for index, image_id in enumerate(selected_scenes)
    }
    image_records = _download_images(selected_scenes, data_root, allow_download)
    samples = []
    for image_id in selected_scenes:
        chosen = sorted(grouped[image_id], key=lambda value: value["question_id"])[
            :REFERENCES_PER_SCENE
        ]
        for annotation in chosen:
            box = parse_vrsbench_box(str(annotation["ground_truth"]))
            assert box is not None
            samples.append(
                {
                    "sample_id": f"vrsbench_ref_{annotation['question_id']}",
                    "scene_id": image_records[image_id]["sha256"],
                    "source_image_id": image_id,
                    "source_question_id": int(annotation["question_id"]),
                    "image_path": (
                        Path("data/benchmarks/vrsbench_grounding_phase3a")
                        / image_id
                    ).as_posix(),
                    "image_sha256": image_records[image_id]["sha256"],
                    "image_width": image_records[image_id]["width"],
                    "image_height": image_records[image_id]["height"],
                    "query": str(annotation["question"]).strip(),
                    "object_class": str(annotation["obj_cls"]),
                    "unique": bool(annotation["unique"]),
                    "ground_truth_normalized_xyxy": box,
                    "split": assignments[image_id],
                }
            )
    samples.sort(key=lambda sample: sample["source_question_id"])
    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "annotation_file": ANNOTATION_FILE,
            "annotation_sha256": ANNOTATION_SHA256,
            "image_archive": IMAGE_ARCHIVE,
            "text_annotation_license": "CC-BY-4.0",
            "image_use": "academic_research_only; verify source-image terms",
        },
        "coordinate_contract": {
            "source": "VRSBench ground_truth tokens normalized to 0-100",
            "stored": "xyxy normalized to 0-1",
        },
        "selection": {
            "seed": SELECTION_SEED,
            "eligible_rule": "valid_box_and_at_least_two_references_per_scene",
            "scenes": SCENE_COUNT,
            "references_per_scene": REFERENCES_PER_SCENE,
        },
        "scene_assignments": assignments,
        "counts": _counts(samples),
        "samples": samples,
    }
    _validate_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_vrsbench_box(value: str) -> list[float] | None:
    coordinates = [int(match) for match in re.findall(r"<(\d+)>", value)]
    if len(coordinates) != 4 or not all(0 <= item <= 100 for item in coordinates):
        return None
    x_min, y_min, x_max, y_max = coordinates
    if x_min >= x_max or y_min >= y_max:
        return None
    return [item / 100 for item in coordinates]


def _load_annotations(*, allow_download: bool) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                DATASET_ID,
                ANNOTATION_FILE,
                repo_type="dataset",
                revision=DATASET_REVISION,
                cache_dir=PROJECT_ROOT / "models/cache",
                local_files_only=not allow_download,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Pinned VRSBench annotations are unavailable; use --allow-download"
        ) from exc
    if _sha256(path) != ANNOTATION_SHA256:
        raise RuntimeError("VRSBench annotation checksum is invalid")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("VRSBench grounding annotations have an invalid schema")
    return payload


def _materialize_images(
    manifest: dict[str, Any], data_root: Path, *, allow_download: bool
) -> None:
    expected = {
        str(sample["source_image_id"]): str(sample["image_sha256"])
        for sample in manifest["samples"]
    }
    missing = [
        image_id
        for image_id, checksum in expected.items()
        if not (data_root / image_id).is_file()
        or _sha256(data_root / image_id) != checksum
    ]
    if missing:
        _download_images(missing, data_root, allow_download)
    for image_id, checksum in expected.items():
        if _sha256(data_root / image_id) != checksum:
            raise RuntimeError(f"VRSBench image checksum changed: {image_id}")


def _download_images(
    image_ids: list[str], data_root: Path, allow_download: bool
) -> dict[str, dict[str, int | str]]:
    if not allow_download:
        raise RuntimeError("VRSBench subset images are missing; use --allow-download")
    try:
        import fsspec

        archive = fsspec.filesystem("zip", fo=IMAGE_ARCHIVE_URL)
        try:
            data_root.mkdir(parents=True, exist_ok=True)
            records = {}
            for image_id in image_ids:
                with archive.open(f"Images_val/{image_id}", "rb") as source:
                    image_bytes = source.read()
                path = data_root / image_id
                path.write_bytes(image_bytes)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                records[image_id] = {
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "width": width,
                    "height": height,
                }
        finally:
            archive.close()
        return records
    except Exception as exc:
        raise RuntimeError("Could not materialize VRSBench subset images") from exc


def _validate_manifest(manifest: dict[str, Any]) -> None:
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("id") != DATASET_ID
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("annotation_sha256") != ANNOTATION_SHA256
    ):
        raise RuntimeError("VRSBench subset provenance is invalid")
    samples = manifest.get("samples")
    assignments = manifest.get("scene_assignments")
    if not isinstance(samples, list) or not samples or not isinstance(assignments, dict):
        raise RuntimeError("VRSBench subset manifest is incomplete")
    seen: dict[str, str] = {}
    for sample in samples:
        scene_id = str(sample["source_image_id"])
        split = str(sample["split"])
        if split not in {"validation", "test"}:
            raise RuntimeError("VRSBench subset contains an unsupported split")
        if seen.setdefault(scene_id, split) != split:
            raise RuntimeError("A VRSBench scene crosses split boundaries")
        if assignments.get(scene_id) != split:
            raise RuntimeError("VRSBench scene assignment is inconsistent")
        box = sample.get("ground_truth_normalized_xyxy")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise RuntimeError("VRSBench subset contains an invalid box")
    if set(seen) != set(assignments) or manifest.get("counts") != _counts(samples):
        raise RuntimeError("VRSBench subset counts or assignments are inconsistent")


def _counts(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "scenes": len(
                {
                    str(sample["source_image_id"])
                    for sample in samples
                    if sample["split"] == split
                }
            ),
            "references": sum(1 for sample in samples if sample["split"] == split),
        }
        for split in ("validation", "test")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare_subset(
        args.manifest.resolve(),
        args.data_root.resolve(),
        allow_download=args.allow_download,
    )
    print(json.dumps({"counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
