#!/usr/bin/env python3
"""P4-E04: SAR Temporal Flood Inundation Validation on Modified Sen1Floods11.

Downloads and verifies the authoritative Modified Sen1Floods11 Dataset for
Change Detection from Zenodo (DOI: 10.5281/zenodo.7946594), then evaluates
deterministic SAR backscatter-differencing flood detection against pixel-level
post-event water extent ground truth labels.

All metrics are derived from real per-scene TP/FP/TN/FN computed from actual
raster evaluation. No synthetic or placeholder data is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


BENCHMARK_PROVENANCE: dict[str, Any] = {
    "benchmark": "Modified Sen1Floods11 Dataset for Change Detection",
    "doi": "10.5281/zenodo.7946594",
    "concept_doi": "10.5281/zenodo.7946593",
    "version": "v1",
    "publication_date": "2023-05-17",
    "creator": "Ritu Yadav / KTH Royal Institute of Technology",
    "license": "CC BY 4.0",
    "paper_doi": "10.1109/IGARSS46834.2022.9883132",
    "paper_title": "Attentive Dual Stream Siamese U-Net for Flood Detection on Multi-Temporal Sentinel-1 Data",
    "label_audit": {
        "ground_truth_target": "post_event_water_extent",
        "values": {"0": "non_water", "1": "water_inundated", "-1": "nodata"},
        "expansion_evidence": "computed_separately_as_pre_post_expansion",
    },
    "pinned_files": [
        {
            "filename": "PRE_S1-20230517T191707Z-001.zip",
            "size_bytes": 1520699949,
            "md5": "4a32637c56ea519bd3c4baca208b289d",
        },
        {
            "filename": "POST_S1-20230517T191716Z-001.zip",
            "size_bytes": 729719788,
            "md5": "40a505cfbd5318d94a9d5bd7aef88561",
        },
        {
            "filename": "Labels-20230517T191741Z-001.zip",
            "size_bytes": 2462219,
            "md5": "069b4c05eefb7a6e72c1adb34aaf1a24",
        },
    ],
}

ZENODO_BASE_URL = "https://zenodo.org/records/7946594/files"


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_file_robust(
    url: str,
    dest: Path,
    expected_size: int,
    expected_md5: str,
    max_attempts: int = 6,
) -> None:
    """Download a file with HTTP Range resume, exponential backoff, and hash verification."""
    import urllib.request
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected_size:
        actual_md5 = _md5_file(dest)
        if actual_md5 == expected_md5:
            print(f"[P4-E04] Verified existing archive: {dest.name}")
            return
        print(f"[P4-E04] MD5 mismatch for existing {dest.name}, re-downloading")
        dest.unlink()

    part_path = dest.parent / f"{dest.name}.part"

    for attempt in range(1, max_attempts + 1):
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if existing_bytes > expected_size:
            print(f"[P4-E04] .part size ({existing_bytes}) exceeds expected ({expected_size}), resetting")
            part_path.unlink()
            existing_bytes = 0

        headers = {"User-Agent": "satquery-P4-E04"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            print(f"[P4-E04] Attempt {attempt}/{max_attempts}: Resuming {dest.name} from byte {existing_bytes}...")
        else:
            print(f"[P4-E04] Attempt {attempt}/{max_attempts}: Downloading {dest.name}...")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status_code = getattr(resp, "status", getattr(resp, "code", 200))
                if status_code == 206:
                    # Partial Content: server accepted Range, append
                    write_mode = "ab"
                else:
                    # 200 OK: server ignored Range header, overwrite from byte 0
                    write_mode = "wb"
                    existing_bytes = 0

                with open(part_path, write_mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            # Check downloaded .part
            actual_size = part_path.stat().st_size
            if actual_size == expected_size:
                actual_md5 = _md5_file(part_path)
                if actual_md5 == expected_md5:
                    shutil.move(str(part_path), str(dest))
                    print(f"[P4-E04] Verified & promoted {dest.name}: {actual_size} bytes, MD5={actual_md5}")
                    return
                else:
                    print(f"[P4-E04] MD5 mismatch for {part_path.name}: {actual_md5} != {expected_md5}, restarting")
                    if part_path.exists():
                        part_path.unlink()
            else:
                print(f"[P4-E04] Incomplete transfer: {actual_size}/{expected_size} bytes")

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                # Range Not Satisfiable: check if .part is already complete
                if part_path.exists() and part_path.stat().st_size == expected_size:
                    if _md5_file(part_path) == expected_md5:
                        shutil.move(str(part_path), str(dest))
                        print(f"[P4-E04] Range 416: verified complete archive {dest.name}")
                        return
                print(f"[P4-E04] Range 416 not satisfiable, resetting .part")
                if part_path.exists():
                    part_path.unlink()
            elif exc.code in (429, 500, 502, 503, 504):
                print(f"[P4-E04] Transient HTTP Error {exc.code}: {exc.reason}")
            else:
                raise RuntimeError(f"Non-retryable HTTP error {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"[P4-E04] Transient network error: {exc}")

        if attempt < max_attempts:
            backoff = min(60.0, (2.0 ** (attempt - 1)) * 2.0)
            print(f"[P4-E04] Backing off for {backoff:.1f}s before retry...")
            time.sleep(backoff)

    raise RuntimeError(f"Failed to acquire {dest.name} after {max_attempts} attempts from {url}")


def _safe_zip_extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            extracted_path = (dest_dir / info.filename).resolve()
            try:
                extracted_path.relative_to(dest_dir.resolve())
            except ValueError:
                raise RuntimeError(f"Unsafe path in zip (traversal): {info.filename}")
            if os.path.islink(extracted_path) and extracted_path.exists():
                raise RuntimeError(f"Symlink in zip: {info.filename}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _acquire_modified_sen1floods11(base_dir: Path) -> Path:
    cache_dir = base_dir / "sen1floods11_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = cache_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = cache_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    archive_env = os.environ.get("SEN1FLOODS11_ARCHIVE_DIR")
    if archive_env:
        archive_dir = Path(archive_env)
        print(f"[P4-E04] Checking SEN1FLOODS11_ARCHIVE_DIR: {archive_dir}")
        for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
            fname = pinned["filename"]
            src_file = archive_dir / fname
            if not src_file.exists():
                raise RuntimeError(f"SEN1FLOODS11_ARCHIVE_DIR missing required archive: {fname}")
            actual_size = src_file.stat().st_size
            if actual_size != pinned["size_bytes"]:
                raise RuntimeError(
                    f"SEN1FLOODS11_ARCHIVE_DIR size mismatch for {fname}: "
                    f"expected {pinned['size_bytes']}, got {actual_size}"
                )
            actual_md5 = _md5_file(src_file)
            if actual_md5 != pinned["md5"]:
                raise RuntimeError(
                    f"SEN1FLOODS11_ARCHIVE_DIR MD5 mismatch for {fname}: "
                    f"expected {pinned['md5']}, got {actual_md5}"
                )
            dest = downloads_dir / fname
            if not dest.exists() or dest.stat().st_size != pinned["size_bytes"]:
                shutil.copy2(src_file, dest)
            print(f"[P4-E04] Verified pre-provisioned archive: {fname}")
    else:
        for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
            fname = pinned["filename"]
            dest = downloads_dir / fname
            url = f"{ZENODO_BASE_URL}/{fname}?download=1"
            _download_file_robust(url, dest, pinned["size_bytes"], pinned["md5"])

    for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
        fname = pinned["filename"]
        target_dir = extracted_dir / fname.replace(".zip", "")
        if not target_dir.exists() or not any(target_dir.rglob("*.tif")):
            zip_path = downloads_dir / fname
            print(f"[P4-E04] Extracting {fname} safely...")
            target_dir.mkdir(parents=True, exist_ok=True)
            _safe_zip_extract(zip_path, target_dir)
            print(f"[P4-E04] Extracted to {target_dir}")

    return extracted_dir


LAYER_PATTERNS = [
    ("_S1Hand_post", "post_sar"),
    ("_S1Hand", "pre_sar"),
    ("_LabelHand", "label"),
]


def _derive_scene_key(filename: str) -> str | None:
    stem = Path(filename).stem
    for suffix, _layer_type in LAYER_PATTERNS:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _classify_raster(filename: str) -> str | None:
    """Classify a raster file as 'pre_sar', 'post_sar', 'label', or None."""
    stem = Path(filename).stem
    for suffix, layer_type in LAYER_PATTERNS:
        if stem.endswith(suffix):
            return layer_type
    return None


def _is_pre_sar(filename: str) -> bool:
    return _classify_raster(filename) == "pre_sar"


def _is_post_sar(filename: str) -> bool:
    return _classify_raster(filename) == "post_sar"


def _is_label(filename: str) -> bool:
    return _classify_raster(filename) == "label"


def _check_duplicate(mapping: dict[str, Path], key: str, path: Path) -> None:
    if key in mapping:
        raise RuntimeError(
            f"Duplicate scene keys detected for '{key}': "
            f"{mapping[key]} and {path}"
        )


def _find_raster_pairs(extracted_dir: Path) -> dict[str, dict[str, Any]]:
    pre_by_scene: dict[str, Path] = {}
    post_by_scene: dict[str, Path] = {}
    label_by_scene: dict[str, Path] = {}

    for tif in sorted(extracted_dir.rglob("*.tif")):
        scene_key = _derive_scene_key(tif.name)
        if scene_key is None:
            continue

        if _is_pre_sar(tif.name):
            _check_duplicate(pre_by_scene, scene_key, tif)
            pre_by_scene[scene_key] = tif
        elif _is_post_sar(tif.name):
            _check_duplicate(post_by_scene, scene_key, tif)
            post_by_scene[scene_key] = tif
        elif _is_label(tif.name):
            _check_duplicate(label_by_scene, scene_key, tif)
            label_by_scene[scene_key] = tif

    all_keys = set(pre_by_scene.keys()) | set(post_by_scene.keys()) | set(label_by_scene.keys())
    paired = {}
    for key in all_keys:
        paired[key] = {
            "pre": pre_by_scene.get(key),
            "post": post_by_scene.get(key),
            "label": label_by_scene.get(key),
        }

    return paired


def _load_raster_band(path: Path) -> tuple[np.ndarray, Affine, str]:
    import rasterio
    from affine import Affine

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
        if nodata is not None:
            arr[nodata == arr] = np.nan
    return arr, transform, crs


def _compute_post_water_mask(post_arr: np.ndarray, water_max_threshold_db: float = -16.0) -> np.ndarray:
    """Compute post-event water mask from SAR backscatter.

    Post-event water detection: pixels where post-event backscatter (in dB)
    is below the water threshold. This matches Modified Sen1Floods11's
    post-event water extent ground truth definition.
    """
    valid = np.isfinite(post_arr)
    post_water = valid & (post_arr <= water_max_threshold_db)
    return post_water.astype(np.uint8)


def _compute_flood_expansion_mask(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    water_max_threshold_db: float = -16.0,
    flood_decrease_db: float = 3.0,
) -> np.ndarray:
    """Compute newly inundated flood expansion mask.

    Secondary evidence product: pixels where backscatter decreased by >= flood_decrease_db
    AND post-event backscatter indicates water. This is NOT the benchmark comparison
    target — it is a separate SatQuery deterministic evidence product.
    """
    from satquery.analytics.sar import SarTemporalAnalytics
    from satquery.sensors.semantics import SemanticBandRole
    from affine import Affine

    mask, sar_result, evidence = SarTemporalAnalytics.detect_flood(
        bands_pre={"vv": pre_arr},
        bands_post={"vv": post_arr},
        transform=Affine(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
        polarization_role=SemanticBandRole.SAR_CO_POL,
        decrease_threshold_db=flood_decrease_db,
        water_max_threshold_db=water_max_threshold_db,
        radiometric_domain="SAR_DB_POWER",
    )
    return mask.astype(np.uint8)


def _compute_confusion_matrix(
    pred_mask: np.ndarray,
    label_arr: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compute TP, FP, TN, FN against ground truth.

    Benchmark target: predicted post-event water mask vs ground truth post-event water extent.
    Label values: 1 = water/inundated, 0 = non-water, -1 = NoData (excluded).
    """
    valid = (label_arr >= 0) & np.isfinite(label_arr)
    label_water = (label_arr == 1) & valid
    label_nonwater = (label_arr == 0) & valid
    pred_water = pred_mask == 1

    tp = int(np.count_nonzero(pred_water & label_water))
    fp = int(np.count_nonzero(pred_water & label_nonwater))
    tn = int(np.count_nonzero(~pred_water & label_nonwater))
    fn = int(np.count_nonzero(~pred_water & label_water))

    return tp, fp, tn, fn


def run_p4_e04_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run P4-E04 evaluation suite against Modified Sen1Floods11.

    Returns metrics dict with real per-scene TP/FP/TN/FN computed from actual rasters.
    Fails closed if dataset acquisition or hash verification fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    base_dir = output_dir.parent
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        extracted_dir = _acquire_modified_sen1floods11(base_dir)
    except Exception as exc:
        failure_meta = {
            "experiment": "P4-E04",
            "task": "sar_temporal_flood_validation",
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": f"Modified Sen1Floods11 acquisition failed: {exc}",
            "timestamp": timestamp,
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e04-modified-sen1floods11-validation",
            "task": "sar_temporal_flood_validation",
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": timestamp,
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print(f"[P4-E04] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

    all_tifs = sorted(extracted_dir.rglob("*.tif"))
    pre_sample = [p.name for p in all_tifs if _is_pre_sar(p.name)][:5]
    post_sample = [p.name for p in all_tifs if _is_post_sar(p.name)][:5]
    label_sample = [p.name for p in all_tifs if _is_label(p.name)][:5]
    print(f"[P4-E04] Representative PRE files ({len(pre_sample)}): {pre_sample}")
    print(f"[P4-E04] Representative POST files ({len(post_sample)}): {post_sample}")
    print(f"[P4-E04] Representative LABEL files ({len(label_sample)}): {label_sample}")

    scene_pairs = _find_raster_pairs(extracted_dir)

    pre_count = sum(1 for v in scene_pairs.values() if v["pre"] is not None)
    post_count = sum(1 for v in scene_pairs.values() if v["post"] is not None)
    label_count = sum(1 for v in scene_pairs.values() if v["label"] is not None)
    paired_count = sum(1 for v in scene_pairs.values() if v["pre"] and v["post"] and v["label"])

    unmatched_pre = sum(1 for v in scene_pairs.values() if v["pre"] and not v["post"])
    unmatched_post = sum(1 for v in scene_pairs.values() if v["post"] and not v["pre"])
    unmatched_label = sum(1 for v in scene_pairs.values() if v["label"] and not v["pre"])

    audit = {
        "pre_count": pre_count,
        "post_count": post_count,
        "label_count": label_count,
        "paired_count": paired_count,
        "unmatched_pre": unmatched_pre,
        "unmatched_post": unmatched_post,
        "unmatched_label": unmatched_label,
        "duplicate_keys": 0,
        "representative_samples": {
            "pre": pre_sample,
            "post": post_sample,
            "label": label_sample,
        },
    }
    print(f"[P4-E04] Scene pairing audit: {audit}")

    if paired_count == 0:
        failure_meta = {
            "experiment": "P4-E04",
            "task": "sar_temporal_flood_validation",
            "status": "PAIRING_FAILED",
            "failure_reason": "Zero paired scenes found (pre/post/label all required)",
            "scene_pairing_audit": audit,
            "timestamp": timestamp,
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e04-modified-sen1floods11-validation",
            "task": "sar_temporal_flood_validation",
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": timestamp,
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print("[P4-E04] Aborting: zero paired scenes found")
        return failure_meta

    predictions: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    sample_count = 0

    for scene_key, paths in sorted(scene_pairs.items()):
        if not (paths["pre"] and paths["post"] and paths["label"]):
            continue

        pre_path = paths["pre"]
        post_path = paths["post"]
        label_path = paths["label"]

        try:
            pre_arr, transform, crs = _load_raster_band(pre_path)
            post_arr, _, _ = _load_raster_band(post_path)
            label_arr, _, _ = _load_raster_band(label_path)

            if pre_arr.shape != post_arr.shape or pre_arr.shape != label_arr.shape:
                print(f"[P4-E04] Shape mismatch for {scene_key}: "
                      f"{pre_arr.shape} vs {post_arr.shape} vs {label_arr.shape}")
                pred_record = {
                    "pair_id": scene_key,
                    "evaluation_split": "val",
                    "status": "SHAPE_MISMATCH",
                    "pre_raster_path": str(pre_path),
                    "post_raster_path": str(post_path),
                    "label_raster_path": str(label_path),
                    "pre_shape": list(pre_arr.shape),
                    "post_shape": list(post_arr.shape),
                    "label_shape": list(label_arr.shape),
                }
                predictions.append(pred_record)
                continue

            pred_water_mask = _compute_post_water_mask(post_arr, water_max_threshold_db=-16.0)

            flood_expansion_mask = _compute_flood_expansion_mask(
                pre_arr, post_arr,
                water_max_threshold_db=-16.0,
                flood_decrease_db=3.0,
            )

            tp, fp, tn, fn = _compute_confusion_matrix(pred_water_mask, label_arr)

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            sample_count += 1

            total_pixels = tp + fp + tn + fn
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            flood_expansion_pixels = int(np.count_nonzero(flood_expansion_mask))

            pred_record = {
                "pair_id": scene_key,
                "pre_raster_path": str(pre_path),
                "post_raster_path": str(post_path),
                "label_raster_path": str(label_path),
                "evaluation_split": "val",
                "crs": crs,
                "raster_shape": list(pre_arr.shape),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "post_event_water_iou": round(iou, 6),
                "post_event_water_accuracy": round(accuracy, 6),
                "post_event_water_precision": round(precision, 6),
                "post_event_water_recall": round(recall, 6),
                "post_event_water_f1": round(f1, 6),
                "flood_expansion_pixels": flood_expansion_pixels,
                "evaluation_status": "SUCCESS",
            }
            predictions.append(pred_record)

        except Exception as exc:
            pred_record = {
                "pair_id": scene_key,
                "evaluation_split": "val",
                "pre_raster_path": str(pre_path),
                "post_raster_path": str(post_path),
                "label_raster_path": str(label_path),
                "evaluation_status": "FAILURE",
                "failure_reason": str(exc),
            }
            predictions.append(pred_record)
            print(f"[P4-E04] Error processing {scene_key}: {exc}")

    elapsed = time.time() - start_time

    total = total_tp + total_fp + total_tn + total_fn
    aggregate_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    aggregate_accuracy = (total_tp + total_tn) / total if total > 0 else 0.0
    aggregate_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    aggregate_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    aggregate_f1 = 2 * aggregate_precision * aggregate_recall / (aggregate_precision + aggregate_recall) if (aggregate_precision + aggregate_recall) > 0 else 0.0

    metrics = {
        "experiment": "P4-E04",
        "task": "sar_temporal_flood_validation",
        "benchmark": BENCHMARK_PROVENANCE["benchmark"],
        "doi": BENCHMARK_PROVENANCE["doi"],
        "license": BENCHMARK_PROVENANCE["license"],
        "sample_count": sample_count,
        "evaluation_split": "val",
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_tn": total_tn,
        "total_fn": total_fn,
        "post_event_water_iou": round(aggregate_iou, 6),
        "post_event_water_accuracy": round(aggregate_accuracy, 6),
        "post_event_water_precision": round(aggregate_precision, 6),
        "post_event_water_recall": round(aggregate_recall, 6),
        "post_event_water_f1": round(aggregate_f1, 6),
        "flood_detected_count": sum(1 for p in predictions if p.get("tp", 0) + p.get("fp", 0) > 0),
        "label_audit": BENCHMARK_PROVENANCE["label_audit"]["ground_truth_target"],
        "expansion_evidence_separated": True,
        "scene_pairing_audit": audit,
        "benchmark_target": "post_event_water_mask",
        "secondary_evidence": "probable_new_water_mask",
        "status": "PASS" if sample_count > 0 and total > 0 else "FAIL",
        "execution_time_seconds": round(elapsed, 2),
    }

    metrics_path = output_dir / "sar_validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / "sar_validation_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    runner_meta = {
        "experiment": "phase4-e04-sar-validation",
        "benchmark_provenance": BENCHMARK_PROVENANCE,
        "status": "success" if sample_count > 0 else "failure",
        "sample_count": sample_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    existing_meta_path = output_dir / "runner_meta.json"
    if existing_meta_path.exists():
        with open(existing_meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        existing_meta["sample_count"] = sample_count
        existing_meta["status"] = "success" if sample_count > 0 else "failure"
        existing_meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(existing_meta_path, "w", encoding="utf-8") as f:
            json.dump(existing_meta, f, indent=2)
    else:
        with open(existing_meta_path, "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)

    evaluation_meta = {
        "experiment": "P4-E04",
        "task": "sar_temporal_flood_validation",
        "benchmark": BENCHMARK_PROVENANCE["benchmark"],
        "dataset": "Modified Sen1Floods11 Dataset for Change Detection",
        "dataset_hashes": BENCHMARK_PROVENANCE["pinned_files"],
        "runtime": {
            "execution_time_seconds": round(elapsed, 2),
            "device": "cpu",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        },
        "git_sha": _get_git_sha(),
        "reproducible": True,
        "dirty_worktree": _is_dirty_worktree(),
        "scene_pairing_audit": audit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "evaluation_meta.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_meta, f, indent=2)

    print(f"[P4-E04] Completed in {elapsed:.2f}s. Samples: {sample_count}")
    print(f"[P4-E04] Post-event water IoU: {aggregate_iou:.4f}, Accuracy: {aggregate_accuracy:.4f}")
    return metrics


def _get_git_sha() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _is_dirty_worktree() -> bool:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return True


if __name__ == "__main__":
    out_dir = Path(os.environ.get("OUTPUT_DIR", "experiments/phase4_sar_validation/results"))
    run_p4_e04_evaluation(out_dir)
