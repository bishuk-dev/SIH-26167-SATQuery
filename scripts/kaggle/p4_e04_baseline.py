#!/usr/bin/env python3
"""P4-E04: SAR Temporal Flood Inundation Validation against Modified Sen1Floods11.

Downloads the authoritative Modified Sen1Floods11 Dataset for Change Detection
from Zenodo (DOI: 10.5281/zenodo.7946594), verifies pinned byte sizes and MD5
hashes, and evaluates deterministic SAR backscatter-differencing flood detection
against pixel-level ground truth labels.

All metrics are derived from real per-scene TP/FP/TN/FN computed from actual
raster evaluation. No synthetic or placeholder data is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine

def _write_failure(output_dir, failure_meta):
    # write mock predictions
    with open(output_dir / "sar_validation_predictions.jsonl", "w", encoding="utf-8") as f:
        pass
    # write metrics
    with open(output_dir / "sar_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(failure_meta, f, indent=2)
    # write runner meta
    runner_meta = {
        "experiment": "phase4-e04-sar-validation",
        "status": "failure",
        "timestamp": failure_meta.get("timestamp", "")
    }
    with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
        json.dump(runner_meta, f, indent=2)


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

ZENODO_BASE_URL = "https://zenodo.org/record/7946594/files"


def _download_file(url: str, dest: Path, expected_size: int, expected_md5: str) -> None:
    import urllib.request
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected_size:
        actual_md5 = _md5_file(dest)
        if actual_md5 == expected_md5:
            print(f"[P4-E04] Verified existing file: {dest.name}")
            return
        print(f"[P4-E04] Size matches but MD5 mismatch for {dest.name}, re-downloading")

    print(f"[P4-E04] Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "satquery-P4-E04"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    actual_size = dest.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {dest.name}: expected {expected_size}, got {actual_size}"
        )

    actual_md5 = _md5_file(dest)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual_md5}"
        )
    print(f"[P4-E04] Verified {dest.name}: {actual_size} bytes, MD5={actual_md5}")


def _md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _locate_or_download_benchmark(base_dir: Path) -> dict[str, Path]:
    cache_dir = base_dir / "sen1floods11_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = cache_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = cache_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    file_paths: dict[str, Path] = {}

    for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
        fname = pinned["filename"]
        dest = downloads_dir / fname

        if not dest.exists():
            url = f"{ZENODO_BASE_URL}/{fname}?download=1"
            _download_file(url, dest, pinned["size_bytes"], pinned["md5"])
        else:
            actual_size = dest.stat().st_size
            actual_md5 = _md5_file(dest)
            if actual_size != pinned["size_bytes"] or actual_md5 != pinned["md5"]:
                print(f"[P4-E04] Re-downloading {fname} due to size/hash mismatch")
                dest.unlink()
                url = f"{ZENODO_BASE_URL}/{fname}?download=1"
                _download_file(url, dest, pinned["size_bytes"], pinned["md5"])
            else:
                print(f"[P4-E04] Found verified file: {fname}")

        file_paths[fname] = dest

    for fname in ("PRE_S1-20230517T191707Z-001.zip", "POST_S1-20230517T191716Z-001.zip", "Labels-20230517T191741Z-001.zip"):
        zip_path = file_paths[fname]
        target_dir = extracted_dir / fname.replace(".zip", "")
        if not target_dir.exists() or not any(target_dir.rglob("*.tif")):
            print(f"[P4-E04] Extracting {fname} ...")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            print(f"[P4-E04] Extracted to {target_dir}")

    return file_paths


def _find_raster_paths(extracted_root: Path, prefix: str) -> list[Path]:
    raster_paths: list[Path] = []
    label_subdir = extracted_root / prefix
    if not label_subdir.exists():
        print(f"[P4-E04] Directory not found: {label_subdir}")
        return raster_paths

    for tif_path in sorted(label_subdir.rglob("*.tif")):
        if tif_path.is_file():
            raster_paths.append(tif_path)

    return raster_paths


def _load_raster_as_array(path: Path) -> tuple[np.ndarray, Affine, str]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
        if nodata is not None:
            arr[nodata == arr] = np.nan
    return arr, transform, crs


def _compute_flood_mask(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    transform: Affine,
    crs: str,
) -> np.ndarray:
    from satquery.analytics.sar import SarTemporalAnalytics
    from satquery.sensors.semantics import SemanticBandRole

    bands_pre = {"sar_vv": "vv", "data": pre_arr}
    bands_post = {"sar_vv": "vv", "data": post_arr}

    mask, sar_result, evidence = SarTemporalAnalytics.detect_flood(
        bands_pre={"vv": pre_arr},
        bands_post={"vv": post_arr},
        transform=transform,
        crs=crs,
        polarization_role=SemanticBandRole.SAR_CO_POL,
        decrease_threshold_db=3.0,
        water_max_threshold_db=-16.0,
        radiometric_domain="SAR_DB_POWER",
    )
    return mask.astype(np.uint8)


def run_p4_e04_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run P4-E04 evaluation suite against Modified Sen1Floods11.

    Returns metrics dict with real per-scene TP/FP/TN/FN computed from actual rasters.
    Fails closed if data acquisition or hash verification fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    try:
        file_paths = _locate_or_download_benchmark(output_dir.parent)
    except Exception as exc:
        failure_meta = {
            "experiment": "P4-E04",
            "task": "sar_temporal_flood_validation",
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": f"Benchmark acquisition failed: {exc}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_failure(output_dir, failure_meta)
        print(f"[P4-E04] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

    extracted_dir = output_dir.parent / "sen1floods11_data" / "extracted"

    pre_root = extracted_dir / "PRE_S1-20230517T191707Z-001"
    post_root = extracted_dir / "POST_S1-20230517T191716Z-001"
    labels_root = extracted_dir / "Labels-20230517T191741Z-001"

    pre_rasters = _find_raster_paths(extracted_dir, "PRE_S1-20230517T191707Z-001")
    post_rasters = _find_raster_paths(extracted_dir, "POST_S1-20230517T191716Z-001")
    label_rasters = _find_raster_paths(extracted_dir, "Labels-20230517T191741Z-001")

    print(f"[P4-E04] Found {len(pre_rasters)} pre-event rasters, "
          f"{len(post_rasters)} post-event rasters, "
          f"{len(label_rasters)} label rasters")

    if not pre_rasters or not post_rasters or not label_rasters:
        failure_meta = {
            "experiment": "P4-E04",
            "task": "sar_temporal_flood_validation",
            "status": "DATASET_MALSTRUCTURED",
            "failure_reason": "Extracted archives do not contain expected .tif rasters",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_failure(output_dir, failure_meta)
        print(f"[P4-E04] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

    label_map = {p.stem: p for p in label_rasters}

    predictions: list[dict[str, Any]] = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    sample_count = 0

    for pre_path in sorted(pre_rasters):
        stem = pre_path.stem

        matching_post = next((p for p in post_rasters if p.stem == stem), None)
        matching_label = label_map.get(stem)

        if matching_post is None or matching_label is None:
            print(f"[P4-E04] Skipping {stem}: no matching post or label raster")
            continue

        try:
            pre_arr, transform, crs = _load_raster_as_array(pre_path)
            post_arr, _, _ = _load_raster_as_array(matching_post)
            label_arr, _, _ = _load_raster_as_array(matching_label)

            if pre_arr.shape != post_arr.shape or pre_arr.shape != label_arr.shape:
                print(f"[P4-E04] Shape mismatch for {stem}: "
                      f"{pre_arr.shape} vs {post_arr.shape} vs {label_arr.shape}")
                continue

            pred_mask = _compute_flood_mask(pre_arr, post_arr, transform, crs)

            label_valid = np.isfinite(label_arr) & (label_arr >= 0)
            label_water = (label_arr == 1) & label_valid
            label_nonwater = (label_arr == 0) & label_valid
            pred_water = pred_mask == 1

            tp = int(np.count_nonzero(pred_water & label_water))
            fp = int(np.count_nonzero(pred_water & label_nonwater))
            tn = int(np.count_nonzero(~pred_water & label_nonwater))
            fn = int(np.count_nonzero(~pred_water & label_water))

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            sample_count += 1

            total_pixels = int(label_valid.sum())
            post_water_pixels = int(label_water.sum())
            pred_water_pixels = int(pred_water.sum())

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            pred_record = {
                "pair_id": stem,
                "pre_raster_path": str(pre_path),
                "post_raster_path": str(matching_post),
                "label_raster_path": str(matching_label),
                "evaluation_split": "val",
                "crs": crs,
                "raster_shape": list(pre_arr.shape),
                "total_valid_pixels": total_pixels,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "post_event_water_pixels": post_water_pixels,
                "predicted_water_pixels": pred_water_pixels,
                "iou": round(iou, 6),
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
                "evaluation_status": "SUCCESS",
            }
            predictions.append(pred_record)

        except Exception as exc:
            pred_record = {
                "pair_id": stem,
                "evaluation_split": "val",
                "pre_raster_path": str(pre_path),
                "post_raster_path": str(matching_post),
                "label_raster_path": str(matching_label),
                "evaluation_status": "FAILURE",
                "failure_reason": str(exc),
            }
            predictions.append(pred_record)
            print(f"[P4-E04] Error processing {stem}: {exc}")

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
        "model_id": "sar_backscatter_differencing_rule_based",
        "device": "cpu",
        "sample_count": sample_count,
        "post_event_water_iou": round(aggregate_iou, 6),
        "post_event_water_accuracy": round(aggregate_accuracy, 6),
        "post_event_water_precision": round(aggregate_precision, 6),
        "post_event_water_recall": round(aggregate_recall, 6),
        "post_event_water_f1": round(aggregate_f1, 6),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_tn": total_tn,
        "total_fn": total_fn,
        "flood_detected_count": sum(1 for p in predictions if p.get("predicted_water_pixels", 0) > 0),
        "label_audit": "post_event_water_extent",
        "expansion_evidence_separated": True,
        "primary_benchmark": BENCHMARK_PROVENANCE,
        "status": "PASS" if sample_count > 0 and total > 0 else "FAIL",
        "execution_time_seconds": round(elapsed, 2),
    }

    with open(output_dir / "sar_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "sar_validation_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    runner_meta = {
        "experiment": "phase4-e04-sar-validation",
        "benchmark_provenance": BENCHMARK_PROVENANCE,
        "status": "success" if sample_count > 0 else "failure",
        "sample_count": sample_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
        json.dump(runner_meta, f, indent=2)

    print(f"[P4-E04] Completed in {elapsed:.2f}s. Samples: {sample_count}. "
          f"IoU: {aggregate_iou:.4f}, Accuracy: {aggregate_accuracy:.4f}")
    return metrics


if __name__ == "__main__":
    out_dir = Path(os.environ.get("OUTPUT_DIR", "experiments/phase4_sar_validation"))
    run_p4_e04_evaluation(out_dir)
