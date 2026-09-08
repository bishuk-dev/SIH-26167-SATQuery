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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from affine import Affine
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
    max_attempts: int = 8,
) -> None:
    """Download a file with HTTP Range resume, exponential backoff, and hash verification."""
    import urllib.request
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == expected_size:
        if _md5_file(dest) == expected_md5:
            print(f"[P4-E04] Verified existing {dest.name}")
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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
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

    role_dirs: dict[str, Path] = {}
    for pinned in BENCHMARK_PROVENANCE["pinned_files"]:
        fname = pinned["filename"]
        target_dir = extracted_dir / fname.replace(".zip", "")
        has_tifs = target_dir.exists() and any(
            p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
            for p in target_dir.rglob("*")
        )
        if not has_tifs:
            zip_path = downloads_dir / fname
            print(f"[P4-E04] Extracting {fname} safely...")
            target_dir.mkdir(parents=True, exist_ok=True)
            _safe_zip_extract(zip_path, target_dir)
            print(f"[P4-E04] Extracted to {target_dir}")

        fn_upper = fname.upper()
        if "PRE" in fn_upper:
            role_dirs["pre"] = target_dir
        elif "POST" in fn_upper:
            role_dirs["post"] = target_dir
        elif "LABEL" in fn_upper:
            role_dirs["label"] = target_dir

    return DatasetRoles(
        pre_dir=role_dirs.get("pre", extracted_dir / "PRE_S1-20230517T191707Z-001"),
        post_dir=role_dirs.get("post", extracted_dir / "POST_S1-20230517T191716Z-001"),
        label_dir=role_dirs.get("label", extracted_dir / "Labels-20230517T191741Z-001"),
        extracted_dir=extracted_dir,
    )


@dataclass
class DatasetRoles:
    pre_dir: Path
    post_dir: Path
    label_dir: Path
    extracted_dir: Path | None = None

    def exists(self) -> bool:
        return self.pre_dir.exists() and self.post_dir.exists() and self.label_dir.exists()


@dataclass
class RasterMetadata:
    shape: tuple[int, ...]
    crs: str
    transform: Affine
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    nodata: float | None
    array: np.ndarray


def _enumerate_rasters(dir_path: Path) -> list[Path]:
    """Enumerate raster files robustly accepting case-insensitive .tif and .tiff."""
    if not dir_path.exists():
        return []
    return sorted(
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )


KNOWN_SUFFIXES: tuple[str, ...] = (
    "_S1Hand_post",
    "_S1Hand_pre",
    "_S1Hand",
    "_LabelHand",
    "_Label",
)


def _derive_scene_key(filename: str) -> str | None:
    """Derive scene key from raster filename.

    Strips known SAR/Label naming suffixes to obtain the base scene key
    (e.g., 'Bolivia_103757').
    Fails closed: returns None if the filename does not match a known pattern.
    """
    stem = Path(filename).stem
    for suffix in KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            key = stem[:-len(suffix)]
            if key:
                return key
    return None


def _build_role_map(
    files: list[Path],
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Build a mapping of scene_key -> Path for a specific role directory.

    Rejects duplicate scene keys from the mapping and tracks them.
    Tracks unrecognized files that could not be normalized.
    """
    mapping: dict[str, Path] = {}
    duplicate_keys: list[str] = []
    unrecognized_files: list[str] = []
    seen_keys: set[str] = set()

    for f in files:
        key = _derive_scene_key(f.name)
        if key is None:
            unrecognized_files.append(f.name)
            continue
        if key in seen_keys:
            if key not in duplicate_keys:
                duplicate_keys.append(key)
            mapping.pop(key, None)  # Reject duplicate: ambiguous scene key must not be used
        else:
            seen_keys.add(key)
            mapping[key] = f

    return mapping, duplicate_keys, unrecognized_files


def _check_duplicate(mapping: dict[str, Path], key: str, path: Path) -> None:
    """Check if key is already in mapping, raising RuntimeError if duplicate."""
    if key in mapping:
        raise RuntimeError(f"Duplicate scene keys detected for '{key}': {mapping[key]} and {path}")



def _load_raster_data(path: Path) -> RasterMetadata:
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        res = src.res
        b = src.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return RasterMetadata(
            shape=arr.shape,
            crs=crs,
            transform=transform,
            resolution=res,
            bounds=bounds,
            nodata=nodata,
            array=arr,
        )


GRID_TOLERANCE: float = 1e-5


def _check_grid_alignment(
    m1: RasterMetadata,
    m2: RasterMetadata,
    label1: str,
    label2: str,
    tolerance: float = GRID_TOLERANCE,
) -> tuple[bool, str]:
    """Strictly verify spatial grid compatibility between two rasters."""
    if m1.shape != m2.shape:
        return False, f"{label1} vs {label2} shape mismatch: {m1.shape} != {m2.shape}"
    crs1 = m1.crs.strip().upper()
    crs2 = m2.crs.strip().upper()
    if crs1 != crs2:
        return False, f"{label1} vs {label2} CRS mismatch: {m1.crs} != {m2.crs}"
    for idx, (v1, v2) in enumerate(zip(m1.transform, m2.transform)):
        if abs(v1 - v2) > tolerance:
            return False, f"{label1} vs {label2} affine transform coeff {idx} mismatch: {v1} vs {v2} (diff {abs(v1-v2):.6e} > {tolerance})"
    return True, "OK"


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


def run_p4_e04_evaluation(
    output_dir: Path,
    roles: DatasetRoles | None = None,
) -> dict[str, Any]:
    """Run P4-E04 evaluation suite against Modified Sen1Floods11.

    Returns metrics dict with real per-scene TP/FP/TN/FN computed from actual rasters.
    Fails closed if dataset acquisition or hash verification fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    base_dir = output_dir.parent
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if roles is None:
        try:
            roles = _acquire_modified_sen1floods11(base_dir)
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

    # Robust raster enumeration per source directory
    raw_pre_files = _enumerate_rasters(roles.pre_dir)
    raw_post_files = _enumerate_rasters(roles.post_dir)
    raw_label_files = _enumerate_rasters(roles.label_dir)

    print(f"[P4-E04] RAW PRE ({len(raw_pre_files)}): {[p.name for p in raw_pre_files[:5]]}")
    print(f"[P4-E04] RAW POST ({len(raw_post_files)}): {[p.name for p in raw_post_files[:5]]}")
    print(f"[P4-E04] RAW LABEL ({len(raw_label_files)}): {[p.name for p in raw_label_files[:5]]}")

    # Build three independent maps
    pre_by_scene_key, dup_pre, unrec_pre = _build_role_map(raw_pre_files)
    post_by_scene_key, dup_post, unrec_post = _build_role_map(raw_post_files)
    label_by_scene_key, dup_label, unrec_label = _build_role_map(raw_label_files)

    paired_keys = sorted(
        set(pre_by_scene_key.keys())
        & set(post_by_scene_key.keys())
        & set(label_by_scene_key.keys())
    )

    pre_count = len(pre_by_scene_key)
    post_count = len(post_by_scene_key)
    label_count = len(label_by_scene_key)
    paired_count = len(paired_keys)

    unmatched_pre = sorted(set(pre_by_scene_key.keys()) - set(paired_keys))
    unmatched_post = sorted(set(post_by_scene_key.keys()) - set(paired_keys))
    unmatched_label = sorted(set(label_by_scene_key.keys()) - set(paired_keys))

    audit = {
        "pre_count": pre_count,
        "post_count": post_count,
        "label_count": label_count,
        "paired_count": paired_count,
        "unmatched_pre": len(unmatched_pre),
        "unmatched_post": len(unmatched_post),
        "unmatched_label": len(unmatched_label),
        "unmatched_pre_samples": unmatched_pre[:5],
        "unmatched_post_samples": unmatched_post[:5],
        "unmatched_label_samples": unmatched_label[:5],
        "duplicate_pre_keys": dup_pre,
        "duplicate_post_keys": dup_post,
        "duplicate_label_keys": dup_label,
        "unrecognized_pre_files": unrec_pre,
        "unrecognized_post_files": unrec_post,
        "unrecognized_label_files": unrec_label,
        "representative_samples": {
            "pre": [p.name for p in raw_pre_files[:5]],
            "post": [p.name for p in raw_post_files[:5]],
            "label": [p.name for p in raw_label_files[:5]],
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
    grid_valid_count = 0
    grid_mismatch_count = 0

    for scene_key in paired_keys:
        pre_path = pre_by_scene_key[scene_key]
        post_path = post_by_scene_key[scene_key]
        label_path = label_by_scene_key[scene_key]

        try:
            pre_meta = _load_raster_data(pre_path)
            post_meta = _load_raster_data(post_path)
            label_meta = _load_raster_data(label_path)

            ok_pre_post, msg_pre_post = _check_grid_alignment(pre_meta, post_meta, "PRE", "POST")
            ok_post_label, msg_post_label = _check_grid_alignment(post_meta, label_meta, "POST", "LABEL")

            if not ok_pre_post or not ok_post_label:
                mismatch_msg = msg_pre_post if not ok_pre_post else msg_post_label
                grid_mismatch_count += 1
                pred_record = {
                    "pair_id": scene_key,
                    "evaluation_split": "val",
                    "status": "GRID_MISMATCH",
                    "evaluation_status": "GRID_MISMATCH",
                    "failure_reason": mismatch_msg,
                    "pre_raster_path": str(pre_path),
                    "post_raster_path": str(post_path),
                    "label_raster_path": str(label_path),
                    "pre_shape": list(pre_meta.shape),
                    "post_shape": list(post_meta.shape),
                    "label_shape": list(label_meta.shape),
                    "pre_crs": pre_meta.crs,
                    "post_crs": post_meta.crs,
                    "label_crs": label_meta.crs,
                }
                predictions.append(pred_record)
                continue

            grid_valid_count += 1

            pred_water_mask = _compute_post_water_mask(post_meta.array, water_max_threshold_db=-16.0)

            flood_expansion_mask = _compute_flood_expansion_mask(
                pre_meta.array, post_meta.array,
                water_max_threshold_db=-16.0,
                flood_decrease_db=3.0,
            )

            tp, fp, tn, fn = _compute_confusion_matrix(pred_water_mask, label_meta.array)

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
                "crs": post_meta.crs,
                "raster_shape": list(post_meta.shape),
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

    audit["grid_valid_count"] = grid_valid_count
    audit["grid_mismatch_count"] = grid_mismatch_count

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
        "grid_valid_count": grid_valid_count,
        "grid_mismatch_count": grid_mismatch_count,
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
