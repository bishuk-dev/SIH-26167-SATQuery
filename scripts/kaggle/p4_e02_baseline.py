#!/usr/bin/env python3
"""P4-E02: Learned Bi-Temporal Change-VQA Specialist Baseline Evaluation.

Runs multi-image Vision-Language model (SmolVLM / Idefics3) over bi-temporal
remote sensing image pairs (CDVQA / LEVIR-CC benchmark) to evaluate learned
change reasoning performance.

License Gate & Dataset Audit:
- cdvqa_annotation_license: Apache-2.0
- second_dataset_access: public
- second_image_license_status: UNRESOLVED
- cdvqa_full_dataset_license_gate: BLOCKED

Active SIH MVP Change Intelligence Benchmark:
- LEVIR-CC Change Description
- Underlying imagery: LEVIR-CD (academic / non-commercial research use only)
- Test set policy: Sealed (evaluation is performed strictly on the validation split)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any
import time

from PIL import Image
import torch

from satquery.evaluation.rsicc_eval import compute_rsicc_caption_metrics


LEVIR_CC_HF_REPO = "lcybuaa/LEVIR-CC"
LEVIR_CC_HF_REVISION = "881887bfc8a0f856f9059bcedf74c388e0d92ad7"
LEVIR_CC_ZIP_NAME = "Levir-CC-dataset.zip"
LEVIR_CC_EXPECTED_SIZE = 2683666867
LEVIR_CC_EXPECTED_SHA256 = "e05d38c0fdfda8c9b2048d314e5f95974d8b81e1b9f83f107acc39d55015e130"

VALIDATION_SUBSET_SIZE = 100


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


def _discover_dataset_root(extracted_dir: Path) -> Path:
    """Discover dataset root requiring captions.json + val/A + val/B.

    Fails closed if any required component is missing.
    Never accesses the test split.
    """
    caption_file = None
    val_a_dir = None
    val_b_dir = None

    for json_path in extracted_dir.rglob("LevirCCcaptions.json"):
        caption_file = json_path
        break

    for dir_path in extracted_dir.rglob("val"):
        if (dir_path / "A").is_dir() and (dir_path / "B").is_dir():
            val_a_dir = dir_path / "A"
            val_b_dir = dir_path / "B"
            break

    if not caption_file:
        raise RuntimeError("Dataset root discovery failed: LevirCCcaptions.json not found")
    if not val_a_dir:
        raise RuntimeError("Dataset root discovery failed: val/A directory not found")
    if not val_b_dir:
        raise RuntimeError("Dataset root discovery failed: val/B directory not found")

    dataset_root = caption_file.parent
    if not (dataset_root / "images" / "val" / "A").is_dir():
        dataset_root = caption_file.parent.parent
    print(f"[P4-E02] Dataset root discovered: {dataset_root}")
    print(f"  Captions: {caption_file}")
    print(f"  val/A: {val_a_dir}")
    print(f"  val/B: {val_b_dir}")

    if (dataset_root / "test").exists():
        print("[P4-E02] WARNING: test split exists but will NOT be accessed during Phase 4")

    return dataset_root


def _acquire_levir_cc(base_dir: Path) -> Path:
    levir_cc_root = os.environ.get("LEVIR_CC_ROOT")
    if levir_cc_root:
        dataset_root = Path(levir_cc_root)
        if not (dataset_root / "LevirCCcaptions.json").exists():
            raise RuntimeError(
                f"LEVIR_CC_ROOT points to invalid dataset: {dataset_root} "
                "(LevirCCcaptions.json not found)"
            )
        if not (dataset_root / "images" / "val" / "A").is_dir():
            raise RuntimeError(
                f"LEVIR_CC_ROOT points to invalid dataset: {dataset_root} "
                "(images/val/A not found)"
            )
        print(f"[P4-E02] Using LEVIR_CC_ROOT: {dataset_root}")
        return dataset_root

    cache_dir = base_dir / "levir_cc_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / LEVIR_CC_ZIP_NAME
    extracted_dir = cache_dir / "extracted"

    if zip_path.exists():
        actual_size = zip_path.stat().st_size
        if actual_size != LEVIR_CC_EXPECTED_SIZE:
            print(f"[P4-E02] Size mismatch ({actual_size} != {LEVIR_CC_EXPECTED_SIZE}), re-downloading")
            zip_path.unlink()
        else:
            actual_sha = _sha256_file(zip_path)
            if actual_sha != LEVIR_CC_EXPECTED_SHA256:
                print(f"[P4-E02] SHA256 mismatch ({actual_sha} != {LEVIR_CC_EXPECTED_SHA256}), re-downloading")
                zip_path.unlink()
            else:
                print(f"[P4-E02] Verified existing ZIP: {LEVIR_CC_ZIP_NAME}")

    if not zip_path.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download LEVIR-CC") from exc

        print("[P4-E02] Downloading LEVIR-CC dataset from HuggingFace...")
        downloaded_path = hf_hub_download(
            repo_id=LEVIR_CC_HF_REPO,
            repo_type="dataset",
            revision=LEVIR_CC_HF_REVISION,
            filename=LEVIR_CC_ZIP_NAME,
        )
        shutil.copy2(downloaded_path, zip_path)

        actual_size = zip_path.stat().st_size
        if actual_size != LEVIR_CC_EXPECTED_SIZE:
            raise RuntimeError(
                f"LEVIR-CC ZIP size mismatch: expected {LEVIR_CC_EXPECTED_SIZE}, got {actual_size}"
            )

        actual_sha = _sha256_file(zip_path)
        if actual_sha != LEVIR_CC_EXPECTED_SHA256:
            raise RuntimeError(
                f"LEVIR-CC ZIP SHA256 mismatch: expected {LEVIR_CC_EXPECTED_SHA256}, got {actual_sha}"
            )
        print("[P4-E02] Verified LEVIR-CC ZIP: size and SHA256 match")

    if not extracted_dir.exists() or not any(extracted_dir.rglob("LevirCCcaptions.json")):
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        extracted_dir.mkdir(parents=True)
        print("[P4-E02] Extracting LEVIR-CC ZIP safely...")
        _safe_zip_extract(zip_path, extracted_dir)
        print(f"[P4-E02] Extracted to {extracted_dir}")

    return _discover_dataset_root(extracted_dir)


def _build_authoritative_val_pairs(
    caption_file: Path,
    val_a_dir: Path,
    val_b_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive authoritative validation membership from LevirCCcaptions.json.

    Benchmark membership is strictly defined by `split == 'val'`.
    Surfaces the 1333 observed vs 1332 historically documented discrepancy.
    Test entries are sealed and never opened.
    """
    with open(caption_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    images = raw.get("images", []) if isinstance(raw, dict) else []
    val_entries = [img for img in images if img.get("split") == "val"]
    annotation_val_count = len(val_entries)

    val_a_files = {p.name for p in val_a_dir.glob("*.png")}
    val_b_files = {p.name for p in val_b_dir.glob("*.png")}
    val_a_stems = {p.stem for p in val_a_dir.glob("*.png")}
    val_b_stems = {p.stem for p in val_b_dir.glob("*.png")}

    duplicate_annotation_filenames = 0
    seen_filenames = set()
    for e in val_entries:
        fn = e.get("filename")
        if fn in seen_filenames:
            duplicate_annotation_filenames += 1
        seen_filenames.add(fn)

    all_val_a_list = list(val_a_dir.glob("*.png"))
    all_val_b_list = list(val_b_dir.glob("*.png"))
    duplicate_a_filenames = len(all_val_a_list) - len(val_a_files)
    duplicate_b_filenames = len(all_val_b_list) - len(val_b_files)

    pairs = []
    missing_a = 0
    missing_b = 0
    entries_without_references = 0

    for item in val_entries:
        filename = item.get("filename") or ""
        stem = Path(filename).stem
        has_a = filename in val_a_files or stem in val_a_stems
        has_b = filename in val_b_files or stem in val_b_stems

        if not has_a:
            missing_a += 1
        if not has_b:
            missing_b += 1

        sentences = item.get("sentences", [])
        if isinstance(sentences, list):
            refs = [s.get("raw", "").strip() if isinstance(s, dict) else str(s).strip() for s in sentences]
        elif isinstance(sentences, dict):
            refs = [sentences.get("raw", "").strip()]
        else:
            refs = [str(sentences).strip()]
        refs = [r for r in refs if r]

        if not refs:
            entries_without_references += 1
            continue

        if has_a and has_b:
            img_a_path = val_a_dir / filename if (val_a_dir / filename).exists() else val_a_dir / f"{stem}.png"
            img_b_path = val_b_dir / filename if (val_b_dir / filename).exists() else val_b_dir / f"{stem}.png"
            pairs.append({
                "pair_id": stem,
                "filename": filename,
                "before_image_path": str(img_a_path),
                "after_image_path": str(img_b_path),
                "reference_captions": refs,
                "evaluation_split": "val",
            })

    # Sort deterministically by filename
    pairs.sort(key=lambda p: p["filename"])
    matched_annotated_pairs = len(pairs)

    audit_meta = {
        "annotation_val_count": annotation_val_count,
        "unique_annotation_filename_count": len(seen_filenames),
        "val_A_file_count": len(val_a_files),
        "val_B_file_count": len(val_b_files),
        "matched_annotated_pairs": matched_annotated_pairs,
        "missing_A": missing_a,
        "missing_B": missing_b,
        "duplicate_annotation_filenames": duplicate_annotation_filenames,
        "duplicate_A_filenames": duplicate_a_filenames,
        "duplicate_B_filenames": duplicate_b_filenames,
        "entries_without_references": entries_without_references,
        "historical_documented_count": 1332,
        "pinned_artifact_observed_count": annotation_val_count,
        "discrepancy_resolution": (
            f"LevirCCcaptions.json defines {annotation_val_count} items with split=='val' "
            f"(6815 train + {annotation_val_count} val + 1929 test = 10077 total). "
            "Historical documentation colloquially cited 1332/1930, but the pinned authoritative "
            f"annotation file assigns {annotation_val_count} items to val and 1929 to test."
        ),
    }

    print(f"[P4-E02] Authoritative Validation Membership Audit:")
    print(f"  Annotation val items: {annotation_val_count}")
    print(f"  Filesystem val/A images: {len(val_a_files)}, val/B images: {len(val_b_files)}")
    print(f"  Matched annotated pairs: {matched_annotated_pairs}")
    print(f"  Discrepancy resolution: {audit_meta['discrepancy_resolution']}")

    return pairs, audit_meta


def run_p4_e02_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run P4-E02 evaluation suite and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[P4-E02] Running evaluation on device: {device}")

    # 1. Acquire LEVIR-CC
    try:
        dataset_root = _acquire_levir_cc(output_dir.parent)
    except Exception as exc:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": f"LEVIR-CC acquisition failed: {exc}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e02-levircc-change-description",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print(f"[P4-E02] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

    # 2. Build Authoritative Validation Pairs
    try:
        caption_file = dataset_root / "LevirCCcaptions.json"
        val_a_dir = dataset_root / "images" / "val" / "A"
        val_b_dir = dataset_root / "images" / "val" / "B"

        if not caption_file.exists():
            raise RuntimeError(f"Captions file not found: {caption_file}")
        if not val_a_dir.is_dir():
            raise RuntimeError(f"val/A directory not found: {val_a_dir}")
        if not val_b_dir.is_dir():
            raise RuntimeError(f"val/B directory not found: {val_b_dir}")

        val_pairs, audit_meta = _build_authoritative_val_pairs(caption_file, val_a_dir, val_b_dir)
    except Exception as exc:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_MALSTRUCTURED",
            "failure_reason": f"Dataset structure invalid: {exc}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e02-levircc-change-description",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print(f"[P4-E02] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

    if not val_pairs:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_MALSTRUCTURED",
            "failure_reason": "No validation pairs with reference captions found",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e02-levircc-change-description",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print("[P4-E02] Aborting: no validation pairs found.")
        return failure_meta

    # 3. Select deterministic subset
    is_subset = len(val_pairs) > VALIDATION_SUBSET_SIZE
    if is_subset:
        eval_pairs = val_pairs[:VALIDATION_SUBSET_SIZE]
        subset_label = "VALIDATION_SUBSET"
        selection_rule = f"deterministic_first_{VALIDATION_SUBSET_SIZE}_sorted_by_filename"
        print(f"[P4-E02] Selected {VALIDATION_SUBSET_SIZE} pairs ({subset_label}, rule: {selection_rule})")
    else:
        eval_pairs = val_pairs
        subset_label = "FULL_VALIDATION"
        selection_rule = "all_matched_validation_pairs"

    # 4. Load model ONCE and perform explicit readiness check before entering inference loop
    try:
        from satquery.models.change_vqa.baseline import load_change_vqa_model
        from satquery.inference.config import VqaRuntimeSettings

        settings = VqaRuntimeSettings(
            device=device,
            allow_remote_network=True,
        )
        backend = load_change_vqa_model(settings=settings)
        backend.load()
        model_loaded = True
        model_revision = backend.registration.revision
        preprocessing_profile = backend.profile_id
        model_id = backend.registration.model_id
        print(f"[P4-E02] Model loaded and readiness verified once: {model_id} (rev: {model_revision})")
    except Exception as exc:
        print(f"[P4-E02] Model load failed: {exc}")
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "MODEL_UNAVAILABLE",
            "failure_reason": str(exc),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e02-levircc-change-description",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "FAIL",
            "failure_reason": str(exc),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print("[P4-E02] Aborting: cannot evaluate without loaded model.")
        return failure_meta

    # 5. Run single-pair smoke test before running the full loop
    print("[P4-E02] Executing 1-pair inference smoke test...")
    smoke_pair = eval_pairs[0]
    smoke_t1 = Image.open(smoke_pair["before_image_path"]).convert("RGB")
    smoke_t2 = Image.open(smoke_pair["after_image_path"]).convert("RGB")
    smoke_res = backend.describe_changes(
        smoke_t1,
        smoke_t2,
        pair_id=smoke_pair["pair_id"],
        evaluation_split="val",
    )
    if not smoke_res.description or not smoke_res.description.strip():
        raise RuntimeError("Inference smoke test failed: empty caption produced")
    print(f"[P4-E02] Inference smoke test PASSED: '{smoke_res.description[:80]}...'")

    # 6. Run inference loop
    predictions: list[dict[str, Any]] = []
    for pair in eval_pairs:
        try:
            img_t1 = Image.open(pair["before_image_path"]).convert("RGB")
            img_t2 = Image.open(pair["after_image_path"]).convert("RGB")
            desc_res = backend.describe_changes(
                img_t1,
                img_t2,
                pair_id=pair["pair_id"],
                evaluation_split="val",
            )
            pred_text = desc_res.description

            record = {
                "pair_id": pair["pair_id"],
                "filename": pair["filename"],
                "before_image_path": pair["before_image_path"],
                "after_image_path": pair["after_image_path"],
                "reference_captions": pair["reference_captions"],
                "generated_caption": pred_text,
                "evaluation_split": pair["evaluation_split"],
                "model_id": model_id,
                "model_revision": model_revision,
                "preprocessing_profile": preprocessing_profile,
            }
            predictions.append(record)
        except Exception as exc:
            print(f"[P4-E02] Warning: pair {pair['pair_id']} failed: {exc}")
            continue

    sample_count = len(predictions)
    elapsed = time.time() - start_time

    if sample_count == 0:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "MODEL_EXECUTION_FAILED",
            "failure_reason": "Zero predictions generated during evaluation loop",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
        runner_meta = {
            "experiment": "phase4-e02-levircc-change-description",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "FAIL",
            "failure_reason": failure_meta["failure_reason"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
            json.dump(runner_meta, f, indent=2)
        print("[P4-E02] Aborting: zero successful predictions generated.")
        return failure_meta

    # 7. Compute corpus-level caption metrics from predictions
    hyps = [p["generated_caption"] for p in predictions]
    refs = [p["reference_captions"] for p in predictions]
    caption_metrics = compute_rsicc_caption_metrics(refs, hyps)

    metrics = {
        "experiment": "P4-E02",
        "task": "bitemporal_change_description",
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "sample_count": sample_count,
        "validation_subset_size": len(eval_pairs),
        "validation_subset_label": subset_label,
        "selection_rule": selection_rule,
        "evaluation_split": "val",
        "test_set_policy": "SEALED (evaluated on validation split only; test set never accessed)",
        "test_accessed": False,
        "status": "PASS",
        "caption_metrics": caption_metrics,
        "validation_membership_audit": audit_meta,
        "license_gates": {
            "cdvqa_annotation_license": "Apache-2.0",
            "second_dataset_access": "public",
            "second_image_license_status": "UNRESOLVED",
            "cdvqa_full_dataset_license_gate": "BLOCKED",
        },
        "primary_benchmark": {
            "name": "LEVIR-CC",
            "task": "change_description",
            "provenance": "Chenyang Liu et al. (IEEE TGRS 2022) / LEVIR Lab (Beihang University)",
            "upstream_imagery": "LEVIR-CD (Academic / non-commercial research use only)",
            "test_set_policy": "SEALED (evaluated on validation split)",
            "huggingface_repo": LEVIR_CC_HF_REPO,
            "huggingface_revision": LEVIR_CC_HF_REVISION,
            "zip_sha256": LEVIR_CC_EXPECTED_SHA256,
            "split_pairs": {
                "train": 6815,
                "val": audit_meta["annotation_val_count"],
                "test": 1929,
            },
        },
        "execution_time_seconds": round(elapsed, 2),
    }

    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / "validation_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    runner_meta = {
        "experiment": "phase4-e02-levircc-change-description",
        "task": "bitemporal_change_description",
        "primary_benchmark": "LEVIR-CC",
        "model_id": model_id,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "sample_count": sample_count,
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
        json.dump(runner_meta, f, indent=2)

    print(f"[P4-E02] Completed evaluation in {elapsed:.2f}s. Samples: {sample_count}")
    print(f"[P4-E02] Caption metrics: {caption_metrics}")
    return metrics


if __name__ == "__main__":
    out_dir = Path(os.environ.get("OUTPUT_DIR", "experiments/phase4_bitemporal_vqa"))
    run_p4_e02_evaluation(out_dir)
