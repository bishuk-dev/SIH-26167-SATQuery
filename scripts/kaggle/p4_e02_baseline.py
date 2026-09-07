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


def _load_captions(caption_file: Path) -> dict[str, list[str]]:
    with open(caption_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    references: dict[str, list[str]] = {}

    if isinstance(raw, dict) and "annotations" in raw:
        for ann in raw["annotations"]:
            filename = ann.get("filename") or ann.get("image_id")
            sentence = ann.get("sentence") or ann.get("caption")
            if filename and sentence:
                if filename not in references:
                    references[filename] = []
                if sentence not in references[filename]:
                    references[filename].append(sentence)
    elif isinstance(raw, dict) and "images" in raw:
        for img in raw["images"]:
            filename = img.get("filename") or img.get("image_id")
            if filename:
                sentences = img.get("sentences", [])
                if isinstance(sentences, list):
                    refs = [s.get("raw", "") if isinstance(s, dict) else str(s) for s in sentences]
                elif isinstance(sentences, dict):
                    refs = [sentences.get("raw", "")]
                else:
                    refs = [str(sentences)]
                references[filename] = [r for r in refs if r]
    elif isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, dict):
                filename = val.get("filename") or val.get("image_id") or key
                sent_field = val.get("sentences") or val.get("caption") or val.get("sentence")
                if filename and sent_field:
                    if isinstance(sent_field, list):
                        refs = sent_field
                    elif isinstance(sent_field, dict):
                        refs = [sent_field.get("raw", "")]
                    else:
                        refs = [str(sent_field)]
                    references[filename] = [r for r in refs if r]
    elif isinstance(raw, list):
        for item in raw:
            filename = item.get("filename") or item.get("image_id")
            sentence = item.get("sentence") or item.get("caption")
            if filename and sentence:
                if filename not in references:
                    references[filename] = []
                if sentence not in references[filename]:
                    references[filename].append(sentence)

    split_filter = None
    if isinstance(raw, dict) and "images" in raw:
        for img in raw["images"]:
            if img.get("split") == "val":
                filename = img.get("filename") or img.get("image_id")
                if filename and filename not in references:
                    sentences = img.get("sentences", [])
                    if isinstance(sentences, list):
                        refs = [s.get("raw", "") if isinstance(s, dict) else str(s) for s in sentences]
                    else:
                        refs = [str(sentences)]
                    references[filename] = [r for r in refs if r]

    print(f"[P4-E02] Loaded {len(references)} caption entries from {caption_file}")
    return references


def _build_val_pairs(val_a_dir: Path, val_b_dir: Path, references: dict[str, list[str]]) -> list[dict[str, Any]]:
    pairs = []

    for img_a in sorted(val_a_dir.glob("*.png")):
        stem = img_a.stem
        img_b = val_b_dir / f"{stem}.png"
        if not img_b.exists():
            continue

        filename_key = f"{stem}.png"
        ref_captions = references.get(filename_key, references.get(stem, []))

        if not ref_captions:
            print(f"[P4-E02] WARNING: no reference captions for pair {stem}, skipping")
            continue

        pairs.append({
            "pair_id": stem,
            "filename": filename_key,
            "before_image_path": str(img_a),
            "after_image_path": str(img_b),
            "reference_captions": ref_captions,
            "evaluation_split": "val",
        })

    print(f"[P4-E02] Found {len(pairs)} validation pairs with reference captions")
    return pairs


def _bleu4_impl(reference_captions: list[str], candidate: str) -> float:
    import math
    import collections
    import re

    def tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        return re.findall(r"\w+", text)

    references = [tokenize(c) for c in reference_captions]
    hypothesis = tokenize(candidate)

    if len(hypothesis) == 0:
        return 0.0

    max_n = 4
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = collections.Counter(zip(*[hypothesis[i:] for i in range(n)])) if len(hypothesis) >= n else collections.Counter()
        ref_ngram_counters = [collections.Counter(zip(*[ref[i:] for i in range(n)])) if len(ref) >= n else collections.Counter() for ref in references]

        matched = 0
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        for ngram, count in hyp_ngrams.items():
            max_ref_count = max((r.get(ngram, 0) for r in ref_ngram_counters), default=0)
            matched += min(count, max_ref_count)
        precisions.append(matched / total)

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda l: abs(l - len(hypothesis)))
    bp = 1.0 if len(hypothesis) > closest_ref_len else math.exp(1 - closest_ref_len / len(hypothesis)) if len(hypothesis) > 0 else 0.0

    valid_precisions = [p for p in precisions if p > 0]
    if not valid_precisions:
        return 0.0

    geometric_mean = math.exp(sum(math.log(p) for p in valid_precisions) / len(valid_precisions))
    return bp * geometric_mean


def _rouge_l_score(reference_captions: list[str], candidate: str) -> float:
    import re

    def tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        return re.findall(r"\w+", text)

    refs_tokenized = [tokenize(c) for c in reference_captions]
    hyp = tokenize(candidate)

    if not hyp:
        return 0.0

    best_f1 = 0.0
    for ref in refs_tokenized:
        if not ref:
            continue
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if ref[i] == hyp[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        lcs_length = dp[0][0]

        precision = lcs_length / n if n > 0 else 0.0
        recall = lcs_length / m if m > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def _cider_score(reference_captions: list[str], candidate: str, n: int = 4) -> float:
    import math
    import collections
    import re
    from math import sqrt

    def tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        return re.findall(r"\w+", text)

    def get_ngrams(tokens: list[str], n_val: int) -> collections.Counter:
        return collections.Counter(zip(*[tokens[i:] for i in range(n_val)])) if len(tokens) >= n_val else collections.Counter()

    def tf(document_ngrams: collections.Counter) -> collections.Counter:
        tf_vec = collections.Counter()
        total = sum(document_ngrams.values())
        if total == 0:
            return tf_vec
        for ngram, count in document_ngrams.items():
            tf_vec[ngram] = count / total
        return tf_vec

    refs_tokenized = [tokenize(c) for c in reference_captions]
    hyp = tokenize(candidate)

    if not hyp or not refs_tokenized:
        return 0.0

    refs_tf = [tf(get_ngrams(ref, n)) for ref in refs_tokenized]
    hyp_tf = tf(get_ngrams(hyp, n))

    if not refs_tf:
        return 0.0

    doc_freq = collections.Counter()
    for ref in refs_tokenized:
        unique_ngrams = set(get_ngrams(ref, n).keys())
        for ngram in unique_ngrams:
            doc_freq[ngram] += 1

    num_docs = len(refs_tokenized) + 1
    idf = collections.Counter()
    for ngram, df in doc_freq.items():
        idf[ngram] = math.log(max(1.0, num_docs / (1 + df)))

    hyp_tfidf = collections.Counter()
    for ngram, tf_val in hyp_tf.items():
        hyp_tfidf[ngram] = tf_val * idf.get(ngram, 0.0)

    ref_tfidf_list = []
    for ref_tf in refs_tf:
        ref_tfidf = collections.Counter()
        for ngram, tf_val in ref_tf.items():
            ref_tfidf[ngram] = tf_val * idf.get(ngram, 0.0)
        ref_tfidf_list.append(ref_tfidf)

    if not ref_tfidf_list or not hyp_tfidf:
        return 0.0

    def cosine_sim(vec1: collections.Counter, vec2: collections.Counter) -> float:
        dot = sum(vec1[k] * vec2[k] for k in vec1 if k in vec2)
        norm1 = sqrt(sum(v * v for v in vec1.values()))
        norm2 = sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    scores = [cosine_sim(hyp_tfidf, ref_tfidf) for ref_tfidf in ref_tfidf_list]
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_captions(predictions: list[dict[str, Any]], references_map: dict[str, list[str]]) -> dict[str, float]:
    bleu4_scores = []
    rouge_l_scores = []
    cider_scores = []

    for pred in predictions:
        pair_id = pred["pair_id"]
        ref_captions = references_map.get(pair_id, [])
        if not ref_captions:
            print(f"[P4-E02] No reference captions for {pair_id}, skipping metric")
            continue

        generated = pred.get("predicted_answer", "")
        if not generated:
            print(f"[P4-E02] Empty generated caption for {pair_id}, skipping metric")
            continue

        bleu4_scores.append(_bleu4_impl(ref_captions, generated))
        rouge_l_scores.append(_rouge_l_score(ref_captions, generated))
        cider_scores.append(_cider_score(ref_captions, generated))

    result = {}
    if bleu4_scores:
        result["bleu4"] = round(sum(bleu4_scores) / len(bleu4_scores), 6)
    if rouge_l_scores:
        result["rouge_l"] = round(sum(rouge_l_scores) / len(rouge_l_scores), 6)
    if cider_scores:
        result["cider"] = round(sum(cider_scores) / len(cider_scores), 6)

    return result


def run_p4_e02_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run P4-E02 evaluation suite and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[P4-E02] Running evaluation on device: {device}")

    try:
        from satquery.models.change_vqa.baseline import load_change_vqa_model
        from satquery.inference.config import VqaRuntimeSettings

        settings = VqaRuntimeSettings(
            device=device,
            allow_remote_network=True,
        )
        backend = load_change_vqa_model(settings=settings)
        model_loaded = True
        model_revision = backend.registration.revision
        preprocessing_profile = backend.profile_id
        model_id = backend.registration.model_id
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
        print("[P4-E02] Aborting: cannot evaluate without loaded model.")
        return failure_meta

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
        print(f"[P4-E02] Aborting: {failure_meta['failure_reason']}")
        return failure_meta

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

        references = _load_captions(caption_file)
        val_pairs = _build_val_pairs(val_a_dir, val_b_dir, references)
    except Exception as exc:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": f"Dataset structure invalid: {exc}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "evaluation_failure.json", "w", encoding="utf-8") as f:
            json.dump(failure_meta, f, indent=2)
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
        print("[P4-E02] Aborting: no validation pairs found.")
        return failure_meta

    if len(val_pairs) > VALIDATION_SUBSET_SIZE:
        val_pairs = val_pairs[:VALIDATION_SUBSET_SIZE]
        print(f"[P4-E02] Using validation subset of {VALIDATION_SUBSET_SIZE} pairs (VALIDATION_SUBSET)")

    references_map = {p["pair_id"]: p["reference_captions"] for p in val_pairs}

    predictions: list[dict[str, Any]] = []
    for pair in val_pairs:
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

    caption_metrics = _evaluate_captions(predictions, references_map)

    metrics = {
        "experiment": "P4-E02",
        "task": "bitemporal_change_description",
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "sample_count": sample_count,
        "validation_subset_size": VALIDATION_SUBSET_SIZE,
        "evaluation_split": "val",
        "test_set_policy": "SEALED (evaluated on validation split only)",
        "status": "PASS" if sample_count > 0 else "FAIL",
        "caption_metrics": caption_metrics,
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
                "val": 1332,
                "test": 1930,
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
        "experiment": "phase4-e02-bitemporal-vqa",
        "task": "bitemporal_change_description",
        "primary_benchmark": "LEVIR-CC",
        "model_id": model_id,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "sample_count": sample_count,
        "status": "PASS" if sample_count > 0 else "FAIL",
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
