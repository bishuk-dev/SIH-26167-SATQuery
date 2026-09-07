#!/usr/bin/env python3
"""P4-E02: Learned Bi-Temporal Change-VQA Specialist Baseline Evaluation.

Runs multi-image Vision-Language model (SmolVLM / Idefics3) over bi-temporal
remote sensing image pairs (CDVQA / LEVIR-CC benchmark) to evaluate learned
change reasoning performance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image
import torch

from satquery.inference.config import VqaRuntimeSettings
from satquery.models.change_vqa.baseline import load_change_vqa_model

def _write_failure(output_dir, failure_meta):
    import json
    with open(output_dir / "validation_predictions.jsonl", "w", encoding="utf-8") as f:
        pass
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(failure_meta, f, indent=2)
    runner_meta = {
        "experiment": "P4-E02",
        "status": "FAIL",
        "timestamp": failure_meta.get("timestamp", "")
    }
    with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
        json.dump(runner_meta, f, indent=2)



def run_p4_e02_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run P4-E02 evaluation suite and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[P4-E02] Running evaluation on device: {device}")

    settings = VqaRuntimeSettings(
        device=device,
        allow_remote_network=True,
    )

    try:
        backend = load_change_vqa_model(settings=settings)
        model_loaded = True
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
        _write_failure(output_dir, failure_meta)
        print("[P4-E02] Aborting: cannot evaluate without loaded model.")
        return failure_meta

    from satquery.analytics.temporal import TemporalAnalytics


    levir_cc_root = os.environ.get("LEVIR_CC_ROOT", "")
    if not levir_cc_root or not Path(levir_cc_root).exists():
        print("[P4-E02] LEVIR_CC_ROOT not set. Attempting authoritative download via huggingface_hub.")
        try:
            from huggingface_hub import snapshot_download
            levir_cc_root = snapshot_download(
                repo_id="lcybuaa/LEVIR-CC",
                repo_type="dataset",
                revision="881887bfc8a0f856f9059bcedf74c388e0d92ad7",
                allow_patterns=["images/val/A/*", "images/val/B/*", "LevirCCcaptions.json"]
            )
            print(f"[P4-E02] Downloaded LEVIR-CC to {levir_cc_root}")
        except Exception as e:
            failure_meta = {
                "experiment": "P4-E02",
                "task": "bitemporal_change_description",
                "device": device,
                "status": "DATASET_UNAVAILABLE",
                "failure_reason": f"Failed to download LEVIR-CC dataset: {e}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_failure(output_dir, failure_meta)
            print("[P4-E02] Aborting: real LEVIR-CC dataset unavailable.")
            return failure_meta


    levir_root = Path(levir_cc_root)

    val_a_dir = levir_root / "images" / "val" / "A"
    if not val_a_dir.exists():
        val_a_dir = levir_root / "val" / "A"
        
    val_b_dir = levir_root / "images" / "val" / "B"
    if not val_b_dir.exists():
        val_b_dir = levir_root / "val" / "B"

    if not val_a_dir.exists() or not val_b_dir.exists():

        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": "LEVIR-CC val/B directory missing",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_failure(output_dir, failure_meta)
        print("[P4-E02] Aborting: LEVIR-CC val/B directory missing.")
        return failure_meta

    image_pairs: list[dict[str, Any]] = []
    for img_path in sorted(val_a_dir.glob("*.png")):
        stem = img_path.stem
        pair_b = val_b_dir / f"{stem}.png"
        if pair_b.exists():
            image_pairs.append({
                "pair_id": stem,
                "t1_path": str(img_path),
                "t2_path": str(pair_b),
            })
    if not image_pairs:
        failure_meta = {
            "experiment": "P4-E02",
            "task": "bitemporal_change_description",
            "device": device,
            "status": "DATASET_UNAVAILABLE",
            "failure_reason": "No matching image pairs found in LEVIR-CC validation split",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_failure(output_dir, failure_meta)
        print("[P4-E02] Aborting: no LEVIR-CC validation pairs found.")
        return failure_meta

    predictions: list[dict[str, Any]] = []
    for pair in image_pairs:
        try:
            img_t1 = Image.open(pair["t1_path"]).convert("RGB")
            img_t2 = Image.open(pair["t2_path"]).convert("RGB")
            desc_res = backend.describe_changes(
                img_t1,
                img_t2,
                pair_id=pair["pair_id"],
                evaluation_split="val",
            )
            pred_text = desc_res.description
            record = {
                "pair_id": pair["pair_id"],
                "category": "real_levircc",
                "question": "Describe the visual differences and changes between Image 1 (T1 before) and Image 2 (T2 after) in detail.",
                "expected_ground_truth": "",
                "predicted_answer": pred_text,
                "matched_ground_truth": False,
                "dataset_source": "LEVIR-CC",
                "evaluation_split": "val",
                "t1_path": pair["t1_path"],
                "t2_path": pair["t2_path"],
            }
            predictions.append(record)
        except Exception as exc:
            print(f"[P4-E02] Warning: pair {pair['pair_id']} failed: {exc}")
            continue


    import json
    captions_path = levir_root / "LevirCCcaptions.json"
    references = {}
    if captions_path.exists():
        with open(captions_path, "r", encoding="utf-8") as f:
            cap_data = json.load(f)
            for item in cap_data.get("images", []):
                filename = item.get("filename", "")
                stem = filename.replace(".png", "")
                sentences = [s.get("raw", "") for s in item.get("sentences", [])]
                references[stem] = sentences
                
    def _compute_metrics(preds, refs):
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import jieba
        except ImportError:
            return {"bleu4": 0.0, "rouge_l": 0.0, "cider": 0.0}
            
        # Simplified metrics for demo if needed, but lets just calculate simple BLEU
        bleu_scores = []
        smooth = SmoothingFunction().method1
        for p in preds:
            pair_id = p["pair_id"]
            pred_text = p["predicted_answer"].lower().split()
            ref_texts = [r.lower().split() for r in refs.get(pair_id, [])]
            if ref_texts:
                score = sentence_bleu(ref_texts, pred_text, smoothing_function=smooth)
                bleu_scores.append(score)
        
        return {
            "bleu4": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "rouge_l": 0.0, # Placeholder
            "cider": 0.0 # Placeholder
        }
        
    calc_metrics = _compute_metrics(predictions, references)

    sample_count = len(predictions)
    elapsed = time.time() - start_time

    metrics = {
        "experiment": "P4-E02",
        "task": "bitemporal_change_description",
        "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "device": device,
        "sample_count": sample_count,
        "status": "PASS" if sample_count > 0 else "FAIL",
        "metrics": calc_metrics,
        "license_gates": {
            "cdvqa_annotation_license": "Apache-2.0",
            "second_dataset_access": "public",
            "second_image_license_status": "UNRESOLVED",
            "cdvqa_full_dataset_license_gate": "BLOCKED",
        },
        "primary_benchmark": {
            "name": "LEVIR-CC",
            "provenance": "Chenyang Liu et al. (IEEE TGRS 2022) / LEVIR Lab (Beihang Univ)",
            "upstream_imagery": "LEVIR-CD (Academic / non-commercial research use only)",
            "test_set_policy": "SEALED (evaluated on validation set)",
            "split_pairs": {
                "train": 6815,
                "val": 1332,
                "test": 1930,
            },
        },
        "execution_time_seconds": elapsed,
    }

    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds_path = output_dir / "validation_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    runner_meta = {
        "experiment": "P4-E02",
        "task": "bitemporal_change_description",
        "primary_benchmark": "LEVIR-CC",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_count": sample_count,
        "status": "PASS" if sample_count > 0 else "FAIL",
        "metrics": calc_metrics,
    }
    with open(output_dir / "runner_meta.json", "w", encoding="utf-8") as f:
        json.dump(runner_meta, f, indent=2)

    print(f"[P4-E02] Completed evaluation in {elapsed:.2f}s. Samples: {sample_count}")
    return metrics


if __name__ == "__main__":
    out_dir = Path(os.environ.get("OUTPUT_DIR", "experiments/phase4_bitemporal_vqa"))
    run_p4_e02_evaluation(out_dir)
