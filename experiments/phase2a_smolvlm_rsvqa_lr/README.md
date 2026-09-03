# Phase 2A — frozen SmolVLM on RSVQA-LR

This is a smoke-scale integration baseline, not a publishable model-quality estimate.

- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`, frozen at the revision and checkpoint SHA-256 in `models/registry.yaml`.
- Dataset: `dmarsili/RSVQA-LR-2k`, revision `35de41f26170edda2ccc4f88c0f62f641bb9e1f1`, sourced from its validation split.
- Grouping: exact image bytes are SHA-256 hashed; every question for a selected scene receives the same split.
- Subset: first 12 unique source-order scenes, two questions each; scene hashes are sorted and assigned 7 train, 2 validation, and 3 test scenes.
- Metric: normalized exact match (case, punctuation, and whitespace normalized).
- Comparator: the most frequent normalized training answer, included as a cheap shortcut baseline.

The held-out frozen model result is 0/6 (0.0) normalized exact match. The train-majority answer `yes` scores 4/6 (0.6667). With only three test scenes, neither number should be generalized; the result establishes a reproducible lower baseline and demonstrates the need for Phase 2B remote-sensing adaptation.

`split_manifest.json`, `results.json`, and `predictions.jsonl` contain the exact sample assignments, provenance, predictions, metrics, and measured runtime. Rebuild/run commands are documented in `ml/README.md`.
