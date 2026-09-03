# Phase 2B — RSVQA-LR adaptation data

This manifest is separate from the Phase 2A smoke benchmark. All 12 Phase 2A scene hashes are excluded before splitting.

- Source: `dmarsili/RSVQA-LR-2k` validation rows, pinned to the dataset revision recorded in `split_manifest.json`.
- Group key: SHA-256 of exact image bytes.
- Selection: 1,767 questions from 88 remaining scenes.
- Train: 1,378 questions / 70 scenes.
- Validation: 180 questions / 9 scenes.
- Test: 209 questions / 9 scenes.
- Assignment: seeded (`42`) scene shuffle followed by an 80/10/10 split.

The committed manifest is the immutable experiment definition and intentionally has no generation timestamp. Normal preparation validates and reuses it while downloading only missing or corrupt image files; `--regenerate` is required to rebuild it. Training and hyperparameter decisions use only train and validation. The test split is evaluated once the configuration is frozen; the configured 100-question comparison samples test questions round-robin across scenes. Images remain under ignored `data/benchmarks/rsvqa_lr_phase2b/`; only the provenance and split manifest is versioned.
