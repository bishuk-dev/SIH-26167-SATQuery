# Phase 3A — Grounding DINO on VRSBench

This experiment freezes `IDEA-Research/grounding-dino-tiny` as SatQuery's first
text-guided grounding baseline. The exact model revision, checkpoint checksum,
and preprocessing profile are registered in `models/registry.yaml` and
`satquery/registry/preprocessing.yaml`.

The committed manifest selects 20 VRSBench scenes at seed 43 and exactly two
valid object references per scene. Scenes, not individual references, are the
split unit: validation contains 12 scenes / 24 references and the untouched test
split contains 8 scenes / 16 references. VRSBench coordinates are supplied as
0–100 tokens and stored as normalized `xyxy` coordinates in the manifest.

Dataset provenance is pinned to Hugging Face dataset revision
`6cee2968fd752a6d51c6cb2d18dded2bc0baa218`; the annotation file and each
materialized image are checksum-verified. VRSBench text annotations are
CC-BY-4.0. Some source imagery inherited from DOTA has academic-use terms, so
image licensing must be reviewed before redistribution or commercial use.

Prepare the small image subset without downloading the 3.98 GB archive in full:

```bash
python -m ml.evaluation.prepare_vrsbench_grounding --allow-download
```

Run the frozen validation benchmark on a GPU:

```bash
python -m ml.evaluation.run_phase3a_grounding \
  --manifest experiments/phase3a_grounding_dino_vrsbench/split_manifest.json \
  --data-root data/benchmarks/vrsbench_grounding_phase3a \
  --output-dir outputs/phase3a_grounding_dino_vrsbench \
  --split validation --device cuda --allow-download
```

The evaluator reports mean IoU and Acc@0.5 IoU against the highest-scoring model
detection and preserves every raw-score detection in JSONL. No oracle selection
against ground truth is performed. Metrics are intentionally not committed until
the registered checkpoint has completed this command.
