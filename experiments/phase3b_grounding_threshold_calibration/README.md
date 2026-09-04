# Phase 3B — Validation-only grounding threshold calibration

Phase 3A established that Grounding DINO Tiny at box threshold 0.40 and text
threshold 0.30 has low validation recall. The completed Phase 3B sweep is:

| Box threshold | Mean IoU | Acc@0.5 | No detection | Detected | Detected-only mIoU | Huge boxes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.1834 | 0.2083 | 4 | 20 | 0.2200 | 8 |
| 0.20 | 0.1834 | 0.2083 | 4 | 20 | 0.2200 | 8 |
| 0.25 | 0.1834 | 0.2083 | 4 | 20 | 0.2200 | 8 |
| **0.30** | **0.1834** | **0.2083** | **4** | **20** | **0.2200** | **8** |
| 0.35 | 0.1314 | 0.1667 | 15 | 9 | 0.3504 | 4 |
| 0.40 | 0.1286 | 0.1667 | 17 | 7 | 0.4410 | 2 |

Threshold 0.30 wins the declared policy. Thresholds 0.15–0.30 tie on every
reported quality/count metric, so the declared final exact-tie rule selects the
higher threshold. Relative to 0.40, mean IoU rises from 0.1286 to 0.1834,
Acc@0.5 rises from 16.67% to 20.83%, and no detections fall from 17 to 4.

The gain is a recall/precision trade-off: huge selected boxes rise from 2 to 8
and detected-only mean IoU falls from 0.4410 to 0.2200. The final validation step
therefore evaluated exactly one guardrail: at threshold 0.30, reject candidates
covering at least 80% of the normalized image and fall back to the
next-highest-score candidate below that cap; abstain if none remains. This uses
only model geometry and scores, never ground truth or `object_class`.

The production selector deterministically reproduced mean IoU 0.2058, Acc@0.5
0.2500, 9 no detections, 15 detections, detected-only mean IoU 0.3293, and zero
huge selected boxes from the stored candidates. No model inference was rerun.
The guardrail is accepted and frozen with threshold 0.30; Phase 3 validation
tuning is closed.

The artifact's four phrase mismatches are heuristic flags, not four confirmed
semantic failures. Both `airplane`/`the plane` cases are aliases, and
`baseball-diamond`/`field` may also refer to the intended object. The clear
semantic failure is the ship query whose top phrase is `the water`; the proposed
guardrail addresses its nearly full-image geometry without adding query rewriting.

Phase 3B holds the model, checkpoint, preprocessing, VRSBench manifest, text
threshold, coordinate mapping, and highest-model-score selection policy fixed.
It evaluates box thresholds `0.15, 0.20, 0.25, 0.30, 0.35, 0.40` on the
validation split only. The CLI intentionally has no split argument, so it cannot
run on the test split.

To avoid six identical model forward passes, inference runs once per reference at
0.15. Higher thresholds filter those same score-sorted detections. The top model
score remains the selected detection at every threshold; IoU never participates
in detection selection.

Run on Kaggle from the repository root:

```bash
python -m ml.evaluation.run_phase3b_grounding_calibration \
  --data-root /kaggle/working/vrsbench-grounding \
  --output-dir /kaggle/working/satquery-output/phase3b-grounding-calibration \
  --device cuda --allow-download
```

`calibration.json` contains the threshold table, frozen selection rule, chosen
threshold, model/dataset/runtime provenance, and diagnostic phrase mismatches.
`validation_candidates.jsonl` preserves all detections available at 0.15.

The selected threshold maximizes Acc@0.5 IoU, then mean IoU, then minimizes the
no-detection count. An otherwise exact tie prefers the higher threshold. Boxes
covering at least 80% of normalized image area are counted as obviously huge for
diagnosis only; that count does not affect selection.

`decision.json` records the final policy and source-artifact hashes, while
`results/final_guardrail_validation.json` preserves the replay and comparisons.
The Kaggle and local manifests are semantically identical; their raw hashes differ
only because of line-ending serialization. No Phase 3B test-split evaluation was
performed. The untouched test split may be evaluated only as the single final
frozen-specialist assessment, never for further tuning.

## One-time frozen test

The final test entrypoint is fail-closed: it requires an explicit confirmation,
validates the production registry against `decision.json`, accepts only the fixed
8-scene / 16-reference test split, and refuses to run when either final artifact
already exists. Ground truth is read only after inference and the shared
production selector has completed for every reference.

The local checkpoint download stalled before inference and created no final-test
artifact, so `test_split_evaluated` remains false. After committing the frozen
implementation, launch the GPU run exactly once from a clean worktree:

```bash
python scripts/kaggle/runner.py run phase3-final-grounding-test
```

The runner retrieves `results/final_test_metrics.json` and
`results/final_test_predictions.jsonl`. Review and record those immutable results
without changing the frozen policy or reopening Phase 3 tuning.
