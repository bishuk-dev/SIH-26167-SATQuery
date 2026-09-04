# Phase 3B — Validation-only grounding threshold calibration

Phase 3A established that Grounding DINO Tiny at box threshold 0.40 and text
threshold 0.30 has low validation recall:

| Box threshold | Mean IoU | Acc@0.5 | No detection | Detected |
|---:|---:|---:|---:|---:|
| 0.40 | 0.1286 | 0.1667 | 17 | 7 |

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

Local execution was attempted, but the model CDN repeatedly timed out before the
checkpoint completed. Therefore no threshold other than the supplied Phase 3A
0.40 result is recorded here yet, and the production registry remains at 0.40.
Run the command above and return the two output files before freezing the chosen
threshold. No Phase 3B test-split evaluation is permitted.
