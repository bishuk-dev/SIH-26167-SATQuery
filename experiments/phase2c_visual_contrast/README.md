# Phase 2C — visual-contrast sampling

Phase 2C diagnoses and addresses the question-prior shortcut without analyzing or evaluating the frozen Phase 2B test split. `diagnostic.json` is deterministic and derived only from the committed Phase 2B train and validation samples.

The single candidate intervention is `visual_contrast_balanced` sampling. It retains real samples from all 70 training scenes where the same normalized question has different answers across scenes, then balances answer frequency within each question. This produces 464 training instances across 100 contrastive question groups. No shuffled image is assigned a fabricated answer.

The candidate is trained from the same frozen SmolVLM base as Phase 2B and uses the same preprocessing, LoRA configuration, learning rate, and optimizer-related training settings. With batch size 1, four-way gradient accumulation, 464 selected samples, and three epochs, it runs 348 optimizer steps. Generate the diagnostic with:

```bash
python -m ml.evaluation.phase2c_diagnostics
```

Run the one candidate training experiment on Kaggle:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2c_smolvlm_visual_contrast.yaml \
  --output-dir /kaggle/working/satquery-output/phase2c-visual-contrast
```

After training, compare the existing Phase 2B adapter and candidate on validation only:

```bash
python -m ml.evaluation.run_phase2c_validation \
  --config ml/configs/phase2c_smolvlm_visual_contrast.yaml \
  --phase2b-adapter /kaggle/working/satquery-output/phase2b/adapter \
  --phase2c-adapter /kaggle/working/satquery-output/phase2c-visual-contrast/adapter \
  --output /kaggle/working/satquery-output/phase2c-visual-contrast/validation.json
```

The comparison covers correct, blank, and deterministically shuffled images plus the exact-question training-prior baseline. A candidate is marked materially improved only if its correct-image advantage over the stronger image control increases by at least five percentage points versus Phase 2B. It does not access the test split.
