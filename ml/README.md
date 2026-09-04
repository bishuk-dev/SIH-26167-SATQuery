# Machine-learning workflows

This tree separates offline model development from production application code:

- `adapters/` — sensor/model adaptation modules.
- `preprocessing/` — versioned model input transformations.
- `training/` — training entry points and configuration integration.
- `inference/` — production-compatible model inference adapters.
- `evaluation/` — benchmark and regression evaluation code.
- `configs/` — experiment configuration, excluding secrets and large artifacts.

Production inference must use approved entries from `models/registry.yaml`; training code must not be required by the API or worker runtime.

## Phase 2A frozen VQA baseline

The first baseline is `HuggingFaceTB/SmolVLM-256M-Instruct` at the immutable revision recorded in `models/registry.yaml`. It is small enough for CPU development, uses an Idefics3-style vision-language architecture suitable for later adapter experiments, has an Apache-2.0 license, and is intentionally treated as domain-shifted until remote-sensing adaptation is measured.

The smoke benchmark uses `dmarsili/RSVQA-LR-2k` (CC-BY-4.0), a compact RSVQA-LR port. Scene identity is the SHA-256 of image bytes, so questions sharing an image cannot cross train/validation/test boundaries.

From the repository root, build the deterministic 24-question manifest and run the six-question held-out baseline:

```bash
python -m ml.evaluation.rsvqa_lr_subset
python -m ml.evaluation.run_phase2a_baseline --allow-download
```

After the first download, omit `--allow-download` to verify fully local execution. The checkpoint is stored under the ignored `models/cache/` directory and its SHA-256 is checked before model loading. Small manifests, predictions, and result summaries are versioned under `experiments/phase2a_smolvlm_rsvqa_lr/`; benchmark images remain under ignored `data/` storage.

## Phase 2B adaptation

Phase 2B uses a separate scene-grouped manifest and excludes every Phase 2A scene. Prepare it with:

```bash
python -m ml.evaluation.prepare_phase2b_rsvqa
```

The checked-in config at `ml/configs/phase2b_smolvlm_lora.yaml` applies rank-8 LoRA only to attention projections. The base checkpoint remains frozen; the CPU smoke run confirmed 1,363,968 trainable parameters out of 257,848,896 total (0.529%). Training requires CUDA by default:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir outputs/phase2b_smolvlm_lora
```

Use `--stability-smoke` before a full GPU run. It exercises eight optimizer steps with the configured gradient accumulation, fails on any non-finite loss/gradient/LoRA parameter, verifies that LoRA weights changed, and records runtime and peak CUDA memory. The older `--smoke-test` remains a minimal wiring check. Kaggle-specific setup, FP32 fallback, and resume instructions are in `docs/KAGGLE.md`.

Mixed precision is hardware-gated in one shared module for training and comparison: CPU uses FP32, CUDA devices below compute capability 8.0 use FP16, and BF16 requires both capability 8.0+ and a positive PyTorch runtime check. This prevents the P100/Pascal false-positive BF16 report observed on Kaggle. Both entrypoints accept `--precision fp32` as an explicit numerical-stability fallback.

## Phase 2C visual-dependence experiment

Phase 2C keeps the Phase 2B test result frozen. Its train/validation diagnostic and single visual-contrast sampling intervention are documented in `experiments/phase2c_visual_contrast/README.md`. The candidate uses `ml/configs/phase2c_smolvlm_visual_contrast.yaml`; validation comparison is performed by `ml.evaluation.run_phase2c_validation` and cannot select the test split.
