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

Use `--smoke-test` for one optimizer step and one validation inference. Kaggle-specific setup and resume instructions are in `docs/KAGGLE.md`.
