# Kaggle GPU execution — Phase 2B

Kaggle is only an execution host. The notebook does not contain dataset, preprocessing, LoRA, training, or evaluation logic; it calls versioned modules from this repository.

## Notebook setup

1. Create a Kaggle notebook and import `notebooks/kaggle_phase2b.ipynb`, or upload that file directly.
2. In **Settings**, select a GPU accelerator. A T4 or P100 with 16 GB VRAM is recommended; the configuration uses batch size 1, gradient accumulation, 512 × 512 inputs, and LoRA on the 256M SmolVLM checkpoint. The pipeline requires CUDA and is designed for at least 8 GB VRAM, but the first Kaggle smoke run must confirm the measured peak on the assigned GPU.
3. Enable **Internet**. It is required to clone/pull GitHub, install training extras, and download the pinned model and RSVQA-LR data. Internet can be disabled only when the repository, Python packages, model snapshot, manifest, and images are supplied through `/kaggle/input`.
4. Run all cells. The notebook prints Python, PyTorch, CUDA, GPU name, VRAM, and BF16 support before doing any work. It then runs exactly one optimizer step and one validation inference. The disposable base-model cache is placed under `/kaggle/temp`; adapter checkpoints and metrics remain under `/kaggle/working`.

Override `SATQUERY_REPO_URL` when running a fork or private mirror. For a prepared Kaggle Dataset, set `SATQUERY_DATA_ROOT` to its read-only `/kaggle/input/<dataset>` directory; otherwise the repository preparation command downloads images to `/kaggle/working`.

## Commands

From the cloned repository root, prepare the separate Phase 2B manifest:

```bash
python -m ml.evaluation.prepare_phase2b_rsvqa
```

Run the same one-step smoke command used by the notebook:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b-smoke \
  --smoke-test
```

After inspecting the smoke artifacts and GPU peak memory, start the actual Phase 2B run explicitly:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b
```

If images are supplied through a Kaggle Dataset, append:

```bash
--data-root /kaggle/input/<dataset-directory>
```

Do not run the test comparison while selecting hyperparameters. Use training loss and the validation split until the configuration is frozen. Then run the single frozen/adapted/control comparison:

```bash
python -m ml.evaluation.run_phase2b_comparison \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b
```

## Resume

Kaggle sessions are ephemeral. Attach a previous output archive as a Kaggle Dataset, copy its run directory into `/kaggle/working`, and resume from an intact Trainer checkpoint:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b \
  --resume-from-checkpoint /kaggle/working/satquery-output/phase2b/checkpoints/checkpoint-<step>
```

The resumed run must use the same config, manifest, base-model revision, and preprocessing profile. The entrypoint records their hashes in `resume_guard.json` before training and refuses incompatible resumes. `run.json` additionally records the Git commit when a run completes.

## Outputs and retrieval

All writable artifacts are below `/kaggle/working/satquery-output/<run>`:

```text
adapter/       final PEFT adapter and processor files
checkpoints/   resumable Trainer checkpoints
metrics/       train metrics, GPU peak memory, and smoke prediction
logs/          TensorBoard event logs
evaluation/    frozen/adapted/control metrics and predictions
run.json       Git, dataset, model, preprocessing, hardware, and artifact provenance
resume_guard.json
               hashes used to reject incompatible checkpoint resumes
```

Use Kaggle's **Save Version** / **Save & Run All** so files under `/kaggle/working` appear in the notebook Output tab. Download the output archive or create a private Kaggle Dataset from it for resuming. Do not copy model caches, downloaded images, or checkpoints into Git; only reviewed small manifests and result summaries belong in the repository.
