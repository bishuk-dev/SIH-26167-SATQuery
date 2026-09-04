# Kaggle GPU execution — Phase 2B

Kaggle is only an execution host. The notebook does not contain dataset, preprocessing, LoRA, training, or evaluation logic; it calls versioned modules from this repository.

## Notebook setup

1. Create a Kaggle notebook and import `notebooks/kaggle_phase2b.ipynb`, or upload that file directly.
2. In **Settings**, select a GPU accelerator. The verified target is `Tesla P100-PCIE-16GB` (Pascal, compute capability 6.0). The configuration uses batch size 1, gradient accumulation, 512 × 512 inputs, and LoRA on the 256M SmolVLM checkpoint.
3. Enable **Internet**. It is required to clone/pull GitHub, install training extras, and download the pinned model and RSVQA-LR data. Internet can be disabled only when the repository, Python packages, model snapshot, manifest, and images are supplied through `/kaggle/input`.
4. Run all cells from a fresh session. The first code cell removes incompatible `torchao` before any PyTorch import and pins `torch==2.8.0`, `torchvision==0.23.0`, and `torchaudio==2.8.0` from the CUDA 12.6 wheel index. The next cell asserts `torch==2.8.0+cu126`, then prints Python, PyTorch, CUDA, GPU name, VRAM, compute capability, and the runtime's BF16 report. It then runs the eight-optimizer-step stability smoke with the real configured gradient accumulation. The disposable base-model cache is placed under `/kaggle/temp`; adapter checkpoints and metrics remain under `/kaggle/working`.

The P100 selects FP16 by default. SatQuery does not trust `torch.cuda.is_bf16_supported()` by itself: BF16 is selected only when the runtime reports support **and** the GPU compute capability is at least 8.0 (Ampere or newer). CPU debugging remains FP32. FP16 is not considered safe for a full P100 run until the stability smoke passes; `--precision fp32` is the explicit fallback if it fails.

Equivalent bootstrap commands are:

```bash
python -m pip uninstall -y torchao
python -m pip install --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Override `SATQUERY_REPO_URL` when running a fork or private mirror. For a prepared Kaggle Dataset, set `SATQUERY_DATA_ROOT` to its read-only `/kaggle/input/<dataset>` directory; otherwise the repository preparation command downloads images to `/kaggle/working`.

## Commands

From the cloned repository root, validate the committed Phase 2B manifest and materialize any missing images:

```bash
python -m ml.evaluation.prepare_phase2b_rsvqa
```

Normal preparation never rewrites the committed split manifest. It validates dataset provenance, Phase 2A exclusions, scene assignments, split counts, and image hashes. Use `--regenerate` only when deliberately defining a new experiment; deterministic regeneration no longer embeds a timestamp.

Run the same stability smoke command used by the notebook:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b-stability-fp16 \
  --stability-smoke
```

This executes eight optimizer steps and the configured four-way gradient accumulation. Every microbatch loss and every optimizer-step gradient/LoRA parameter is checked for finiteness; the run also requires at least one LoRA parameter to change. Results are written to `metrics/stability_smoke.json`. A non-finite run stops immediately and writes `metrics/numerical_failure.json`.

If the FP16 stability smoke fails on the P100, rerun it in FP32 in a separate output directory:

```bash
python -m ml.training.phase2b \
  --config ml/configs/phase2b_smolvlm_lora.yaml \
  --output-dir /kaggle/working/satquery-output/phase2b-stability-fp32 \
  --stability-smoke \
  --precision fp32
```

Only after inspecting a passing stability report and its GPU peak memory, start the actual Phase 2B run explicitly. If only the FP32 smoke passes, append `--precision fp32` to this command:

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

The resumed run must use the same config, frozen manifest, base-model revision, preprocessing profile, and selected precision. The entrypoint records them in `resume_guard.json` before training and refuses incompatible resumes. `run.json` additionally records the Git commit when a run completes.

## Outputs and retrieval

All writable artifacts are below `/kaggle/working/satquery-output/<run>`:

```text
adapter/       final PEFT adapter and processor files
checkpoints/   resumable Trainer checkpoints
metrics/       train metrics, stability/failure report, GPU peak memory, smoke prediction
logs/          TensorBoard event logs
evaluation/    frozen/adapted/control metrics and predictions
run.json       Git, dataset, model, preprocessing, hardware, and artifact provenance
resume_guard.json
               hashes used to reject incompatible checkpoint resumes
```

Use Kaggle's **Save Version** / **Save & Run All** so files under `/kaggle/working` appear in the notebook Output tab. Download the output archive or create a private Kaggle Dataset from it for resuming. Do not copy model caches, downloaded images, or checkpoints into Git; only reviewed small manifests and result summaries belong in the repository.

## Phase 3A grounding baseline

Import `notebooks/kaggle_phase3a.ipynb`, select a P100/T4-class GPU, and enable
Internet. The notebook retains the verified PyTorch 2.8.0 + CUDA 12.6 setup,
installs the repository's `grounding` extra, range-downloads only the 20 selected
VRSBench images, and runs the frozen validation benchmark. No training logic is
present.

Equivalent repository commands are:

```bash
python -m ml.evaluation.prepare_vrsbench_grounding \
  --data-root /kaggle/working/vrsbench-grounding --allow-download
python -m ml.evaluation.run_phase3a_grounding \
  --data-root /kaggle/working/vrsbench-grounding \
  --output-dir /kaggle/working/satquery-output/phase3a-grounding-dino \
  --split validation --device cuda --allow-download
```

Metrics appear in `validation_metrics.json`; all detections and per-reference IoU
appear in `validation_predictions.jsonl`. Retrieve the directory from Kaggle's
Output tab. Do not run the untouched test split while changing the model,
thresholds, preprocessing, or subset definition.
