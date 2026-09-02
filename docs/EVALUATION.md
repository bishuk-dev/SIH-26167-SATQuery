# SatQuery AI — Evaluation Protocol

> **Source-of-truth evaluation plan for SatQuery AI.**

**Project:** SatQuery AI  
**Problem Statement ID:** 26167  
**Document:** `docs/EVALUATION.md`  
**Status:** Evaluation freeze — v1  
**Last updated:** 2026-09-03

---

# 1. Purpose

This document defines **how SatQuery is evaluated**.

Its purpose is to prevent vague claims such as:

```text
"the model looks accurate"
"fusion seems better"
"confidence is high"
"the agent worked"
```

without measurable evidence.

SatQuery is a multi-component analytical system. Therefore, evaluation must separately measure:

1. perception quality,
2. language-answer quality,
3. spatial correctness,
4. temporal correctness,
5. multimodal contribution,
6. quantitative measurement accuracy,
7. confidence calibration,
8. cross-region / cross-sensor / cross-scale robustness,
9. routing and workflow correctness,
10. execution validity,
11. latency and resource cost,
12. failure handling.

No single metric is sufficient.

---

# 2. Evaluation principles

## E-P-001 — Evaluate the capability, not just the final sentence

A fluent answer is not enough.

Example:

```text
Question:
"Where did water increase?"

Answer:
"Water increased in the south."
```

The answer may sound correct while the predicted mask is geographically wrong.

Therefore SatQuery evaluates:

```text
semantic answer
+
spatial evidence
+
workflow validity
```

separately.

---

## E-P-002 — Different tasks require different metrics

SatQuery MUST NOT use one "overall accuracy" metric for every capability.

Examples:

```text
VQA            → accuracy
grounding      → IoU / mIoU
segmentation   → IoU / F1
change         → precision / recall / F1 / IoU
measurement    → MAE / relative error
calibration    → ECE / NLL
routing        → routing accuracy
```

---

## E-P-003 — Test-set integrity

The test set MUST NOT be used repeatedly to tune:

- prompts,
- hyperparameters,
- thresholds,
- model architecture,
- preprocessing,
- routing rules.

Development decisions should use training/validation data.

Final test sets remain untouched until the relevant design is frozen.

---

## E-P-004 — Scene-level splitting

Annotations derived from the same underlying image or temporal pair MUST remain in the same split.

Bad:

```text
Scene A, question 1 → train
Scene A, question 2 → test
```

Good:

```text
Scene A → one split only
```

Where possible, spatially adjacent tiles should also be grouped to reduce geographic leakage.

---

## E-P-005 — Domain shift is a first-class metric

SatQuery must report:

- in-domain performance,
- cross-region performance,
- cross-sensor performance,
- cross-scale performance,

where relevant data exist.

A Sentinel score must not be presented as proof of Cartosat/RISAT performance.

---

## E-P-006 — Fusion must beat or complement unimodal baselines

Every optical/SAR fusion experiment must include:

```text
Optical only
SAR only
Optical + SAR
```

Accepting two modalities is not evidence that both are used.

---

## E-P-007 — Invalid requests are evaluation cases

Correct refusal is a capability.

Examples:

```text
1 image + change query
RGB + NDVI request
no CRS + area request
non-overlapping pair
```

These must be part of evaluation.

---

## E-P-008 — No invented official weighting

If the official challenge does not publish exact metric weights, SatQuery MUST NOT invent them.

Internal summaries may be used for engineering convenience but must be explicitly labeled as internal.

---

# 3. Evaluation layers

SatQuery evaluation is divided into five layers.

| Layer | Question |
|---|---|
| Perception | Did the model understand the imagery? |
| Language | Did it answer the question correctly? |
| Spatial / physical | Is the evidence geographically and numerically correct? |
| Reliability | Is confidence meaningful and robust under shift? |
| Workflow | Did the system use a valid analysis trajectory? |

---

# 4. Dataset split policy

Each benchmark should use:

```text
TRAIN
VALIDATION
TEST
```

where available.

For additional internal robustness tests:

```text
CROSS_REGION_TEST
CROSS_SENSOR_TEST
CROSS_SCALE_TEST
FAILURE_TEST
```

may be maintained separately.

---

## 4.1 Split grouping unit

Preferred grouping order:

```text
underlying scene / pair
        >
location / tile group
        >
individual annotation
```

The same physical scene should never be represented independently in train and test merely because it has different questions or captions.

---

## 4.2 Temporal grouping

For temporal datasets:

```text
T1 + T2 pair
```

is one indivisible unit.

All questions / masks derived from that pair remain in the same split.

---

## 4.3 Cross-region split

A cross-region test should contain geographic regions not used during training.

Its purpose is to answer:

> Did the model learn general visual concepts or regional appearance priors?

---

## 4.4 Cross-sensor split

A cross-sensor test should contain an unseen or held-out sensor family where possible.

Its purpose is to answer:

> Does perception survive a change in sensing characteristics?

---

## 4.5 Cross-scale split

A cross-scale test should differ meaningfully in GSD / object scale.

Its purpose is to answer:

> Does the model understand physical structures or only patterns at familiar pixel scales?

---

# 5. Baseline requirements

Every model family must be compared against credible baselines.

---

## 5.1 VQA baselines

At minimum:

```text
majority-answer baseline
question-only baseline
frozen generic VLM
remote-sensing adapted model
```

Where practical:

```text
blank-image baseline
shuffled-image baseline
```

---

## 5.2 Grounding baselines

At minimum:

```text
frozen / zero-shot grounding model
remote-sensing fine-tuned grounding model
```

Where multiple models exist, compare under identical image scaling / tiling conditions.

---

## 5.3 Fusion baselines

Required:

```text
O
S
O+S
```

---

## 5.4 Change baselines

At minimum:

```text
simple image-difference baseline
trained change specialist
```

The difference baseline is not expected to win; it provides a sanity reference.

---

## 5.5 Calibration baseline

Compare:

```text
raw model probability
post-hoc calibrated probability
```

where calibration is applicable.

---

# 6. Classification and VQA metrics

## 6.1 Accuracy

For closed-form answers:

\[
Accuracy =
\frac{\text{correct predictions}}
{\text{total predictions}}
\]

Use for:

- binary VQA,
- multiple-choice VQA,
- constrained classification.

---

## 6.2 Per-question-type accuracy

Overall accuracy can hide weak categories.

Report accuracy by question type where labels exist, for example:

```text
presence
count
area
comparison
spatial relation
scene type
change direction
```

---

## 6.3 Class / answer imbalance

If one answer dominates, report:

- majority baseline,
- class distribution,
- per-class accuracy where meaningful.

A model that barely beats the majority baseline is not considered strong.

---

# 7. Precision, recall and F1

For positive class evaluation:

\[
Precision =
\frac{TP}{TP + FP}
\]

\[
Recall =
\frac{TP}{TP + FN}
\]

\[
F1 =
2\frac{Precision \cdot Recall}{Precision + Recall}
\]

Use especially for:

- binary change masks,
- rare classes,
- event detection,
- imbalanced segmentation.

---

# 8. IoU and mIoU

For predicted region \(P\) and ground truth \(G\):

\[
IoU =
\frac{|P \cap G|}{|P \cup G|}
\]

Use for:

- grounding boxes,
- segmentation masks,
- change masks.

Mean IoU:

\[
mIoU =
\frac{1}{N}
\sum_i IoU_i
\]

The averaging unit must always be documented.

---

## 8.1 Grounding threshold metrics

Where relevant, report:

```text
Acc@IoU 0.25
Acc@IoU 0.50
Acc@IoU 0.75
```

or the thresholds used by the benchmark.

Do not compare threshold metrics across papers unless the definitions match.

---

# 9. Dice score

For binary masks:

\[
Dice =
\frac{2|P \cap G|}{|P| + |G|}
\]

Equivalent to pixel-level F1 in the binary case.

If Dice is used, also report IoU where practical because IoU is easier to compare across grounding / segmentation workflows.

---

# 10. Object detection metrics

Use:

```text
AP
mAP
```

with the benchmark's defined IoU protocol.

Also report where possible:

- per-class AP,
- small-object AP,
- medium-object AP,
- large-object AP.

Remote-sensing images often contain extreme scale variation, so a single mAP can hide failure on small objects.

---

# 11. Caption / free-text metrics

Where captioning is enabled, use multiple metrics.

Potential lexical metrics:

```text
BLEU
ROUGE
METEOR
CIDEr
```

Potential semantic metrics:

```text
BERTScore
sentence-embedding similarity
```

Language similarity MUST NOT be treated as scientific correctness.

Example:

```text
reference:
"water increased"

prediction:
"water decreased"
```

may remain lexically / semantically similar despite opposite scientific meaning.

Therefore important factual fields should be evaluated separately where structured labels exist.

---

# 12. Structured factual evaluation

When possible, extract or directly store structured targets such as:

```text
object
class
presence
count
direction
change_type
spatial_relation
area
```

Evaluate these separately from prose quality.

Example:

```yaml
reference:
  class: water
  change: increase
  region: south

prediction:
  class: water
  change: decrease
  region: south
```

Text may look similar, but the structured change result is wrong.

---

# 13. Numerical measurement evaluation

For numeric predictions:

\[
MAE =
\frac{1}{N}
\sum_i |\hat{y_i} - y_i|
\]

Where meaningful:

\[
RelativeError =
\frac{|\hat{y} - y|}{|y|}
\]

Use for:

- area,
- distance,
- count-derived quantity,
- change extent.

Relative error must be handled carefully when the true value approaches zero.

---

## 13.1 Measurement validity

A numeric result is not accepted solely because its value is close.

The workflow must also pass:

```text
valid CRS / geodesic path
valid evidence geometry
valid unit conversion
valid source mask
```

A numerically lucky result from an invalid workflow is still invalid.

---

# 14. Change-detection evaluation

Minimum metrics:

```text
Precision
Recall
F1
IoU
```

Where semantic classes exist:

```text
per-class IoU
mIoU
```

---

## 14.1 Temporal sanity controls

Every relevant change model must be tested on:

### Normal pair

```text
T1 + T2
```

### Identity pair

```text
T1 + T1
```

Expected:

```text
little / no meaningful change
```

### Reversed pair

```text
T2 + T1
```

For direction-sensitive tasks, semantic direction should reverse appropriately.

---

## 14.2 Misalignment test

Deliberately misalign a temporal pair.

Expected behavior:

- pair validator detects invalid / degraded alignment,
- change workflow warns or rejects,
- model output is not blindly trusted.

---

# 15. Change-VQA evaluation

Report:

```text
answer accuracy
+
linked change-mask quality
```

A correct text answer with an incorrect change region is considered partially failed.

Suggested record:

```yaml
qa_correct: true
mask_iou: 0.18
final_status: spatial_failure
```

---

# 16. VQA shortcut / leakage tests

These tests determine whether the model relies on imagery.

---

## 16.1 Blank-image test

Replace the image with a blank image while retaining the question.

A strong visual model should show meaningful degradation.

---

## 16.2 Shuffled-image test

Pair each question with an unrelated image.

If performance remains close to the real-image score, the benchmark/model may contain strong language shortcuts.

---

## 16.3 Question-only baseline

Run a text-only answer model using the question.

This reveals dataset answer priors.

---

## 16.4 Interpretation

Example:

```text
real image        81%
blank image       79%
question only     78%
```

This is a serious warning.

Example:

```text
real image        81%
blank image       52%
question only     50%
```

This provides stronger evidence that visual information matters.

No universal minimum gap is assumed; compare empirically.

---

# 17. Optical-SAR fusion evaluation

Every cross-modal experiment MUST include:

\[
Score(O)
\]

\[
Score(S)
\]

\[
Score(O,S)
\]

Define internal diagnostic:

\[
\Delta_{fusion} =
Score(O,S) - \max(Score(O), Score(S))
\]

This is an internal diagnostic, not an official benchmark metric.

---

## 17.1 Fusion success

Strong evidence of complementary fusion:

```text
O      70
S      68
O+S    84
```

---

## 17.2 Possible modality dominance

Warning case:

```text
O      83
S      59
O+S    83
```

Additional tests are required to determine whether SAR is being ignored.

---

## 17.3 Corrupted-modality test

Use:

```text
correct O + wrong S
correct S + wrong O
```

If fusion output barely reacts to one modality being replaced, that modality may not meaningfully contribute.

---

## 17.4 Missing-modality test

Evaluate graceful degradation:

```text
both modalities
optical only
SAR only
```

Missing modality should be explicitly represented, not silently replaced with a valid-valued zero image.

---

# 18. Modality-agreement evaluation

Where optical and SAR specialists produce compatible spatial evidence:

\[
AgreementIoU =
IoU(M_O, M_S)
\]

This measures prediction agreement, **not ground-truth accuracy**.

Low agreement can indicate:

- one model is wrong,
- modalities observe different aspects,
- data quality differs,
- registration problems,
- domain shift.

Therefore agreement is a reliability signal, not a correctness metric.

---

# 19. Calibration evaluation

A confidence value is meaningful only if it is calibrated.

---

## 19.1 Reliability diagram

Group predictions into confidence bins.

Compare:

```text
average confidence
vs
actual accuracy
```

A calibrated model should lie near the diagonal.

---

## 19.2 Expected Calibration Error

\[
ECE =
\sum_{m=1}^{M}
\frac{|B_m|}{N}
\left|
acc(B_m)-conf(B_m)
\right|
\]

Document binning strategy.

---

## 19.3 Negative log-likelihood

\[
NLL =
-\frac{1}{N}
\sum_i \log P(y_i)
\]

Useful where probabilistic classification outputs are available.

---

## 19.4 Calibration policy

Compare:

```text
uncalibrated
vs
calibrated
```

on validation/calibration data.

Never calibrate using the final test set.

---

# 20. Cross-region evaluation

Report:

```text
in-domain score
cross-region score
absolute drop
relative drop
```

Example:

```text
in-domain:      84
cross-region:   72
absolute drop:  12 points
```

---

# 21. Cross-sensor evaluation

This is a critical SatQuery metric.

Report:

```text
source-sensor score
held-out sensor score
degradation
```

Where possible, isolate sensor shift from geography and scale.

---

## 21.1 Internal domain-shift diagnostic

Optional internal ratio:

\[
D_{shift} =
1 -
\frac{Score_{shifted}}{Score_{in-domain}}
\]

This is not an official metric.

Report both raw scores alongside any diagnostic ratio.

---

# 22. Cross-scale evaluation

Group results by:

- GSD,
- object size,
- image scale,
- tiling / resize strategy.

Compare:

```text
global resize
tile inference
coarse-to-fine
```

for high-resolution tasks.

Metrics must include both quality and computational cost.

---

# 23. Grounding coordinate correctness

Grounding evaluation has two independent checks.

### Model-space correctness

Does the predicted box/mask overlap reference evidence?

### Geospatial mapping correctness

Does the model-space evidence map back to the correct source/world coordinates?

A unit/integration test MUST verify:

```text
crop pixel
→ source pixel
→ world coordinate
```

with known expected values.

---

# 24. Router evaluation

Maintain a labeled routing test set.

Example categories:

```text
SINGLE_VQA
GROUND_OBJECT
CROSS_MODAL_VQA
CHANGE_VQA
CHANGE_LOCALIZE
CHANGE_MEASURE
METADATA_QUERY
INVALID_REQUEST
```

Primary metric:

\[
RoutingAccuracy =
\frac{\text{correct workflow selections}}{\text{total routing cases}}
\]

Also report per-intent accuracy.

---

# 25. Input-validator evaluation

The validator must be tested as a classifier over:

```text
ALLOW
ALLOW_WITH_WARNING
REJECT
```

Examples:

| Case | Expected |
|---|---|
| one image + change query | REJECT |
| RGB + NDVI | REJECT |
| no CRS + semantic VQA | ALLOW_WITH_WARNING |
| no CRS + area | REJECT |
| valid optical/SAR pair | ALLOW |
| non-overlap pair | REJECT |

Metrics:

```text
overall decision accuracy
false acceptance rate
false rejection rate
reason-code accuracy
```

False acceptance is particularly important because it may permit scientifically invalid output.

---

# 26. Workflow evaluation

Final-answer correctness is not enough.

Each workflow should be evaluated at:

```text
input validation
workflow selection
model selection
tool selection
parameter validity
intermediate evidence
measurement validity
verification
final answer
```

---

## 26.1 Pipeline Integrity

Internal implementation may track:

\[
PI =
\frac{\text{valid transitions}}{\text{total transitions}}
\]

This is inspired by recent agentic-EO research and is an internal workflow metric, not an official challenge metric.

---

## 26.2 Intermediate state audit

Every sampled analysis should allow inspection of:

- source observations,
- derived assets,
- models used,
- masks/boxes,
- measurements,
- verifier output.

---

# 27. Execution trace evaluation

Trace completeness checklist:

```text
task present
input IDs present
model ID present
model version present
tool IDs present
important parameters present
evidence IDs present
verification present
warnings present
```

Trace completeness can be scored internally as percentage of required fields present.

---

# 28. Performance evaluation

Every approved model/workflow should record:

```text
total latency
preprocessing latency
model latency
GIS latency
verification latency
answer-generation latency

CPU memory
GPU VRAM
```

Where applicable:

```text
throughput
tiles/sec
samples/sec
```

---

## 28.1 Cost-quality tradeoff

Do not automatically prefer a model that gains a tiny metric improvement at enormous runtime/memory cost.

Record both:

```text
quality
resource cost
```

for model-selection decisions.

---

# 29. Required evaluation matrices

## 29.1 Single-image VQA matrix

| Model | Dataset | In-domain Acc | Blank-image | Shuffled-image | Cross-region | Cross-sensor | Latency |
|---|---|---:|---:|---:|---:|---:|---:|

---

## 29.2 Grounding matrix

| Model | Dataset | mIoU | Acc@0.5 | Acc@0.75 | Small Obj | Cross-scale | Latency |
|---|---|---:|---:|---:|---:|---:|---:|

---

## 29.3 Fusion matrix

| Model | Optical | SAR | O+S | Fusion Gain | Agreement | Cross-sensor |
|---|---:|---:|---:|---:|---:|---:|

---

## 29.4 Change matrix

| Model | Precision | Recall | F1 | IoU | T1+T1 FP | Reverse-order | Cross-region |
|---|---:|---:|---:|---:|---:|---:|---:|

---

## 29.5 Calibration matrix

| Model | Accuracy | ECE Raw | ECE Calibrated | NLL Raw | NLL Calibrated |
|---|---:|---:|---:|---:|---:|

---

## 29.6 System matrix

| Workflow | Routing | Validation | Evidence | Measurement | Verification | Latency |
|---|---:|---:|---:|---:|---:|---:|

---

# 30. Experiment record format

Every experiment SHOULD store:

```yaml
experiment_id: exp_...

date: ...

dataset:
  name: ...
  version: ...
  split: ...

model:
  id: ...
  version: ...
  checkpoint_hash: ...

preprocessing:
  profile: ...

training:
  adapter: ...
  learning_rate: ...
  batch_size: ...
  epochs: ...
  seed: ...

evaluation:
  metrics: {}
  domain: ...
  latency: ...
  peak_vram: ...

notes:
  - ...
```

---

# 31. Model promotion policy

A new checkpoint may replace the production checkpoint only if:

1. benchmark evaluation is complete,
2. no critical regression appears,
3. cross-domain performance is inspected,
4. failure tests pass,
5. resource use remains acceptable,
6. model/preprocessing version is registered.

A model should not be promoted because one demonstration example looks better.

---

# 32. Golden regression suite

Before competition hardening, maintain a small manually inspected set covering:

- optical VQA,
- SAR VQA,
- grounding,
- optical/SAR fusion,
- change,
- no-change,
- measurements,
- invalid requests,
- OOD sensor cases.

Every release runs this set automatically.

---

# 33. Competition evaluation policy

If the official evaluator defines:

- benchmark subsets,
- metrics,
- normalization,
- hidden test behavior,

those rules take precedence for competition reporting.

Internal SatQuery evaluation remains broader because it is designed to expose failure modes the official score may not isolate.

---

# 34. Interpretation rules

## High benchmark score does not imply

- cross-sensor generalization,
- calibrated confidence,
- correct grounding,
- valid measurement,
- valid workflow.

## High VQA accuracy does not imply

- image dependence,
- spatial correctness,
- multimodal use.

## High fusion score does not imply

- both modalities contributed.

## Correct final number does not imply

- valid GIS pipeline.

## Fluent explanation does not imply

- scientific correctness.

---

# 35. Minimum evaluation required before MVP claim

The project cannot claim MVP completion until it has:

- VQA benchmark,
- grounding benchmark,
- optical/SAR O-S-O+S ablation,
- temporal change benchmark,
- temporal T1+T1 sanity test,
- routing test,
- invalid-input test,
- execution trace validation,
- basic latency profiling.

---

# 36. Minimum evaluation required before competition-ready claim

Competition-ready status additionally requires:

- cross-region evaluation,
- cross-sensor evaluation,
- cross-scale evaluation,
- confidence calibration or explicit non-calibrated status,
- failure/adversarial suite,
- model-regression suite,
- frozen checkpoints,
- frozen preprocessing,
- reproducible experiment records,
- demo-case verification.

---

# 37. Research basis

This evaluation design is informed by the reviewed research on:

- multimodal geospatial foundation models,
- BigEarthNet.txt,
- VRSBench,
- RSVQA,
- CDVQA,
- CROMA,
- confidence calibration,
- agentic EO workflow validation.

Important status distinction:

- task metrics such as accuracy, IoU, mIoU, F1, mAP, ECE, and NLL are established evaluation concepts,
- workflow metrics such as pipeline integrity / trajectory validity are used here as internal research-inspired diagnostics,
- no unsupported official SIH weighting is assumed.

---

# 38. Final evaluation principle

The project should prefer:

```text
a lower but trustworthy score
```

over:

```text
a higher score produced by leakage,
shortcut learning,
invalid geometry,
or hidden sensor assumptions.
```

> **SatQuery is successful only when the answer, the evidence, and the workflow are all defensible.**
