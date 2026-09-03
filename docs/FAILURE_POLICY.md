# SatQuery AI — Failure Policy

> **Source-of-truth policy for refusal, warning, degradation, retry, and abstention behavior in SatQuery AI.**

**Project:** SatQuery AI  
**Problem Statement ID:** 26167  
**Document:** `docs/FAILURE_POLICY.md`  
**Status:** Failure policy freeze — v1  
**Last updated:** 2026-09-03

---

# 1. Purpose

SatQuery is an analytical system, not a system that must always produce an answer.

This document defines what SatQuery should do when:

- required data are missing,
- observations are incompatible,
- metadata are incomplete,
- a model is out of domain,
- evidence is weak,
- modalities disagree,
- a GIS operation is invalid,
- a model/tool crashes,
- a user requests an unsupported scientific operation.

The default principle is:

> **A correct refusal is better than an unsupported answer.**

---

# 2. Failure outcomes

Every detected issue is classified into one of five outcomes.

---

## 2.1 `ALLOW`

The requested workflow is scientifically and technically supported.

Example:

```text
2 aligned georeferenced temporal images
+
change query
```

---

## 2.2 `ALLOW_WITH_WARNING`

The task can still be performed, but some guarantees are unavailable.

Example:

```text
JPEG without CRS
+
"Are buildings visible?"
```

Pixel-space semantic analysis is possible, but geographic measurement is not.

---

## 2.3 `REQUEST_INPUT`

The request could become valid if the user provides missing information or data.

Example:

```text
1 image
+
"What changed?"
```

Request:

> Upload a second observation of the same area from another time.

---

## 2.4 `ABSTAIN`

The workflow is technically executable, but the available evidence is too weak or conflicting for a defensible conclusion.

Example:

```text
optical says flood
SAR says no flood
optical has heavy cloud contamination
```

---

## 2.5 `REJECT`

The requested operation is scientifically invalid, unsafe, unsupported, or impossible with the supplied evidence.

Example:

```text
RGB image
+
"Compute NDVI"
```

No NIR band exists.

---

# 3. Failure severity

Failures are categorized as:

```text
INFO
WARNING
ERROR
CRITICAL
```

### INFO

No analytical limitation; informational notice only.

### WARNING

Analysis may proceed, but result must expose the warning.

### ERROR

The current workflow cannot proceed.

### CRITICAL

The input or operation creates security/system-integrity risk and must be terminated immediately.

---

# 4. Error response contract

Failures should be structured.

```yaml
Failure:
  code: string
  severity: INFO|WARNING|ERROR|CRITICAL

  outcome:
    ALLOW|ALLOW_WITH_WARNING|REQUEST_INPUT|ABSTAIN|REJECT

  user_message: string
  technical_message: string

  affected_requirement: string|null

  recoverable: boolean

  required_action:
    type: string|null
    details: object|null

  evidence_ids: [string]

  warnings: [string]
```

The UI renders `user_message`.

Technical logs retain `technical_message`.

---

# 5. General rules

## F-P-001 — Never fabricate missing evidence

SatQuery MUST NOT fabricate:

- missing bands,
- missing polarization,
- missing sensor name,
- missing acquisition time,
- missing georeferencing,
- missing temporal image,
- missing mask,
- missing measurement.

---

## F-P-002 — Never replace scientific evidence with prose

If a required analytical step fails, the answer composer does not "fill in" the missing result.

Example:

```text
change model failed
```

Forbidden:

> "The region appears to have changed by approximately 20%."

Required:

> The change-analysis model failed, so SatQuery cannot provide a verified change result for this request.

---

## F-P-003 — Unknown is not failure by itself

Unknown metadata may be acceptable if the requested task does not require it.

Example:

```text
sensor = unknown
modality = optical
question = "Are buildings visible?"
```

May proceed with warning if the selected model supports generic optical input.

---

## F-P-004 — Failure should be local where possible

If one optional capability fails, do not automatically discard valid independent evidence.

Example:

```text
optical analysis succeeds
SAR analysis fails
```

Possible:

- return optical-only result,
- explicitly state SAR could not be evaluated,
- do not describe the result as fused evidence.

---

## F-P-005 — Quantitative claims require stronger validity than qualitative claims

A semantic question may be answerable without georeferencing.

A physical-area question is not.

---

# 6. Upload and file failures

## F-UPLOAD-001 — Unsupported file type

**Condition**

Actual raster driver is not in the allowed list.

**Outcome**

```text
REJECT
```

**Code**

```text
UNSUPPORTED_RASTER_DRIVER
```

**User message**

> This raster format is not supported by the current SatQuery ingestion pipeline. Please upload a supported GeoTIFF/TIFF or approved PNG/JPEG input.

---

## F-UPLOAD-002 — Extension/driver mismatch

**Condition**

Filename extension suggests an allowed raster but detected driver is different or disallowed.

**Outcome**

```text
REJECT
```

**Code**

```text
RASTER_DRIVER_MISMATCH
```

This is treated as a security/integrity issue.

---

## F-UPLOAD-003 — Corrupt raster

**Condition**

GDAL/Rasterio cannot safely read required metadata/data.

**Outcome**

```text
REJECT
```

**Code**

```text
INVALID_RASTER
```

---

## F-UPLOAD-004 — Resource limits exceeded

**Condition**

Input exceeds configured:

- bytes,
- width,
- height,
- band count,
- total pixel count,
- processing timeout.

**Outcome**

```text
REJECT
```

**Code**

```text
RASTER_RESOURCE_LIMIT_EXCEEDED
```

---

## F-UPLOAD-005 — Suspicious driver / unsafe raster feature

**Condition**

Input requires a disallowed driver or potentially unsafe external reference/execution behavior.

**Outcome**

```text
REJECT
```

**Severity**

```text
CRITICAL
```

---

## F-UPLOAD-006 — Asset storage failure

**Condition**

The inspected raster cannot be promoted from quarantine into immutable observation storage, or its filesystem registration record cannot be written safely.

**Outcome**

```text
REJECT
```

**Code**

```text
ASSET_STORAGE_FAILED
```

This is a server-side failure. Partial observation state and remaining quarantine files must be cleaned before a retry.

---

## F-UPLOAD-007 — Visualization derivative failure

**Condition**

The source raster passes inspection, but its bounded display-only derivative cannot be generated or exceeds the configured derivative-size limit.

**Outcome**

```text
REJECT
```

**Codes**

```text
VISUALIZATION_GENERATION_FAILED
VISUALIZATION_RESOURCE_LIMIT_EXCEEDED
```

The original and derivative quarantine artifacts must be cleaned; no partial observation is registered.

---

# 7. Metadata failures

## F-META-001 — Missing CRS

### Semantic/pixel-space task

Example:

> "Is an airplane visible?"

Outcome:

```text
ALLOW_WITH_WARNING
```

Warning:

> Georeferencing is unavailable. The result is limited to pixel-space interpretation.

### Geographic measurement

Example:

> "How many hectares does this region cover?"

Outcome:

```text
REJECT
```

Code:

```text
CRS_REQUIRED_FOR_MEASUREMENT
```

---

## F-META-002 — Missing acquisition time

### Non-temporal task

Outcome:

```text
ALLOW
```

or warning if time matters to interpretation.

### Temporal task where ordering cannot be otherwise established

Outcome:

```text
REQUEST_INPUT
```

Code:

```text
TEMPORAL_ORDER_UNKNOWN
```

---

## F-META-003 — Unknown sensor

If modality is known and model supports generic modality:

```text
ALLOW_WITH_WARNING
```

Set:

```text
domain_status = unknown
```

If model requires a specific sensor profile:

```text
REJECT
```

Code:

```text
MODEL_INPUT_UNSUPPORTED
```

---

## F-META-004 — Unknown SAR polarization

Generic SAR analysis MAY proceed if the model supports it.

Polarization-specific interpretation MUST NOT proceed.

Outcome:

```text
ALLOW_WITH_WARNING
```

or:

```text
REJECT
```

depending on requested task.

Code:

```text
UNKNOWN_SAR_POLARIZATION
```

---

# 8. Spectral-band failures

## F-BAND-001 — Required band missing

Example:

```text
NDVI requires RED + NIR
input contains RGB only
```

Outcome:

```text
REJECT
```

Code:

```text
MISSING_REQUIRED_BAND
```

User message:

> This operation requires a band that is not available in the uploaded observation.

---

## F-BAND-002 — Band identity unknown

If the file contains multiple bands but their meaning cannot be established reliably:

- band-specific calculation → REJECT,
- generic model input → only if model supports unidentified channel configuration.

SatQuery MUST NOT guess band semantics from band position alone unless that mapping is guaranteed by the dataset/product contract.

---

## F-BAND-003 — Estimated synthetic band

SatQuery does not treat a predicted/synthetic band as a measured band.

If future research enables spectral reconstruction:

```text
estimated_NIR
```

must remain explicitly marked as estimated.

It MUST NOT silently satisfy a "measured NIR" precondition.

---

# 9. Single-image task failures

## F-SINGLE-001 — Unsupported modality

If selected model does not support the observed modality:

Outcome:

```text
REJECT
```

Code:

```text
UNSUPPORTED_MODALITY
```

Planner MAY route to another registered model first.

---

## F-SINGLE-002 — Model out of validated scale range

If the model can technically run but the input GSD/scale is far outside validated range:

Outcome:

```text
ALLOW_WITH_WARNING
```

or:

```text
ABSTAIN
```

depending on evidence quality.

Set:

```text
domain_status = shifted
```

---

# 10. Temporal failures

## F-TEMP-001 — Missing second observation

Condition:

```text
one image
+
change query
```

Outcome:

```text
REQUEST_INPUT
```

Code:

```text
MISSING_TEMPORAL_PAIR
```

User message:

> Change analysis requires a second observation of the same area from another time.

No change model should run.

---

## F-TEMP-002 — Unknown temporal order

If the query depends on direction:

> "Did built-up area increase?"

and order cannot be determined:

```text
REQUEST_INPUT
```

Code:

```text
TEMPORAL_ORDER_UNKNOWN
```

---

## F-TEMP-003 — Duplicate observation

If T1 and T2 resolve to the same exact source file / hash:

Outcome:

```text
ALLOW_WITH_WARNING
```

for a sanity/no-change test only.

For ordinary change requests:

```text
REQUEST_INPUT
```

or `REJECT`, because a second distinct observation is required.

---

## F-TEMP-004 — Non-overlapping temporal observations

Outcome:

```text
REJECT
```

Code:

```text
NO_SPATIAL_OVERLAP
```

---

## F-TEMP-005 — Severe misregistration

If pair alignment is insufficient for spatial change analysis:

Outcome:

```text
REJECT
```

unless an approved registration/alignment workflow can repair it.

Code:

```text
PAIR_ALIGNMENT_INVALID
```

The system must not interpret misregistration edges as change.

---

## F-TEMP-006 — Temporal comparability warning

Examples:

- very different season,
- very different atmospheric condition,
- acquisition geometry likely to affect comparison.

Outcome:

```text
ALLOW_WITH_WARNING
```

where analysis remains scientifically defensible.

Warning must appear in final response.

---

# 11. Optical-SAR pair failures

## F-FUSION-001 — Missing modality

If query explicitly requests both modalities but only one exists:

Outcome:

```text
REQUEST_INPUT
```

If query can be answered unimodally:

```text
ALLOW_WITH_WARNING
```

but result MUST NOT be described as fused.

---

## F-FUSION-002 — No overlap

Outcome:

```text
REJECT
```

Code:

```text
NO_SPATIAL_OVERLAP
```

---

## F-FUSION-003 — Bad registration

Outcome:

```text
REJECT
```

or execute approved alignment workflow.

No fused spatial claim before alignment passes.

---

## F-FUSION-004 — Modality disagreement

Example:

```text
Optical → flood
SAR     → no flood
```

Outcome depends on evidence quality.

### Strong evidence conflict

```text
ABSTAIN
```

Response:

> The optical and SAR analyses disagree, so SatQuery cannot make a high-confidence fused conclusion.

### One modality degraded

Example:

```text
optical heavy clouds
SAR valid
```

May:

```text
ALLOW_WITH_WARNING
```

and identify SAR as the stronger supported modality.

---

## F-FUSION-005 — Fusion model ignores modality

This is primarily an evaluation failure rather than runtime failure.

If O+S behavior consistently matches O-only despite SAR variation:

- fusion model is not promoted,
- runtime system may continue using unimodal specialists,
- "multimodal fusion" claim must not be made until fixed.

---

# 12. Geometric failures

## F-GEO-001 — Invalid crop/source mapping

If a grounding mask/box cannot be reliably mapped from model coordinates back to the source raster:

Outcome:

```text
REJECT SPATIAL EVIDENCE
```

Text-only semantic answer MAY survive if independently valid.

Code:

```text
INVALID_EVIDENCE_GEOMETRY
```

---

## F-GEO-002 — Invalid world-coordinate mapping

If source pixels are valid but CRS/transform is unavailable:

- pixel-space evidence may be shown,
- geographic coordinate claims must be omitted.

Outcome:

```text
ALLOW_WITH_WARNING
```

---

## F-GEO-003 — Area requested in angular CRS

SatQuery MUST NOT square degree units.

Planner must:

1. choose approved projected/equal-area/geodesic path,
2. or reject if none is available.

If no valid path:

```text
REJECT
```

Code:

```text
CRS_REQUIRED_FOR_MEASUREMENT
```

---

## F-GEO-004 — Invalid evidence mask

Examples:

- unexpected dimensions,
- corrupted raster,
- missing transform,
- impossible values.

Outcome:

```text
REJECT MEASUREMENT
```

The answer composer must not estimate around it.

---

# 13. Model failures

## F-MODEL-001 — Model execution error

Outcome:

```text
REJECT current workflow
```

Code:

```text
MODEL_EXECUTION_FAILED
```

A bounded retry MAY occur for transient failures.

No fabricated fallback answer.

---

## F-MODEL-002 — Model unavailable

Planner MAY select another compatible registered model.

If no valid substitute exists:

```text
REJECT
```

Code:

```text
MODEL_UNAVAILABLE
```

---

## F-MODEL-003 — Input unsupported by model

Examples:

- unexpected band configuration,
- unsupported modality,
- unsupported dimensions without tiling path.

Planner MAY route to another model.

Otherwise:

```text
REJECT
```

Code:

```text
MODEL_INPUT_UNSUPPORTED
```

---

## F-MODEL-004 — Low confidence

If confidence is calibrated and below an application threshold:

Outcome may be:

```text
ALLOW_WITH_WARNING
```

or:

```text
ABSTAIN
```

depending on task severity.

SatQuery should not turn low-confidence evidence into high-certainty language.

---

## F-MODEL-005 — Out-of-domain input

Examples:

- unseen sensor,
- extreme GSD shift,
- new geographic distribution.

Outcome:

```text
ALLOW_WITH_WARNING
```

if analysis remains useful.

Potentially:

```text
ABSTAIN
```

if evidence is unreliable.

Final answer must expose the domain warning.

---

# 14. GIS tool failures

## F-GIS-001 — Tool precondition not satisfied

Example:

```text
compute_ndvi
requires RED + NIR
```

Outcome:

```text
REJECT
```

No tool execution.

---

## F-GIS-002 — GIS operation execution error

Outcome:

```text
REJECT measurement-dependent claim
```

Code:

```text
GIS_OPERATION_FAILED
```

Other independent evidence MAY still be returned.

---

## F-GIS-003 — Unit conversion unsupported

Outcome:

```text
REJECT requested unit
```

May return a supported unit if user intent permits and this is clearly stated.

---

# 15. Verification failures

## F-VERIFY-001 — Geometry FAIL

Any measurement or spatial claim depending on geometry is blocked.

Semantic answer may remain if independent.

---

## F-VERIFY-002 — Temporal FAIL

Change claim is blocked.

---

## F-VERIFY-003 — Physical FAIL

Band/index/measurement claim is blocked.

---

## F-VERIFY-004 — Provenance FAIL

Result should not be presented as reproducible.

For competition/production paths:

```text
ABSTAIN
```

or fail analysis depending on severity.

---

## F-VERIFY-005 — Statistical WARN

Result may be returned with explicit uncertainty.

Examples:

- OOD sensor,
- low calibrated probability,
- modality disagreement.

---

# 16. Answer-composition failures

## F-ANSWER-001 — Evidence absent

If no evidence supports requested claim:

```text
ABSTAIN
```

The answer should explain why evidence is insufficient.

---

## F-ANSWER-002 — Evidence conflict

Language must preserve the conflict.

Forbidden:

> Flooding definitely occurred.

Allowed:

> The available evidence is conflicting: optical analysis suggests flooding, while SAR analysis does not corroborate it.

---

## F-ANSWER-003 — Numeric mismatch

If generated prose alters a structured numeric value:

```text
structured:
3.14 ha

generated:
3.41 ha
```

The structured value wins.

Production implementation should insert measurements programmatically rather than relying on model memory.

---

## F-ANSWER-004 — Unsupported certainty

If verification contains warnings, the answer must not use stronger certainty than evidence supports.

---

# 17. Confidence failures

## F-CONF-001 — Raw probability not calibrated

If calibration is unavailable:

Label it:

```text
raw model score
```

not:

```text
scientific confidence
```

---

## F-CONF-002 — Calibration domain mismatch

A calibrator trained on one sensor/domain may not remain valid under major domain shift.

When detected:

```text
confidence_status = uncertain
```

and domain warning is shown.

---

## F-CONF-003 — Fake aggregate confidence

Until validated, SatQuery must not combine:

```text
geometry
temporal
model confidence
modality agreement
domain status
```

using arbitrary fixed weights.

Expose them separately.

---

# 18. Agent/orchestrator failures

## F-AGENT-001 — Unknown intent

If query cannot be mapped reliably to a supported intent:

Outcome:

```text
REQUEST_INPUT
```

Ask for clarification or explain supported capabilities.

---

## F-AGENT-002 — No feasible workflow

Outcome:

```text
REJECT
```

Code:

```text
NO_FEASIBLE_WORKFLOW
```

---

## F-AGENT-003 — Attempted unregistered tool/model

Outcome:

```text
REJECT
```

Severity:

```text
ERROR
```

The orchestrator is not allowed to execute it.

---

## F-AGENT-004 — Invalid parameters

Tool/model call is rejected before execution.

Code:

```text
INVALID_TOOL_PARAMETERS
```

---

## F-AGENT-005 — Arbitrary code request

The orchestration layer does not execute:

- arbitrary shell,
- arbitrary Python,
- arbitrary GDAL CLI flags,
- arbitrary remote commands.

Outcome:

```text
REJECT
```

---

# 19. Security failures

## F-SEC-001 — Processing timeout

Terminate sandboxed raster/model operation.

Code:

```text
PROCESSING_TIMEOUT
```

---

## F-SEC-002 — Memory/resource violation

Terminate operation.

Code:

```text
RESOURCE_LIMIT_EXCEEDED
```

---

## F-SEC-003 — Unexpected external access attempt

Terminate processing.

Severity:

```text
CRITICAL
```

Log security event.

---

## F-SEC-004 — Unsafe path / filesystem access

Reject access and terminate relevant operation.

---

# 20. Retry policy

Retries are allowed only for failures likely to be transient.

Potentially retryable:

```text
GPU temporary OOM after cache eviction
worker restart
temporary file lock
transient service error
```

Not retryable without changed inputs:

```text
missing NIR
missing second image
non-overlap
bad CRS for requested measurement
unsupported format
invalid model modality
```

Default automated retry count should remain small.

Repeated retries must not hide deterministic failure.

---

# 21. Degradation policy

SatQuery may degrade from a richer workflow to a simpler one only if:

1. the simpler workflow remains scientifically valid,
2. the UI clearly states the degraded mode,
3. the result is not mislabeled as the richer capability.

Example:

```text
optical + SAR requested
SAR model unavailable
```

Possible response:

> SAR analysis is unavailable. SatQuery is returning an optical-only result.

Forbidden:

> Fused optical-SAR analysis confirms...

---

# 22. Abstention policy

SatQuery SHOULD abstain when:

- evidence conflicts strongly,
- calibrated confidence is very low,
- input is severely OOD,
- alignment quality is insufficient,
- model result is unstable,
- validation cannot establish required scientific assumptions.

Abstention text should state:

1. what could not be established,
2. which evidence is available,
3. what additional data/action could resolve it.

---

# 23. Warning propagation

Warnings generated early in the workflow must propagate to the final response.

Example:

```text
Input Inspector:
sensor unknown

Model:
generic optical model

Verifier:
domain unknown
```

Final response must include the domain/sensor uncertainty.

Warnings must not disappear because later stages succeeded.

---

# 24. Failure event logging

Each failure event records:

```yaml
failure_id: ...
analysis_id: ...
step_id: ...

code: ...
severity: ...
outcome: ...

component: ...
model_or_tool: ...

input_ids: [...]
evidence_ids: [...]

technical_message: ...
user_message: ...

timestamp: ...
```

---

# 25. User-facing language style

Failure messages should be:

- clear,
- specific,
- non-accusatory,
- actionable where possible.

Bad:

> Invalid data.

Better:

> The two images do not overlap geographically, so SatQuery cannot perform a valid change comparison. Upload observations covering the same area.

---

# 26. Failure matrix

| Scenario | Outcome | Code |
|---|---|---|
| 1 image + change query | REQUEST_INPUT | MISSING_TEMPORAL_PAIR |
| RGB + NDVI | REJECT | MISSING_REQUIRED_BAND |
| missing CRS + VQA | ALLOW_WITH_WARNING | GEOREFERENCE_UNAVAILABLE |
| missing CRS + hectares | REJECT | CRS_REQUIRED_FOR_MEASUREMENT |
| no pair overlap | REJECT | NO_SPATIAL_OVERLAP |
| severe misalignment | REJECT / repair | PAIR_ALIGNMENT_INVALID |
| unknown temporal order | REQUEST_INPUT | TEMPORAL_ORDER_UNKNOWN |
| unknown polarization | WARN / REJECT task | UNKNOWN_SAR_POLARIZATION |
| model OOD | WARN / ABSTAIN | OUT_OF_DOMAIN_WARNING |
| optical/SAR conflict | ABSTAIN / WARN | MODALITY_CONFLICT |
| low confidence | WARN / ABSTAIN | LOW_EVIDENCE_CONFIDENCE |
| invalid mask | REJECT measurement | INVALID_EVIDENCE_GEOMETRY |
| model crash | REJECT workflow | MODEL_EXECUTION_FAILED |
| GIS error | REJECT measurement | GIS_OPERATION_FAILED |
| corrupt TIFF | REJECT | INVALID_RASTER |
| oversized TIFF | REJECT | RASTER_RESOURCE_LIMIT_EXCEEDED |
| unsupported driver | REJECT | UNSUPPORTED_RASTER_DRIVER |
| asset storage failure | REJECT | ASSET_STORAGE_FAILED |

---

# 27. Required failure tests

Before MVP completion, automated/integration tests must cover:

```text
corrupt raster
fake extension
oversized raster
missing CRS
missing transform
missing band
unknown sensor
unknown SAR polarization
one-image change query
non-overlapping pair
misaligned pair
unknown temporal order
invalid mask
model crash
GIS failure
low-confidence evidence
modality disagreement
OOD input
```

---

# 28. Competition demo failure case

At least one deliberate refusal should appear in the internal/demo test suite.

Recommended example:

```text
Upload one image.

Ask:
"What changed?"
```

Expected:

> Change analysis requires a second observation of the same area from another time.

This demonstrates that SatQuery does not hallucinate missing temporal evidence.

---

# 29. Incident priority

## P0 — Scientific integrity

Examples:

- fabricated measurement,
- wrong CRS area calculation,
- silent non-overlap comparison,
- invented missing band,
- wrong temporal order without warning.

Block release.

---

## P1 — Evidence integrity

Examples:

- grounding overlay offset,
- wrong evidence ID,
- fusion result not linked to unimodal evidence.

Block relevant capability.

---

## P2 — Reliability / UX

Examples:

- warning not displayed,
- ambiguous error message,
- missing execution metadata.

Fix before competition hardening.

---

## P3 — Cosmetic

Examples:

- minor layout issue,
- non-critical animation bug.

Does not override scientific work.

---

# 30. Release gate

A release is blocked if any of the following occur:

1. invalid scientific workflow is accepted,
2. measurement is generated without valid evidence,
3. original source data are overwritten,
4. provenance is missing for production analysis,
5. spatial evidence cannot be mapped correctly,
6. known critical failure test regresses,
7. model/tool errors lead to fabricated output,
8. agent can invoke unregistered arbitrary actions.

---

# 31. Final failure principle

SatQuery is allowed to say:

```text
"I cannot determine this reliably from the supplied data."
```

That is a successful system behavior when the evidence is insufficient.

The project should never optimize for:

```text
always answer
```

at the expense of:

```text
answer only what the evidence supports
```

> **If the workflow cannot support the claim, SatQuery must not make the claim.**
