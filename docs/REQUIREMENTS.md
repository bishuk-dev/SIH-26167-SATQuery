# SatQuery AI — Requirements Specification

> **Source-of-truth requirements for the SatQuery AI project.**

**Project:** SatQuery AI  
**Problem Statement ID:** 26167  
**Document:** `docs/REQUIREMENTS.md`  
**Status:** Requirements freeze — v1  
**Last updated:** 2026-09-03

---

# 1. Purpose

This document defines what SatQuery **must**, **should**, and **may** implement.

Its purpose is to prevent feature creep and to clearly distinguish:

- competition-critical requirements,
- engineering requirements needed to satisfy those capabilities reliably,
- research extensions that are useful but not mandatory,
- explicitly unsupported behavior.

This document should be updated only when:

1. the official problem statement clarifies or changes a requirement,
2. implementation evidence shows that a requirement needs a more precise engineering interpretation,
3. a planned feature is promoted or removed after team review.

---

# 2. Requirement levels

## MUST

A capability required to satisfy the problem statement or required to make another mandatory capability scientifically valid.

Failure to implement a MUST requirement means SatQuery is incomplete.

## SHOULD

A capability that materially improves reliability, usability, evaluation, or competitiveness, but is not strictly necessary for the minimum solution.

## MAY

A research or product enhancement that should only be attempted after mandatory functionality is stable.

## MUST NOT

Behavior that would make SatQuery scientifically unreliable, misleading, insecure, or inconsistent with the project architecture.

---

# 3. Core project objective

SatQuery MUST provide a natural-language interface for multimodal remote-sensing imagery that can automatically determine the requested analysis, select an appropriate workflow/model/tool, and return an answer supported by spatial or analytical evidence.

The target interaction is:

```text
REMOTE-SENSING OBSERVATIONS
        +
NATURAL-LANGUAGE QUERY
        ↓
INPUT / SENSOR VALIDATION
        ↓
TASK CLASSIFICATION
        ↓
SPECIALIST MODEL / TOOL SELECTION
        ↓
STRUCTURED EVIDENCE
        ↓
VERIFICATION
        ↓
ANSWER + VISUAL EVIDENCE + EXECUTION SUMMARY
```

---

# 4. Mandatory input modes

## R-INPUT-001 — Single observation

**Level:** MUST

SatQuery MUST support analysis of one remote-sensing observation.

Supported modality categories:

- optical,
- multispectral,
- SAR.

Example tasks:

- VQA,
- scene description,
- text-guided grounding.

---

## R-INPUT-002 — Optical/SAR pair

**Level:** MUST

SatQuery MUST support a spatially corresponding optical/multispectral + SAR pair for complementary cross-modal analysis.

The system MUST NOT assume that optical and SAR use identical preprocessing or image physics.

---

## R-INPUT-003 — Bi-temporal pair

**Level:** MUST

SatQuery MUST support two spatially corresponding observations acquired at different times.

The system MUST be able to use such pairs for:

- change description,
- change VQA,
- change localization where supported.

---

## R-INPUT-004 — Scientific raster formats

**Level:** MUST

SatQuery MUST support:

- GeoTIFF,
- TIFF.

PNG/JPEG MAY be accepted for approved benchmark datasets, previews, or explicitly non-georeferenced analysis.

---

# 5. Mandatory AI capabilities

## R-AI-001 — Remote-sensing adaptation

**Level:** MUST

The project MUST demonstrate remote-sensing-specific model adaptation using BigEarthNet.txt or another justified open-source remote-sensing training source.

A generic natural-image VLM used without remote-sensing adaptation is insufficient as the complete solution.

---

## R-AI-002 — Single-image VQA

**Level:** MUST

SatQuery MUST support natural-language visual question answering over a single remote-sensing image.

Example:

> "Is water present in this scene?"

---

## R-AI-003 — Captioning or grounding

**Level:** MUST

In addition to single-image VQA, SatQuery MUST implement at least one of:

- image captioning,
- text-guided grounding.

### Project decision

SatQuery will prioritize:

```text
VQA + TEXT-GUIDED GROUNDING
```

because grounding produces reusable spatial evidence.

Captioning MAY be supported as an additional capability.

---

## R-AI-004 — Bi-temporal analysis

**Level:** MUST

SatQuery MUST support at least one of:

- change description,
- change VQA.

### Project decision

SatQuery will target:

```text
change representation / mask
        +
change VQA
```

where feasible.

This provides stronger evidence than direct free-form temporal description alone.

---

## R-AI-005 — Optical-SAR complementary reasoning

**Level:** MUST

SatQuery MUST extract and combine complementary information from optical/multispectral and SAR observations.

The system MUST NOT satisfy this requirement merely by accepting two files.

The fusion pathway MUST be evaluated against:

```text
Optical only
SAR only
Optical + SAR
```

to verify that multimodal input provides actual value.

---

# 6. Agentic orchestration requirements

## R-AGENT-001 — Query classification

**Level:** MUST

The system MUST classify a natural-language query into a supported analysis intent.

Initial intent set:

```text
SINGLE_VQA
GROUND_OBJECT
DESCRIBE_SCENE
CROSS_MODAL_VQA
CHANGE_VQA
CHANGE_LOCALIZE
MEASURE
CHANGE_MEASURE
METADATA_QUERY
```

The exact set MAY evolve, but the controller MUST remain bounded.

---

## R-AGENT-002 — Input validation

**Level:** MUST

Before model execution, the controller MUST inspect whether the supplied inputs satisfy workflow requirements.

Validation includes, where relevant:

- number of images,
- modality,
- format,
- required bands,
- acquisition time,
- CRS,
- extent,
- spatial overlap,
- alignment,
- resolution,
- sensor metadata,
- SAR polarization.

---

## R-AGENT-003 — Model/tool selection

**Level:** MUST

The controller MUST automatically select an appropriate model/tool from a predefined registry.

It MUST NOT invent model names or arbitrary tools.

---

## R-AGENT-004 — Parameter configuration

**Level:** MUST

The controller MUST be able to configure permitted parameters for registered workflows/tools.

Parameters MUST be schema-constrained.

The system MUST NOT allow arbitrary shell, Python, or unrestricted code execution.

---

## R-AGENT-005 — Multi-step workflow

**Level:** MUST

The system MUST be capable of combining multiple analytical steps when necessary.

Example:

```text
change query
   ↓
change specialist
   ↓
change mask
   ↓
GIS area tool
   ↓
verifier
   ↓
answer
```

---

## R-AGENT-006 — Invalid request handling

**Level:** MUST

The controller MUST reject, qualify, or request additional information when a requested operation is unsupported by the available evidence.

Examples:

```text
1 image + "What changed?"
→ require second image

RGB only + "Compute NDVI"
→ required NIR unavailable

no CRS/GSD + "How many hectares?"
→ physical measurement unsupported
```

---

# 7. Evidence requirements

## R-EVIDENCE-001 — Structured evidence

**Level:** MUST

Analytical outputs MUST be represented internally as structured evidence before final language generation.

Possible evidence types:

- class label,
- VQA answer,
- bounding box,
- mask,
- polygon,
- point,
- measurement.

---

## R-EVIDENCE-002 — Spatial evidence

**Level:** MUST for grounding and spatial change outputs

Spatial outputs MUST retain a valid mapping back to the original source raster.

The system MUST preserve:

```text
model/crop coordinates
→ source pixel coordinates
→ geographic coordinates (when georeferenced)
```

---

## R-EVIDENCE-003 — Evidence-language separation

**Level:** MUST

The final LLM/VLM explanation MUST NOT invent:

- masks,
- boxes,
- physical measurements,
- spectral bands,
- sensor metadata,
- temporal observations.

The language layer MAY explain only evidence available from approved analytical components.

---

# 8. Geospatial requirements

## R-GEO-001 — Metadata preservation

**Level:** MUST

For scientific raster inputs, SatQuery MUST preserve available:

- CRS,
- transform,
- bounds,
- resolution/GSD,
- band metadata,
- NoData,
- acquisition time,
- modality,
- sensor,
- polarization,
- provenance.

---

## R-GEO-002 — Immutable original

**Level:** MUST

Uploaded scientific source data MUST remain immutable.

All preprocessing outputs MUST be stored or tracked as derived assets.

---

## R-GEO-003 — Pair compatibility

**Level:** MUST

Before pair analysis, SatQuery MUST determine whether inputs are sufficiently compatible.

Checks may include:

- overlapping geography,
- transformable CRS,
- registration/alignment,
- temporal order,
- compatible analysis resolution.

---

## R-GEO-004 — Deterministic measurements

**Level:** MUST

Physical measurements such as:

- area,
- distance,
- pixel count,
- geometry intersection,
- unit conversion,

MUST be computed using deterministic GIS functions.

The LLM MUST NOT estimate these quantities directly.

---

## R-GEO-005 — CRS-safe area

**Level:** MUST

SatQuery MUST NOT treat angular degrees as metric area.

Area calculations MUST use an appropriate:

- projected CRS,
- equal-area method,
- or valid geodesic method.

---

# 9. Verification requirements

## R-VERIFY-001 — Independent verification

**Level:** MUST

SatQuery MUST verify analytical outputs independently from final prose generation.

---

## R-VERIFY-002 — Geometric verification

**Level:** MUST where spatial output exists

Checks SHOULD include:

- CRS validity,
- overlap,
- grid compatibility,
- output dimensions,
- source-coordinate mapping,
- measurement geometry.

---

## R-VERIFY-003 — Temporal verification

**Level:** MUST for temporal workflows

Checks SHOULD include:

- T1/T2 order,
- duplicate observations,
- comparable time windows,
- required temporal relationship.

---

## R-VERIFY-004 — Physical verification

**Level:** MUST where applicable

Checks SHOULD include:

- required band existence,
- valid units,
- valid index ranges,
- sensor-specific preconditions.

---

## R-VERIFY-005 — Provenance verification

**Level:** MUST

Every analysis MUST retain enough information to identify:

- source observations,
- model ID,
- model version,
- checkpoint/version reference,
- preprocessing profile,
- tool names,
- important tool parameters.

---

## R-VERIFY-006 — Statistical reliability

**Level:** SHOULD

The system SHOULD expose:

- calibrated model confidence where available,
- domain-shift status,
- modality disagreement,
- low-confidence warnings.

---

# 10. Confidence requirements

## R-CONF-001 — No fake precision

**Level:** MUST

SatQuery MUST NOT present an arbitrary scalar such as:

```text
Confidence = 93.47%
```

unless the value has a defined and validated interpretation.

---

## R-CONF-002 — Separate reliability dimensions

**Level:** SHOULD

Initial UI/API should expose separate fields such as:

```text
model confidence
data validity
domain status
geometric validity
temporal validity
modality agreement
```

---

## R-CONF-003 — Calibration

**Level:** SHOULD

Where probabilistic model outputs are exposed as confidence, they SHOULD be calibrated on held-out data.

Potential evaluation:

- ECE,
- NLL,
- reliability diagrams.

---

# 11. Execution-summary requirements

## R-TRACE-001 — Auditable summary

**Level:** MUST

Each analysis MUST produce an execution summary including:

- task selected,
- source observations,
- validation outcome,
- model/tool names,
- model/tool versions,
- key parameters,
- evidence generated,
- verification status,
- warnings.

---

## R-TRACE-002 — No private reasoning requirement

**Level:** MUST

The execution summary MUST describe operational steps and provenance.

It MUST NOT depend on exposing internal chain-of-thought.

---

# 12. UI requirements

## R-UI-001 — Upload and validation

**Level:** MUST

The UI MUST support image upload and display:

- input status,
- detected metadata,
- warnings,
- compatibility results for pairs.

---

## R-UI-002 — Imagery-first visualization

**Level:** MUST

The primary analysis interface MUST be centered on the imagery/map rather than a chat-only view.

---

## R-UI-003 — Evidence overlays

**Level:** MUST

The UI MUST display supported spatial evidence such as:

- bounding boxes,
- masks,
- change regions,
- polygons.

---

## R-UI-004 — Pair comparison

**Level:** SHOULD

For temporal or multimodal data, the UI SHOULD support one or more:

- side-by-side,
- swipe,
- opacity,
- flicker,
- layer toggles.

---

## R-UI-005 — Result panel

**Level:** MUST

The result UI MUST show:

- answer,
- evidence,
- confidence/reliability status,
- warnings,
- execution summary.

---

## R-UI-006 — Downloadable report

**Level:** MUST

Users MUST be able to export/download a structured analysis report.

---

# 13. Evaluation requirements

## R-EVAL-001 — Task-specific metrics

**Level:** MUST

Different tasks MUST be evaluated using appropriate metrics.

| Capability | Required/Preferred Metrics |
|---|---|
| VQA | accuracy |
| Grounding | IoU / mIoU / Acc@IoU |
| Change detection | precision / recall / F1 / IoU |
| Detection | AP / mAP where applicable |
| Numeric measurement | MAE / relative error |
| Calibration | ECE / NLL where applicable |
| Routing | routing accuracy |
| Input validation | acceptance/rejection accuracy |

---

## R-EVAL-002 — Multimodal ablation

**Level:** MUST

Optical-SAR evaluation MUST compare:

```text
O
S
O+S
```

---

## R-EVAL-003 — VQA shortcut tests

**Level:** SHOULD

VQA evaluation SHOULD include:

- normal image,
- blank image,
- shuffled image,
- question-only baseline.

---

## R-EVAL-004 — Temporal sanity tests

**Level:** MUST

Change evaluation MUST include:

```text
T1 + T2
T1 + T1
T2 + T1
```

where the task semantics make them appropriate.

---

## R-EVAL-005 — Cross-domain evaluation

**Level:** MUST

The project MUST explicitly measure generalization across at least relevant subsets of:

- region,
- sensor,
- resolution/scale.

Sentinel benchmark performance MUST NOT be presented as proof of Cartosat/RISAT performance.

---

## R-EVAL-006 — No invented official weights

**Level:** MUST

The project MUST NOT invent SIH judging/evaluation weights not present in official material.

Internal composite scores, if ever used, MUST be labeled as internal.

---

# 14. Security requirements

## R-SEC-001 — Untrusted uploads

**Level:** MUST

Raster uploads MUST be treated as untrusted input.

---

## R-SEC-002 — Driver validation

**Level:** MUST

The backend MUST validate actual raster driver/format rather than trusting file extension alone.

---

## R-SEC-003 — Resource limits

**Level:** MUST

The ingestion path MUST enforce limits for:

- file bytes,
- width,
- height,
- bands,
- total pixel count,
- processing time,
- memory where practical.

---

## R-SEC-004 — Restricted raster processing

**Level:** SHOULD

Untrusted GDAL/raster processing SHOULD run in a restricted environment with limited:

- filesystem access,
- network access,
- CPU,
- memory.

---

## R-SEC-005 — No arbitrary execution

**Level:** MUST

The agent/controller MUST NOT expose arbitrary:

- shell execution,
- Python execution,
- user-defined GDAL arguments,
- unvalidated tool invocation.

---

# 15. Deployment requirements

## R-DEPLOY-001 — Offline-capable core

**Level:** SHOULD

Core image analysis SHOULD be capable of operating without mandatory external web APIs.

---

## R-DEPLOY-002 — CPU/GPU separation

**Level:** SHOULD

The architecture SHOULD permit:

```text
CPU:
GIS / metadata / verification

GPU:
vision / VLM / change / fusion inference
```

---

## R-DEPLOY-003 — Reproducible deployment

**Level:** SHOULD

The project SHOULD provide Docker-based deployment.

Initial target:

```text
Docker Compose
```

Kubernetes is not a requirement.

---

# 16. Data / model requirements

## R-DATA-001 — Dataset provenance

**Level:** MUST

Training/evaluation records MUST track dataset source and split.

---

## R-DATA-002 — Scene-level leakage prevention

**Level:** MUST

Annotations derived from the same underlying scene/pair MUST NOT be split across train/test in a way that leaks imagery.

---

## R-DATA-003 — Sensor provenance

**Level:** MUST

Where available, training samples MUST retain:

- sensor,
- modality,
- bands/polarization,
- GSD,
- time,
- source dataset.

---

## R-MODEL-001 — Model registry

**Level:** MUST

Every production model MUST have a registry entry defining:

- ID,
- version,
- supported tasks,
- supported modalities,
- preprocessing profile,
- known training domain,
- output schema.

---

## R-MODEL-002 — Versioned preprocessing

**Level:** MUST

Preprocessing is part of model provenance and MUST be versioned.

---

# 17. Initial project decisions

The following choices are frozen for the first implementation unless experiments justify changes.

## D-001 — VQA + grounding over VQA + captioning

Rationale:

Grounding creates reusable visual evidence and better supports the project's evidence-first architecture.

---

## D-002 — One constrained orchestrator

Rationale:

The problem requires agentic orchestration, but a multi-agent swarm is unnecessary and harder to validate.

---

## D-003 — Deterministic GIS tools

Rationale:

Physical/geometric calculations should not be generative.

---

## D-004 — Evidence contract between models and language

Rationale:

Allows specialists to evolve independently while preventing unsupported language claims.

---

## D-005 — Optical-only / SAR-only / fusion inference

Rationale:

Necessary to audit whether both modalities are actually being used.

---

## D-006 — Change perception before change language

Rationale:

A spatial change representation/mask enables verification and measurement.

---

## D-007 — PASS/WARN/FAIL verification before weighted reliability

Rationale:

No scientifically validated weights currently exist for combining all verifier dimensions.

---

# 18. Explicit non-requirements

The first SatQuery release does **not** require:

```text
Kubernetes
Kafka
vector database
multi-agent swarm
RL-trained planner
custom LLM pretraining
custom billion-scale EO pretraining
arbitrary Python execution
arbitrary shell execution
unrestricted web retrieval
```

These MAY be reconsidered only if a concrete future requirement justifies them.

---

# 19. Must-not rules

SatQuery MUST NOT:

1. fabricate missing bands,
2. fabricate sensor metadata,
3. infer physical area from pixels without valid spatial scale,
4. claim temporal change from a single observation,
5. silently compare non-overlapping scenes,
6. silently ignore major pair misalignment,
7. present raw LLM probability as calibrated scientific certainty,
8. claim optical-SAR fusion success without unimodal ablation,
9. present Sentinel-domain performance as proof of RISAT/Cartosat generalization,
10. let prose overwrite structured measurements,
11. expose arbitrary code execution through the agent,
12. discard source-to-world coordinate mappings for spatial evidence,
13. overwrite original scientific inputs,
14. invent official competition weights or scores.

---

# 20. Definition of minimum viable SatQuery

The MVP is considered functionally complete only when all of the following work end-to-end:

```text
1. GeoTIFF ingestion
2. metadata extraction
3. map visualization
4. single-image VQA
5. text-guided grounding
6. optical-SAR pair analysis
7. bi-temporal analysis
8. automatic query/workflow routing
9. structured evidence
10. input validation
11. confidence/warnings
12. execution summary
13. downloadable report
```

The MVP does not need to be the final best-performing model configuration.

It must prove the complete architecture.

---

# 21. Definition of competition-ready SatQuery

Competition-ready status additionally requires:

- benchmarked model performance,
- optical/SAR/fusion ablation,
- temporal sanity tests,
- cross-sensor tests,
- failure cases,
- stable model versions,
- deterministic geospatial tests,
- secure upload limits,
- performance profiling,
- polished evidence visualization,
- reproducible demonstration cases.

---

# 22. Requirement traceability

Each implementation issue / pull request SHOULD reference one or more requirement IDs.

Examples:

```text
feat: implement pair overlap validation

Requirements:
R-INPUT-003
R-GEO-003
R-VERIFY-002
```

This makes it possible to answer:

> Which code satisfies each problem-statement requirement?

and:

> Which mandatory capability is still missing?

---

# 23. Final requirement principle

When requirements conflict with convenience, use the following order:

```text
scientific validity
        >
evidence integrity
        >
problem-statement compliance
        >
evaluation quality
        >
system simplicity
        >
visual polish
        >
extra features
```

SatQuery should prefer a correct refusal over a fluent unsupported answer.

> **If the evidence cannot support the claim, SatQuery does not make the claim.**
