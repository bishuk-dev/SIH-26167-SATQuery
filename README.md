# SatQuery AI

> **An evidence-grounded, sensor-aware Vision-Language Assistant for multimodal remote-sensing analysis through natural-language queries.**

**Problem Statement ID:** 26167  
**Problem Statement:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries

---

## Table of Contents

- [Overview](#overview)
- [Why SatQuery?](#why-satquery)
- [Problem Statement](#problem-statement)
- [Core Design Philosophy](#core-design-philosophy)
- [What SatQuery Can Do](#what-satquery-can-do)
- [Supported Analysis Modes](#supported-analysis-modes)
- [System Architecture](#system-architecture)
- [How a Query Is Processed](#how-a-query-is-processed)
- [Remote-Sensing Data Model](#remote-sensing-data-model)
- [Input Validation](#input-validation)
- [Vision and Language Stack](#vision-and-language-stack)
- [Optical-SAR Fusion](#optical-sar-fusion)
- [Bi-Temporal Change Analysis](#bi-temporal-change-analysis)
- [Grounding and Spatial Evidence](#grounding-and-spatial-evidence)
- [GIS and Deterministic Computation](#gis-and-deterministic-computation)
- [Agentic Orchestration](#agentic-orchestration)
- [Verification Layer](#verification-layer)
- [Evidence Contract](#evidence-contract)
- [Confidence and Uncertainty](#confidence-and-uncertainty)
- [Execution Trace and Provenance](#execution-trace-and-provenance)
- [Datasets](#datasets)
- [Model Strategy](#model-strategy)
- [Training Strategy](#training-strategy)
- [Evaluation](#evaluation)
- [Cross-Sensor Generalization](#cross-sensor-generalization)
- [Technology Stack](#technology-stack)
- [Backend Architecture](#backend-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Raster Visualization](#raster-visualization)
- [Storage](#storage)
- [API Design](#api-design)
- [Security](#security)
- [Repository Structure](#repository-structure)
- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Development Roadmap](#development-roadmap)
- [Failure Policy](#failure-policy)
- [What SatQuery Will Not Do](#what-satquery-will-not-do)
- [Research Risks](#research-risks)
- [Demo Scenarios](#demo-scenarios)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

# Overview

Remote-sensing imagery is widely used for:

- agriculture monitoring,
- flood and disaster assessment,
- urban planning,
- forest monitoring,
- water-resource analysis,
- infrastructure mapping,
- environmental monitoring,
- land-cover analysis,
- change detection.

However, most existing remote-sensing AI systems are designed for one predefined task:

```text
classification
object detection
segmentation
change detection
captioning
visual question answering
```

A user must often understand:

- satellite sensors,
- spectral bands,
- SAR characteristics,
- coordinate reference systems,
- image registration,
- model selection,
- GIS workflows,
- preprocessing requirements,
- task-specific parameters.

SatQuery attempts to provide a simpler interface:

```text
REMOTE-SENSING DATA
        +
NATURAL-LANGUAGE QUERY
        ↓
SENSOR-AWARE ANALYSIS
        ↓
SPATIAL EVIDENCE
        ↓
VERIFIED RESULT
        ↓
NATURAL-LANGUAGE EXPLANATION
```

The goal is **not** merely to build a chatbot capable of looking at satellite screenshots.

SatQuery is designed as an:

> **Evidence-grounded remote-sensing analysis system with a natural-language interface.**

---

# Why SatQuery?

Consider the query:

> "How much water area increased after the flood?"

A generic multimodal chatbot may try to visually compare two images and produce a plausible number.

That is not scientifically reliable.

A defensible workflow should instead be:

```text
Before image
      +
After image
      ↓
Validate geographic compatibility
      ↓
Align observations if required
      ↓
Detect water/change
      ↓
Produce spatial mask
      ↓
Count changed pixels
      ↓
Convert pixels to physical area using geospatial metadata
      ↓
Verify result
      ↓
Explain result
```

SatQuery therefore separates:

- **perception**
- **reasoning**
- **measurement**
- **verification**
- **language generation**

instead of forcing one model to perform everything implicitly.

---

# Problem Statement

SatQuery is designed for natural-language interaction with remote-sensing imagery across multiple input configurations.

The system must support analysis involving:

### Single observations

One optical, multispectral, or SAR image.

Possible tasks include:

- visual question answering,
- scene understanding,
- text-guided grounding,
- captioning.

### Cross-modal observations

Co-registered imagery of the same region from complementary sensors, such as:

```text
optical / multispectral
+
SAR
```

The system should exploit complementary sensor information rather than treating them as interchangeable images.

### Bi-temporal observations

Two spatially corresponding observations from different times:

```text
T1
+
T2
```

for:

- change detection,
- change localization,
- change description,
- change visual question answering,
- quantitative change analysis.

### Supported input formats

Primary scientific formats:

```text
GeoTIFF
TIFF
```

with PNG/JPEG used where permitted for benchmark datasets or visualization.

---

# Core Design Philosophy

The most important rule in SatQuery is:

> **The language model may explain evidence. It may not invent evidence.**

This rule affects the entire architecture.

If the user asks:

> "How many hectares changed?"

The LLM is not allowed to estimate the area visually.

A valid answer must originate from something like:

```text
verified change mask
      ↓
pixel count
      ↓
raster transform / projected CRS
      ↓
physical area
      ↓
unit conversion
```

Likewise:

### No NIR band

No NDVI calculation.

### One temporal image

No change claim.

### Missing georeferencing

No physical-area measurement.

### Unsupported sensor interpretation

The system reports uncertainty instead of silently guessing.

### Low model confidence

The system may abstain or qualify the result.

---

# What SatQuery Can Do

The target SatQuery system supports the following capability families.

## Natural-language remote-sensing VQA

Examples:

> "Is forest present?"

> "Is this region predominantly urban?"

> "Are buildings visible?"

---

## Text-guided grounding

Examples:

> "Show me the airport."

> "Locate the largest water body."

> "Where are the buildings near the river?"

Returns spatial evidence such as:

```text
bounding box
mask
polygon
```

---

## Optical-SAR analysis

Examples:

> "Is the settlement supported by both optical and SAR observations?"

> "What information does the SAR image add?"

> "Do both sensors indicate flooding?"

---

## Change analysis

Examples:

> "What changed between these observations?"

> "Did built-up area increase?"

> "Where did water expand?"

---

## Quantitative geospatial analysis

Examples:

> "How many hectares of water were added?"

> "What is the area of the detected region?"

> "How far is this detected object from the river?"

These measurements are performed by deterministic GIS tools.

---

# Supported Analysis Modes

| Mode | Inputs | Example Query | Primary Evidence |
|---|---|---|---|
| Single-image VQA | 1 optical/MS/SAR | "Is water present?" | semantic prediction |
| Grounding | image + phrase | "Where are the buildings?" | bbox/mask |
| Cross-modal VQA | optical + SAR | "Do both sensors support flooding?" | modality-specific evidence |
| Change VQA | T1 + T2 | "Did urban area increase?" | temporal representation |
| Change localization | T1 + T2 | "Where did water expand?" | change mask |
| Change measurement | T1 + T2 | "How many hectares changed?" | mask + GIS measurement |
| Metadata query | image | "What is the GSD?" | raster metadata |

---

# System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                           USER                              │
│             Images + Natural-Language Query                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  INGESTION / INSPECTOR                     │
│                                                             │
│ Format │ Bands │ Sensor │ Modality │ CRS │ GSD │ Time      │
│ Extent │ NoData │ Polarization │ Metadata                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                   STRUCTURED EO STATE
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  INPUT / PAIR VALIDATOR                    │
│                                                             │
│ overlap │ alignment │ modalities │ temporal order          │
│ required bands │ task feasibility                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY INTERPRETER                        │
│                                                             │
│ intent │ target │ requested output │ spatial constraint     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                CONSTRAINED ORCHESTRATOR                     │
│                                                             │
│                   MODEL + TOOL REGISTRY                     │
└───────────┬─────────────────┬─────────────────┬─────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
       RS-VLM Core       SAR/Optical         Temporal
                           Specialists       Specialist
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                    STRUCTURED EVIDENCE
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
                Boxes       Masks       Classes
                  │           │           │
                  └───────────┼───────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       GIS OPERATORS                         │
│                                                             │
│ area │ count │ distance │ intersection │ CRS │ band math   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                         VERIFIER                            │
│                                                             │
│ geometric │ temporal │ physical │ provenance │ statistical │
└────────────────────────────┬────────────────────────────────┘
                             │
                        PASS/WARN/FAIL
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANSWER COMPOSER                          │
│                                                             │
│ Query + Verified Evidence + Measurements + Warnings         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                       USER RESULT                           │
│                                                             │
│ Answer │ Evidence │ Confidence │ Warnings │ Execution Trace │
└─────────────────────────────────────────────────────────────┘
```

---

# How a Query Is Processed

Consider:

> "Where did built-up area increase and by how much?"

SatQuery processes the request as follows.

```text
1. Inspect uploaded observations
        ↓
2. Determine sensor, CRS, GSD, bounds and acquisition time
        ↓
3. Confirm two valid temporal observations exist
        ↓
4. Validate spatial overlap and alignment
        ↓
5. Interpret query:
      task = CHANGE_MEASURE
      target = BUILT_UP
        ↓
6. Select temporal specialist
        ↓
7. Produce built-up gain mask
        ↓
8. Verify geometry and temporal validity
        ↓
9. Calculate physical area using GIS
        ↓
10. Generate evidence record
        ↓
11. Compose natural-language explanation
        ↓
12. Return answer + mask + measurement + execution trace
```

---

# Remote-Sensing Data Model

Each uploaded observation is represented internally using structured geospatial state.

Conceptually:

```text
ObservationState
{
    id

    source_asset

    modality
    sensor

    bands
    polarizations

    width
    height

    crs
    transform
    bounds
    native_gsd

    acquisition_time

    nodata
    validity_mask

    uncertainty

    provenance
    transform_history
}
```

This prevents critical metadata from disappearing when imagery enters AI preprocessing.

---

# Input Validation

SatQuery validates whether a requested operation is scientifically possible.

Examples:

| Request | Inputs | Behavior |
|---|---|---|
| "What changed?" | 1 image | request second image |
| NDVI | RGB only | reject: NIR unavailable |
| area in hectares | no CRS/GSD | reject physical measurement |
| visual VQA | no CRS | allow pixel-space analysis with warning |
| optical-SAR fusion | no overlap | reject comparison |
| temporal change | unknown ordering | request/resolve order |
| SAR interpretation | unknown polarization | avoid polarization-specific claims |

Validation statuses:

```text
PASS
WARN
FAIL
```

---

# Vision and Language Stack

The vision-language component is responsible for:

```text
semantic interpretation
visual question answering
language-conditioned perception
cross-modal semantic reasoning
answer generation
```

It is **not** responsible for deterministic GIS calculations.

A candidate multisensor architecture follows the principle demonstrated by remote-sensing VLM research:

```text
Optical / RGB
      ↓
visual encoder
      ↓
visual tokens
          \
           \
SAR         \
 ↓           → language model
SAR encoder /
 ↓         /
SAR tokens
          +
question tokens
```

SatQuery's implementation may evolve as experiments identify stronger backbones.

---

# Optical-SAR Fusion

Optical and SAR are not treated as equivalent image channels.

They arise from fundamentally different sensing mechanisms.

### Optical / multispectral imagery

Useful for:

```text
spectral reflectance
vegetation
water
land-cover appearance
context
```

### SAR

Useful for:

```text
microwave backscatter
surface structure
roughness-related response
day/night acquisition
cloud-penetrating observation
```

SatQuery therefore uses modality-aware processing.

---

## Fusion evaluation

A fused model is always compared against:

```text
Optical only   P(O)
SAR only       P(S)
Optical + SAR  P(O,S)
```

This is necessary to verify that the second modality contributes information rather than merely being accepted as input.

---

## Modality evidence

SatQuery aims to preserve modality identity:

```text
OPTICAL_SUPPORTED
SAR_SUPPORTED
BOTH_SUPPORTED
CONFLICTING
UNCERTAIN
```

It does not report invented contribution percentages unless a validated attribution method is available.

---

# Bi-Temporal Change Analysis

Change analysis uses two spatially corresponding observations:

```text
T1
+
T2
```

The preferred design is:

```text
T1 ──► temporal encoder ──┐
                          ├──► change representation
T2 ──► temporal encoder ──┘
                                  ↓
                             change mask
                                  ↓
                          semantic reasoning
                                  ↓
                           language answer
```

This implements the principle:

> **Change first, language second.**

---

## Why not compare raw pixels directly?

Raw differences can represent:

```text
real land-cover change
illumination
season
clouds
sensor differences
registration errors
SAR geometry
radiometric differences
```

The system must distinguish semantic change from acquisition differences.

---

# Grounding and Spatial Evidence

A natural-language answer alone is difficult to audit.

SatQuery therefore treats spatial evidence as a first-class output.

Supported evidence forms may include:

```text
bounding boxes
segmentation masks
change masks
polygons
points
```

Example:

```text
Query:
"Where are the buildings?"

Answer:
"Buildings are concentrated in the northeastern region."

Evidence:
bbox / mask overlay
```

---

## Coordinate preservation

Model inference may happen on crops:

```text
model crop pixels
      ↓
source raster pixels
      ↓
geographic coordinates
```

Every crop must preserve its mapping back to the original raster.

This is essential for correct grounding.

---

# GIS and Deterministic Computation

Certain operations should never depend on generative language models.

Examples:

```text
pixel counting
area calculation
distance
coordinate transformation
geometry intersection
band math
unit conversion
CRS reprojection
```

---

## Example: area calculation

Given binary mask:

\[
M(x,y)\in\{0,1\}
\]

positive pixel count:

\[
N=\sum M(x,y)
\]

If projected pixel dimensions are:

\[
\Delta x,\Delta y
\]

pixel area:

\[
A_p=|\Delta x\Delta y|
\]

total area:

\[
A=N\times A_p
\]

Conversion:

\[
1\text{ hectare}=10,000\,m^2
\]

The LLM explains the result.

It does not calculate it independently.

---

# Agentic Orchestration

SatQuery uses a **constrained orchestration model**.

It is not intended to be an unrestricted autonomous agent.

The orchestrator receives:

```text
query intent
input metadata
available modalities
valid workflows
model registry
tool registry
```

and selects a permitted workflow.

Conceptually:

```text
Workflow =
f(
    QueryIntent,
    InputState,
    AvailableModalities,
    ModelRegistry,
    ToolRegistry
)
```

---

## Initial workflow types

```text
SINGLE_VQA
SINGLE_GROUND
CROSS_MODAL_VQA
CHANGE_VQA
CHANGE_LOCALIZE
CHANGE_MEASURE
METADATA_QUERY
```

---

# Verification Layer

Every significant result is checked independently from the LLM.

Five major verification categories are used.

## Geometric

```text
CRS
extent
alignment
grid
resolution
coordinate mapping
mask dimensions
```

## Temporal

```text
T1/T2 order
duplicate observations
time compatibility
seasonal warning
```

## Physical

```text
band availability
polarization
valid ranges
units
sensor requirements
```

## Provenance

```text
input source
model
version
parameters
derived assets
tool history
```

## Statistical

```text
calibrated model confidence
domain shift
out-of-distribution warning
modality disagreement
```

SatQuery initially exposes these separately instead of inventing an unsupported weighted "overall reliability score."

---

# Evidence Contract

Every specialist should return a common structured evidence representation.

```text
Evidence
{
    evidence_id

    task

    prediction

    source_observations[]

    source_modalities[]

    spatial_evidence
    {
        bbox?
        mask?
        polygon?
        point?
    }

    measurements[]

    model
    {
        id
        version
        checkpoint_hash
    }

    confidence

    domain_status

    verification

    provenance
}
```

This interface allows models to be replaced without rewriting the entire application.

---

# Confidence and Uncertainty

SatQuery does not equate:

```text
largest softmax probability
```

with:

```text
scientific confidence
```

Instead the system tracks several signals independently.

```text
model confidence
input validity
geometric validity
temporal validity
domain status
modality agreement
evidence quality
```

Where possible, classifier confidence should be calibrated using held-out validation data.

Potential calibration techniques include:

```text
temperature scaling
reliability analysis
ECE
NLL
```

A result may therefore look like:

```text
Model confidence: HIGH
Geometry: PASS
Temporal validity: PASS
Domain: OUT-OF-DOMAIN WARNING
Optical/SAR agreement: MODERATE
```

rather than an unjustified:

```text
Confidence: 93.42%
```

---

# Execution Trace and Provenance

SatQuery exposes an auditable execution summary.

This is **not** hidden chain-of-thought.

Example:

```text
Task
CHANGE_MEASURE

Inputs
2 GeoTIFF observations

Validation
Spatial overlap: PASS
Temporal ordering: PASS
Grid compatibility: PASS

Model
change_model_v3

Output
built_up_gain_mask.tif

GIS operation
mask area calculation

Measurement
3.142 ha

Verification
Geometry: PASS
Temporal: PASS
Physical: PASS
Provenance: PASS
Statistical: WARN

Warnings
Cross-sensor deployment domain
```

This provides reproducibility without exposing private model reasoning.

---

# Datasets

SatQuery requires multiple datasets because no public dataset covers the complete task.

| Dataset | Main Purpose |
|---|---|
| BigEarthNet.txt | multisensor S1/S2 vision-language adaptation |
| BigEarthNet | multispectral/SAR EO representation |
| SSL4EO-S12 | multimodal self-supervised EO representation |
| VRSBench | high-resolution VQA and grounding |
| RSVQA | remote-sensing VQA diversity |
| CDVQA | bi-temporal visual question answering |
| SECOND | semantic change supervision |
| CROMA training resources | SAR-optical representation alignment |
| SpaceNet 6 | high-resolution SAR/EO geometry |
| QAG-360K / VisTA research | grounded temporal-change research |

---

## BigEarthNet.txt

Reported characteristics:

```text
464,044 co-registered Sentinel-1 / Sentinel-2 pairs
~9.6 million textual annotations
15 tasks
```

Major task groups:

```text
captioning
binary VQA
multiple-choice VQA
referring-expression detection
```

It is particularly relevant to the multisensor VLM component.

However:

> BigEarthNet.txt is based on Sentinel imagery and does not prove direct generalization to Cartosat/RISAT imagery.

---

## VRSBench

Useful for:

```text
high-resolution VQA
object understanding
visual grounding
spatial language
```

---

## RSVQA

Useful for:

```text
presence questions
counting
area reasoning
spatial relations
general remote-sensing VQA
```

---

## CDVQA / SECOND

Useful for:

```text
bi-temporal reasoning
semantic change
change VQA
temporal feature learning
```

---

## SSL4EO / CROMA-style data

Useful for:

```text
SAR-optical alignment
self-supervised EO representations
multimodal pretraining
```

---

## SpaceNet 6

Useful for:

```text
high-resolution SAR
building geometry
SAR/optical transfer
fine-scale remote-sensing perception
```

It should not be treated as direct RISAT equivalence.

---

# Model Strategy

SatQuery does not assume one model should solve every problem.

Potential components include:

| Component | Responsibility |
|---|---|
| Remote-sensing VLM | semantic VQA and language interaction |
| Optical encoder | optical/MS representation |
| SAR encoder | radar representation |
| Fusion model | optical-SAR combination |
| Grounding specialist | boxes/masks from text |
| Change specialist | temporal representations/masks |
| Segmentation specialist | pixel-level evidence |
| GIS operators | deterministic spatial computations |
| Planner | workflow selection |
| Verifier | independent validity checks |
| Answer composer | natural-language explanation |

---

# Training Strategy

SatQuery should reuse pretrained models whenever possible.

Training a new LLM or giant visual foundation model from scratch is not a project requirement.

Preferred adaptation hierarchy:

```text
frozen pretrained model
        ↓
linear/projector adaptation
        ↓
LoRA / PEFT
        ↓
sensor adapter
        ↓
partial visual unfreezing
        ↓
full fine-tuning only if justified
```

---

## LoRA

For a frozen weight matrix:

\[
W_0
\]

LoRA learns:

\[
\Delta W = BA
\]

and uses:

\[
W' = W_0 + \frac{\alpha}{r}BA
\]

with low rank:

\[
r\ll d
\]

This can dramatically reduce the number of trainable parameters during domain adaptation.

---

# Training Roadmap

```text
E0  Frozen baseline
 ↓
E1  Sensor representation evaluation
 ↓
E2  Multisensor VLM adaptation
 ↓
E3  VQA + grounding
 ↓
E4  Temporal/change specialist
 ↓
E5  Optical-SAR fusion
 ↓
E6  Cross-sensor adaptation
 ↓
E7  Calibration
 ↓
E8  Orchestration
 ↓
E9  Integrated evaluation
```

---

# Evaluation

No single metric is sufficient.

---

## VQA

Primary:

```text
Accuracy
```

Additional:

```text
per-question-type accuracy
blank-image test
shuffled-image test
question-only baseline
```

---

## Grounding

```text
IoU
mIoU
Acc@0.5
higher IoU thresholds
```

---

## Segmentation / Change Detection

```text
Precision
Recall
F1
IoU
mIoU
```

---

## Object Detection

```text
AP
mAP
per-class AP
object-size breakdown
```

---

## Captioning

Potential metrics:

```text
BLEU
ROUGE
METEOR
CIDEr
BERTScore
semantic evaluation
```

Text metrics should never substitute for factual/geospatial verification.

---

## Numerical outputs

```text
MAE
relative error
```

plus verification of:

```text
CRS
units
source mask
```

---

## Calibration

```text
ECE
NLL
reliability diagram
```

---

# Cross-Sensor Generalization

One of the highest-risk technical problems is domain shift.

Typical public training:

```text
Sentinel-1
Sentinel-2
```

Potential hidden/deployment distribution:

```text
RISAT
Cartosat
other unseen sensors
```

SatQuery therefore evaluates separately:

```text
in-domain
cross-region
cross-sensor
cross-scale
```

Performance degradation is tracked explicitly.

---

## Cross-modal ablation

Every fusion experiment should report:

```text
Optical only
SAR only
Optical + SAR
```

A multimodal model is not considered successful simply because it accepts multiple images.

---

# Technology Stack

## Frontend

```text
React
TypeScript
Vite
OpenLayers
Tailwind CSS
```

## Python domain/backend environment

```text
Python
FastAPI
```

## Geospatial

```text
GDAL
Rasterio
pyproj
Shapely
GeoPandas
rio-tiler / TiTiler
```

## AI

```text
PyTorch
Transformers
PEFT
timm where required
```

## Storage

Initial:

```text
filesystem / object-storage abstraction
SQLite
```

Scale-up option:

```text
PostgreSQL
```

## Deployment

```text
Docker
Docker Compose
```

---

# Backend Architecture

```text
                     FastAPI
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      Ingestion     Orchestration    Reports
          │             │
          ▼             ▼
       Raster/GIS     Job Queue
                        │
                        ▼
                  GPU Model Worker
                        │
                        ▼
                    Evidence
```

The API server remains separate from heavy model inference where practical.

---

# Frontend Architecture

The UI is **imagery-first**, not chat-first.

```text
┌──────────────┬─────────────────────────────┬──────────────┐
│ INPUTS       │                             │ QUERY        │
│              │                             │              │
│ Optical      │                             │ Ask...       │
│ SAR          │            MAP              │              │
│ Time         │                             │ ANSWER       │
│ CRS / GSD    │                             │              │
│ Layers       │                             │ Evidence     │
│              │                             │ Confidence   │
├──────────────┴─────────────────────────────┴──────────────┤
│ Evidence │ Execution │ Technical Details │ Export         │
└───────────────────────────────────────────────────────────┘
```

---

## Temporal viewing

Supported UI concepts:

```text
side-by-side
swipe comparison
opacity slider
flicker
change overlay
```

---

## Optical/SAR viewing

```text
Optical
SAR
Fusion
Evidence
```

should be independently togglable.

---

# Raster Visualization

Large GeoTIFFs should not be transferred directly to the browser in full.

Preferred architecture:

```text
GeoTIFF
   ↓
Rasterio/GDAL
   ↓
COG / raster source
   ↓
bounded Rasterio tile renderer
   ↓
XYZ tiles
   ↓
OpenLayers
```

The browser loads only the imagery required for the current viewport and zoom level.

The MVP uses Rasterio directly. `rio-tiler` or TiTiler remains an optional scale-up path rather than a required dependency.

---

# Storage

Scientific source files remain immutable.

Conceptual layout:

```text
data/
├── observations/
│   └── <observation_id>/
│       ├── original.tif
│       ├── visualization.tif
│       ├── visualization.json
│       ├── thumbnail.png
│       └── metadata.json
│
├── assets/
│   └── <visualization_asset_id>.json
│
├── analyses/
│   └── <analysis_id>/
│       ├── evidence/
│       ├── masks/
│       ├── overlays/
│       └── report/
│
└── models/
```

---

## Important asset distinction

Do not confuse:

```text
original scientific raster
analysis-ready raster
visualization image
thumbnail
model tensor
```

An 8-bit RGB preview must never silently replace multispectral/SAR data for scientific inference.

---

# API Design

Example endpoints:

```http
POST /api/observations
POST /api/analyses

GET /api/jobs/{job_id}
GET /api/analyses/{analysis_id}
GET /api/evidence/{evidence_id}

GET /tiles/{asset_id}/{z}/{x}/{y}.png

GET /api/reports/{analysis_id}
```

---

## Analysis request

Conceptually:

```json
{
  "observation_ids": [
    "obs_pre",
    "obs_post"
  ],
  "query": "Where did water increase and by how much?"
}
```

---

## Analysis result

Conceptually:

```json
{
  "analysis_id": "analysis_001",
  "task": "CHANGE_MEASURE",
  "answer": "Water extent increased primarily in the southern area.",
  "evidence": [],
  "measurements": [],
  "warnings": [],
  "verification": {},
  "execution_summary": []
}
```

---

# Security

Remote-sensing files must be treated as untrusted input.

GDAL supports many complex formats and drivers; therefore ingestion should be sandboxed and constrained.

---

## Upload controls

Validate:

```text
file size
actual raster driver
width
height
band count
uncompressed pixel count
allowed format
```

Do not trust filename extensions alone.

---

## Recommended allowed formats

Initial:

```text
GeoTIFF / GTiff
PNG
JPEG
```

Reject unnecessary formats such as user-supplied VRT for the competition build.

---

## Raster sandboxing

Raster inspection should run with:

```text
restricted filesystem
restricted network
CPU limit
RAM limit
execution timeout
output-size limit
minimal required GDAL drivers
```

The process should not have access to:

```text
SSH keys
model secrets
database admin credentials
host filesystem
```

---

## Command execution

Do not concatenate user-controlled strings into shell commands.

Prefer:

```text
Python library APIs
typed parameters
enumerated arguments
```

---

# Repository Structure

Suggested structure:

```text
satquery/
│
├── apps/
│   ├── web/
│   │   ├── src/
│   │   ├── public/
│   │   └── package.json
│   │
│   └── api/
│       ├── app/
│       │   ├── routes/
│       │   ├── schemas/
│       │   ├── services/
│       │   └── main.py
│       └── requirements.txt
│
├── satquery/
│   ├── ingestion/
│   ├── geo/
│   ├── orchestration/
│   ├── verification/
│   ├── evidence/
│   ├── registry/
│   └── reporting/
│
├── ml/
│   ├── adapters/
│   ├── inference/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── configs/
│
├── models/
│   └── registry.yaml
│
├── experiments/
│
├── tests/
│   ├── geo/
│   ├── ingestion/
│   ├── routing/
│   ├── evidence/
│   ├── models/
│   └── integration/
│
├── data/
│
├── docker/
│
├── docs/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

# Local Development

> The exact commands may evolve as the implementation lands. Keep this section synchronized with the repository.

## Requirements

Recommended:

```text
Python 3.11+
Node.js 20+
GDAL
Docker
NVIDIA GPU + CUDA for accelerated inference
```

CPU-only development should still support:

```text
ingestion
metadata inspection
GIS operations
frontend
routing tests
```

even if model inference is slower or disabled.

---

## Clone

```bash
git clone <repository-url>
cd satquery
```

---

## Environment

```bash
cp .env.example .env
```

---

## Backend

Example:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -e ".[dev]"
```

Start the Phase 1B API:

```bash
uvicorn apps.api.app.main:app --reload
```

Run the current test suite with `python -m pytest`.

---

## Frontend

The frontend package has not been scaffolded yet. It will be added with the imagery-viewer phase rather than maintained as unused boilerplate.

---

## Docker

When Docker Compose configuration is available:

```bash
docker compose up --build
```

---

# Environment Variables

Example `.env.example`:

```env
SATQUERY_ENV=development

DATA_ROOT=./data
MODEL_ROOT=./models

DATABASE_URL=sqlite:///./data/satquery.db

MAX_UPLOAD_SIZE_MB=512
MAX_RASTER_WIDTH=50000
MAX_RASTER_HEIGHT=50000
MAX_RASTER_PIXELS=150000000
MAX_RASTER_BANDS=32

GPU_DEVICE=cpu

ENABLE_REMOTE_NETWORK=false
```

Do not commit secrets.

---

# Testing

SatQuery requires three different testing layers.

---

## 1. Deterministic GIS tests

Examples:

```text
pixel → world coordinate
world → pixel
CRS conversion
area calculation
tile offset mapping
raster overlap
band validation
temporal ordering
```

These should have strict unit tests.

---

## 2. Model evaluation

Maintain fixed benchmark/golden datasets.

Evaluate every approved checkpoint against:

```text
VQA
grounding
SAR
optical-SAR
change
cross-sensor
```

---

## 3. Orchestration tests

Examples:

```text
1 image + "What changed?"
→ MISSING_TEMPORAL_PAIR

RGB + "Calculate NDVI"
→ MISSING_NIR_BAND

Optical + SAR + comparison query
→ CROSS_MODAL_VQA

Non-overlapping temporal pair
→ NO_SPATIAL_OVERLAP
```

---

# Development Roadmap

## Phase 0 — Research and benchmark freeze

- [ ] Freeze required task matrix
- [ ] Freeze evaluation protocol
- [ ] Define failure policy
- [ ] Define benchmark splits
- [ ] Establish frozen-model baselines

---

## Phase 1 — Geospatial foundation

- [x] GeoTIFF upload
- [x] Raster metadata inspector
- [x] CRS/GSD/bounds extraction
- [x] Pair validator
- [x] COG/tiling support
- [ ] OpenLayers viewer
- [x] Coordinate-mapping tests

---

## Phase 2 — Single-image VQA

- [ ] Integrate baseline VLM
- [ ] Remote-sensing preprocessing
- [ ] VQA inference
- [ ] Evaluation harness
- [ ] Blank/shuffled-image baselines

---

## Phase 3 — Grounding

- [ ] Grounding model
- [ ] Bounding-box output
- [ ] Original-raster coordinate mapping
- [ ] Map overlays
- [ ] IoU evaluation

---

## Phase 4 — Multisensor VLM

- [ ] Optical/MS branch
- [ ] SAR branch
- [ ] Sensor projection layers
- [ ] LoRA adaptation
- [ ] BigEarthNet.txt baseline reproduction
- [ ] Single-sensor ablations

---

## Phase 5 — Optical-SAR fusion

- [ ] Optical-only inference
- [ ] SAR-only inference
- [ ] Fused inference
- [ ] Modality-support audit
- [ ] Modality disagreement handling

---

## Phase 6 — Change analysis

- [ ] Temporal pair workflow
- [ ] Change specialist
- [ ] Change mask
- [ ] Change VQA
- [ ] T1+T1 sanity test
- [ ] T2+T1 temporal-direction test

---

## Phase 7 — Cross-sensor robustness

- [ ] Cross-region evaluation
- [ ] Cross-scale evaluation
- [ ] Cross-sensor evaluation
- [ ] Sensor-adapter experiments
- [ ] Partial vision adaptation if required

---

## Phase 8 — Verification and calibration

- [ ] Geometric verifier
- [ ] Temporal verifier
- [ ] Physical verifier
- [ ] Provenance verifier
- [ ] Statistical/OOD verifier
- [ ] Confidence calibration

---

## Phase 9 — Agentic orchestration

- [ ] Query intent schema
- [ ] Model registry
- [ ] Tool registry
- [ ] Workflow selection
- [ ] Parameter validation
- [ ] Structured execution trace

---

## Phase 10 — Product experience

- [ ] Evidence panel
- [ ] Temporal swipe viewer
- [ ] Optical/SAR layers
- [ ] Confidence/warnings
- [ ] Execution trace
- [ ] Downloadable analysis report

---

## Phase 11 — Red-team and ablation testing

- [ ] Corrupt TIFF
- [ ] Missing CRS
- [ ] Missing NIR
- [ ] Unknown SAR polarization
- [ ] Misaligned imagery
- [ ] Non-overlapping scenes
- [ ] Low-confidence cases
- [ ] Modality-drop experiments
- [ ] OOD sensor experiments
- [ ] Very large rasters

---

## Phase 12 — Competition hardening

- [ ] Freeze checkpoints
- [ ] Freeze preprocessing
- [ ] Freeze routing rules
- [ ] Golden regression suite
- [ ] Demo datasets
- [ ] Offline deployment verification
- [ ] Performance profiling
- [ ] Documentation
- [ ] Final reports

---

# Failure Policy

SatQuery prefers abstention over unsupported certainty.

| Situation | Behavior |
|---|---|
| One image + change query | request second observation |
| Missing required spectral band | reject calculation |
| Missing CRS + semantic VQA | allow with warning |
| Missing CRS + area request | reject physical measurement |
| Non-overlapping images | reject comparison |
| Unknown temporal order | request/resolve order |
| Unknown SAR polarization | avoid polarization-specific claims |
| Misaligned pair | reject/repair before spatial comparison |
| Out-of-domain sensor | warn/reduce confidence |
| Optical/SAR conflict | report disagreement |
| Low-confidence evidence | qualify or abstain |
| No supported change | report no significant supported change |
| Tool failure | fail analysis; do not fabricate result |
| Invalid evidence geometry | do not calculate measurement |

---

# What SatQuery Will Not Do

SatQuery intentionally avoids several common over-engineering patterns.

For the initial system:

```text
❌ No arbitrary Python execution by the LLM
❌ No autonomous shell commands
❌ No multi-agent swarm
❌ No RL-trained planner requirement
❌ No Kubernetes requirement
❌ No Kafka requirement
❌ No vector database requirement
❌ No fake missing spectral bands
❌ No LLM-generated physical measurements
❌ No hidden change claim from one image
❌ No treating SAR as ordinary grayscale RGB
```

These can be reconsidered only if future requirements provide a clear technical justification.

---

# Research Risks

## 1. Cross-sensor domain shift

**Risk:** Critical

Training resources are dominated by public sensors such as Sentinel-1 and Sentinel-2, while deployment may involve different sensors.

Mitigations:

```text
sensor-aware adapters
cross-sensor evaluation
multi-sensor representation learning
high-resolution proxy datasets
partial visual fine-tuning
```

---

## 2. SAR generalization

**Risk:** High

SAR appearance depends on:

```text
wavelength
polarization
incidence geometry
surface roughness
moisture
acquisition mode
```

Avoid universal rules such as:

```text
water is always dark
urban is always bright
```

---

## 3. Grounded change

**Risk:** High

Language models may correctly describe change while localizing it incorrectly.

Mitigation:

```text
change mask first
language second
```

---

## 4. Multimodal collapse

**Risk:** High

Fusion model may ignore one sensor.

Mitigation:

```text
O
S
O+S
```

ablation.

---

## 5. Scale shift

**Risk:** High

The same pixel/patch dimensions may correspond to drastically different ground areas.

Mitigation:

```text
GSD-aware preprocessing
multi-scale inference
scale-stratified evaluation
```

---

## 6. Confidence calibration

**Risk:** Medium-high

Raw neural probabilities may be overconfident.

Mitigation:

```text
held-out calibration
reliability diagrams
ECE/NLL
explicit domain warnings
abstention
```

---

## 7. Agent routing

**Risk:** Medium

Much easier to control than perception.

Mitigation:

```text
bounded intents
typed tools
preconditions
deterministic tests
```

---

# Demo Scenarios

A strong SatQuery demo should demonstrate capability **and failure awareness**.

---

## Demo 1 — Single image VQA

Upload:

```text
optical.tif
```

Ask:

> "Is water present?"

Show:

```text
answer
evidence
confidence
model
```

---

## Demo 2 — Grounding

Ask:

> "Where is the largest built-up region?"

Show:

```text
bounding box / mask
map zoom
```

---

## Demo 3 — Optical + SAR

Upload:

```text
optical.tif
sar.tif
```

Ask:

> "Is the settlement supported by both modalities?"

Show:

```text
optical result
SAR result
fusion result
agreement
```

---

## Demo 4 — Temporal change

Upload:

```text
before.tif
after.tif
```

Ask:

> "Where did water expand?"

Show:

```text
before/after
change mask
```

---

## Demo 5 — Quantitative analysis

Ask:

> "How many hectares of water were added?"

Show:

```text
mask
pixel count
area
calculation provenance
```

---

## Demo 6 — Correct refusal

Upload one image.

Ask:

> "What changed?"

Expected:

> A second observation of the same area from another time is required for change analysis.

This demonstrates reliability rather than hallucination.

---

# Key Research References

The architecture is informed by research across vision-language modeling, multimodal Earth observation, change detection, and agentic geospatial systems.

## Remote-sensing multimodal learning

### BigEarthNet.txt

**BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation**

Relevant for:

```text
Sentinel-1 + Sentinel-2
multisensor VLM adaptation
VQA
captioning
grounding
```

Important reported characteristics:

```text
464,044 co-registered S1/S2 pairs
~9.6M annotations
15 tasks
```

> The current research version used in this project is a preprint; results should not be treated as proof of universal sensor generalization.

---

### CROMA

**Contrastive Radar-Optical Masked Autoencoders**

NeurIPS 2023.

Relevant for:

```text
SAR-optical contrastive learning
multimodal EO representation
masked reconstruction
```

---

### AnySat

**AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**

CVPR 2025.

Relevant for:

```text
heterogeneous EO sensors
resolution generalization
multi-scale representation
```

---

# Vision-language foundations

### CLIP

Radford et al.

**Learning Transferable Visual Models From Natural Language Supervision**

ICML 2021.

Relevant for:

```text
image-text contrastive learning
shared embedding spaces
zero-shot visual classification
```

---

### LLaVA

**Visual Instruction Tuning**

NeurIPS 2023.

Relevant for:

```text
vision encoder
projection layer
LLM
visual instruction tuning
```

---

### InternVL

Relevant as a scalable multimodal VLM foundation and as the backbone adapted by RS-InternVL-style research.

---

# Efficient adaptation

### LoRA

Hu et al.

**LoRA: Low-Rank Adaptation of Large Language Models**

ICLR 2022.

Relevant for:

```text
parameter-efficient fine-tuning
low-rank weight updates
```

---

# Remote-sensing VQA / grounding

### VRSBench

NeurIPS 2024 Datasets and Benchmarks.

Relevant for:

```text
high-resolution remote-sensing VQA
captioning
visual grounding
```

---

### RSVQA

Relevant for:

```text
remote-sensing visual question answering
presence
counting
spatial questions
```

---

# Change analysis

### CDVQA

**Change Detection Meets Visual Question Answering**

Relevant for:

```text
multi-temporal visual encoding
change VQA
temporal fusion
```

---

### SECOND

Relevant for semantic change-detection supervision.

---

# Agentic Earth Observation

### Agentic AI for Remote Sensing: Technical Challenges and Research Directions

Used as a conceptual reference for:

```text
geospatial state
tool feasibility
planner-executor-verifier architecture
trajectory validity
provenance
```

> This work is a position paper and should be treated as architectural guidance rather than empirical proof of one optimal agent design.

---

# Evaluation References

Relevant foundations include:

```text
Precision / Recall / F1
IoU / mIoU
mAP
BLEU
CIDEr
BERTScore
confidence calibration
temperature scaling
```

Evaluation in SatQuery is task-specific; no single score is treated as universal system quality.

---

# Project Principles

The following principles should remain true even as individual models change.

```text
1. Sensor metadata is first-class information.

2. Optical and SAR are different physical measurements.

3. Remote-sensing images must retain CRS, GSD and provenance.

4. Spatial evidence must survive resizing and tiling.

5. Numeric claims come from deterministic computations.

6. A model cannot use bands that do not exist.

7. Change requires compatible temporal observations.

8. Multimodal fusion must be proven through ablation.

9. Confidence must be calibrated or qualified.

10. Cross-sensor performance must be tested explicitly.

11. Execution traces expose operations, not chain-of-thought.

12. Invalid inputs should be rejected instead of hallucinated around.

13. Evidence exists independently of language.

14. The language model explains evidence; it does not manufacture it.
```

---

# Final Project Definition

SatQuery can be summarized as:

\[
\boxed{
\text{SatQuery}
=
\text{Sensor-Aware Perception}
+
\text{Geospatial State}
+
\text{Multimodal Fusion}
+
\text{Temporal Analysis}
+
\text{Structured Evidence}
+
\text{Deterministic GIS}
+
\text{Verification}
+
\text{Natural Language}
}
\]

or, operationally:

```text
UNDERSTAND SENSOR
       +
UNDERSTAND QUESTION
       +
VALIDATE INPUT
       +
SELECT SPECIALIST
       +
PROCESS DATA CORRECTLY
       +
GENERATE EVIDENCE
       +
COMPUTE REQUIRED QUANTITIES
       +
VERIFY
       +
EXPLAIN
```

---

# Contributing

The project is currently being developed around a research-first approach.

Before contributing:

1. Understand the remote-sensing assumptions of the component being modified.
2. Do not remove geospatial metadata from processing pipelines.
3. Add tests for deterministic GIS transformations.
4. Benchmark model changes rather than relying on visual impressions.
5. Preserve model/checkpoint/preprocessing provenance.
6. Document new supported sensors and known limitations.
7. Never silently invent missing sensor information.

More detailed contribution guidelines can be added in:

```text
CONTRIBUTING.md
```

---

# License

License selection is pending final confirmation of:

```text
project ownership
competition requirements
model licenses
dataset licenses
third-party code licenses
```

Do not assume that every research dataset or pretrained checkpoint permits unrestricted commercial redistribution.

A final project license should be selected only after all dependencies have been audited.

---

# Status

SatQuery is currently under active research and development.

The target is not simply to demonstrate that a language model can talk about satellite images.

The target is to build a system where:

> **every important answer can be connected to the sensor data, spatial evidence, analytical operation and execution history that produced it.**

---

<p align="center">
  <strong>SatQuery AI</strong><br/>
  Ask the Earth. Verify the answer.
</p>
