# SatQuery AI — Architecture

> **Architecture specification for an evidence-grounded, sensor-aware remote-sensing assistant.**

**Project:** SatQuery AI  
**Problem Statement ID:** 26167  
**Document:** `docs/ARCHITECTURE.md`  
**Status:** Target architecture / implementation specification  
**Last updated:** 2026-09-03

---

## 1. Purpose

This document defines the technical architecture of **SatQuery AI**.

SatQuery is not designed as a generic chatbot that receives a satellite screenshot and produces free-form text. It is designed as a **remote-sensing analysis system with a natural-language interface**.

The core architectural principle is:

> **The language model may explain evidence; it may not manufacture evidence.**

Every important analytical answer should therefore be traceable to:

1. the source observation(s),
2. the selected workflow,
3. the model or deterministic tool that produced the evidence,
4. the spatial output used to support the claim,
5. the measurements derived from that evidence,
6. verification checks,
7. confidence / uncertainty information,
8. the execution history.

The architecture is intentionally modular so that a stronger VLM, SAR encoder, grounding model, change detector, or fusion model can replace an existing one without requiring a complete rewrite of the platform.

---

## 2. Scope

SatQuery targets the following analysis modes.

### 2.1 Single-observation analysis

Inputs:

- one optical image,
- one multispectral image,
- or one SAR image.

Primary capabilities:

- visual question answering,
- scene understanding,
- captioning where useful,
- text-guided grounding,
- object / region localization.

Example:

> "Where is the largest built-up region?"

---

### 2.2 Cross-modal analysis

Inputs:

- co-registered optical / multispectral imagery,
- and SAR imagery of the same region.

Primary capabilities:

- complementary information extraction,
- modality-specific interpretation,
- optical-only / SAR-only / fused reasoning,
- disagreement detection,
- evidence attribution at the modality level.

Example:

> "Is the settlement supported by both optical and SAR observations?"

---

### 2.3 Bi-temporal analysis

Inputs:

- observation at time `T1`,
- observation at time `T2`,
- covering the same spatial region.

Primary capabilities:

- change detection,
- change localization,
- change description,
- change VQA,
- quantitative change measurement.

Example:

> "Where did water expand and by how much?"

---

### 2.4 Geospatial measurement

Inputs:

- validated spatial evidence,
- georeferenced raster metadata.

Primary capabilities:

- area,
- count,
- distance,
- intersection,
- coordinate conversion,
- approved band math / indices.

Measurements must come from deterministic GIS operators, not from free-form LLM generation.

---

## 3. Evidence status of this architecture

This document contains two kinds of content.

### 3.1 Source-derived design principles

The following principles are directly motivated by the research reviewed for SatQuery:

- remote-sensing modalities should not be treated as interchangeable RGB images,
- optical and SAR require modality-aware representation,
- multi-sensor VLM adaptation can use modality-specific visual branches and projections,
- change analysis requires explicit temporal reasoning,
- remote-sensing workflows should preserve CRS, resolution, extent, time, modality, uncertainty, and provenance,
- tool calls should be constrained by geospatial / physical preconditions,
- verification should be external to language-model self-critique,
- spatial evidence should be evaluated independently from linguistic fluency,
- cross-sensor generalization must be measured rather than assumed.

Relevant research includes:

- **BigEarthNet.txt** — multi-sensor S1/S2 image-text data and RS-InternVL adaptation,
- **CROMA** — radar-optical contrastive and masked representation learning,
- **AnySat** — heterogeneous EO sensors, scales, resolutions, and modalities,
- **VRSBench** — remote-sensing VQA and grounding,
- **RSVQA** — remote-sensing visual question answering,
- **CDVQA** — bi-temporal change VQA,
- **Agentic AI for Remote Sensing** — structured EO state, parameterized tools, planner/executor/verifier design,
- multimodal geospatial foundation-model surveys reviewed during project research.

### 3.2 SatQuery engineering proposals

The following are **our architecture choices**, not claims that a paper has proved this exact implementation is optimal:

- the exact service boundaries,
- the `ObservationState` schema,
- the `Evidence` contract,
- the workflow names,
- the registry formats,
- the API design,
- the storage layout,
- the failure codes,
- the UI composition,
- the caching strategy,
- the deployment topology.

These should evolve only when experiments or implementation constraints justify a change.

---

# 4. Architectural principles

## 4.1 Evidence before language

The system should prefer:

```text
observation
    ↓
specialist model
    ↓
mask / bbox / class / score
    ↓
GIS measurement
    ↓
verification
    ↓
language explanation
```

over:

```text
observation
    ↓
LLM/VLM
    ↓
plausible prose
```

---

## 4.2 Metadata is first-class state

The system must preserve:

- CRS,
- affine transform,
- spatial extent,
- native resolution / GSD,
- band descriptions,
- sensor name where available,
- modality,
- SAR polarization where available,
- acquisition time,
- NoData information,
- data type,
- provenance.

AI preprocessing must not erase the scientific state needed later for measurement and verification.

---

## 4.3 Original observations are immutable

Uploaded scientific inputs are stored as immutable source assets.

All transformations produce derived assets.

Example:

```text
original.tif
   ↓
analysis_ready_v1.tif
   ↓
preview_rgb.tif
   ↓
web_cog.tif
```

Every derived asset records:

- parent asset,
- operation,
- parameters,
- software version,
- time,
- output metadata.

---

## 4.4 Deterministic operations remain deterministic

Operations such as:

- CRS transforms,
- clipping,
- raster overlap,
- pixel counting,
- mask area,
- coordinate conversion,
- unit conversion,
- band-presence validation,
- approved spectral index calculations,

are implemented as tested GIS operators.

They are not delegated to generative models.

---

## 4.5 Unknown is a valid value

If the system does not know a sensor, band meaning, polarization, acquisition date, or CRS, it stores:

```text
unknown
```

rather than inventing a value.

---

## 4.6 Sensor-specific perception

Optical / multispectral and SAR imagery arise from different sensing physics.

The system should therefore permit:

- separate optical encoders,
- separate SAR encoders,
- separate preprocessing profiles,
- sensor-specific adaptation,
- fusion only after modality-appropriate processing.

---

## 4.7 Fusion must be demonstrated, not assumed

Every multimodal model should be evaluated as:

```text
Optical only
SAR only
Optical + SAR
```

A system that accepts two modalities but ignores one is not considered successful fusion.

---

## 4.8 Geometric validity is independent of semantic validity

A model can be semantically correct while its bounding box, mask, coordinates, or area are wrong.

Therefore:

- language correctness,
- spatial correctness,
- geospatial validity,
- confidence,
- workflow validity

are evaluated separately.

---

## 4.9 Constrained orchestration

The planner may choose only from registered:

- models,
- workflows,
- deterministic tools,
- parameter ranges.

It may not invent arbitrary model names, shell commands, Python programs, or unsupported processing chains.

---

## 4.10 Fail closed for unsupported scientific claims

Examples:

- one image + "what changed?" → request second image,
- RGB-only + "compute NDVI" → reject,
- no georeferencing + "area in hectares" → reject,
- non-overlapping pair → reject spatial comparison,
- invalid mask geometry → do not calculate area,
- model failure → do not fabricate an answer.

---

# 5. System context

```text
┌──────────────────────────────────────────────────────────────┐
│                           USER                               │
│              observations + natural language                │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       WEB CLIENT                             │
│ React + TypeScript + OpenLayers                             │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTPS / JSON
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                         API                                  │
│ FastAPI                                                      │
└──────────────────────────────┬───────────────────────────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
      INGESTION / GIS    ORCHESTRATION       REPORTING
             │                 │
             │                 ▼
             │            JOB DISPATCH
             │                 │
             │                 ▼
             │          GPU MODEL WORKER
             │                 │
             └────────────┬────┘
                          ▼
                 STRUCTURED EVIDENCE
                          │
                          ▼
                       VERIFIER
                          │
                          ▼
                  ANSWER COMPOSER
                          │
                          ▼
                        RESULT
```

---

# 6. Logical architecture

```text
USER QUERY + OBSERVATIONS
          │
          ▼
┌────────────────────┐
│ Ingestion / Inspect │
└─────────┬──────────┘
          ▼
   ObservationState[]
          │
          ▼
┌────────────────────┐
│ Input / Pair Verify │
└─────────┬──────────┘
          ▼
      InputContext
          │
          ▼
┌────────────────────┐
│ Query Interpreter   │
└─────────┬──────────┘
          ▼
        QueryPlan
          │
          ▼
┌────────────────────┐
│ Constrained Planner │
└─────────┬──────────┘
          ▼
       WorkflowPlan
          │
   ┌──────┼────────────────┐
   │      │                │
   ▼      ▼                ▼
 VLM   Specialist(s)   GIS operators
   │      │                │
   └──────┼────────────────┘
          ▼
        Evidence[]
          │
          ▼
┌────────────────────┐
│ Verifier            │
└─────────┬──────────┘
          ▼
 VerificationReport
          │
          ▼
┌────────────────────┐
│ Answer Composer     │
└─────────┬──────────┘
          ▼
  AnalysisResponse
```

---

# 7. Component responsibilities

## 7.1 Web client

Responsibilities:

- upload observations,
- show upload / validation state,
- display raster imagery,
- switch optical / SAR / temporal layers,
- ask natural-language questions,
- show evidence overlays,
- show warnings,
- show confidence / domain status,
- show execution trace,
- export report.

The browser must not be treated as the source of scientific truth.

Visual state such as:

- map zoom,
- layer opacity,
- selected tab,

must remain separate from scientific evidence.

---

## 7.2 API service

Responsibilities:

- authentication / session boundary if enabled,
- upload handling,
- observation registration,
- analysis creation,
- job status,
- result retrieval,
- report requests,
- validated communication between web, GIS, orchestration, and model workers.

The API should not perform expensive GPU inference inline with user HTTP request handling.

---

## 7.3 Ingestion service

Responsibilities:

1. quarantine upload,
2. identify actual raster format / driver,
3. apply size / dimension / band limits,
4. extract metadata,
5. classify modality where supported,
6. create immutable source asset,
7. create visualization derivative if needed,
8. create `ObservationState`,
9. store warnings.

---

## 7.4 Geospatial service

Responsibilities:

- raster metadata,
- window reads,
- tile selection,
- spatial overlap,
- reprojection,
- alignment support,
- coordinate mapping,
- AOI clipping,
- mask area,
- distance,
- intersection,
- band validation,
- supported band math,
- raster-to-vector conversion where useful.

Preferred libraries:

- GDAL,
- Rasterio,
- pyproj,
- Shapely,
- GeoPandas,
- rio-tiler / TiTiler.

---

## 7.5 Query interpreter

Responsibilities:

- convert user request into bounded structured intent,
- identify target concept,
- identify requested output,
- identify spatial constraints,
- determine whether measurement is requested.

It does not execute tools.

---

## 7.6 Orchestrator

Responsibilities:

- combine query plan with available input state,
- check workflow feasibility,
- select a registered workflow,
- select registered model(s),
- select registered tool(s),
- build ordered execution plan,
- react to allowed intermediate outcomes,
- terminate on hard failure,
- record execution facts.

The orchestrator is constrained by registries and schemas.

---

## 7.7 GPU model worker

Responsibilities:

- load approved checkpoints,
- keep hot models resident where feasible,
- run preprocessing profiles,
- run inference,
- postprocess model outputs,
- transform outputs into canonical `Evidence`,
- expose model version / checkpoint hash,
- return no prose unless the registered model's task is inherently textual.

---

## 7.8 Verifier

Responsibilities:

- geometric checks,
- temporal checks,
- physical checks,
- provenance checks,
- statistical / reliability checks,
- produce `PASS`, `WARN`, or `FAIL` per category,
- block unsupported quantitative or scientific claims.

The verifier is independent from the final answer composer.

---

## 7.9 Answer composer

Responsibilities:

- receive user query,
- receive only verified / permitted evidence,
- generate concise natural-language explanation,
- faithfully represent measurements,
- include uncertainty / warnings,
- never invent unsupported spatial or numeric claims.

---

## 7.10 Reporting service

Responsibilities:

- render structured analysis results,
- include metadata,
- include evidence snapshots,
- include measurements,
- include model/tool provenance,
- include warnings,
- include execution summary,
- produce HTML first,
- optionally generate PDF from structured HTML.

---

# 8. Core domain models

The schemas below are conceptual. Production code should use typed equivalents such as Pydantic models.

---

## 8.1 `ObservationState`

```yaml
ObservationState:
  observation_id: string

  source_asset:
    asset_id: string
    original_name: string
    path: string
    sha256: string
    immutable: true

  raster:
    driver: string
    width: integer
    height: integer
    band_count: integer
    dtype: [string]
    nodata: [number|null]

  sensor:
    modality: optical|multispectral|sar|unknown
    sensor_name: string|null
    platform: string|null
    product_level: string|null
    bands:
      - index: integer
        name: string|null
        wavelength_nm: number|null
        resolution_m: number|null
    polarizations:
      - string

  geo:
    crs: string|null
    transform: [number]|null
    bounds: [number, number, number, number]|null
    native_gsd_x: number|null
    native_gsd_y: number|null
    units: string|null

  temporal:
    acquisition_time: datetime|null

  validity:
    has_crs: boolean
    has_transform: boolean
    has_nodata: boolean
    metadata_quality: high|medium|low|unknown
    warnings: [string]

  provenance:
    created_at: datetime
    ingestion_version: string
```

---

## 8.2 `PairCompatibility`

```yaml
PairCompatibility:
  observation_a: string
  observation_b: string

  overlap:
    known: boolean
    overlap_fraction: number|null
    sufficient: boolean|null

  crs:
    equal: boolean|null
    transformable: boolean|null

  grid:
    same_shape: boolean|null
    aligned: boolean|null
    same_resolution: boolean|null

  temporal:
    order_known: boolean
    first: string|null
    second: string|null
    time_delta_seconds: number|null

  modality:
    pair_type: optical_sar|temporal_same_modality|temporal_cross_modal|unknown
    compatible: boolean|null

  registration:
    status: verified|approximate|unknown|invalid

  result:
    status: PASS|WARN|FAIL
    reasons: [string]
```

---

## 8.3 `QueryIntent`

```yaml
QueryIntent:
  intent: SINGLE_VQA|GROUND_OBJECT|DESCRIBE_SCENE|CROSS_MODAL_VQA|CHANGE_VQA|CHANGE_LOCALIZE|MEASURE|CHANGE_MEASURE|METADATA_QUERY

  target:
    concept: string|null

  spatial_constraint:
    relation: string|null
    region: string|null

  output:
    wants_text: boolean
    wants_spatial_evidence: boolean
    wants_measurement: boolean
    unit: string|null

  raw_query: string
```

---

## 8.4 `WorkflowPlan`

```yaml
WorkflowPlan:
  workflow_id: string
  workflow_type: string

  observations:
    - string

  requirements:
    required_modalities: [string]
    required_count: integer
    requires_georeferencing: boolean
    requires_alignment: boolean
    required_bands: [string]

  selected_models:
    - model_id: string
      version: string

  selected_tools:
    - tool_id: string

  steps:
    - step_id: string
      kind: model|tool|verify
      operation: string
      depends_on: [string]

  feasibility:
    status: PASS|WARN|FAIL
    reasons: [string]
```

---

## 8.5 `Evidence`

```yaml
Evidence:
  evidence_id: string

  task: string
  prediction:
    label: string|null
    answer: string|null
    score: number|null

  source_observations:
    - string

  source_modalities:
    - optical|multispectral|sar|unknown

  spatial:
    coordinate_space: source_pixel|world|analysis_grid|null

    bbox:
      x1: number
      y1: number
      x2: number
      y2: number
    mask_asset_id: string|null
    polygon_asset_id: string|null
    point: [number, number]|null

  measurements:
    - name: string
      value: number
      unit: string
      source_operation: string

  model:
    model_id: string|null
    version: string|null
    checkpoint_hash: string|null
    preprocessing_profile: string|null

  confidence:
    raw: number|null
    calibrated: number|null
    calibration_id: string|null

  domain:
    status: in_domain|shifted|unknown
    reasons: [string]

  provenance:
    created_at: datetime
    parent_evidence_ids: [string]
    operation_id: string|null
```

---

## 8.6 `VerificationReport`

```yaml
VerificationReport:
  geometry:
    status: PASS|WARN|FAIL
    checks: [object]

  temporal:
    status: PASS|WARN|FAIL
    checks: [object]

  physical:
    status: PASS|WARN|FAIL
    checks: [object]

  provenance:
    status: PASS|WARN|FAIL
    checks: [object]

  statistical:
    status: PASS|WARN|FAIL
    checks: [object]

  overall_policy:
    may_answer: boolean
    may_measure: boolean
    must_warn: boolean
    must_abstain: boolean

  reasons: [string]
```

---

## 8.7 `ExecutionEvent`

```yaml
ExecutionEvent:
  event_id: string
  analysis_id: string
  step_id: string

  type: validation|model|tool|verification|composition

  name: string
  version: string|null

  inputs:
    observation_ids: [string]
    evidence_ids: [string]

  parameters: object

  outputs:
    evidence_ids: [string]
    asset_ids: [string]

  status: STARTED|COMPLETED|FAILED|SKIPPED

  started_at: datetime
  finished_at: datetime|null

  warnings: [string]
  error_code: string|null
```

---

## 8.8 `AnalysisResponse`

```yaml
AnalysisResponse:
  analysis_id: string
  task: string

  answer: string|null

  evidence:
    - evidence_id: string
      role: primary|supporting|conflicting

  measurements:
    - name: string
      value: number
      unit: string
      evidence_id: string

  confidence_status:
    model: string|null
    data_validity: string
    domain: string
    modality_agreement: string|null

  verification:
    geometry: PASS|WARN|FAIL
    temporal: PASS|WARN|FAIL
    physical: PASS|WARN|FAIL
    provenance: PASS|WARN|FAIL
    statistical: PASS|WARN|FAIL

  warnings: [string]

  execution_summary:
    - step: string
      model_or_tool: string
      parameters: object
      status: string
```

---

# 9. State machine

Each analysis is managed as an explicit state machine.

```text
CREATED
  ↓
VALIDATING_INPUT
  ↓
INTERPRETING_QUERY
  ↓
PLANNING
  ↓
PREPROCESSING
  ↓
RUNNING_MODELS
  ↓
POSTPROCESSING
  ↓
RUNNING_GIS
  ↓
VERIFYING
  ↓
COMPOSING
  ↓
COMPLETED
```

Failure can occur from any active state:

```text
VALIDATING_INPUT ──► FAILED
PLANNING ──────────► FAILED
RUNNING_MODELS ────► FAILED
VERIFYING ─────────► FAILED
```

The system must not skip from a failed evidence step directly to answer generation.

---

# 10. Workflow catalogue

The initial system supports a deliberately bounded set of workflow templates.

---

## 10.1 `SINGLE_VQA`

### Preconditions

- exactly one supported observation,
- modality supported by selected model,
- no requirement for missing scientific metadata unless query demands it.

### Flow

```text
Observation
   ↓
Input validation
   ↓
Modality-specific preprocessing
   ↓
VLM / VQA specialist
   ↓
Semantic evidence
   ↓
Statistical/domain verification
   ↓
Answer composer
```

### Output

- text answer,
- optional spatial evidence,
- confidence / warnings,
- execution summary.

---

## 10.2 `SINGLE_GROUND`

### Preconditions

- one supported observation,
- grounding specialist compatible with modality / display representation.

### Flow

```text
Observation
   ↓
Tile / crop selection
   ↓
Grounding model
   ↓
bbox/mask in crop coordinates
   ↓
map to source pixels
   ↓
map to world coordinates if georeferenced
   ↓
geometric verification
   ↓
Evidence
```

### Output

- box / mask / polygon,
- label / textual description,
- coordinates where valid.

---

## 10.3 `CROSS_MODAL_VQA`

### Preconditions

- at least optical/MS + SAR,
- sufficient spatial overlap,
- pair registration acceptable,
- compatible temporal relationship for intended question.

### Flow

```text
Optical/MS ─► optical preprocessing ─► optical evidence ─┐
                                                         │
SAR ────────► SAR preprocessing ─────► SAR evidence ─────┤
                                                         ▼
                                                 Fusion / reconciliation
                                                         │
                                                         ▼
                                                   fused evidence
                                                         │
                                                         ▼
                                            agreement / conflict analysis
                                                         │
                                                         ▼
                                                      verifier
                                                         │
                                                         ▼
                                                   answer composer
```

### Required audit

Where practical, store:

- optical-only result,
- SAR-only result,
- fused result.

---

## 10.4 `CHANGE_VQA`

### Preconditions

- two temporal observations,
- temporal order known,
- overlapping extent,
- alignment verified or repaired,
- model-compatible modalities.

### Flow

```text
T1 ─► preprocessing ─► temporal encoder ─┐
                                         ├─► change representation
T2 ─► preprocessing ─► temporal encoder ─┘
                                             ↓
                                      change evidence
                                             ↓
                                      question fusion
                                             ↓
                                         answer
                                             ↓
                                         verifier
```

### Output

- change answer,
- linked temporal evidence,
- optional change mask,
- warnings.

---

## 10.5 `CHANGE_LOCALIZE`

### Preconditions

Same as `CHANGE_VQA`.

### Flow

```text
T1 + T2
   ↓
temporal/change specialist
   ↓
target-specific change mask
   ↓
map to source/world coordinates
   ↓
verify
```

### Output

- change mask / polygons,
- change class,
- confidence,
- geographic location if valid.

---

## 10.6 `CHANGE_MEASURE`

### Preconditions

- all `CHANGE_LOCALIZE` requirements,
- valid measurement geometry,
- suitable CRS / geodesic computation path,
- valid mask.

### Flow

```text
T1 + T2
   ↓
change mask
   ↓
mask verification
   ↓
GIS measurement
   ↓
unit conversion
   ↓
physical verification
   ↓
answer composer
```

### Output

- change evidence,
- numeric measurement,
- unit,
- calculation provenance.

---

## 10.7 `METADATA_QUERY`

### Preconditions

- observation exists.

### Flow

```text
Query
  ↓
ObservationState
  ↓
metadata answer
```

No model inference is required for deterministic metadata questions.

---

# 11. Query routing

Routing must use both query intent and input state.

Conceptually:

```text
workflow =
    route(
        intent,
        observation_count,
        modalities,
        pair_compatibility,
        metadata_availability,
        registry_capabilities
    )
```

Examples:

| Inputs | Query | Route |
|---|---|---|
| 1 optical | "Is water present?" | `SINGLE_VQA` |
| 1 SAR | "Are structures visible?" | `SINGLE_VQA` with SAR-capable model |
| optical + SAR | "Do both sensors support flooding?" | `CROSS_MODAL_VQA` |
| T1 + T2 | "Where did water expand?" | `CHANGE_LOCALIZE` |
| T1 + T2 | "How many hectares changed?" | `CHANGE_MEASURE` |
| 1 image | "What changed?" | fail: `MISSING_TEMPORAL_PAIR` |
| RGB only | "Compute NDVI" | fail: `MISSING_REQUIRED_BAND` |

---

# 12. Model registry

The model registry is the canonical source of model capabilities.

Example:

```yaml
models:

  rs_vqa_v1:
    kind: vlm
    version: 1.0.0

    checkpoint:
      path: models/rs_vqa_v1
      sha256: "..."

    supported_tasks:
      - SINGLE_VQA
      - CROSS_MODAL_VQA

    supported_modalities:
      - optical
      - multispectral
      - sar

    training_domain:
      sensors:
        - Sentinel-1
        - Sentinel-2
      geography:
        - mixed
      resolution_notes: "..."

    preprocessing_profile: rs_vqa_v1

    outputs:
      - text_answer

    calibration:
      id: rs_vqa_cal_v1

    resource_profile:
      device: cuda
      expected_vram_gb: null

  grounding_v1:
    kind: grounding
    version: 1.0.0
    supported_tasks:
      - GROUND_OBJECT
    supported_modalities:
      - optical
    outputs:
      - bbox
```

The planner chooses only registered models.

---

# 13. Tool registry

Deterministic scientific tools are registered separately.

Example:

```yaml
tools:

  compute_mask_area:
    version: 1.0.0

    input:
      - georeferenced_mask

    preconditions:
      - valid_mask_geometry
      - geospatial_area_computable

    parameters:
      output_unit:
        enum:
          - m2
          - ha
          - km2

    output:
      - measurement

  compute_ndvi:
    version: 1.0.0

    preconditions:
      - RED band exists
      - NIR band exists

    output:
      - raster
```

---

# 14. Preprocessing registry

Preprocessing is part of model provenance.

Example:

```yaml
profiles:

  generic_optical_v1:
    modality: optical
    nodata_policy: mask
    tile_size: 512
    overlap: 64
    normalization: percentile

  s1_croma_v1:
    modality: sar
    channels:
      - VV
      - VH
    normalization: model_specific

  bigearthnet_s2_v1:
    modality: multispectral
    band_policy: model_specific
    normalization: model_specific
```

A model cannot silently change preprocessing without versioning the profile.

---

# 15. Coordinate systems and geometry

This section is non-negotiable.

## 15.1 Three coordinate spaces

SatQuery distinguishes:

### Model / crop coordinates

Coordinates inside a resized tile / crop.

### Source raster pixel coordinates

Coordinates in the original raster grid.

### World coordinates

Coordinates produced through the source affine transform and CRS.

The relationship is:

```text
model/crop pixel
      ↓
crop-to-source transform
      ↓
source pixel
      ↓
affine transform
      ↓
world coordinate
```

---

## 15.2 Resizing

If a crop is resized before inference, the reverse scale must be preserved.

Example:

```text
source crop: 1024 × 1024
model input: 448 × 448
```

A predicted point `(xm, ym)` maps back through known scale factors before applying the crop offset.

---

## 15.3 Tiling

Each tile stores:

```yaml
TileState:
  source_observation_id: string
  row_offset: integer
  col_offset: integer
  width: integer
  height: integer
  source_transform: [...]
  tile_transform: [...]
  preprocessing_profile: string
```

Predictions from overlapping tiles are reconciled before final evidence is committed.

---

## 15.4 Area computation

For a projected raster with square / rectangular pixels:

\[
A_p = |\Delta x \times \Delta y|
\]

For a binary mask:

\[
A = \sum_{x,y} M(x,y) \times A_p
\]

For geographic CRS in angular units, SatQuery must use an appropriate projected / equal-area or geodesic calculation path.

Degrees must never be treated directly as meters.

---

# 16. Raster ingestion pipeline

```text
UPLOAD
  ↓
QUARANTINE
  ↓
FILE TYPE / DRIVER DETECTION
  ↓
RESOURCE LIMIT CHECK
  ↓
METADATA INSPECTION
  ↓
SCIENTIFIC VALIDATION
  ↓
IMMUTABLE ORIGINAL STORAGE
  ↓
VISUALIZATION DERIVATIVE
  ↓
OBSERVATION REGISTRATION
```

---

## 16.1 Supported initial drivers

The competition build should prefer a narrow allow-list such as:

- GeoTIFF / GTiff,
- PNG,
- JPEG,

plus only explicitly required formats.

User-supplied VRT should be rejected in the initial system.

---

## 16.2 Limits

Configure:

- max upload bytes,
- max raster width,
- max raster height,
- max band count,
- max total pixel count,
- processing timeout,
- max derived output size.

---

# 17. Raster visualization

Large rasters should be served as tiles rather than transferred whole.

```text
Scientific raster
      ↓
COG / raster source
      ↓
rio-tiler / TiTiler
      ↓
XYZ tile endpoint
      ↓
OpenLayers
```

Dense evidence masks should normally be rendered as raster overlays.

Bounding boxes, points, selected polygons, and AOIs can use vector layers.

---

# 18. Optical / multispectral pipeline

```text
ObservationState
      ↓
validate modality
      ↓
validate required bands
      ↓
apply optical preprocessing profile
      ↓
window / tile
      ↓
optical encoder / VLM
      ↓
postprocess
      ↓
Evidence
```

The analysis input remains distinct from the RGB visualization asset.

A multispectral model must not silently consume an 8-bit RGB preview.

---

# 19. SAR pipeline

```text
ObservationState
      ↓
validate SAR metadata
      ↓
polarization availability
      ↓
SAR preprocessing profile
      ↓
SAR encoder / specialist
      ↓
postprocess
      ↓
Evidence
      ↓
domain-shift verification
```

The system avoids universal claims such as:

- "water is always dark",
- "urban is always bright",

because SAR response depends on acquisition geometry, wavelength, polarization, roughness, moisture, and target structure.

---

# 20. Cross-modal fusion

SatQuery treats fusion as an auditable process.

```text
Optical input
    ↓
Optical encoder
    ↓
Optical evidence ─────────────┐
                              │
                              ▼
                         Fusion module
                              ▲
                              │
SAR input                     │
    ↓                         │
SAR encoder                   │
    ↓                         │
SAR evidence ─────────────────┘
                              ↓
                         Fused evidence
```

Recommended inference record:

```yaml
FusionAudit:
  optical_result: evidence_id
  sar_result: evidence_id
  fused_result: evidence_id

  agreement:
    status: strong|moderate|conflicting|unknown

  support:
    optical: supported|unsupported|uncertain
    sar: supported|unsupported|uncertain
```

Do not output arbitrary "percentage contribution" without validated attribution.

---

# 21. Temporal / change pipeline

```text
T1
 ↓
validate time
 ↓
preprocess
 ↓
temporal encoder ───────┐
                        │
                        ▼
                   change module
                        ▲
                        │
T2                      │
 ↓                      │
validate time           │
 ↓                      │
preprocess              │
 ↓                      │
temporal encoder ───────┘
                        ↓
                  change evidence
                        ↓
                     verifier
                        ↓
                     language
```

Required checks:

- same / overlapping area,
- known temporal order,
- acceptable alignment,
- compatible analysis grid,
- sensor/model compatibility,
- NoData handling,
- domain status.

---

# 22. Change evidence

Preferred change output:

```yaml
ChangeEvidence:
  change_type: string
  target_class: string|null
  mask_asset_id: string
  score: number|null

  temporal:
    t1_observation_id: string
    t2_observation_id: string

  model:
    id: string
    version: string

  geometry:
    source_grid: string
    world_mapping_valid: boolean
```

Language is generated after this evidence exists.

---

# 23. Measurement pipeline

Example query:

> "How much water was added?"

```text
water-gain mask
      ↓
validate mask
      ↓
validate area computation path
      ↓
count positive pixels / geometry area
      ↓
convert units
      ↓
MeasurementEvidence
      ↓
answer composer
```

Example measurement record:

```yaml
Measurement:
  name: water_gain_area
  value: 3.142
  unit: ha

  source_evidence_id: ev_change_001

  method:
    tool: compute_mask_area
    version: 1.0.0

  geo:
    calculation_crs: EPSG:xxxx
```

---

# 24. Verification architecture

The verifier does not produce chain-of-thought.

It runs explicit checks.

---

## 24.1 Geometric verifier

Checks may include:

- CRS present where required,
- CRS transformable,
- extent overlap,
- grid alignment,
- mask dimensions,
- crop/source mapping,
- world-coordinate bounds,
- valid area calculation path.

---

## 24.2 Temporal verifier

Checks may include:

- dates known,
- order valid,
- no accidental duplicate observation,
- requested "before/after" relationship supported,
- time gap plausible,
- seasonality warning if relevant.

---

## 24.3 Physical verifier

Checks may include:

- required bands exist,
- SAR polarization assumptions supported,
- index output within expected mathematical range,
- measurement units valid,
- impossible operation rejected.

---

## 24.4 Provenance verifier

Checks may include:

- source observation IDs present,
- input hashes known,
- model version known,
- checkpoint hash known,
- preprocessing version known,
- tool version known,
- intermediate evidence linked.

---

## 24.5 Statistical verifier

Checks may include:

- calibrated confidence where available,
- OOD / sensor shift warning,
- modality disagreement,
- abnormal output distribution,
- low-confidence / abstention policy.

---

# 25. Answer policy

The answer composer receives:

- original user query,
- approved evidence,
- approved measurements,
- warnings,
- verification report.

It does **not** receive permission to invent additional measurements.

---

## 25.1 Example: successful answer

Evidence:

```yaml
target: water_gain
area: 3.142 ha
location: southern AOI
optical_support: supported
sar_support: supported
geometry: PASS
temporal: PASS
```

Possible answer:

> Water extent increased mainly in the southern part of the area of interest. The verified change mask corresponds to approximately **3.14 hectares** of additional water coverage. Both optical and SAR analyses support the change.

---

## 25.2 Example: disagreement

Evidence:

```yaml
optical: supported
sar: unsupported
optical_cloud_warning: true
agreement: conflicting
```

Possible answer:

> The evidence is inconclusive. The optical observation suggests possible flooding, but the SAR result does not corroborate it. The optical scene also contains significant cloud contamination, so the result should be treated with caution.

---

## 25.3 Example: unsupported request

Query:

> "What changed?"

Inputs:

```text
1 observation
```

Response:

> A second observation of the same area from another time is required for change analysis.

No model is called.

---

# 26. Confidence architecture

SatQuery initially keeps confidence dimensions separate.

```yaml
ConfidenceStatus:
  model_confidence:
    value: number|null
    calibrated: boolean

  data_validity:
    status: high|medium|low|unknown

  domain_status:
    status: in_domain|shifted|unknown

  modality_agreement:
    status: strong|moderate|conflicting|unknown

  geometric_validity:
    status: PASS|WARN|FAIL

  temporal_validity:
    status: PASS|WARN|FAIL
```

A single "overall confidence percentage" should not be introduced until an aggregation method is validated against held-out correctness data.

---

# 27. Domain-shift handling

Model registry entries define their known training domains.

When an observation falls outside them, SatQuery records:

```yaml
domain:
  status: shifted
  reasons:
    - sensor_not_seen_in_training
    - resolution_outside_validation_range
```

Possible policy:

- allow inference,
- lower trust,
- require warning,
- route to a more sensor-general model if available,
- abstain for highly sensitive calculations if evidence is unreliable.

Cross-sensor performance must be measured explicitly during evaluation.

---

# 28. Execution trace

The execution trace records **operations**, not private reasoning.

Example:

```yaml
analysis_id: analysis_42

steps:

  - step: inspect_inputs
    tool: raster_inspector
    version: 1.0.0
    status: COMPLETED

  - step: validate_pair
    tool: pair_validator
    version: 1.0.0
    status: COMPLETED

  - step: classify_query
    component: query_interpreter
    output: CHANGE_MEASURE

  - step: run_change_model
    model: change_v3
    version: 3.1.0
    parameters:
      target: water

  - step: calculate_area
    tool: compute_mask_area
    parameters:
      output_unit: ha

  - step: verify
    output:
      geometry: PASS
      temporal: PASS
      physical: PASS
      provenance: PASS
      statistical: WARN
```

---

# 29. Persistence

## 29.1 Immutable scientific assets

Store:

```text
/data/observations/<observation_id>/original.tif
```

Original inputs are immutable.

---

## 29.2 Derived assets

Examples:

```text
analysis_ready.tif
web.cog.tif
preview.png
mask.tif
overlay.png
polygon.geojson
```

Each derived asset must record parent provenance.

---

## 29.3 Metadata database

Initial implementation may use SQLite.

Suggested logical tables:

- `observations`
- `assets`
- `analyses`
- `jobs`
- `evidence`
- `measurements`
- `execution_events`
- `model_versions`
- `verification_reports`

PostgreSQL can replace SQLite if concurrency / scale requires it.

---

# 30. Caching

Evidence-producing operations may be cached.

Cache identity should include:

```text
input hash
+
model ID
+
model version
+
checkpoint hash
+
preprocessing version
+
task
+
parameters
```

Example cache key:

```text
sha256(
  source_sha256
  + model_checkpoint_sha256
  + preprocessing_profile
  + task
  + canonical_parameters
)
```

Never reuse a result across different checkpoints or preprocessing profiles without an explicit equivalence guarantee.

---

# 31. API boundaries

The exact endpoints may evolve, but the API contract should remain resource-oriented.

---

## 31.1 Upload observation

```http
POST /api/observations
```

Response:

```json
{
  "observation_id": "obs_123",
  "status": "READY",
  "metadata": {},
  "warnings": []
}
```

---

## 31.2 Create analysis

```http
POST /api/analyses
```

Request:

```json
{
  "observation_ids": ["obs_pre", "obs_post"],
  "query": "Where did water increase and by how much?"
}
```

Response:

```json
{
  "analysis_id": "analysis_123",
  "job_id": "job_123",
  "status": "QUEUED"
}
```

---

## 31.3 Job status

```http
GET /api/jobs/{job_id}
```

---

## 31.4 Analysis result

```http
GET /api/analyses/{analysis_id}
```

---

## 31.5 Evidence

```http
GET /api/evidence/{evidence_id}
```

---

## 31.6 Raster tiles

```http
GET /tiles/{asset_id}/{z}/{x}/{y}.png
```

---

## 31.7 Export report

```http
GET /api/reports/{analysis_id}
```

---

# 32. Job architecture

Analyses are asynchronous jobs.

Suggested states:

```text
QUEUED
VALIDATING
PREPROCESSING
RUNNING_MODEL
POSTPROCESSING
RUNNING_GIS
VERIFYING
COMPOSING
COMPLETED
FAILED
```

The API remains responsive while the GPU worker executes inference.

---

# 33. GPU worker architecture

```text
worker process
   ↓
load common model(s)
   ↓
keep hot models resident
   ↓
receive inference job
   ↓
load raster window / tensor
   ↓
apply preprocessing profile
   ↓
run inference
   ↓
postprocess
   ↓
emit canonical Evidence
```

Training code and optimizer state do not belong in the production worker.

---

# 34. CPU/GPU responsibility split

## CPU

Prefer CPU for:

- metadata inspection,
- window planning,
- raster overlap,
- CRS / coordinate math,
- mask area,
- distance,
- polygon operations,
- database,
- provenance,
- report rendering.

## GPU

Use GPU for:

- VLM inference,
- ViT / EO encoders,
- SAR models,
- fusion,
- segmentation,
- grounding,
- change detection.

---

# 35. Security architecture

Remote-sensing files are untrusted input.

The ingestion path is therefore sandboxed.

---

## 35.1 Quarantine

Uploads first enter a restricted directory / process.

No scientific analysis occurs before identification and validation.

---

## 35.2 Driver allow-list

Do not trust extension alone.

The initial system should permit only required / audited raster drivers.

User-supplied VRT should be rejected.

---

## 35.3 Resource controls

Raster inspection processes should have limits for:

- memory,
- CPU,
- time,
- output size,
- disk usage.

---

## 35.4 Filesystem isolation

Raster workers should not be able to read:

- SSH keys,
- unrelated host directories,
- model service secrets,
- database admin credentials.

---

## 35.5 Network isolation

Where possible, ingestion / GDAL workers handling untrusted uploads should have restricted outbound network access.

---

## 35.6 No arbitrary shell / Python execution

The orchestrator cannot run free-form user or LLM-generated shell commands.

GIS actions are typed functions from the tool registry.

---

# 36. Error taxonomy

Suggested error codes:

```text
INVALID_UPLOAD
UNSUPPORTED_RASTER_DRIVER
RASTER_RESOURCE_LIMIT_EXCEEDED
RASTER_METADATA_INVALID

UNKNOWN_MODALITY
UNSUPPORTED_MODALITY
MISSING_REQUIRED_BAND
UNKNOWN_SAR_POLARIZATION

MISSING_TEMPORAL_PAIR
TEMPORAL_ORDER_UNKNOWN
NO_SPATIAL_OVERLAP
PAIR_ALIGNMENT_INVALID
CRS_REQUIRED_FOR_MEASUREMENT

MODEL_INPUT_UNSUPPORTED
MODEL_EXECUTION_FAILED
LOW_EVIDENCE_CONFIDENCE
OUT_OF_DOMAIN_WARNING

INVALID_EVIDENCE_GEOMETRY
GIS_OPERATION_FAILED
VERIFICATION_FAILED
```

Errors should be structured and user-facing messages should be generated from known error semantics.

---

# 37. Failure policy

| Condition | Policy |
|---|---|
| one image + change query | request second observation |
| no NIR + NDVI request | fail |
| no CRS + semantic VQA | allow with warning |
| no CRS + area measurement | fail |
| non-overlapping pair | fail |
| unknown temporal order | request / fail until resolved |
| unknown SAR polarization | allow only generic supported analysis |
| invalid alignment | repair through allowed workflow or fail |
| strongly OOD sensor | warn / route / abstain according to policy |
| optical-SAR disagreement | expose conflict |
| low-confidence evidence | qualify or abstain |
| invalid mask | no measurement |
| model crash | analysis fails; no fallback fabrication |

---

# 38. Frontend architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│ SatQuery            Analysis / Dataset                    Export    │
├────────────────┬───────────────────────────────┬───────────────────┤
│ INPUTS         │                               │ QUERY / RESULT    │
│                │                               │                   │
│ Optical        │                               │ Ask...            │
│ SAR            │            MAP                │                   │
│ Time           │                               │ Answer            │
│ CRS / GSD      │                               │ Evidence          │
│ Layers         │                               │ Confidence        │
│                │                               │ Warnings          │
├────────────────┴───────────────────────────────┴───────────────────┤
│ Evidence | Execution | Technical Details | Export                 │
└────────────────────────────────────────────────────────────────────┘
```

The map is the primary visual surface.

---

# 39. Map modes

## 39.1 Temporal

- side-by-side,
- swipe,
- opacity,
- flicker,
- change overlay.

## 39.2 Multimodal

- optical,
- SAR,
- fusion,
- evidence,
- derived mask.

## 39.3 Evidence interaction

Clicking evidence should:

- highlight corresponding map region,
- zoom to region,
- reveal model / source,
- reveal confidence / domain status,
- reveal measurements.

---

# 40. Report architecture

Report generation consumes structured analysis data.

Suggested sections:

1. Query
2. Source observations
3. Observation metadata
4. Analysis result
5. Spatial evidence
6. Measurements
7. Confidence / domain status
8. Verification
9. Models / tools used
10. Execution summary
11. Warnings / limitations

The LLM may write explanatory prose.

Numbers, model IDs, metadata, and measurements are inserted directly from structured records.

---

# 41. Repository boundaries

Suggested structure:

```text
satquery/
│
├── apps/
│   ├── web/
│   └── api/
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
│   ├── ingestion/
│   ├── geo/
│   ├── routing/
│   ├── evidence/
│   ├── models/
│   └── integration/
│
├── data/
├── docker/
├── docs/
└── docker-compose.yml
```

---

# 42. Testing architecture

SatQuery requires four testing layers.

---

## 42.1 Deterministic unit tests

Must cover:

- pixel → world,
- world → pixel,
- crop → source mapping,
- area conversion,
- CRS transform,
- raster overlap,
- pair compatibility,
- band prerequisites,
- temporal order,
- evidence geometry,
- measurement provenance.

---

## 42.2 Model regression tests

Maintain fixed evaluation sets for:

- single-image VQA,
- grounding,
- SAR,
- cross-modal,
- temporal change,
- cross-sensor cases.

Every approved checkpoint should run through the same evaluation harness.

---

## 42.3 Orchestration tests

Examples:

```text
1 optical + "What changed?"
→ MISSING_TEMPORAL_PAIR
```

```text
RGB + "Compute NDVI"
→ MISSING_REQUIRED_BAND
```

```text
optical + SAR + comparison question
→ CROSS_MODAL_VQA
```

```text
T1 + T2 + measurement
→ CHANGE_MEASURE
```

---

## 42.4 Failure / adversarial tests

Test:

- corrupt TIFF,
- fake extension,
- giant raster,
- excessive band count,
- all-NoData raster,
- missing CRS,
- mismatched CRS,
- non-overlap,
- wrong temporal order,
- blank image,
- constant SAR,
- unknown sensor,
- unknown polarization,
- cloud-heavy optical scene,
- misregistration,
- OOD sensor,
- low-confidence outputs.

---

# 43. Evaluation architecture

Metrics are capability-specific.

| Capability | Primary metrics | Critical controls |
|---|---|---|
| VQA | accuracy | blank/shuffled-image |
| MCQ | accuracy | answer-position bias |
| Grounding | mIoU / Acc@IoU | object size |
| Detection | mAP | per-class / scale |
| Segmentation | mIoU / F1 | rare classes |
| Change | F1 / IoU | T1+T1 |
| Change VQA | accuracy | linked mask correctness |
| Numeric | MAE / relative error | CRS / unit validity |
| Fusion | task score | O vs S vs O+S |
| Calibration | ECE / NLL | reliability diagram |
| Cross-sensor | task metric | degradation |
| Router | routing accuracy | invalid-request rate |
| Workflow | validity | intermediate checks |

---

# 44. Cross-sensor robustness architecture

SatQuery must not assume that Sentinel-trained models generalize to Cartosat / RISAT or any unseen sensor.

Adaptation should be escalated gradually:

```text
frozen encoder
   ↓
new projection
   ↓
sensor adapter
   ↓
LoRA / PEFT
   ↓
partial unfreezing
   ↓
full fine-tuning only if necessary
```

The selected level should be driven by cross-sensor evaluation.

---

# 45. Model development strategy

Initial candidate families:

### Multisensor VLM

RS-InternVL-style architecture or stronger experimentally validated successor.

### SAR-optical representation

CROMA / AnySat-style representations or adapted alternatives.

### High-resolution VQA / grounding

VRSBench-trained or compatible remote-sensing VLM / grounding specialist.

### Temporal change

CDVQA / SECOND-derived specialist or stronger validated change model.

### High-resolution SAR

SpaceNet-like proxy training / adaptation where useful.

No model name is architecturally permanent.

The evidence and interface contracts are the stable layer.

---

# 46. Training / inference separation

Training pipeline:

```text
dataset
→ transforms
→ batching
→ forward
→ loss
→ backward
→ optimizer
→ validation
→ checkpoint
→ evaluation
```

Production pipeline:

```text
approved checkpoint
→ approved preprocessing
→ inference
→ postprocess
→ evidence
→ verification
```

Training code must not be required in production.

---

# 47. Observability

Record per analysis:

- upload time,
- metadata time,
- preprocessing time,
- model inference time,
- GIS time,
- verification time,
- answer composition time,
- total latency,
- peak CPU memory,
- peak VRAM,
- selected model,
- selected workflow,
- cache hit/miss.

This allows bottlenecks to be measured rather than guessed.

---

# 48. Deployment topology

Initial competition / local topology:

```text
┌──────────────┐
│ web          │
└──────┬───────┘
       │
┌──────▼───────┐
│ FastAPI API  │
│ + GIS        │
└──────┬───────┘
       │
       ├──────────── shared data ─────────────┐
       │                                      │
┌──────▼────────┐                     ┌───────▼───────┐
│ GPU worker    │                     │ SQLite        │
└───────────────┘                     └───────────────┘
```

Containerized with Docker Compose.

Scale-out components such as Redis, Celery, PostgreSQL, multiple workers, or object storage should be added only when required.

---

# 49. Non-goals for the initial architecture

The initial SatQuery architecture intentionally does **not** depend on:

- Kubernetes,
- Kafka,
- a vector database,
- unrestricted autonomous browsing,
- arbitrary Python execution,
- arbitrary shell execution,
- multi-agent swarms,
- an RL-trained planner,
- training a new LLM from scratch,
- training a giant EO foundation model from scratch,
- converting all modalities to RGB,
- LLM-computed physical measurements.

---

# 50. Architecture invariants

The following should remain true regardless of model changes.

1. Original scientific observations remain immutable.
2. CRS / resolution / extent / time / modality / provenance remain available throughout analysis.
3. Missing sensor information is not invented.
4. A task is validated before model execution.
5. Required bands are checked before band-dependent operations.
6. Temporal analysis requires valid temporal observations.
7. Cross-modal analysis requires compatible spatial observations.
8. Model outputs become structured evidence before final explanation.
9. Measurements come from deterministic geospatial operations.
10. Spatial evidence can be mapped back to source coordinates.
11. Model / tool / preprocessing versions are recorded.
12. Domain shift is surfaced instead of silently ignored.
13. Fusion is tested against unimodal baselines.
14. Verification is independent from final language generation.
15. Execution traces contain operational facts, not chain-of-thought.
16. Unsupported claims are refused rather than guessed.

---

# 51. Architectural decision summary

## Decision: modular specialists over one monolithic VLM

**Reason:** separates semantic, spatial, temporal, and deterministic failure modes.

---

## Decision: structured EO state

**Reason:** remote-sensing validity depends on CRS, scale, extent, time, modality, and provenance.

---

## Decision: deterministic GIS measurements

**Reason:** area, distance, reprojection, and unit conversion do not require generative AI.

---

## Decision: constrained planner

**Reason:** reduces invented tools, illegal operations, and unreproducible workflows.

---

## Decision: evidence contract

**Reason:** stabilizes the system while models evolve.

---

## Decision: separate confidence dimensions

**Reason:** raw model probabilities do not represent all sources of scientific uncertainty.

---

## Decision: imagery-first UI

**Reason:** remote-sensing evidence should remain visually inspectable.

---

## Decision: model / tool registries

**Reason:** capabilities, requirements, versions, and limitations must be explicit.

---

# 52. End-to-end example

User uploads:

```text
pre_flood.tif
post_flood.tif
```

User asks:

> "Where did water increase and by how much?"

Execution:

```text
1. ingest both rasters

2. build ObservationState for both

3. determine T1 / T2

4. validate overlap / CRS / grid

5. parse:
   intent = CHANGE_MEASURE
   target = water
   output = mask + area + explanation

6. choose temporal-change workflow

7. preprocess T1 / T2

8. run change specialist

9. produce water-gain mask

10. map mask into source/world coordinates

11. geometrically verify mask

12. calculate area deterministically

13. run physical / temporal / provenance verification

14. compose answer from verified evidence

15. store execution trace

16. return:
    - answer
    - change overlay
    - area
    - confidence status
    - warnings
    - execution summary
```

Possible response:

> Water extent increased primarily along the southern floodplain. The verified change mask covers approximately **3.14 hectares**. The geometry and temporal pairing passed validation; the result includes the change overlay and full execution summary.

---

# 53. Final architecture definition

SatQuery can be represented as:

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

Operationally:

```text
UNDERSTAND SENSOR
       ↓
UNDERSTAND QUERY
       ↓
VALIDATE FEASIBILITY
       ↓
SELECT WORKFLOW
       ↓
SELECT SPECIALIST
       ↓
PROCESS DATA CORRECTLY
       ↓
PRODUCE EVIDENCE
       ↓
RUN REQUIRED GIS OPERATIONS
       ↓
VERIFY
       ↓
EXPLAIN
```

That is the architecture SatQuery should protect even as individual models, datasets, and deployment technologies evolve.

---

# 54. References

Core research informing this architecture includes:

- **BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation**
- **Survey of Multimodal Geospatial Foundation Models: Techniques, Applications, and Challenges**
- **Agentic AI for Remote Sensing: Technical Challenges and Research Directions**
- **Contrastive Radar-Optical Masked Autoencoders (CROMA)**
- **AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**
- **VRSBench**
- **RSVQA**
- **Change Detection Meets Visual Question Answering (CDVQA)**
- **LoRA: Low-Rank Adaptation of Large Language Models**
- **Visual Instruction Tuning (LLaVA)**
- **InternVL**
- **CLIP**

Important evidence-status note:

- BigEarthNet.txt is currently used as a highly relevant recent preprint.
- Agentic AI for Remote Sensing is a position paper and is used as architectural guidance.
- The exact SatQuery service architecture, schemas, registries, workflows, and failure policy in this document are our engineering design.

---

> **SatQuery architecture rule:**  
> **Evidence exists independently of language. The language layer explains verified evidence; it does not create the physical evidence itself.**
