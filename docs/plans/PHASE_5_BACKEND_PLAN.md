# Phase 5 — SatQuery Agent, Backend API, Evidence Engine, and Product Services

> **Status:** Planned. Start only after canonical Phase 4 closeout is scientifically accepted.
>
> **Intent:** Phase 5 becomes the **complete backend/product-service phase**: constrained natural-language orchestration, persistent analyses, job execution, evidence aggregation, artifacts, reports, system introspection, and a production-quality FastAPI/OpenAPI surface.
>
> **Phase 6 after this plan:** frontend integration, interaction design, end-to-end demo hardening, and presentation polish. Phase 6 should consume this backend rather than inventing backend contracts in the UI.

---

## 1. Why combine the backend into Phase 5?

The older roadmap separated:

```text
Phase 5 → Agent + Evidence Engine
Phase 6 → Product Integration
```

For the clean rebuild, the better boundary is:

```text
Phase 5
├── Agent
├── Evidence/verification
├── execution/jobs
├── persistence/cache
├── complete FastAPI API
├── Swagger/OpenAPI documentation
├── artifacts/tiles/reports
└── backend security/observability

Phase 6
├── frontend
├── map interaction
├── UI workflows
├── demo scripts
└── final product robustness
```

Reason: SatQuery already has a FastAPI transport boundary with observation upload, raster tiles, VQA, and grounding. Phase 5 should extend that existing backend instead of leaving the agent and the product API as two separate partially integrated systems.

---

# 2. Mission

Phase 5 turns the frozen Phase 1–4 scientific capabilities into one constrained backend:

```text
REMOTE-SENSING OBSERVATIONS
          +
NATURAL-LANGUAGE QUERY
          ↓
QUERY INTERPRETER
          ↓
FEASIBILITY VALIDATOR
          ↓
SENSOR / TASK ROUTER
          ↓
BOUNDED WORKFLOW PLANNER
          ↓
REGISTERED TOOLS / SPECIALISTS
          ↓
EVIDENCE AGGREGATOR
          ↓
VERIFICATION + MEASUREMENT
          ↓
PERSISTED ANALYSIS
          ↓
API / MAP / REPORT / EXPLANATION
```

The central invariant remains:

> **The language model may explain evidence. It must never manufacture evidence.**

---

# 3. Phase 5 non-negotiable rules

- Reuse the existing FastAPI application. Do not replace it with another framework.
- API code is a transport/application boundary; scientific behavior remains in `satquery/`.
- Do not duplicate Phase 1–4 model logic inside API routes.
- Only registered tools/models may execute.
- The planner cannot execute arbitrary Python, shell commands, SQL, GDAL strings, URLs, or file paths supplied by a query.
- Natural-language parsing may propose an intent; deterministic validation decides whether it is executable.
- Every analysis has immutable input IDs/hashes, workflow version, registry version/hash, evidence IDs, and execution trace.
- No hidden chain-of-thought is stored or returned. Store only operational trace and explicit decision reasons.
- No aggregate numerical confidence unless calibrated.
- Raw model scores remain raw model scores.
- Agreement IoU remains diagnostic unless a later calibrated policy explicitly proves a threshold.
- Numeric answers require deterministic measurement evidence.
- Unsupported sensor/model combinations must fail closed.
- Existing VQA/grounding APIs remain backward compatible or are explicitly deprecated with a migration period.
- Originals remain immutable.
- Derived rasters/evidence are versioned artifacts with provenance.
- Filesystem + SQLite remain the default storage architecture.
- Do not add Redis, Celery, Kafka, Kubernetes, a vector database, LangGraph, or a multi-agent framework without a demonstrated requirement.
- Long-running jobs must not block the entire API process.
- Model concurrency must be bounded to avoid CPU/GPU exhaustion.
- OpenAPI schema is a product contract and receives regression tests.
- Swagger UI is a first-class developer/testing surface, not an accidental FastAPI default.
- Offline/local deployment must be possible. Swagger/ReDoc assets should be self-hostable.
- Security limits are enforced before expensive raster/model operations.
- Do not expose local filesystem paths in public API responses.
- Every error uses a stable structured error contract.
- Phase 5 ends only when the frontend can build entirely against the published OpenAPI contract.

---

# 4. Existing backend to preserve

The current backend already provides:

```text
POST /api/observations
GET  raster tiles
POST /api/vqa
POST /api/grounding
```

and application state wires:

- `FilesystemObservationStore`
- `ObservationIngestionService`
- `RasterTileService`
- `SingleImageVqaService`
- `TextGuidedGroundingService`

Phase 5 extends this architecture instead of rewriting it.

---

# 5. Backend architecture

```text
                        FastAPI
                           │
           ┌───────────────┴────────────────┐
           │                                │
        API routes                     OpenAPI/docs
           │
     application services
           │
 ┌─────────┼───────────┬──────────────┬──────────────┐
 │         │           │              │              │
Observation Pair    Query/Agent    Analysis/Job   Artifact/Report
 Service    Service    Service        Service        Service
 │         │           │              │              │
 └─────────┴───────────┴──────┬───────┴──────────────┘
                              │
                     bounded execution engine
                              │
               ┌──────────────┼──────────────┐
               │              │              │
         deterministic      models        verification
            tools        / specialists     / measurement
               │              │              │
               └──────────────┴──────────────┘
                              │
                       evidence graph
                              │
             ┌────────────────┴────────────────┐
             │                                 │
          SQLite                           filesystem
  metadata / jobs / analyses        originals / derived assets
```

### Deployment shape

Default:

```text
one FastAPI process
+ bounded local worker executor
+ SQLite
+ filesystem
```

When a specialist requires an incompatible runtime, for example an isolated TensorFlow/Keras environment:

```text
FastAPI execution engine
        ↓
registered isolated worker adapter
        ↓
local subprocess/container
        ↓
validated evidence artifact
```

This is process isolation, not a microservice architecture.

---

# 6. API versioning and route structure

Introduce `/api/v1` as the canonical product API.

Keep old endpoints temporarily:

```text
/api/observations
/api/vqa
/api/grounding
```

as compatibility wrappers if required.

Canonical route groups:

```text
/api/v1/system
/api/v1/observations
/api/v1/pairs
/api/v1/query
/api/v1/analyses
/api/v1/jobs
/api/v1/tools
/api/v1/models
/api/v1/evidence
/api/v1/artifacts
/api/v1/reports
/api/v1/tiles
```

---

# 7. Proposed endpoint surface

## 7.1 System

```http
GET /health/live
GET /health/ready

GET /api/v1/system/version
GET /api/v1/system/capabilities
GET /api/v1/system/status
GET /api/v1/system/limits
```

`/health/live`:
- process is alive.

`/health/ready`:
- SQLite available;
- filesystem writable;
- registries parse;
- required production dependencies initialized;
- model availability summarized without forcing all huge models into memory.

`/capabilities` should return a dynamic capability matrix:

```json
{
  "single_image_vqa": {"status": "AVAILABLE"},
  "grounding": {"status": "AVAILABLE"},
  "structural_change": {"status": "AVAILABLE_WITH_LIMITS"},
  "change_captioning": {"status": "AVAILABLE"},
  "sar_flood": {"status": "MODEL_NOT_LOADED"},
  "ndvi": {"status": "AVAILABLE_IF_NIR"},
  "mask_area": {"status": "AVAILABLE_IF_CRS"}
}
```

Do not expose capability as available merely because code exists; use registry/runtime readiness.

---

## 7.2 Observations

```http
POST   /api/v1/observations
GET    /api/v1/observations
GET    /api/v1/observations/{observation_id}
GET    /api/v1/observations/{observation_id}/metadata
GET    /api/v1/observations/{observation_id}/statistics
GET    /api/v1/observations/{observation_id}/assets
DELETE /api/v1/observations/{observation_id}
```

Useful upload metadata:

- optional sensor/platform;
- acquisition timestamp;
- band semantic mapping;
- SAR polarization mapping;
- SAR radiometric domain;
- user label/description.

The API must distinguish:

```text
metadata read from file
metadata supplied by user
metadata inferred by deterministic known-source adapter
unknown metadata
```

Never silently merge them without provenance.

Observation listing filters:

- modality;
- sensor;
- date range;
- georeferenced yes/no;
- processing status;
- text label;
- pagination.

Deletion:
- logical deletion is preferred;
- reject deletion when required by immutable scientific evidence unless explicit cascade policy permits it;
- analyses retain hashes/provenance even if source is later removed.

Optional high-value endpoint:

```http
POST /api/v1/observations/batch
```

for multi-file drag/drop workflows.

---

## 7.3 Temporal / multimodal pairs

```http
POST /api/v1/pairs/validate
POST /api/v1/pairs
GET  /api/v1/pairs
GET  /api/v1/pairs/{pair_id}
```

Pair types:

```text
TEMPORAL
OPTICAL_SAR
GENERIC_COREGISTERED
```

Validation response should expose:

```json
{
  "compatible": false,
  "outcome": "REQUEST_INPUT",
  "checks": [
    {"check": "temporal_order", "status": "PASS"},
    {"check": "crs", "status": "PASS"},
    {"check": "overlap", "status": "PASS"},
    {"check": "grid", "status": "WARN"},
    {"check": "required_bands", "status": "FAIL"}
  ],
  "required_actions": ["provide NIR semantic mapping"]
}
```

This endpoint is valuable for Swagger testing and frontend diagnostics.

---

## 7.4 Registry introspection

```http
GET /api/v1/tools
GET /api/v1/tools/{tool_id}
GET /api/v1/models
GET /api/v1/models/{model_id}
```

Return only safe metadata:

- ID/version;
- task;
- domain;
- availability;
- supported sensors/modalities;
- input requirements;
- output evidence type;
- limitations.

Do not expose local checkpoint paths or secrets.

---

## 7.5 Main natural-language agent API

### Plan-only

```http
POST /api/v1/query/plan
```

Request:

```json
{
  "query": "How much vegetation disappeared after the event?",
  "observation_ids": ["obs_t1", "obs_t2"],
  "roi": null
}
```

Response:

```json
{
  "intent": "VEGETATION_LOSS_MEASUREMENT",
  "can_execute": true,
  "validation": {...},
  "workflow": [
    {
      "step": 1,
      "tool_id": "compute_ndvi_v1",
      "input_ids": ["obs_t1"]
    },
    {
      "step": 2,
      "tool_id": "compute_ndvi_v1",
      "input_ids": ["obs_t2"]
    },
    {
      "step": 3,
      "tool_id": "vegetation_loss_mask_v1"
    },
    {
      "step": 4,
      "tool_id": "compute_mask_area_v1"
    }
  ],
  "expected_evidence": ["mask", "measurement"],
  "warnings": []
}
```

This is a major debugging/demo feature: users can inspect what SatQuery *would* do before expensive execution.

### Execute

```http
POST /api/v1/query
```

Request includes:

- natural-language query;
- observation IDs or pair ID;
- optional ROI;
- optional execution mode;
- only bounded public options.

Response for long work:

```http
202 Accepted
```

with:

```json
{
  "analysis_id": "ana_...",
  "job_id": "job_...",
  "status": "QUEUED"
}
```

For fast deterministic work, the service may support:

```json
"execution_mode": "auto|sync|async"
```

but `auto` must have a deterministic policy.

---

# 8. Intent schema

Create a strict `QueryIntent`.

Initial intents should cover all demonstrated product capabilities, for example:

```text
SCENE_QUESTION
OBJECT_GROUNDING

SPECTRAL_INDEX
VEGETATION_STATUS
VEGETATION_LOSS
VEGETATION_GAIN
WATER_STATUS
WATER_EXPANSION
WATER_CONTRACTION

GENERIC_CHANGE_LOCALIZATION
CHANGE_DESCRIPTION

FLOOD_EXTENT
SAR_TEMPORAL_CHANGE

AREA_MEASUREMENT
DISTANCE_MEASUREMENT
COUNT_MEASUREMENT

OPTICAL_SAR_COMPARE

CAPABILITY_QUESTION
UNSUPPORTED
AMBIGUOUS
```

Do not create hundreds of intents.

Use composable fields:

```text
task_family
target_semantic
requested_measurement
temporal_direction
spatial_request
```

instead of exploding every query into a unique enum.

---

# 9. Query interpreter strategy

The agent must not depend entirely on an external LLM.

Use:

```text
1. deterministic parser / high-confidence patterns
2. optional structured language-model interpreter
3. strict Pydantic validation
4. deterministic feasibility validator
```

The language-model interpreter may produce only a `QueryIntent` candidate.

It cannot:
- select arbitrary Python;
- invent observation metadata;
- execute tools;
- bypass validation.

If no language model is configured, direct tool APIs and the deterministic interpreter must still work.

Any planner-language model should be audited separately before becoming mandatory. Phase 5 should not casually add a cloud dependency.

---

# 10. Feasibility validator

Implement one central validator before planning/execution.

Checks may include:

- observation count;
- temporal order;
- modality;
- sensor identity;
- bands;
- polarizations;
- radiometric domain;
- CRS;
- transform;
- spatial overlap;
- grid compatibility;
- GSD;
- NoData/valid region;
- ROI validity;
- model availability;
- model-domain compatibility;
- required measurement metadata;
- network/local dependency availability.

Result:

```text
ALLOW
ALLOW_WITH_WARNING
REQUEST_INPUT
ABSTAIN
REJECT
```

This validator should be callable independently from Swagger.

---

# 11. Bounded planner

Planner output is data, not executable code.

```python
ExecutionPlan:
    plan_id
    intent
    input_ids
    steps[]
    expected_evidence[]
    validation_snapshot
    registry_hash
    planner_version
```

Each step:

```python
PlanStep:
    step_id
    tool_id
    input_bindings
    bounded_parameters
    expected_output_type
    dependencies
```

Rules:

- tools must exist in `tools.yaml`;
- models must exist in approved registry;
- params validated against registry schema;
- no arbitrary filenames;
- no arbitrary URLs;
- no arbitrary command strings;
- maximum steps configured;
- no recursion;
- no planner-created tool IDs.

---

# 12. Workflow selection policies

High-value policies to encode explicitly:

### Quantitative requests

```text
query asks "how much / area / distance / count"
        ↓
must include deterministic measurement tool
```

A VLM/caption answer alone can never satisfy the request.

### Change requests

```text
two observations required
+ order known
+ overlap/compatibility validated
```

### Spectral requests

```text
semantic bands must exist
```

No NIR → no NDVI.

### SAR

Never force STURM onto generic/RISAT SAR.
Use learned specialist only when its exact domain contract passes.

### Optical + SAR

Use Phase 3 evidence:

- optical is normally the stronger standalone BENv2 classifier baseline;
- SAR is conditional complementary evidence;
- joint fusion had only marginal global benefit;
- disagreement is evidence requiring caution, not a reason to force fusion.

Do not assume multimodal fusion is always superior.

---

# 13. Tool execution engine

Create a common executor:

```text
ExecutionEngine
├── tool lookup
├── input binding
├── parameter validation
├── resource/concurrency gate
├── execution
├── artifact registration
├── evidence registration
└── trace event
```

Tool classes:

```text
DeterministicToolExecutor
ModelToolExecutor
IsolatedWorkerExecutor
```

The API does not care whether an approved specialist runs:
- in-process;
- in a local worker thread/process;
- in an isolated TensorFlow subprocess.

All return the same typed evidence boundary.

---

# 14. Jobs and long-running execution

Create a persistent job model.

Statuses:

```text
QUEUED
RUNNING
SUCCEEDED
FAILED
CANCEL_REQUESTED
CANCELLED
INTERRUPTED
```

Endpoints:

```http
GET  /api/v1/jobs/{job_id}
POST /api/v1/jobs/{job_id}/cancel
GET  /api/v1/jobs/{job_id}/events
```

Use Server-Sent Events (SSE) for progress unless a concrete bidirectional requirement appears.

Example event stream:

```text
job.queued
plan.validated
step.started
step.progress
artifact.created
step.completed
verification.started
analysis.completed
```

Do not stream private reasoning.

Persist enough state so a server restart can mark orphaned `RUNNING` jobs as `INTERRUPTED`.

Do not add Celery/Redis initially.

---

# 15. Resource control

Create a bounded resource manager.

Examples:

```text
VQA concurrency            1–N from config
Grounding concurrency      bounded
ChangerEx GPU concurrency  1 by default
Chg2Cap GPU concurrency    1 by default
STURM worker concurrency   1 by default
CPU analytics              bounded worker pool
```

Values are operational settings, not scientific constants.

Features:

- per-tool semaphore;
- timeout;
- memory-aware model loading;
- lazy load;
- optional unload after idle interval;
- cancellation checkpoints;
- clear `RESOURCE_BUSY` / `MODEL_UNAVAILABLE` states.

Avoid launching multiple large models simultaneously by accident.

---

# 16. Persistence

Keep:

```text
filesystem → raster/model/evidence artifacts
SQLite     → metadata and relationships
```

Suggested SQLite entities:

```text
observations
observation_assets
pairs
analyses
analysis_inputs
jobs
plan_steps
evidence
artifacts
measurements
execution_events
cache_entries
```

Do not store huge masks/blobs inside SQLite.

All scientific artifacts receive:
- artifact ID;
- SHA-256;
- media type;
- byte size;
- producer;
- parent evidence/input IDs;
- creation time;
- immutable/versioned flag.

---

# 17. Analysis lifecycle

```http
GET  /api/v1/analyses
GET  /api/v1/analyses/{analysis_id}
GET  /api/v1/analyses/{analysis_id}/evidence
GET  /api/v1/analyses/{analysis_id}/trace
GET  /api/v1/analyses/{analysis_id}/artifacts
POST /api/v1/analyses/{analysis_id}/rerun
```

Filters:

- date;
- task/intent;
- sensor/modality;
- status;
- source observation;
- warnings;
- model/tool.

`rerun` must default to:
- same inputs;
- same frozen workflow/config where still available;
- new analysis ID.

Never overwrite an old analysis.

---

# 18. Execution trace

Trace is operational and auditable:

```json
{
  "analysis_id": "ana_...",
  "intent": "WATER_EXPANSION",
  "validation": {...},
  "steps": [
    {
      "step_id": "s1",
      "tool_id": "compute_mndwi_v1",
      "version": "...",
      "input_ids": ["obs_t1"],
      "parameters": {...},
      "status": "SUCCEEDED",
      "duration_ms": 124
    }
  ]
}
```

Include:
- versions;
- model IDs/checkpoint hashes;
- tool IDs;
- allowed params;
- input/artifact hashes;
- warnings;
- durations.

Exclude:
- chain-of-thought;
- hidden prompts;
- secrets;
- local filesystem paths.

---

# 19. Evidence aggregation

The aggregator should create an evidence graph, not flatten everything into one string.

Possible nodes:

```text
VqaEvidence
GroundingEvidence
ChangeMaskEvidence
FloodMaskEvidence
ChangeCaptionEvidence
SpectralIndexEvidence
MeasurementEvidence
AgreementEvidence
DomainAssessment
VerificationEvidence
```

Relationships:

```text
derived_from
measured_from
supports
conflicts_with
describes
localized_by
```

This enables:
- map overlays;
- reports;
- traceability;
- frontend evidence inspection.

---

# 20. Verification layer

Before final analysis completion:

- ensure referenced artifacts exist/hash-match;
- check masks/grid provenance;
- verify measurement source evidence;
- verify quantitative claims match measurement values;
- check domain warnings;
- check no evidence references unknown input;
- ensure caption/VQA did not become the source of a numeric measurement;
- verify requested intent is actually answered.

Failure may yield:

```text
ALLOW_WITH_WARNING
ABSTAIN
REJECT
```

without deleting useful independent evidence.

---

# 21. Answer composition

Create:

```text
AnswerComposer
```

Input:
- query;
- validated intent;
- evidence graph;
- measurements;
- warnings/limitations.

Output:

```json
{
  "answer": "...",
  "support_status": "ALLOW_WITH_WARNING",
  "evidence_ids": [...],
  "measurement_ids": [...],
  "limitations": [...]
}
```

Core answer generation should be possible without a generative LLM using deterministic templates.

An optional language model may improve readability, but:
- receives only verified structured evidence;
- is forbidden from adding unsupported numeric/spatial claims;
- final response undergoes a claim consistency check.

---

# 22. Confidence / support semantics

Do not emit a fake:

```json
"confidence": 0.87
```

unless calibrated.

Instead expose separate evidence:

```json
{
  "raw_model_score": 0.71,
  "calibration_status": "NOT_CALIBRATED",
  "domain_status": "IN_DOMAIN",
  "agreement": {"metric": "mask_iou", "value": 0.62},
  "data_quality_warnings": []
}
```

Product-facing support state is:

```text
ALLOW
ALLOW_WITH_WARNING
REQUEST_INPUT
ABSTAIN
REJECT
```

No invented weighted confidence formula.

---

# 23. Direct tool execution API

Swagger/testing should not require the natural-language agent for every capability.

Provide:

```http
POST /api/v1/tools/{tool_id}/execute
```

The request schema is dynamically/explicitly constrained by the tool registry.

Benefits:
- scientific debugging;
- frontend development;
- judge demos;
- regression testing;
- isolation of routing vs model/tool quality.

Never allow a generic `python_code`, `command`, or free-form params field.

---

# 24. ROI support

Support optional GeoJSON ROI in analyses.

Rules:
- validate geometry;
- validate CRS/coordinate assumption;
- intersect with source extent;
- reject empty/out-of-bounds ROI;
- rasterize only on a verified source/common grid;
- keep ROI artifact/provenance.

Useful queries:

```text
How much vegetation was lost inside this polygon?
How much water covers this district?
Find change only in this region.
```

ROI measurement must use only masked pixels inside the verified ROI.

---

# 25. Raster statistics and preview features

High-value backend endpoints:

```http
GET /api/v1/observations/{id}/statistics
GET /api/v1/observations/{id}/histogram
GET /api/v1/observations/{id}/thumbnail
```

Statistics can include when scientifically meaningful:

- valid pixel fraction;
- band min/max/mean/std;
- percentiles;
- NoData fraction;
- raster dimensions;
- CRS/GSD;
- band/polarization semantics.

Do not compute misleading statistics across incompatible units without labeling them.

---

# 26. Artifact API

```http
GET /api/v1/artifacts/{artifact_id}
GET /api/v1/artifacts/{artifact_id}/download
```

Artifact types:

```text
original_raster
display_cog
thumbnail
mask_geotiff
index_geotiff
preview_png
geojson
report_json
report_html
report_pdf
```

Responses include metadata and provenance.

Download security:
- resolve only registered artifact IDs;
- never accept raw filesystem path;
- safe content disposition;
- range requests optional for large files.

---

# 27. Evidence GeoJSON

```http
GET /api/v1/evidence/{evidence_id}/geojson
```

Convert when possible:
- bbox → polygon;
- change/flood vectorization → polygons;
- ROI;
- grounded object polygons.

No CRS → do not invent lon/lat.
Pixel-space evidence should be explicitly marked pixel-space.

---

# 28. Tiles and overlays

Move/alias canonical tiles under:

```http
GET /api/v1/tiles/{asset_id}/{z}/{x}/{y}.png
```

Support:
- source visualization COG;
- derived index layers;
- masks;
- change overlays.

Optional query params, bounded and typed:

```text
opacity
colormap_id
min
max
```

Do not accept arbitrary colormap code/expression.

Keep pixel-space tile mode for ungeoreferenced benchmark imagery.

---

# 29. Report generation

Provide:

```http
GET /api/v1/analyses/{analysis_id}/report.json
GET /api/v1/analyses/{analysis_id}/report.html
GET /api/v1/analyses/{analysis_id}/report.pdf
```

JSON and HTML are mandatory.

PDF is enabled only with an audited dependency/runtime.

Report sections:

```text
query
inputs
sensor metadata
workflow
answer
evidence
map/preview links
measurements
model/tool provenance
warnings/limitations
verification
execution trace summary
reproducibility identifiers
```

Never include hidden reasoning.

---

# 30. Caching and deduplication

Use exact cache keys based on:

```text
tool/model ID + version
checkpoint/profile hash
input artifact hashes
bounded parameters
ROI hash
```

For natural-language agent runs include:

```text
normalized query
planner version
registry hash
```

Cache rules:
- deterministic tools are strongest cache candidates;
- cached evidence must retain original provenance;
- scientific code/profile change invalidates cache;
- never return stale evidence merely because filenames match.

---

# 31. Idempotency

High-value optional request header:

```http
Idempotency-Key
```

Useful for:
- uploads;
- long analysis submission.

Duplicate submissions with the same user-provided key and exact body/input hashes should return the original job/analysis rather than duplicate work.

---

# 32. Progress and cancellation

SSE endpoint:

```http
GET /api/v1/jobs/{job_id}/events
```

Cancellation:

```http
POST /api/v1/jobs/{job_id}/cancel
```

Cancellation is best effort:
- queued jobs cancel immediately;
- deterministic loops/model pipelines check cancellation between safe stages;
- external subprocess receives controlled termination;
- partial artifacts are not published as complete evidence.

---

# 33. Error contract

Standardize every API error:

```json
{
  "error": {
    "code": "MISSING_NIR_BAND",
    "message": "NDVI requires an NIR band semantic mapping.",
    "outcome": "REQUEST_INPUT",
    "details": {
      "observation_id": "obs_..."
    },
    "request_id": "req_..."
  }
}
```

Categories:

```text
INVALID_REQUEST
UNSUPPORTED_FORMAT
UPLOAD_TOO_LARGE
RASTER_LIMIT_EXCEEDED
OBSERVATION_NOT_FOUND
PAIR_INCOMPATIBLE
MISSING_REQUIRED_BAND
UNKNOWN_SAR_DOMAIN
MODEL_UNAVAILABLE
MODEL_INPUT_UNSUPPORTED
RESOURCE_BUSY
EXECUTION_TIMEOUT
ARTIFACT_NOT_FOUND
VERIFICATION_FAILED
INTERNAL_ERROR
```

Do not expose raw stack traces in normal API responses.

---

# 34. Swagger UI / OpenAPI — mandatory

FastAPI already provides OpenAPI and Swagger UI automatically, but Phase 5 should deliberately productize it.

Canonical docs:

```text
/docs          → Swagger UI, interactive testing
/redoc         → ReDoc, read-oriented documentation
/openapi.json  → machine-readable API contract
```

Use rich metadata:

- title;
- summary;
- full Markdown description;
- version;
- license;
- contact;
- server definitions when appropriate;
- ordered tags.

Suggested tags:

```text
System
Observations
Pairs
Query Agent
Analyses
Jobs
Tools
Models
Evidence
Artifacts
Reports
Tiles
Legacy
```

Every endpoint should have:
- summary;
- meaningful description;
- stable `operation_id`;
- request examples;
- success example;
- structured error responses;
- tags;
- response model;
- status code;
- deprecation marker when needed.

Swagger UI parameters worth enabling for development:

```text
displayRequestDuration
filter
persistAuthorization
tryItOutEnabled
```

For offline deployment, self-host Swagger/ReDoc JS/CSS assets rather than relying on a CDN.

The OpenAPI schema should be snapshot/regression-tested.

This also enables later generated TypeScript API clients for the frontend.

---

# 35. Swagger-specific demo examples

Include example requests that demonstrate SatQuery's range:

### VQA

```json
{
  "query": "Is water visible in this image?",
  "observation_ids": ["obs_optical"]
}
```

### Grounding

```json
{
  "query": "Locate the storage tanks.",
  "observation_ids": ["obs_optical"]
}
```

### Vegetation loss

```json
{
  "query": "How much vegetation disappeared between these dates?",
  "pair_id": "pair_temporal_1"
}
```

### Structural change

```json
{
  "query": "Where are new structures visible?",
  "pair_id": "pair_temporal_2"
}
```

### Change caption

```json
{
  "query": "Describe what changed.",
  "pair_id": "pair_temporal_2"
}
```

### Flood

```json
{
  "query": "Map probable flood/water extent.",
  "observation_ids": ["obs_s1_post"]
}
```

### Invalid sensor

```json
{
  "query": "Run Sentinel-1 flood detection.",
  "observation_ids": ["obs_risat"]
}
```

Expected:
`MODEL_INPUT_UNSUPPORTED`, not forced execution.

---

# 36. API security

For hackathon/local mode do not build user-account infrastructure unless required.

Implement:

- strict CORS allowlist from environment;
- maximum request/body/file limits;
- validated upload filenames;
- no raw path inputs;
- raster safety limits;
- safe archive handling where applicable;
- no arbitrary URLs in analysis requests;
- no arbitrary remote model IDs;
- security headers where practical;
- request IDs;
- optional API-key mode for deployed demos.

If optional API-key auth is enabled, integrate it into Swagger UI's OpenAPI security scheme so authorized testing works from `/docs`.

Do not hardcode credentials.

---

# 37. Rate/resource abuse protection

Without adding a distributed rate-limit stack:

- bound upload size;
- bound concurrent analyses;
- bound queued jobs;
- bound per-model concurrency;
- bound ROI complexity;
- bound planner steps;
- bound result size;
- bound request query length;
- timeout expensive operations.

Return structured `429` or `503` responses when overloaded.

---

# 38. Observability

Structured logging fields:

```text
timestamp
level
request_id
analysis_id
job_id
tool_id
model_id
observation_ids
duration_ms
status
error_code
```

Do not log:
- secrets;
- full uploaded data;
- hidden prompts/reasoning.

Metrics worth exposing if a lightweight dependency is approved:

```text
request count/latency
job queue depth
job success/failure
model load time
model inference latency
cache hit ratio
artifact bytes
```

Optional:

```http
GET /metrics
```

Prometheus is useful but not mandatory for SIH.

---

# 39. Health/readiness details

`/health/live` must remain cheap.

`/health/ready` should report components:

```json
{
  "ready": true,
  "components": {
    "database": "READY",
    "filesystem": "READY",
    "registry": "READY",
    "vqa": "AVAILABLE",
    "grounding": "AVAILABLE",
    "changerex": "NOT_LOADED",
    "chg2cap": "AVAILABLE",
    "sturm_worker": "AVAILABLE"
  }
}
```

Do not load multi-gigabyte models simply to answer health checks.

---

# 40. Model lifecycle

Create a model resource manager:

```text
registered
available_on_disk
loading
ready
failed
unloaded
```

Features:
- lazy loading;
- hash verification before load;
- one load per process;
- load failure cached briefly to prevent request storms;
- explicit retry;
- optional unload;
- concurrency semaphore.

Endpoint:

```http
GET /api/v1/models/{model_id}
```

may expose state and domain limitations.

Admin-style model load/unload endpoints are optional and should not be public by default.

---

# 41. Isolated specialist runtime

If Phase 4 closes with specialists requiring incompatible Python stacks:

```text
satquery/workers/
├── protocol.py
├── local.py
└── subprocess.py
```

Use a minimal typed request/response protocol.

Example request:

```json
{
  "request_id": "...",
  "model_id": "sturm_s1_unet",
  "input_artifact_ids": [...],
  "parameters": {...}
}
```

The worker returns:
- status;
- evidence artifact metadata;
- model provenance;
- logs/error code.

The parent verifies output hashes/contracts before accepting evidence.

Do not create a network microservice unless deployment proves it necessary.

---

# 42. ROI + geometry utilities

Add shared backend geometry validation:

- GeoJSON Polygon/MultiPolygon;
- coordinate bounds;
- self-intersection handling;
- maximum vertex count;
- optional simplification only when explicit;
- CRS contract;
- raster intersection.

Evidence exported to GeoJSON should state CRS/domain clearly.

---

# 43. Search/history

`GET /api/v1/analyses` should support pagination and filters.

Potential filters:

```text
intent
status
observation_id
pair_id
sensor
model_id
tool_id
created_after
created_before
has_warning
```

This gives the frontend a useful history panel nearly for free once persistence exists.

---

# 44. Reproducibility endpoint

High-value feature:

```http
GET /api/v1/analyses/{analysis_id}/reproducibility
```

Return:

```text
Git/application version
registry hash
input hashes
model/checkpoint identities
preprocessing profiles
tool versions
bounded parameters
artifact hashes
```

No local paths or secrets.

---

# 45. Dry-run / plan visualization

`POST /api/v1/query/plan` should be rich enough for frontend display:

```text
Query understood as...
Inputs valid...
Selected workflow...
Why this sensor/tool is applicable...
Expected outputs...
Warnings...
```

This is **operational explanation**, not hidden chain-of-thought.

It is especially valuable to judges because it visibly proves agentic routing.

---

# 46. Capability matrix feature

Generate a capability matrix dynamically from:
- tool registry;
- model registry;
- runtime availability;
- input requirements.

Example:

| Capability | Optical RGB | Multispectral | Sentinel-1 SAR | Generic SAR | Temporal |
|---|---:|---:|---:|---:|---:|
| VQA | ✓ | adapted view | limited | no | no |
| Grounding | ✓ | RGB view | no | no | no |
| NDVI | no | NIR required | no | no | optional |
| Change mask | ✓ domain-limited | maybe | no | no | ✓ |
| Change caption | ✓ domain-limited | RGB adapter | no | no | ✓ |
| Flood mask | no | no | exact contract | no | optional |
| SAR change | no | no | explicit | explicit mapped | ✓ |

Do not hardcode this table in the UI; derive it from backend contracts.

---

# 47. Request IDs and correlation

Every request receives:

```text
X-Request-ID
```

Analysis/job responses return IDs.

Logs, evidence, and trace use these IDs for debugging.

Clients may submit a valid request ID only if policy allows; otherwise generate server-side.

---

# 48. OpenAPI schema quality gate

Add automated tests that verify:

- `/docs`, `/redoc`, `/openapi.json` available in development mode;
- every `/api/v1` route has tags;
- every operation has unique stable `operation_id`;
- every write route documents at least one non-2xx structured error;
- response models are defined;
- no internal filesystem schema leaks;
- deprecated routes marked;
- security scheme appears only when enabled;
- schema generation has no duplicate model names;
- examples remain valid against Pydantic schemas.

Optionally save a reviewed `openapi.snapshot.json` in tests or generate/compare selected structural properties rather than freezing noisy ordering.

---

# 49. Generated frontend client

Because OpenAPI is authoritative, optionally generate a TypeScript client during Phase 6.

Phase 5 must make this possible by:
- stable operation IDs;
- stable schemas;
- API versioning;
- no ad-hoc untyped JSON responses.

Do not manually maintain duplicate TypeScript request/response interfaces if generation can replace them.

---

# 50. Backward compatibility

Do not break existing:

```text
POST /api/observations
POST /api/vqa
POST /api/grounding
GET  /tiles/...
```

without explicit migration.

Preferred:
- implement canonical `/api/v1`;
- make legacy routes thin wrappers;
- mark legacy routes deprecated in OpenAPI;
- remove only after Phase 6 if no client depends on them.

---

# 51. Suggested file map

## API

```text
apps/api/app/
├── main.py
├── dependencies.py
├── errors.py
├── openapi.py
├── schemas/
│   ├── common.py
│   ├── observations.py
│   ├── pairs.py
│   ├── query.py
│   ├── analyses.py
│   ├── jobs.py
│   ├── artifacts.py
│   └── system.py
├── routes/
│   ├── system.py
│   ├── observations.py
│   ├── pairs.py
│   ├── query.py
│   ├── analyses.py
│   ├── jobs.py
│   ├── tools.py
│   ├── models.py
│   ├── evidence.py
│   ├── artifacts.py
│   ├── reports.py
│   ├── tiles.py
│   └── legacy.py
└── services/
    ├── observations.py
    ├── pairs.py
    ├── analyses.py
    ├── jobs.py
    ├── artifacts.py
    └── reports.py
```

Do not mechanically split files unless it improves cohesion. Existing small route modules may remain.

## Scientific/application core

```text
satquery/
├── agent/
│   ├── intents.py
│   ├── interpreter.py
│   ├── validation.py
│   ├── planner.py
│   ├── policies.py
│   └── composer.py
├── execution/
│   ├── engine.py
│   ├── jobs.py
│   ├── resources.py
│   └── workers.py
├── persistence/
│   ├── database.py
│   ├── repositories.py
│   └── migrations.py
├── evidence/
├── verification/
├── reporting/
└── registry/
```

---

# 52. Phase 5 task plan

## Task 0 — Freeze Phase 5 backend contract

Create:
- `experiments/phase5_backend/README.md`
- `experiments/phase5_backend/backend_contract.yaml`
- `experiments/phase5_backend/api_contract.md`

Freeze:
- route families;
- error envelope;
- job states;
- analysis states;
- persistence entities;
- OpenAPI/docs policy;
- agent security boundaries;
- Phase 4 capabilities available at start.

**Stop for review before code.**

---

## Task 1 — FastAPI/OpenAPI foundation

Implement:
- `/api/v1`;
- app metadata;
- route tags;
- stable operation IDs;
- Swagger `/docs`;
- ReDoc `/redoc`;
- `/openapi.json`;
- self-hostable docs assets;
- request IDs;
- version endpoint;
- common response/error schemas.

Tests:
- OpenAPI generation;
- docs availability;
- unique operation IDs;
- error schema.

Commit:
```text
feat: establish versioned backend API contract
```

---

## Task 2 — Persistence foundation

Implement SQLite metadata repository:
- schema creation/version;
- observations;
- pairs;
- analyses;
- jobs;
- artifacts/evidence references.

Use filesystem for binaries.

Test:
- fresh DB;
- migrations;
- transaction rollback;
- uniqueness;
- restart persistence.

Commit:
```text
feat: add persistent analysis metadata store
```

---

## Task 3 — Observation API v1

Wrap existing ingestion safely.

Add:
- upload;
- list/filter/pagination;
- metadata;
- assets;
- stats;
- optional batch;
- safe deletion policy.

Preserve immutable original behavior.

Commit:
```text
feat: expose observation lifecycle API
```

---

## Task 4 — Pair API

Add:
- validation;
- creation;
- listing/get;
- temporal and optical/SAR pair types.

Persist validation snapshot.

Commit:
```text
feat: add validated observation pair workflows
```

---

## Task 5 — System capability/registry API

Add:
- tools;
- models;
- capabilities;
- limits;
- runtime availability.

Do not expose paths/secrets.

Commit:
```text
feat: expose runtime capability registry
```

---

## Task 6 — Agent intent contracts and interpreter

Implement:
- composable `QueryIntent`;
- deterministic high-confidence parser;
- interpreter protocol for optional structured language model;
- adversarial/ambiguous query tests.

No execution yet.

Commit:
```text
feat: interpret remote sensing queries safely
```

---

## Task 7 — Feasibility validator

Centralize:
- sensor;
- band;
- pair;
- grid;
- overlap;
- model domain;
- measurement requirements.

Return repository failure outcomes.

Make validator available independently.

Commit:
```text
feat: validate query feasibility before planning
```

---

## Task 8 — Bounded planner

Implement:
- tool-only plan generation;
- registry-constrained params;
- max steps;
- dependency graph;
- plan-only API.

No arbitrary code.

Commit:
```text
feat: build bounded evidence workflow planner
```

---

## Task 9 — Job/execution engine

Implement:
- queued/running/completed/failed/cancelled states;
- bounded local executors;
- per-model resource semaphore;
- restart interruption handling;
- tool execution protocol.

Commit:
```text
feat: execute registered workflows with persistent jobs
```

---

## Task 10 — Integrate frozen Phase 1–4 tools

Adapters only; no model-science changes.

Wire:
- VQA;
- grounding;
- multisensor evidence;
- spectral analytics;
- structural change;
- change caption;
- SAR deterministic change;
- flood specialist;
- measurement.

Test each via fake backend plus approved integration smoke tests.

Commit:
```text
feat: expose scientific specialists through execution engine
```

---

## Task 11 — Evidence graph and verification

Implement:
- evidence relationships;
- artifact references;
- measurement-source checks;
- final verification;
- support outcomes.

No numeric aggregate confidence.

Commit:
```text
feat: verify and aggregate scientific evidence
```

---

## Task 12 — Main query API

Implement:
- `/api/v1/query/plan`;
- `/api/v1/query`;
- persisted analysis/job;
- synchronous quick path if approved;
- async heavy path.

Commit:
```text
feat: expose agentic SatQuery analysis API
```

---

## Task 13 — Analysis/history API

Add:
- list/filter;
- get;
- evidence;
- trace;
- artifacts;
- reproducibility;
- rerun.

Commit:
```text
feat: persist and inspect reproducible analyses
```

---

## Task 14 — Artifacts, GeoJSON, and overlay tiles

Add:
- artifact metadata/download;
- evidence GeoJSON;
- mask/index tiles;
- safe source/derived visualization.

Commit:
```text
feat: expose spatial evidence artifacts
```

---

## Task 15 — Reports

Add:
- JSON;
- HTML;
- PDF only if dependency approved.

Commit:
```text
feat: generate auditable analysis reports
```

---

## Task 16 — Streaming progress and cancellation

Implement:
- job event stream (SSE);
- cancellation;
- partial-artifact safety.

Commit:
```text
feat: stream analysis progress and support cancellation
```

---

## Task 17 — Cache/idempotency

Implement:
- scientific cache key;
- exact artifact reuse;
- optional `Idempotency-Key`.

Test cache invalidation on:
- registry version;
- checkpoint;
- profile;
- inputs;
- parameters.

Commit:
```text
feat: add reproducible analysis caching
```

---

## Task 18 — Security and abuse limits

Implement:
- CORS;
- optional API key;
- upload/query/ROI limits;
- queue/concurrency limits;
- safe errors;
- security-focused tests.

Commit:
```text
feat: harden backend execution boundaries
```

---

## Task 19 — Observability and readiness

Implement:
- structured logs;
- request/analysis/job correlation;
- liveness;
- readiness;
- model/resource state;
- optional metrics endpoint.

Commit:
```text
feat: add backend health and observability
```

---

## Task 20 — Swagger/OpenAPI polish

Audit every API endpoint in Swagger.

Add:
- descriptions;
- examples;
- documented error responses;
- stable operation IDs;
- deprecated legacy labels;
- capability demo examples.

Test generated schema.

Commit:
```text
docs: complete interactive backend API documentation
```

---

## Task 21 — Full backend integration tests

Test end-to-end flows:

### Single image
```text
upload
→ metadata
→ VQA
→ evidence
→ history
```

### Grounding
```text
upload
→ query object location
→ bbox/world evidence
→ map artifact
```

### Vegetation
```text
upload T1/T2 multispectral
→ pair validate
→ query
→ NDVI
→ loss mask
→ area
→ report
```

### Structural change
```text
pair
→ ChangerEx
→ mask
→ area where valid
→ caption where applicable
```

### SAR flood
```text
S1-compatible observation
→ flood specialist
→ mask
→ measurement
```

### Invalid RISAT
```text
RISAT incompatible with STURM
→ planner rejects model
→ deterministic SAR path only if explicit semantics support it
```

### Optical/SAR disagreement
```text
paired modalities
→ separate evidence
→ diagnostic disagreement
→ warning
→ no forced fusion
```

### Failure tests
- missing NIR;
- unknown SAR domain;
- wrong image count;
- no overlap;
- missing CRS for hectares;
- model unavailable;
- resource busy;
- invalid ROI;
- corrupted artifact;
- cancelled job;
- server restart/interrupted job.

Commit:
```text
test: verify complete backend workflows
```

---

## Task 22 — Deployment packaging

Add/verify:
- Dockerfile;
- Docker Compose only if multiple isolated workers require it;
- environment template;
- volume layout;
- healthcheck;
- startup migrations;
- offline docs assets.

No Kubernetes.

Commit:
```text
build: package SatQuery backend runtime
```

---

## Task 23 — Backend closeout

Create:
```text
experiments/phase5_backend/PHASE_5_CLOSEOUT.json
```

Record:
- Git SHA;
- OpenAPI schema hash;
- route inventory;
- test counts;
- capability matrix;
- Phase 4 model/tool registry hash;
- persistence schema version;
- security limits;
- known domain restrictions;
- external runtime requirements.

Run:
```bash
python -m pytest
python -m compileall -q satquery apps ml scripts
git diff --check
```

Phase 5 is complete only when:
- all mandatory backend workflows pass;
- Swagger UI can execute representative requests;
- OpenAPI is valid and stable;
- frontend no longer needs undocumented backend knowledge;
- analyses are persistent/reproducible;
- no scientific claim bypasses evidence verification.

Commit:
```text
docs: close phase 5 backend and agent services
```

---

# 53. Core vs high-value stretch features

## Core — must ship

- versioned API;
- Swagger UI / OpenAPI;
- observations;
- pair validation;
- capability registry;
- query plan;
- query execution;
- persistent jobs;
- evidence;
- trace;
- measurement;
- artifacts;
- overlay tiles;
- analysis history;
- reports JSON/HTML;
- liveness/readiness;
- structured errors;
- security limits;
- full integration tests.

## High-value — strongly recommended

- SSE progress;
- cancellation;
- cache;
- idempotency;
- reproducibility endpoint;
- dynamic capability matrix;
- ROI analysis;
- GeoJSON evidence;
- runtime model state;
- self-hosted offline Swagger/ReDoc;
- generated frontend client support.

## Stretch — implement only if time remains

- PDF reports;
- optional API-key auth;
- Prometheus metrics;
- batch upload;
- model unload/reload controls;
- advanced analysis search.

Do not sacrifice core scientific correctness for stretch features.

---

# 54. Backend acceptance gate

Phase 5 must prove this complete judge-facing workflow:

```text
1. Upload scientific raster(s)
2. Inspect sensor/bands/CRS
3. Validate pair if required
4. Ask natural-language query
5. Inspect generated plan
6. Execute registered workflow
7. Track progress
8. View evidence on map
9. Read measurements
10. Inspect warnings/domain limits
11. Inspect execution trace
12. Download report/artifacts
13. Re-open analysis from history
14. Reproduce exact inputs/tool/model identities
```

If the backend can do all fourteen without undocumented manual intervention, it is product-ready for Phase 6 frontend integration.

---

# 55. Explicit exclusions

Phase 5 does **not** include:

- a custom web frontend;
- UI animation/polish;
- a multi-agent swarm;
- arbitrary code execution;
- external plugin execution;
- a vector database;
- Kubernetes;
- Kafka;
- mandatory Redis/Celery;
- automatic model retraining;
- unsupported generic RISAT→Sentinel-1 conversion;
- invented confidence scoring;
- hidden chain-of-thought storage.

Those are either unnecessary or belong elsewhere.

---

# 56. Final Phase 5 product statement

At Phase 5 closeout, SatQuery backend should be describable as:

> **A versioned, documented, evidence-grounded remote-sensing analysis API that accepts scientific imagery and natural-language queries, validates sensor/task feasibility, executes only registered deterministic/model specialists, persists reproducible evidence and measurements, and exposes every result through interactive OpenAPI documentation, map-ready artifacts, execution traces, and downloadable reports.**
