# Phase 5 Backend and Evidence Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned FastAPI backend that turns registered observations and bounded natural-language requests into persistent, verified, reproducible evidence without permitting the language layer to manufacture scientific claims.

**Architecture:** Extend the existing modular monolith. FastAPI remains the transport boundary; `satquery/` owns query interpretation, feasibility, planning, execution, evidence, verification, persistence, and reporting. SQLite (through Python's `sqlite3`) stores metadata and state, the existing controlled filesystem stores immutable binary artifacts, and one bounded in-process worker queue executes jobs; isolated subprocess adapters are added only for a registered specialist that cannot run in the API environment.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, stdlib `sqlite3`, stdlib `concurrent.futures`/`threading`, Rasterio, NumPy, PyYAML, pytest.

**Spec:** `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/FAILURE_POLICY.md`, and this plan's frozen contract in Task 0.

## Global Constraints

- Canonical Phase 4 must have a canonical closeout artifact whose hash/reconstruction checks pass and which has been human-reviewed before Phase 5 execution begins. Phase 5 starts in exactly one mode (see Task 0): `FULL_CAPABILITY` when Phase 4 status is `COMPLETE`, or `RESTRICTED_CAPABILITY` when Phase 4 status is `BLOCKED` but independently verified capabilities may be exposed while blocked capabilities remain unavailable. No blocked capability may become runnable merely because Phase 5 started; until the gate passes, this document is planning-only.
- Evidence is the source of scientific claims. A parser, planner, VQA model, caption model, or answer composer may not create masks, measurements, metadata, or confidence values.
- Execute only model and tool IDs loaded from validated repository registries.
- Preserve sensor, modality, band/polarization semantics, radiometric domain, acquisition time, CRS, transform, bounds, GSD, NoData, source hashes, and preprocessing provenance when available.
- Unknown sensor semantics remain unknown. Generic SAR is never treated as RGB, Sentinel-1, VV/VH, or a known radiometric domain.
- Temporal execution requires distinct ordered observations. Equal array shape is not grid alignment.
- Physical measurements come only from deterministic GIS evidence and fail closed when their geospatial preconditions are absent.
- Every denied workflow uses one repository outcome: `ALLOW`, `ALLOW_WITH_WARNING`, `REQUEST_INPUT`, `ABSTAIN`, or `REJECT`.
- Do not add Celery, Redis, Kafka, Kubernetes, a vector database, LangGraph, or a multi-agent framework.
- Keep legacy `/api/observations`, `/api/vqa`, `/api/grounding`, and `/tiles/...` behavior while `/api/v1` becomes canonical.
- Do not expose local paths, stack traces, checkpoint paths, secrets, hidden prompts, or chain-of-thought through API responses or logs.
- Add no dependency for SQLite, job execution, HTML reports, SSE, hashing, or JSON serialization; the standard library and FastAPI already cover them.
- JSON and HTML reports are required. PDF remains disabled unless a separately audited renderer is approved.
- Each task must pass its targeted test before its commit. Run the full affected suite at every milestone gate.

---

## 1. Corrected Architecture Decisions

The previous plan mixed required contracts, speculative product features, and implementation detail. Use these decisions during execution:

1. **One deployable backend, not services by directory.** Application services are ordinary Python objects wired by `create_app`; they are not network services.
2. **One durable queue, not FastAPI background tasks.** A bounded `JobRunner` owns a fixed worker pool and persistent state. `BackgroundTasks` is unsuitable because queued state and restart recovery would be lost.
3. **SQLite is authoritative only for backend metadata.** Existing immutable observation JSON/raster storage remains intact. Phase 5 indexes it in SQLite instead of migrating or rewriting frozen source assets.
4. **Evidence graph is relational, not a graph framework.** Store evidence records plus typed edges in SQLite. No graph database is needed.
5. **Tool request schemas are explicit.** OpenAPI cannot safely describe arbitrary registry-generated parameter objects. Public direct execution uses a discriminated union of approved request models; registry constraints still validate the resulting call.
6. **Deterministic interpretation is the required path.** An optional language-model interpreter may be added behind the same typed protocol only after separate evaluation. Phase 5 correctness cannot depend on network access.
7. **No fake asynchronous fast path.** `POST /api/v1/query` always creates an analysis and job and returns `202`. Direct deterministic tool routes may remain synchronous. This gives one observable lifecycle.
8. **No generic model adapter.** Each frozen capability receives a small typed adapter because input contracts differ materially across optical, multispectral, SAR, and temporal models.
9. **No automatic reprojection in the agent.** Existing pair validation may identify that reprojection is required, but pixelwise tools reject unless an explicit registered alignment tool with tested provenance exists.
10. **Capability means executable now.** A code path without a registered model/tool, verified checkpoint, required runtime, and satisfied input contract is reported unavailable.

## 2. Delivery Milestones

| Milestone | Tasks | Independently testable outcome |
| --- | --- | --- |
| M0 Contract | 0 | Frozen backend/API/security contract consistent with Phase 4 closeout |
| M1 Durable platform | 1–5 | Versioned API, SQLite metadata, observations, pairs, and capability inspection |
| M2 Safe agent | 6–8 | Deterministic intent, feasibility, bounded plans, and plan-only API |
| M3 Execution/evidence | 9–12 | Persistent jobs execute registered tools and return verified answers |
| M4 Product APIs | 13–20 | History, artifacts, reports, progress, caching, security, health, OpenAPI |
| M5 Release gate | 21–23 | End-to-end proof, packaging, and evidence-backed closeout |

A milestone is not complete when only schemas or mocks pass. Its stated outcome must pass through the real application boundary using local deterministic fixtures.

## 3. File Map

Create only the files needed by the tasks below. Do not pre-create empty modules.

```text
apps/api/app/
├── main.py                         # app factory, lifespan, dependency wiring
├── dependencies.py                 # typed state accessors
├── errors.py                       # Failure -> HTTP mapping and handlers
├── openapi.py                      # metadata and schema quality helpers
├── schemas.py                      # legacy schemas; retained unchanged where possible
├── schemas_v1.py                   # v1 transport-only request/response models
└── routes/
    ├── v1_system.py
    ├── v1_observations.py
    ├── v1_pairs.py
    ├── v1_query.py
    ├── v1_jobs.py
    ├── v1_analyses.py
    ├── v1_registry.py
    ├── v1_artifacts.py
    └── v1_reports.py

satquery/
├── agent/
│   ├── models.py                   # intent, validation, plan, answer contracts
│   ├── interpreter.py              # deterministic parser
│   ├── feasibility.py              # centralized scientific gate
│   ├── planner.py                  # registry-constrained DAG construction
│   └── composer.py                 # deterministic evidence-only answer text
├── execution/
│   ├── models.py                   # job/tool call/result/event contracts
│   ├── engine.py                   # ordered registered-tool execution
│   ├── jobs.py                     # bounded durable local runner
│   └── adapters.py                 # approved capability adapters
├── persistence/
│   ├── database.py                 # connection, transaction, migrations
│   └── repositories.py             # backend metadata persistence
├── evidence/
│   ├── models.py                   # existing evidence contracts
│   └── graph.py                    # typed evidence nodes/edges
├── verification/
│   └── analysis.py                 # graph and requested-claim verification
├── artifacts/
│   └── store.py                    # controlled derived-artifact storage
└── reporting/
    └── reports.py                  # deterministic JSON/HTML rendering
```

Tests mirror ownership under `tests/agent`, `tests/execution`, `tests/persistence`, `tests/artifacts`, `tests/reporting`, and `tests/integration`.

---

### Task 0: Freeze the Backend Contract and Phase 4 Inventory

**Files:**
- Create: `experiments/phase5_backend/backend_contract.yaml`
- Create: `experiments/phase5_backend/README.md`
- Modify: `docs/plans/PHASE_5_BACKEND_PLAN.md`
- Test: `tests/integration/test_phase5_contract.py`

**Interfaces:**
- Consumes: canonical `experiments/phase4_temporal_analytics/PHASE_4_CLOSEOUT.json`, `models/registry.yaml`, `satquery/registry/tools.yaml`, and `satquery/registry/preprocessing.yaml`.
- Produces: schema version `1`, the Phase 5 start mode, exact route families, capability states, state enums, limits, legacy compatibility policy, and an inventory whose entries correspond to real registered implementations.

- [ ] **Step 1: Write the contract test**

```python
from pathlib import Path
import yaml


def test_phase5_contract_names_only_registered_capabilities():
    contract = yaml.safe_load(Path("experiments/phase5_backend/backend_contract.yaml").read_text())
    tools = yaml.safe_load(Path("satquery/registry/tools.yaml").read_text())["tools"]
    models = yaml.safe_load(Path("models/registry.yaml").read_text())["models"]
    assert contract["schema_version"] == 1
    assert contract["phase5_start"]["mode"] in {"FULL_CAPABILITY", "RESTRICTED_CAPABILITY"}
    tool_ids = {
        tool_id
        for capability in contract["capabilities"].values()
        for tool_id in capability.get("tool_ids", [])
    }
    model_ids = {
        model_id
        for capability in contract["capabilities"].values()
        for model_id in capability.get("model_ids", [])
    }
    assert tool_ids <= set(tools)
    assert model_ids <= set(models)
    assert contract["job_states"] == [
        "QUEUED", "RUNNING", "SUCCEEDED", "FAILED",
        "CANCEL_REQUESTED", "CANCELLED", "INTERRUPTED",
    ]
    assert contract["analysis_states"] == [
        "PENDING", "RUNNING", "SUCCEEDED", "FAILED",
        "ABSTAINED", "REJECTED", "CANCELLED", "INTERRUPTED",
    ]
    # no blocked or unpromoted learned specialist may appear AVAILABLE
    for capability in contract["capabilities"].values():
        if capability.get("contract_status") == "BLOCKED" or capability.get("promotion_status") == "NOT_PROMOTED":
            assert capability["status"] != "AVAILABLE"
```

- [ ] **Step 2: Run the test and confirm it fails because the contract is absent**

Run: `python -m pytest tests/integration/test_phase5_contract.py -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Audit the canonical Phase 4 closeout and write the contract**

Verify the closeout's reconstruction checks pass and record its SHA-256. Record exact model/tool IDs; do not copy aspirational capabilities from architecture prose. Freeze:

```yaml
schema_version: 1
phase5_start:
  mode: RESTRICTED_CAPABILITY   # or FULL_CAPABILITY when Phase 4 is COMPLETE
  phase4_closeout_status: BLOCKED
  phase4_closeout_sha256: <actual>
  starting_git_sha: <actual>
api_prefix: /api/v1
job_states: [QUEUED, RUNNING, SUCCEEDED, FAILED, CANCEL_REQUESTED, CANCELLED, INTERRUPTED]
analysis_states: [PENDING, RUNNING, SUCCEEDED, FAILED, ABSTAINED, REJECTED, CANCELLED, INTERRUPTED]
failure_outcomes: [ALLOW, ALLOW_WITH_WARNING, REQUEST_INPUT, ABSTAIN, REJECT]
capability_states: [AVAILABLE, AVAILABLE_WITH_LIMITS, UNAVAILABLE_CONTRACT_BLOCKED, UNAVAILABLE_MODEL_NOT_INSTALLED, UNAVAILABLE_RUNTIME, UNSUPPORTED_INPUT, NOT_IMPLEMENTED]
max_plan_steps: 8
legacy_routes: [/api/observations, /api/vqa, /api/grounding, /tiles]
pdf_reports: disabled
```

Learned-model capabilities additionally freeze `contract_status`, `runtime_status`, and `promotion_status`, and must never expose a BLOCKED or NOT_PROMOTED specialist as AVAILABLE. Cite the exact Phase 4 closeout path and hash, inventory only real capabilities, and explain every unavailable specialist in `README.md`. Freeze route families and the single error envelope in `api_contract.md`; add a `carried_blockers` section so no Phase 4 blocker silently disappears. Task 0 must not download datasets/checkpoints, run Kaggle, change Phase 4 results, or promote temporal models.

- [ ] **Step 4: Run the contract test**

Run: `python -m pytest tests/integration/test_phase5_contract.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/plans/PHASE_5_BACKEND_PLAN.md experiments/phase5_backend tests/integration/test_phase5_contract.py
git commit -m "docs: freeze phase 5 backend contract"
```

**Gate:** Stop if canonical Phase 4 lacks a closeout whose hash/reconstruction checks pass and which has been human-reviewed. Record the blocker; do not manufacture an empty closeout or mark models available from source files alone. Phase 5 starts in `RESTRICTED_CAPABILITY` while Phase 4 is `BLOCKED`.

---

### Task 1: Establish the Versioned FastAPI and Error Contract

**Files:**
- Create: `apps/api/app/errors.py`
- Create: `apps/api/app/openapi.py`
- Create: `apps/api/app/schemas_v1.py`
- Create: `apps/api/app/routes/v1_system.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/integration/test_api_v1_foundation.py`

**Interfaces:**
- Produces: `ApiError(error: FailureDetailV1)`, `GET /health/live`, `GET /api/v1/system/version`, `X-Request-ID`, stable operation IDs, and unchanged legacy routes.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_v1_foundation(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert response.headers["X-Request-ID"].startswith("req_")
    version = client.get("/api/v1/system/version").json()
    assert version["api_version"] == "v1"
    assert client.get("/api/observations").status_code != 404


def test_validation_error_uses_one_envelope(client):
    response = client.post("/api/v1/query/plan", json={})
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["outcome"] == "REJECT"
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_api_v1_foundation.py -q`
Expected: FAIL because v1 routes are absent.

- [ ] **Step 3: Add strict v1 schemas and HTTP mapping**

Define `FailureDetailV1` with exactly the frozen envelope fields `code`, `message`, `outcome`, `details`, plus `request_id`; define `failure_response(failure, status_code, request_id)`. HTTP status and scientific outcome are independent (for example HTTP 422 + `REQUEST_INPUT`). Map known domain failures explicitly and map unexpected exceptions to `INTERNAL_ERROR` without exception text.

- [ ] **Step 4: Wire middleware and routes without changing legacy handlers**

Generate server-side IDs with `f"req_{uuid.uuid4().hex}"`. Put the ID on `request.state`, response headers, structured logs, and errors. Configure FastAPI title, description, version, and ordered tags in `openapi.py`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/integration/test_api_v1_foundation.py tests/integration/test_observations_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/app tests/integration/test_api_v1_foundation.py
git commit -m "feat: establish versioned backend API contract"
```

---

### Task 2: Add Transactional SQLite Metadata Persistence

**Files:**
- Create: `satquery/persistence/__init__.py`
- Create: `satquery/persistence/database.py`
- Create: `satquery/persistence/repositories.py`
- Test: `tests/persistence/test_database.py`
- Test: `tests/persistence/test_repositories.py`

**Interfaces:**
- Produces: `Database(path: Path)`, `Database.transaction()`, `migrate()`, and repository methods `create_*`, `get_*`, `list_*`, and state transitions.
- SQLite stores JSON contract payloads as canonical UTF-8 text and all timestamps as UTC ISO-8601 strings.

- [ ] **Step 1: Test schema creation, foreign keys, rollback, and restart**

```python
def test_database_is_durable_and_rolls_back(tmp_path):
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    with pytest.raises(RuntimeError):
        with db.transaction() as connection:
            connection.execute("INSERT INTO schema_meta(version) VALUES (99)")
            raise RuntimeError("rollback")
    reopened = Database(tmp_path / "satquery.db")
    assert reopened.schema_version() == 1
```

Also assert `PRAGMA foreign_keys = 1`, WAL mode, unique IDs, legal state transitions, cursor pagination order `(created_at DESC, id DESC)`, and that `RUNNING` jobs become `INTERRUPTED` at startup.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/persistence -q`
Expected: FAIL because persistence modules are absent.

- [ ] **Step 3: Implement one migration**

Create tables:

```text
schema_meta, observations, pairs, analyses, analysis_inputs, jobs,
plan_steps, evidence, evidence_edges, artifacts, execution_events,
cache_entries, idempotency_keys
```

Use `BEGIN IMMEDIATE`, parameterized SQL, `sqlite3.Row`, and one connection per transaction. Never interpolate values or identifiers from requests.

- [ ] **Step 4: Implement repositories with compare-and-set transitions**

Use signatures:

```python
create_analysis(analysis: AnalysisRecord) -> None
get_analysis(analysis_id: str) -> AnalysisRecord | None
create_job(job: JobRecord) -> None
transition_job(job_id: str, expected: JobStatus, target: JobStatus) -> bool
append_event(event: ExecutionEvent) -> None
mark_running_jobs_interrupted() -> int
```

- [ ] **Step 5: Run persistence tests**

Run: `python -m pytest tests/persistence -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/persistence tests/persistence
git commit -m "feat: add persistent analysis metadata store"
```

---

### Task 3: Index Existing Observations and Expose Observation API v1

**Files:**
- Modify: `satquery/ingestion/storage.py`
- Create: `apps/api/app/routes/v1_observations.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/integration/test_observations_v1_api.py`

**Interfaces:**
- Consumes: `FilesystemObservationStore.register_with_visualization()` and `load_registration()`.
- Produces: `list_registrations()`, `POST/GET /api/v1/observations`, `GET /{id}`, `/metadata`, `/assets`, and cursor pagination.

- [ ] **Step 1: Write failing tests for upload indexing and restart**

Upload a fixture through v1, restart `create_app` against the same `data_root`, list observations, and assert the same ID/hash appears with no public `path` field. Assert the legacy upload still works.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_observations_v1_api.py -q`
Expected: FAIL with 404.

- [ ] **Step 3: Add safe filesystem enumeration**

`list_registrations()` must enumerate only directories matching `OBSERVATION_ID_PATTERN`, call `load_registration`, ignore no malformed entry silently, and return deterministic ID order. Add a startup indexer that inserts missing immutable observation metadata into SQLite without rewriting files.

- [ ] **Step 4: Add v1 routes**

Reuse `ObservationIngestionService`; after successful filesystem registration, index metadata transactionally. If indexing fails, return `ASSET_STORAGE_FAILED` and leave the immutable observation recoverable by startup indexing. Do not add physical deletion in Phase 5; return `405` for DELETE and document immutability.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/integration/test_observations_v1_api.py tests/integration/test_observations_api.py tests/ingestion -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/ingestion/storage.py apps/api/app tests/integration/test_observations_v1_api.py
git commit -m "feat: expose observation lifecycle API"
```

---

### Task 4: Persist and Expose Validated Observation Pairs

**Files:**
- Create: `apps/api/app/routes/v1_pairs.py`
- Modify: `apps/api/app/schemas_v1.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/integration/test_pairs_api.py`

**Interfaces:**
- Consumes: `FilesystemObservationStore.load_registration()` and `PairValidator.validate()`.
- Produces: `PairCreateRequest(observation_a, observation_b, pair_type, explicit_t1_id=None)`, `POST /api/v1/pairs/validate`, `POST /api/v1/pairs`, list/get routes, and immutable validation snapshots.

- [ ] **Step 1: Write failing pair tests**

Test exact-grid temporal success, explicit ordering when dates are absent, duplicate observation rejection, no-overlap rejection, optical/SAR classification, and persisted snapshot equality after restart.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_pairs_api.py -q`
Expected: FAIL with 404.

- [ ] **Step 3: Implement transport mapping**

Map `PairCompatibility` reasons to policy outcomes without changing `PairValidator`: no overlap -> `REJECT`; missing temporal order for temporal pair without explicit mapping -> `REQUEST_INPUT`; transformable but unaligned -> `ALLOW_WITH_WARNING` for non-pixelwise use and infeasible for pixelwise plans.

- [ ] **Step 4: Persist pair and validation snapshot atomically**

Generate `pair_<32 hex>`. Never mutate source observations or claim registration occurred.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/integration/test_pairs_api.py tests/geo/test_pairing.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/app tests/integration/test_pairs_api.py
git commit -m "feat: add validated observation pair workflows"
```

---

### Task 5: Load and Expose the Runtime Capability Registry

**Files:**
- Modify: `satquery/registry/models.py`
- Create: `satquery/registry/tools.py`
- Modify: `satquery/registry/__init__.py`
- Modify: `satquery/registry/tools.yaml`
- Create: `apps/api/app/routes/v1_registry.py`
- Test: `tests/registry/test_tools.py`
- Test: `tests/integration/test_registry_api.py`

**Interfaces:**
- Produces: `ToolRegistration`, `ToolRegistry`, `load_tool_registry()`, `RuntimeCapability`, safe model/tool list/get routes, and `GET /api/v1/system/capabilities`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_tool_registry_rejects_unbounded_parameters(tmp_path):
    path = tmp_path / "tools.yaml"
    path.write_text("schema_version: 1\ntools:\n  bad:\n    parameters:\n      command: {type: string}\n")
    with pytest.raises(ValueError, match="command"):
        load_tool_registry(path)
```

Also test unknown executors, duplicate IDs, missing implementation keys, safe API projection, hash stability, and that availability requires both registration and runtime probe success.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/registry/test_tools.py tests/integration/test_registry_api.py -q`
Expected: FAIL because the loader/routes do not exist.

- [ ] **Step 3: Implement strict tool contracts**

Allowed executor IDs are an enum defined in code. Allowed parameter types are booleans, bounded numbers, enums, server IDs, and validated ROI; reject `command`, `code`, `url`, and `path`. Populate `tools.yaml` only with implementations confirmed by Task 0.

- [ ] **Step 4: Compute availability dynamically**

Use the frozen capability states from Task 0. `AVAILABLE` requires a valid registration, known executor adapter, present dependency/checkpoint, and successful cheap readiness probe; otherwise `UNAVAILABLE_MODEL_NOT_INSTALLED` or `UNAVAILABLE_RUNTIME`. Learned-model capabilities expose `contract_status`, `runtime_status`, and `promotion_status`, and a BLOCKED contract yields `UNAVAILABLE_CONTRACT_BLOCKED` regardless of runtime state.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/registry tests/integration/test_registry_api.py tests/inference -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/registry apps/api/app tests/registry tests/integration/test_registry_api.py
git commit -m "feat: expose runtime capability registry"
```

**Milestone M1 gate:** Run `python -m pytest tests/ingestion tests/geo tests/registry tests/integration -q` and preserve all legacy route tests.

---

### Task 6: Define and Deterministically Interpret Query Intents

**Files:**
- Create: `satquery/agent/__init__.py`
- Create: `satquery/agent/models.py`
- Create: `satquery/agent/interpreter.py`
- Test: `tests/agent/test_interpreter.py`

**Interfaces:**
- Produces:

```python
class QueryInterpreter(Protocol):
    def interpret(self, query: str) -> QueryIntent: ...

class DeterministicQueryInterpreter:
    def interpret(self, query: str) -> QueryIntent: ...
```

`QueryIntent` fields are `task_family`, `target_semantic`, `requested_measurement`, `temporal_direction`, `spatial_request`, `matched_rule`, and `ambiguities`.

- [ ] **Step 1: Write a table-driven failing test**

Cover VQA, grounding, metadata, area, NDVI, water gain/loss, generic change, change description, optical/SAR comparison, capability question, ambiguous “bank”, negation, oversized input, and prompt-injection text. Assert parser output contains no tool ID.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/agent/test_interpreter.py -q`
Expected: FAIL because agent modules are absent.

- [ ] **Step 3: Implement ordered rules**

Normalize with `unicodedata.normalize("NFKC", query).casefold()`, cap query length from settings, use compiled word-boundary regular expressions, detect measurement before semantic task, and return `AMBIGUOUS` when rules conflict. Keep rules in one module; do not create a plugin framework.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/agent/test_interpreter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add satquery/agent tests/agent/test_interpreter.py
git commit -m "feat: interpret remote sensing queries safely"
```

---

### Task 7: Centralize Scientific Feasibility Decisions

**Files:**
- Create: `satquery/agent/feasibility.py`
- Modify: `satquery/agent/models.py`
- Test: `tests/agent/test_feasibility.py`

**Interfaces:**
- Produces `FeasibilityValidator.validate(intent, observations, pair, roi, capabilities) -> FeasibilityResult` containing ordered checks, one policy outcome, failures, warnings, and allowed capability IDs.

- [ ] **Step 1: Write failing policy tests**

Assert:
- one observation + change -> `REQUEST_INPUT`;
- RGB without NIR + NDVI -> `REJECT`;
- no CRS + area -> `REJECT`;
- unknown SAR semantics + SAR specialist -> `REQUEST_INPUT` or `REJECT` per failure policy;
- non-overlap -> `REJECT`;
- misaligned grids + pixelwise tool -> `REJECT` until an alignment tool exists;
- JPEG VQA -> `ALLOW_WITH_WARNING`;
- unavailable model -> `ABSTAIN`;
- valid deterministic area chain -> `ALLOW`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/agent/test_feasibility.py -q`
Expected: FAIL because validator is absent.

- [ ] **Step 3: Implement check composition and precedence**

Use named pure checks and precedence `CRITICAL/REJECT`, `REQUEST_INPUT`, `ABSTAIN`, `ALLOW_WITH_WARNING`, `ALLOW`. Preserve all check results rather than returning at the first warning. Reference stable failure codes from `docs/FAILURE_POLICY.md`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/agent/test_feasibility.py tests/geo tests/analytics -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add satquery/agent tests/agent/test_feasibility.py
git commit -m "feat: validate query feasibility before planning"
```

---

### Task 8: Build Registry-Constrained Plans and Plan-Only API

**Files:**
- Create: `satquery/agent/planner.py`
- Create: `apps/api/app/routes/v1_query.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/agent/test_planner.py`
- Test: `tests/integration/test_query_plan_api.py`

**Interfaces:**
- Produces `BoundedPlanner.plan(intent, feasibility, inputs) -> ExecutionPlan` and `POST /api/v1/query/plan`.
- `PlanStep` contains only `step_id`, `tool_id`, `input_bindings`, `parameters`, `depends_on`, and `expected_evidence_type`.

- [ ] **Step 1: Write failing planner tests**

Test deterministic plan equality, max eight steps, known tool IDs, acyclic dependencies, parameter bounds, area requests ending in measurement, denied feasibility producing no steps, and query text never appearing as code/path/URL.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/agent/test_planner.py tests/integration/test_query_plan_api.py -q`
Expected: FAIL because planner/API are absent.

- [ ] **Step 3: Implement explicit workflow policies**

Use a dictionary keyed by composable intent fields whose values are factories for known plans. Resolve every tool against `ToolRegistry`; validate bindings and parameters with the registration before returning. Hash canonical plan JSON plus registry hash and planner version.

- [ ] **Step 4: Expose plan-only route**

Load observations/pair server-side by ID, interpret, validate, plan, and return operational reasons. Never persist a dry run and never execute a step.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/agent tests/integration/test_query_plan_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/agent apps/api/app tests/agent tests/integration/test_query_plan_api.py
git commit -m "feat: build bounded evidence workflow planner"
```

**Milestone M2 gate:** Demonstrate in `/docs` that valid and invalid plan requests produce deterministic plans or structured refusals without executing models.

---

### Task 9: Implement Persistent Jobs and the Bounded Execution Engine

**Files:**
- Create: `satquery/execution/__init__.py`
- Create: `satquery/execution/models.py`
- Create: `satquery/execution/engine.py`
- Create: `satquery/execution/jobs.py`
- Create: `satquery/artifacts/__init__.py`
- Create: `satquery/artifacts/store.py`
- Create: `apps/api/app/routes/v1_jobs.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/execution/test_engine.py`
- Test: `tests/execution/test_jobs.py`
- Test: `tests/artifacts/test_store.py`

**Interfaces:**
- Produces:

```python
class ToolAdapter(Protocol):
    def execute(self, call: ToolCall, context: ExecutionContext) -> ToolResult: ...

class ArtifactStore:
    def stage(self, suffix: str) -> StagedArtifact: ...
    def publish(self, staged: StagedArtifact, metadata: ArtifactMetadata) -> ArtifactRecord: ...
    def resolve(self, artifact_id: str) -> tuple[ArtifactRecord, Path]: ...

class ExecutionEngine:
    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> tuple[ToolResult, ...]: ...

class JobRunner:
    def submit(self, analysis_id: str, plan: ExecutionPlan) -> JobRecord: ...
    def request_cancel(self, job_id: str) -> JobRecord: ...
    def start(self) -> None: ...
    def stop(self, timeout_seconds: float) -> None: ...
```

- [ ] **Step 1: Write failing execution tests**

Use fake adapters to prove dependency order, unknown tool rejection, timeout failure, per-tool semaphore enforcement, bounded queue rejection, cancellation between steps, no publication after failure, event order, and restart interruption recovery.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/execution -q`
Expected: FAIL because execution modules are absent.

- [ ] **Step 3: Implement the engine**

Topologically execute the already validated DAG. Resolve only adapters registered at app startup. Before and after each step check cancellation, persist event/state transitions, and register outputs only after adapter result validation.

- [ ] **Step 4: Implement private artifact staging and publication**

Use `data/artifacts/<artifact_id>/artifact.<approved suffix>` plus `metadata.json`. Generate IDs server-side, resolve under the controlled root, hash before publication, atomically promote with `os.replace`, and remove staging data after failure/cancellation. This task establishes execution safety; public artifact/GeoJSON/tile routes arrive in Task 14.

- [ ] **Step 5: Implement the runner**

Use `queue.Queue(maxsize=settings.max_queued_jobs)` plus a fixed number of daemon worker threads. Each worker claims a SQLite `QUEUED` job with compare-and-set. App lifespan calls `mark_running_jobs_interrupted()`, then `start()`, and calls `stop()` on shutdown.

- [ ] **Step 6: Add job get/cancel routes**

Return `409` for cancellation of terminal jobs and `404` for unknown IDs. Do not expose worker thread details.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/execution tests/artifacts/test_store.py tests/persistence -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add satquery/execution satquery/artifacts satquery/persistence apps/api/app tests/execution tests/artifacts/test_store.py
git commit -m "feat: execute registered workflows with persistent jobs"
```

---

### Task 10: Adapt Frozen Scientific Capabilities to Registered Tools

**Files:**
- Create: `satquery/execution/adapters.py`
- Modify: `satquery/registry/tools.yaml`
- Test: `tests/execution/test_adapters.py`

**Interfaces:**
- Consumes existing `SingleImageVqaService`, `TextGuidedGroundingService`, `satquery.analytics.spectral`, `satquery.analytics.temporal`, and `satquery.analytics.measurement` APIs plus only Phase 4 specialists accepted in Task 0.
- Produces one adapter per materially different input contract and canonical evidence objects from `satquery.evidence.models`.

- [ ] **Step 1: Write failing adapter contract tests**

For each Task 0 tool, assert valid evidence type, source IDs/hashes, model/tool provenance, exact profile ID, immutable artifact reference where relevant, and failure on wrong modality, missing semantics, wrong temporal order, or unverified grid.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/execution/test_adapters.py -q`
Expected: FAIL because adapters are absent.

- [ ] **Step 3: Implement minimal typed adapters**

Do not wrap tools that are not registered. VQA and grounding delegate to existing services. Deterministic adapters read controlled assets, call existing pure analytics, and publish derived masks through the `ArtifactStore` created in Task 9. Learned temporal/SAR adapters must preserve their exact source contracts.

- [ ] **Step 4: Register implementation keys**

Map fixed registry `executor` values to constructors in a code-owned dictionary. Reject duplicate and missing mappings at startup.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/execution/test_adapters.py tests/inference tests/analytics tests/evidence -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/execution/adapters.py satquery/registry/tools.yaml tests/execution/test_adapters.py
git commit -m "feat: expose scientific specialists through execution engine"
```

---

### Task 11: Persist an Evidence Graph and Verify Requested Claims

**Files:**
- Create: `satquery/evidence/graph.py`
- Create: `satquery/verification/analysis.py`
- Modify: `satquery/verification/__init__.py`
- Test: `tests/evidence/test_graph.py`
- Test: `tests/verification/test_analysis.py`

**Interfaces:**
- Produces `EvidenceGraph(nodes, edges)`, edge types `DERIVED_FROM`, `MEASURED_FROM`, `SUPPORTS`, `CONFLICTS_WITH`, `DESCRIBES`, `LOCALIZED_BY`, and `AnalysisVerifier.verify(intent, graph) -> VerificationReport`.

- [ ] **Step 1: Write failing verifier tests**

Reject dangling edges, duplicate IDs with different payloads, hash-missing artifacts, measurement sourced from caption/VQA, area value inconsistent with its deterministic result, numeric answer without measurement, unknown input IDs, and intent not answered. Preserve valid independent evidence when another branch fails.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/evidence/test_graph.py tests/verification/test_analysis.py -q`
Expected: FAIL because graph/verifier are absent.

- [ ] **Step 3: Implement a typed adjacency representation**

Use tuples and dictionaries; reject cycles only for derivation edges. Persist nodes as their discriminated Pydantic JSON and edges as rows. Do not add NetworkX.

- [ ] **Step 4: Implement deterministic verification rules**

Rules operate on types and provenance, not prose similarity. A change caption can support description only; VQA cannot support area; `MeasurementEvidence.source_evidence_id` must identify spatial mask evidence; all artifact hashes must verify.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/evidence tests/verification -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/evidence satquery/verification tests/evidence tests/verification
git commit -m "feat: verify and aggregate scientific evidence"
```

---

### Task 12: Compose Verified Answers and Submit Agent Queries

**Files:**
- Create: `satquery/agent/composer.py`
- Modify: `apps/api/app/routes/v1_query.py`
- Test: `tests/agent/test_composer.py`
- Test: `tests/integration/test_query_submission.py`

**Interfaces:**
- Produces `AnswerComposer.compose(query, intent, graph, report) -> AnalysisAnswer` and `POST /api/v1/query -> 202` with `analysis_id`, `job_id`, and `QUEUED`.

- [ ] **Step 1: Write failing composer and submission tests**

Assert exact measurement values/units come from `MeasurementEvidence`, limitations are preserved, abstention contains no positive claim, raw model score is labeled uncalibrated, submission persists immutable input hashes/plan/registry hash, and duplicate arbitrary text cannot inject a tool.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/agent/test_composer.py tests/integration/test_query_submission.py -q`
Expected: FAIL because composer/submission are absent.

- [ ] **Step 3: Implement deterministic templates**

Templates branch on intent and verified evidence type. Format numbers from stored values with a declared display precision while retaining the full numeric value in response data. Never parse numbers from model prose.

- [ ] **Step 4: Implement submission transaction**

In one transaction create analysis, inputs, frozen plan, and queued job. Submit the job only after commit. If queue submission fails, transition both records to failed with `RESOURCE_BUSY`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/agent tests/integration/test_query_submission.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/agent apps/api/app tests/agent tests/integration/test_query_submission.py
git commit -m "feat: expose agentic SatQuery analysis API"
```

**Milestone M3 gate:** Run one local deterministic query from upload through persisted verified answer. Model-dependent tests may use injected fake backends, but no test may synthesize benchmark claims.

---

### Task 13: Expose Analysis History, Trace, and Reproducibility

**Files:**
- Create: `apps/api/app/routes/v1_analyses.py`
- Modify: `apps/api/app/schemas_v1.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/integration/test_analyses_api.py`

**Interfaces:**
- Produces list/get, `/evidence`, `/trace`, `/artifacts`, `/reproducibility`, and `POST /{id}/rerun`.

- [ ] **Step 1: Write failing history tests**

Assert stable cursor pagination, filters by status/intent/observation, no local paths, ordered events, exact registry/input/artifact hashes in reproducibility, and rerun creates new IDs while retaining the original frozen plan when registrations remain available.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_analyses_api.py -q`
Expected: FAIL with 404.

- [ ] **Step 3: Implement read routes and rerun guard**

Rerun returns `409 WORKFLOW_VERSION_UNAVAILABLE` if the original tool/model/profile cannot be loaded. It never silently upgrades a workflow and never overwrites an analysis.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/integration/test_analyses_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/app tests/integration/test_analyses_api.py
git commit -m "feat: persist and inspect reproducible analyses"
```

---

### Task 14: Store and Serve Derived Artifacts, GeoJSON, and Tiles

**Files:**
- Modify: `satquery/artifacts/store.py`
- Create: `apps/api/app/routes/v1_artifacts.py`
- Modify: `satquery/execution/adapters.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/artifacts/test_store.py`
- Test: `tests/integration/test_artifacts_api.py`

**Interfaces:**
- Extends the Task 9 `ArtifactStore` with safe API projection and produces metadata/download routes, `GET /api/v1/evidence/{id}/geojson`, and canonical tile aliases.

- [ ] **Step 1: Extend artifact tests for the public boundary**

Keep Task 9 staging/publication coverage and add registered-ID-only HTTP resolution, traversal rejection, safe content disposition, mask grid provenance, bbox-to-polygon conversion, pixel-space labeling, and no invented geographic GeoJSON without CRS.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/artifacts tests/integration/test_artifacts_api.py -q`
Expected: FAIL because public artifact, GeoJSON, and tile routes are absent.

- [ ] **Step 3: Harden controlled artifact resolution for HTTP use**

Resolve registered IDs through `ArtifactStore.resolve()`, verify metadata and SHA-256 before serving, enforce `relative_to(artifacts_root)`, and return only approved media types and suffixes.

- [ ] **Step 4: Verify adapter publication boundaries**

Execution stages outputs per step. Publish immutable artifacts only after adapter contract checks pass; link evidence and mark the analysis complete only after Task 11 verification. Failed/cancelled jobs remove staging data, while an already-published but unlinked artifact is retained for audit and later garbage-collection policy.

- [ ] **Step 5: Add safe API projection and tile alias**

Reuse `RasterTileService` for supported raster artifacts. Do not accept arbitrary colormap expressions; accept only registered `colormap_id` values.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/artifacts tests/integration/test_artifacts_api.py tests/visualization -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add satquery/artifacts satquery/execution/adapters.py apps/api/app tests/artifacts tests/integration/test_artifacts_api.py
git commit -m "feat: expose spatial evidence artifacts"
```

---

### Task 15: Generate Deterministic JSON and HTML Reports

**Files:**
- Create: `satquery/reporting/reports.py`
- Modify: `satquery/reporting/__init__.py`
- Create: `apps/api/app/routes/v1_reports.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/reporting/test_reports.py`
- Test: `tests/integration/test_reports_api.py`

**Interfaces:**
- Produces `ReportBuilder.build(analysis) -> AnalysisReport`, `render_html(report) -> str`, and JSON/HTML endpoints. PDF endpoint returns `501 PDF_REPORT_DISABLED`.

- [ ] **Step 1: Write failing report tests**

Assert all required sections, HTML escaping of query/model text, exact provenance, no local paths/hidden prompts, correct content types, immutable report artifact hash, and disabled PDF behavior.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/reporting tests/integration/test_reports_api.py -q`
Expected: FAIL because reporting implementation is absent.

- [ ] **Step 3: Implement Pydantic report and stdlib HTML rendering**

Use `html.escape` and explicit markup; do not add a template dependency for one report. Include query, sanitized inputs, workflow, answer, evidence links, measurements, warnings, verification, trace summary, and reproducibility IDs.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/reporting tests/integration/test_reports_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add satquery/reporting apps/api/app tests/reporting tests/integration/test_reports_api.py
git commit -m "feat: generate auditable analysis reports"
```

---

### Task 16: Stream Persistent Progress and Support Cancellation

**Files:**
- Modify: `satquery/execution/jobs.py`
- Modify: `apps/api/app/routes/v1_jobs.py`
- Test: `tests/integration/test_job_events.py`

**Interfaces:**
- Produces `GET /api/v1/jobs/{id}/events` as SSE with `Last-Event-ID` resume and best-effort cancellation.

- [ ] **Step 1: Write failing SSE tests**

Create persisted events, connect after event 2, assert only later events arrive in ID order, heartbeat comments do not contain data, disconnect does not cancel a job, queued cancellation is immediate, and running cancellation occurs at a safe checkpoint.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_job_events.py -q`
Expected: FAIL because event streaming is absent.

- [ ] **Step 3: Implement database-backed SSE polling**

Use `StreamingResponse` with an async generator, short `asyncio.sleep`, bounded heartbeat interval, terminal-state exit, and request disconnect checks. Events are read from SQLite, so restart/resume works without an in-memory broker.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/integration/test_job_events.py tests/execution/test_jobs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add satquery/execution/jobs.py apps/api/app/routes/v1_jobs.py tests/integration/test_job_events.py
git commit -m "feat: stream analysis progress and support cancellation"
```

---

### Task 17: Add Exact Scientific Caching and Idempotent Submission

**Files:**
- Create: `satquery/execution/cache.py`
- Modify: `satquery/execution/engine.py`
- Modify: `apps/api/app/routes/v1_query.py`
- Test: `tests/execution/test_cache.py`
- Test: `tests/integration/test_idempotency.py`

**Interfaces:**
- Produces `build_cache_key(tool, inputs, parameters, roi) -> str` and `Idempotency-Key` handling for query submission.

- [ ] **Step 1: Write failing cache tests**

Assert key changes with tool version, checkpoint hash, preprocessing hash, input hashes, canonical parameters, ROI, planner version, and registry hash; assert dictionary order does not change the key. Assert same idempotency key + same body returns original IDs, while same key + different body returns `409 IDEMPOTENCY_CONFLICT`.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/execution/test_cache.py tests/integration/test_idempotency.py -q`
Expected: FAIL because cache/idempotency are absent.

- [ ] **Step 3: Implement canonical hashing**

Serialize with `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)` and SHA-256. Cache only immutable successful evidence whose artifact hashes still verify. Never cache failures or mutable staging paths.

- [ ] **Step 4: Implement atomic idempotency records**

Store request-body hash and response IDs in the query submission transaction. Do not apply idempotency to GET routes.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/execution/test_cache.py tests/integration/test_idempotency.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add satquery/execution apps/api/app/routes/v1_query.py tests/execution/test_cache.py tests/integration/test_idempotency.py
git commit -m "feat: add reproducible analysis caching"
```

---

### Task 18: Enforce API and Resource-Abuse Boundaries

**Files:**
- Create: `apps/api/app/security.py`
- Modify: `apps/api/app/main.py`
- Modify: `satquery/ingestion/config.py`
- Modify: `satquery/execution/jobs.py`
- Test: `tests/integration/test_api_security.py`

**Interfaces:**
- Produces environment-backed settings for exact CORS origins, query bytes, ROI vertices, queue size, result bytes, and optional API key. Uses existing raster limits for uploads.

- [ ] **Step 1: Write failing security tests**

Test disallowed origin, oversized query, excessive ROI vertices, raw path/URL/model ID injection, full queue (`429`), busy model (`503`), invalid API key (`401`), redacted internal exception, and absent auth scheme when API-key mode is disabled.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_api_security.py -q`
Expected: FAIL because security settings are absent.

- [ ] **Step 3: Implement strict settings and optional auth**

Parse comma-separated origins exactly; never use wildcard with credentials. Compare API keys with `secrets.compare_digest`. Put API-key auth on `/api/v1` only when configured and expose FastAPI `APIKeyHeader` so OpenAPI matches runtime behavior.

- [ ] **Step 4: Enforce limits before expensive work**

Reject request/query/ROI/queue violations before raster reads or model loading. Preserve existing upload streaming limits.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/integration/test_api_security.py tests/ingestion tests/execution -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/app/security.py apps/api/app/main.py satquery/ingestion/config.py satquery/execution/jobs.py tests/integration/test_api_security.py
git commit -m "feat: harden backend execution boundaries"
```

---

### Task 19: Add Structured Observability and Cheap Readiness

**Files:**
- Create: `satquery/observability.py`
- Modify: `apps/api/app/routes/v1_system.py`
- Modify: `apps/api/app/main.py`
- Test: `tests/integration/test_health_and_logging.py`

**Interfaces:**
- Produces JSON log records with correlation fields, `GET /health/ready`, `/api/v1/system/status`, and `/limits`.

- [ ] **Step 1: Write failing health/log tests**

Assert liveness performs no registry/database/model load, readiness checks SQLite, filesystem, registries, queue, and cheap model state without loading checkpoints; assert logs include request/job/analysis IDs and omit query text, secrets, paths, and hidden prompts.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_health_and_logging.py -q`
Expected: FAIL because readiness/logging are absent.

- [ ] **Step 3: Implement stdlib JSON logging and probes**

Use `logging.Formatter`; no metrics dependency. Return `200` ready or `503` with component statuses. Model status comes from registry/resource state and file/hash checks, never inference.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/integration/test_health_and_logging.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add satquery/observability.py apps/api/app tests/integration/test_health_and_logging.py
git commit -m "feat: add backend health and observability"
```

---

### Task 20: Make OpenAPI a Tested Product Contract

**Files:**
- Modify: `apps/api/app/openapi.py`
- Modify: all `apps/api/app/routes/v1_*.py`
- Test: `tests/integration/test_openapi_contract.py`

**Interfaces:**
- Produces `/docs`, `/redoc`, `/openapi.json`, stable tagged operations, schema-valid examples, structured non-2xx responses, and deprecated legacy markers.

- [ ] **Step 1: Write failing schema-quality tests**

Iterate over `/api/v1` operations and assert unique explicit `operationId`, tags, summary, response schema, at least one structured non-2xx response for writes, no property named `path`, examples validate with their Pydantic model, legacy routes are marked deprecated, and security appears only when enabled.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_openapi_contract.py -q`
Expected: FAIL with missing metadata.

- [ ] **Step 3: Complete route metadata**

Set explicit `operation_id`, response models, status codes, descriptions, and examples. Configure Swagger options `displayRequestDuration`, `filter`, `persistAuthorization`, and `tryItOutEnabled`.

- [ ] **Step 4: Handle offline docs without adding a package**

Vendor the exact Swagger/ReDoc static files only if deployment tests prove CDN access unavailable; record their source/version/hash under `apps/api/static/docs/`. Otherwise keep FastAPI defaults and document that offline docs assets remain deployment-gated. The OpenAPI JSON endpoint must always work offline.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/integration/test_openapi_contract.py tests/integration -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/app tests/integration/test_openapi_contract.py
git commit -m "docs: complete interactive backend API documentation"
```

**Milestone M4 gate:** Start the app locally, open `/docs`, submit one plan, submit one deterministic job, follow SSE, inspect evidence/history, and download JSON/HTML reports. Record only observed behavior.

---

### Task 21: Prove Complete Backend Workflows

**Files:**
- Create: `tests/integration/test_backend_workflows.py`
- Create: `tests/integration/test_backend_failures.py`

**Interfaces:**
- Consumes the public API only, with injected model backends where downloading a checkpoint is not part of the test.
- Produces end-to-end regression proof for every Task 0 available capability and every required failure boundary.

- [ ] **Step 1: Add failing happy-path scenarios**

Implement fixture-driven flows for:

```text
upload -> inspect -> plan -> execute -> evidence -> trace -> report -> restart -> history
```

Cover single-image VQA, grounding, deterministic spectral/temporal analysis, deterministic area measurement, and each accepted Phase 4 specialist. Optical/SAR scenarios retain separate modality-attributed evidence and never call marginal fusion “strong improvement.”

- [ ] **Step 2: Add failing refusal scenarios**

Cover missing NIR, unknown SAR domain, wrong image count, unknown temporal order, no overlap, unverified pixel grid, missing CRS for area, model unavailable, resource busy, invalid ROI, corrupted artifact, cancellation, interrupted restart, unsupported RISAT-to-Sentinel-1 specialist request, and numeric caption text without measurement evidence.

- [ ] **Step 3: Run and diagnose failures**

Run: `python -m pytest tests/integration/test_backend_workflows.py tests/integration/test_backend_failures.py -q`
Expected before final wiring: FAIL at the first unintegrated boundary; fix only wiring/contract defects exposed by these tests, not scientific model behavior.

- [ ] **Step 4: Run the complete affected suite**

Run:

```bash
python -m pytest tests/agent tests/artifacts tests/evidence tests/execution tests/geo tests/ingestion tests/inference tests/integration tests/persistence tests/registry tests/reporting tests/verification -q
```

Expected: PASS. Record skipped model-runtime tests with exact missing dependency/checkpoint reason.

- [ ] **Step 5: Commit**

```bash
git add tests/integration satquery apps/api/app
git commit -m "test: verify complete backend workflows"
```

---

### Task 22: Package the Single Backend Runtime

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.env.example`
- Modify: `apps/api/README.md`
- Test: `tests/integration/test_runtime_config.py`

**Interfaces:**
- Produces one API image, documented volume layout, startup migration, healthcheck, and environment contract. Compose is not created unless Task 10 proves an incompatible isolated worker is required.

- [ ] **Step 1: Write failing runtime configuration tests**

Assert `.env.example` contains names but no secrets, every setting parses, data/database paths live under the mounted data root by default, startup migration is idempotent, and missing writable storage makes readiness fail.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/integration/test_runtime_config.py -q`
Expected: FAIL because packaging/config documentation is absent.

- [ ] **Step 3: Add minimal multi-stage Dockerfile**

Install the project from `pyproject.toml`, run as a non-root user, expose the configured port, mount `/data`, start `uvicorn apps.api.app.main:app`, and use `/health/ready` for healthcheck. Do not copy local data, model cache, `.env`, Git metadata, or experiment outputs into the image.

- [ ] **Step 4: Document local and container startup**

Include exact environment names, storage ownership, optional model cache mount, API-key mode, CORS configuration, docs URLs, and shutdown semantics.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest tests/integration/test_runtime_config.py -q
docker build -t satquery-backend:phase5 .
docker run --rm -d --name satquery-phase5 -p 8000:8000 -v satquery-data:/data satquery-backend:phase5
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready').status)"
docker stop satquery-phase5
```

Expected: tests PASS, image builds, readiness returns `200`, container stops cleanly. If Docker is unavailable, record that the image was not built; do not claim packaging verification.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore .env.example apps/api/README.md tests/integration/test_runtime_config.py
git commit -m "build: package SatQuery backend runtime"
```

---

### Task 23: Audit and Close Phase 5

**Files:**
- Create: `experiments/phase5_backend/PHASE_5_CLOSEOUT.json`
- Modify: `experiments/phase5_backend/README.md`
- Modify: `docs/DEVELOPMENT_PLAN.md`

**Interfaces:**
- Produces an evidence-backed closeout with Git SHA, contract/schema/registry hashes, route inventory, observed test results, runtime checks, capability matrix, schema version, limits, known restrictions, and external dependencies.

- [ ] **Step 1: Generate auditable inventories**

From code/runtime rather than hand entry, capture:

```text
Git SHA
OpenAPI SHA-256 and operation IDs
model/tool/preprocessing registry SHA-256 values
SQLite schema version
available/unavailable capabilities and reasons
configured security/resource limits
pytest commands, return codes, pass/fail/skip counts
container verification result
```

- [ ] **Step 2: Reconstruct representative evidence**

For each required local workflow, verify analysis input hashes, plan hash, evidence rows, edges, artifact hashes, measurement source, verification report, and report links agree. Do not create benchmark metrics during backend closeout.

- [ ] **Step 3: Run final verification**

```bash
python -m pytest
python -m compileall -q satquery apps ml scripts
git diff --check
git status --short
```

Expected: pytest PASS except explicitly justified runtime skips; compileall and diff check exit `0`; status contains only intended closeout/documentation changes.

- [ ] **Step 4: Write closeout from observed results**

Set `status` to `COMPLETE` only if all acceptance conditions below are met. Otherwise use `BLOCKED`, list exact blockers, and leave `docs/DEVELOPMENT_PLAN.md` unchanged for Phase 5 status.

- [ ] **Step 5: Update the canonical roadmap and commit**

Canonical Phase 4 must already be `COMPLETE` before Task 0 permits implementation. After a complete Phase 5 closeout, set Phase 5 to `COMPLETE` and Phase 6 to `NEXT`; do not rewrite Phase 4's earlier closeout state.

```bash
git add experiments/phase5_backend docs/DEVELOPMENT_PLAN.md
git commit -m "docs: close phase 5 backend and evidence engine"
```

---

## 4. Acceptance Conditions

Phase 5 is complete only when all statements below are evidenced:

1. Existing legacy APIs retain their tested behavior.
2. `/api/v1` has stable strict schemas and one structured error envelope.
3. Uploaded immutable assets survive process restart and are indexed without path leakage.
4. Pair validation preserves temporal, modality, overlap, CRS, resolution, and alignment facts.
5. The deterministic interpreter routes all supported frozen examples and refuses ambiguous/adversarial requests safely.
6. Every executable plan contains only registered tools, bounded parameters, at most eight steps, and an acyclic dependency graph.
7. Quantitative requests cannot complete without deterministic `MeasurementEvidence` linked to spatial evidence.
8. SAR and temporal preconditions fail closed; no generic sensor substitution occurs.
9. Jobs are persistent, bounded, cancellable at safe checkpoints, and interrupted correctly after restart.
10. Evidence and derived artifacts are immutable, hash-verified, and traceable to source inputs and producers.
11. Verification runs before answer completion and prevents unsupported numeric/spatial claims.
12. Analysis history, trace, reproducibility, artifacts, GeoJSON where valid, tiles, and JSON/HTML reports are available through documented APIs.
13. Cache and idempotency keys include every scientific input/version that can change a result.
14. Security limits execute before expensive work, and responses/logs reveal no secrets, local paths, stack traces, or private reasoning.
15. Liveness is cheap; readiness reports actual dependency and capability state without loading large models.
16. OpenAPI operations are uniquely identified, tagged, typed, example-backed, and usable by the Phase 6 frontend without undocumented knowledge.
17. The complete affected test suite passes, and unavailable external runtimes are reported as unavailable rather than mocked into production capability.
18. Closeout claims are reconstructible from stored test output and hashes.

## 5. Explicitly Deferred

The following are not Phase 5 completion requirements:

- custom frontend work;
- generated TypeScript client code (Phase 6 consumes the stable OpenAPI contract);
- arbitrary workflow/plugin execution;
- cloud-only language-model planning;
- generic sensor conversion;
- calibrated aggregate confidence without a separate calibration experiment;
- PDF reports without an audited renderer;
- Prometheus metrics without an operational consumer;
- model administration endpoints;
- physical observation deletion;
- automatic reprojection/registration without a registered, tested scientific tool;
- distributed workers or infrastructure.

Add any deferred item only when a measured product or deployment requirement makes the simpler architecture insufficient.
