# Phase 5 API Contract (frozen at Task 0)

This document freezes **route families and cross-cutting API behavior only**.
Per-endpoint request/response details are implemented in Tasks 1–20 and pinned
by the OpenAPI regression tests. Legacy routes stay backward-compatible for
the whole of Phase 5.

## Canonical prefix

`/api/v1`

## Route families

| Family | Purpose (Phase 5 plan tasks) |
| --- | --- |
| `/api/v1/system` | version, capabilities, status, limits (Tasks 1, 5, 19) |
| `/api/v1/observations` | immutable upload, list/get, metadata/assets (Task 3) |
| `/api/v1/pairs` | validated temporal pair creation and inspection (Task 4) |
| `/api/v1/query` | plan-only and 202-query submission (Tasks 8, 12, 17) |
| `/api/v1/analyses` | history, trace, evidence, reproducibility, rerun (Task 13) |
| `/api/v1/jobs` | job state, cancellation, SSE progress (Tasks 9, 16) |
| `/api/v1/tools` | registered tool introspection (Task 5) |
| `/api/v1/models` | registered model introspection with per-field status (Task 5) |
| `/api/v1/evidence` | evidence graph nodes/edges; GeoJSON where valid (Tasks 11, 14) |
| `/api/v1/artifacts` | immutable derived-artifact metadata and download (Task 14) |
| `/api/v1/reports` | deterministic JSON/HTML; PDF returns `501 PDF_REPORT_DISABLED` (Task 15) |
| `/api/v1/tiles` | canonical tile aliases over registered raster artifacts (Task 14) |

Families are unique; no two families overlap in resource semantics.

## Documentation (mandatory, not optional polish)

- `/docs` — interactive Swagger UI
- `/redoc` — human-readable reference
- `/openapi.json` — machine-readable contract (must work offline)

Every canonical endpoint eventually has: stable `operation_id`, tag, summary,
description, typed request, typed response, structured error responses,
example payloads, documented status codes. OpenAPI receives automated
regression tests (Task 20). For offline/local SIH deployment, docs assets
must be self-hostable.

## Error envelope (single, frozen)

```json
{
  "error": {
    "code": "STABLE_ERROR_CODE",
    "message": "human-readable; no paths, stack traces, or secrets",
    "outcome": "ALLOW | ALLOW_WITH_WARNING | REQUEST_INPUT | ABSTAIN | REJECT",
    "details": {},
    "request_id": "req_<32 hex>"
  }
}
```

HTTP status and scientific outcome are independent concepts. Example:
`HTTP 422` + `outcome: REQUEST_INPUT` for a missing semantic NIR mapping.

## State enums (frozen)

Analysis: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `ABSTAINED`,
`REJECTED`, `CANCELLED`, `INTERRUPTED`.

Job: `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCEL_REQUESTED`,
`CANCELLED`, `INTERRUPTED`.

## Cross-cutting guarantees

- `X-Request-ID` on every response; server-generated `req_` IDs.
- Numeric/spatial claims come only from deterministic measurement evidence.
- No local paths, checkpoint paths, secrets, hidden prompts, or
  chain-of-thought in any response or log.
- Idempotency applies to query submission; GETs are never idempotency-guarded.
- Security limits (origins, query bytes, ROI vertices, queue size, API key)
  execute before expensive work.
