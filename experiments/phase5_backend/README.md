# Phase 5 Backend — Experiment Record

Phase 5 backend artifacts (contracts, closeout) live in this directory.
`PHASE_5_CLOSEOUT.json` does **not** exist yet and is created only at Task 23.

## Start state (frozen at Task 0)

- **Mode:** `RESTRICTED_CAPABILITY`
- **Phase 4 closeout:** `experiments/phase4_temporal_analytics/PHASE_4_CLOSEOUT.json`
  (SHA-256 `84b890f56c990ad83497668e1393a3086248ae8af9babb1eab081e9770e582e3`,
  status `BLOCKED`, closeout git SHA `660466161cb187208a63111698fd11a14f3412bc`)
- **Starting git SHA:** `5d2adc7a95fb3853788a905782a8e5f32b058031`
- **Human review:** the Phase 4 closeout (BLOCKED status, measured P4-E01 lanes
  A–C, three contract-blocked learned lanes) was reviewed when Task 0 froze
  this contract; its hash/reconstruction checks are enforced by
  `tests/phase4/test_phase4_closeout.py`.

## Why RESTRICTED_CAPABILITY

Phase 4 is truthfully `BLOCKED`, not `COMPLETE`. Independently verified
capabilities are exposed; blocked capabilities stay unavailable:

**Locally verified (exposed):**
- deterministic multispectral formulas (NDVI/NDWI/MNDWI)
- temporal pair/grid preparation and signed temporal differences
- CRS-safe deterministic area measurement
- deterministic SAR temporal change (explicit radiometric contracts)
- diagnostic mask compatibility/agreement
- Phase 1–3 frozen capabilities (ingestion, metadata, pairing, tiles,
  SmolVLM VQA, Grounding DINO, BIFOLD S1/S2/joint v0.2.0 with their frozen
  scientific limitations)

**Blocked / not available (never exposed as runnable):**
- ChangerEx structural-change specialist
- Chg2Cap change-caption specialist
- STURM learned flood specialist
- OSCD Lane-D generic deterministic benchmark

The full frozen inventory, state enums, route families, error envelope,
security boundary, and blocker carry-forward are in `backend_contract.yaml`;
the API surface is frozen in `api_contract.md`.

## Task 0 tool-registry audit finding

`satquery/registry/tools.yaml` contains two declarative entries
(`sar_temporal_change_v1`, `mask_agreement_v1`) that map to real, tested
implementations, but there is **no strict ToolRegistry parser/schema yet**.
Implementing the strict loader is a **Phase 5 Task 5 prerequisite**. The
entries are retained.

## Task 0 boundaries

Task 0 froze contracts only. It did not: download any dataset or checkpoint,
run Kaggle, change Phase 4 results, promote temporal models, implement any
backend route, or fabricate capability readiness. Runtime readiness probing
for learned models happens at Phase 5 Task 5, never here.
