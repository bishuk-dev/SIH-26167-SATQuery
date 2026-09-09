"""Task 17 — exact scientific cache keys and reproducible result caching.

Cache keys must change with every scientific input that can change a result
(tool identity/version, checkpoint hash, preprocessing hash, input hashes,
parameters, ROI, planner version, registry hash) and must be invariant to
dictionary insertion order. Cache entries store only successful, immutable
results whose published artifact hashes still verify.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from satquery.artifacts import ArtifactStore
from satquery.evidence.graph import EvidenceGraph
from satquery.evidence.models import (
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    MaskAsset,
    MeasurementEvidence,
    TemporalPairEvidence,
)
from satquery.execution import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionPlan,
    JobRunner,
    PlanStep,
    ToolExecutionError,
    ToolResult,
)
from satquery.execution.cache import AnalysisCache, CacheKeyError, build_cache_key
from satquery.execution.jobs import _evidence_edges
from satquery.execution.models import ArtifactMetadata, ArtifactOutput
from satquery.persistence import (
    AnalysisRecord,
    AnalysisStatus,
    Database,
    JobStatus,
    MetadataRepository,
)
from satquery.registry.tools import load_tool_registry

_REGISTRY_HASH = "a" * 64
_TOOL_ID = "compute_mask_area_v1"


def _tool(**overrides: object) -> dict[str, object]:
    tool: dict[str, object] = {
        "tool_id": _TOOL_ID,
        "version": "v1",
        "checkpoint_sha256": "b" * 64,
        "preprocessing_sha256": "c" * 64,
    }
    tool.update(overrides)
    return tool


def _key_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "inputs": {"observation": "d" * 64},
        "parameters": {"threshold": 2.0},
        "roi": {"type": "polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]},
        "planner_version": "1",
        "registry_hash": _REGISTRY_HASH,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# build_cache_key
# ---------------------------------------------------------------------------


def test_cache_key_is_sha256_of_canonical_serialization() -> None:
    payload = {
        "cache_schema_version": 1,
        "tool": _tool(),
        "inputs": {"observation": "d" * 64},
        "parameters": {"threshold": 2.0},
        "roi": {"type": "polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]},
        "planner_version": "1",
        "registry_hash": _REGISTRY_HASH,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert build_cache_key(_tool(), **_key_kwargs()) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"tool": {"tool_id": "mask_agreement_v1", "version": "v1"}},
        {"tool": _tool(version="v2")},
        {"tool": _tool(checkpoint_sha256="e" * 64)},
        {"tool": _tool(preprocessing_sha256="f" * 64)},
        {"inputs": {"observation": "1" * 64}},
        {"inputs": {"other_observation": "d" * 64}},
        {"parameters": {"threshold": 3.0}},
        {"parameters": {"threshold": 2.0, "unit": "ha"}},
        {"roi": {"type": "polygon", "coordinates": [[[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]]}},
        {"roi": None},
        {"planner_version": "2"},
        {"registry_hash": "b" * 64},
    ],
)
def test_cache_key_changes_with_every_scientific_input(overrides: dict[str, object]) -> None:
    baseline = build_cache_key(_tool(), **_key_kwargs())
    kwargs = {**_key_kwargs(), **overrides}
    changed = build_cache_key(kwargs.pop("tool", _tool()), **kwargs)
    assert changed != baseline


def test_dictionary_insertion_order_does_not_change_cache_key() -> None:
    first = build_cache_key(
        _tool(),
        inputs={"a": "1" * 64, "b": "2" * 64},
        parameters={"threshold": 2.0, "unit": "ha"},
        roi={"lat": 1.0, "lon": 2.0},
        planner_version="1",
        registry_hash=_REGISTRY_HASH,
    )
    second = build_cache_key(
        _tool(),
        inputs={"b": "2" * 64, "a": "1" * 64},
        parameters={"unit": "ha", "threshold": 2.0},
        roi={"lon": 2.0, "lat": 1.0},
        planner_version="1",
        registry_hash=_REGISTRY_HASH,
    )
    assert first == second


def test_nan_and_non_serializable_values_fail_closed() -> None:
    with pytest.raises(CacheKeyError):
        build_cache_key(**{**_key_kwargs(), "tool": _tool(), "parameters": {"threshold": float("nan")}})
    with pytest.raises(CacheKeyError):
        build_cache_key(**{**_key_kwargs(), "tool": _tool(), "parameters": {"value": object()}})


def test_registry_hash_and_planner_version_are_required() -> None:
    with pytest.raises(CacheKeyError):
        build_cache_key(**{**_key_kwargs(), "tool": _tool(), "registry_hash": ""})
    with pytest.raises(CacheKeyError):
        build_cache_key(**{**_key_kwargs(), "tool": _tool(), "planner_version": ""})
    with pytest.raises(CacheKeyError):
        build_cache_key(**{**_key_kwargs(), "tool": {}})


# ---------------------------------------------------------------------------
# AnalysisCache
# ---------------------------------------------------------------------------


def _repository(tmp_path: Path) -> MetadataRepository:
    database = Database(tmp_path / "satquery.db")
    database.migrate()
    return MetadataRepository(database)


def _entry_payload() -> dict[str, object]:
    return {"output": {"area": 1.5, "unit": "ha"}, "evidence": None, "artifacts": []}


def test_cache_roundtrip_returns_stored_payload(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    assert cache.put("0" * 64, _entry_payload()) is True
    assert cache.get("0" * 64) == _entry_payload()


def test_cache_rejects_malformed_key_or_payload(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    with pytest.raises(CacheKeyError):
        cache.put("not-a-cache-key", _entry_payload())
    with pytest.raises(CacheKeyError):
        cache.put("0" * 64, {"output": {"value": object()}, "evidence": None, "artifacts": []})


def test_cache_get_returns_none_for_missing_or_corrupt_entry(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    cache = AnalysisCache(repository)
    assert cache.get("0" * 64) is None
    with repository._db.transaction() as connection:
        connection.execute(
            "INSERT INTO cache_entries(cache_key, created_at, payload_json) VALUES (?, ?, ?)",
            ("0" * 64, "2026-01-01T00:00:00Z", "{not json"),
        )
    assert cache.get("0" * 64) is None


def test_cached_artifacts_must_still_verify_against_store(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "data")
    staged = store.stage("tif")
    staged.path.write_bytes(b"mask-bytes")
    record = store.publish(staged, ArtifactMetadata())

    cache = AnalysisCache(_repository(tmp_path))
    payload = {
        "output": {},
        "evidence": None,
        "artifacts": [{"artifact_id": record.artifact_id, "sha256": record.sha256}],
    }
    assert cache.put("0" * 64, payload) is True
    assert cache.get("0" * 64, artifact_store=store) == payload

    artifact_file = next((store.artifacts_root / record.artifact_id).glob("artifact.*"))
    artifact_file.write_bytes(b"tampered-bytes")
    assert cache.get("0" * 64, artifact_store=store) is None


def test_cached_artifacts_with_wrong_recorded_hash_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "data")
    staged = store.stage("tif")
    staged.path.write_bytes(b"mask-bytes")
    record = store.publish(staged, ArtifactMetadata())

    cache = AnalysisCache(_repository(tmp_path))
    payload = {
        "output": {},
        "evidence": None,
        "artifacts": [{"artifact_id": record.artifact_id, "sha256": "9" * 64}],
    }
    cache.put("0" * 64, payload)
    assert cache.get("0" * 64, artifact_store=store) is None


def test_cache_get_without_store_rejects_entries_with_artifacts(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    payload = {
        "output": {},
        "evidence": None,
        "artifacts": [{"artifact_id": "artifact_" + "0" * 32, "sha256": "a" * 64}],
    }
    cache.put("0" * 64, payload)
    assert cache.get("0" * 64) is None


# ---------------------------------------------------------------------------
# engine integration
# ---------------------------------------------------------------------------


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        steps=(PlanStep("a", _TOOL_ID, parameters={"unit": "ha"}),),
        planner_version="1",
    )


def _cached_engine(tmp_path: Path, adapter: object, cache: AnalysisCache) -> ExecutionEngine:
    store = ArtifactStore(tmp_path / "data")
    return ExecutionEngine(
        {_TOOL_ID: adapter},  # type: ignore[dict-item]
        artifact_store=store,
        tool_registry=load_tool_registry(),
        cache=cache,
    )


def test_engine_reuses_cached_step_without_reexecution(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    calls: list[object] = []

    class Counter:
        def execute(self, call: object, context: object) -> ToolResult:
            calls.append(call)
            return ToolResult(output={"area": 1.5, "unit": "ha"})

    context = ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32)
    first = _cached_engine(tmp_path, Counter(), cache).execute(_plan(), context)
    assert len(calls) == 1

    events: list[tuple[str, dict[str, object]]] = []
    replay_context = ExecutionContext(
        "job_" + "b" * 32,
        "ana_" + "b" * 32,
        metadata={"event_callback": lambda event_type, payload: events.append((event_type, payload))},
    )
    second = _cached_engine(tmp_path, Counter(), cache).execute(_plan(), replay_context)
    assert len(calls) == 1
    assert second[0].output == first[0].output
    assert any(
        event_type == "STEP_SUCCEEDED" and payload.get("cache_hit") is True
        for event_type, payload in events
    )


def test_failed_step_is_never_cached(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    state = {"calls": 0}

    class Flaky:
        def execute(self, call: object, context: object) -> ToolResult:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("tool failure")
            return ToolResult(output={"ok": True})

    engine = _cached_engine(tmp_path, Flaky(), cache)
    with pytest.raises(ToolExecutionError):
        engine.execute(_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    recovered = engine.execute(_plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32))
    assert state["calls"] == 2
    assert recovered[0].output == {"ok": True}

    cached_engine = _cached_engine(tmp_path, Flaky(), cache)
    replay = cached_engine.execute(_plan(), ExecutionContext("job_" + "c" * 32, "ana_" + "c" * 32))
    assert state["calls"] == 2
    assert replay[0].output == {"ok": True}


def test_non_serializable_step_output_is_not_cached(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    state = {"calls": 0}

    class Unserializable:
        def execute(self, call: object, context: object) -> ToolResult:
            state["calls"] += 1
            return ToolResult(output={"value": object()})

    engine = _cached_engine(tmp_path, Unserializable(), cache)
    engine.execute(_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    engine.execute(_plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32))
    assert state["calls"] == 2


def test_engine_with_artifacts_replays_only_while_hashes_verify(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    state = {"calls": 0}

    class MaskProducing:
        def execute(self, call: object, context: object) -> ToolResult:
            state["calls"] += 1
            store = ArtifactStore(tmp_path / "data")
            staged = store.stage("tif")
            staged.path.write_bytes(b"mask-bytes-" + str(state["calls"]).encode())
            return ToolResult(
                output={"artifact_id": staged.artifact_id},
                artifacts=(ArtifactOutput(staged, ArtifactMetadata()),),
            )

    engine = _cached_engine(tmp_path, MaskProducing(), cache)
    first = engine.execute(_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    second = _cached_engine(tmp_path, MaskProducing(), cache).execute(
        _plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32)
    )
    assert state["calls"] == 1
    assert second[0].output == first[0].output

    store = ArtifactStore(tmp_path / "data")
    artifact_file = next(store.artifacts_root.glob("artifact_*/artifact.*"))
    artifact_file.write_bytes(b"corrupted")
    _cached_engine(tmp_path, MaskProducing(), cache).execute(
        _plan(), ExecutionContext("job_" + "c" * 32, "ana_" + "c" * 32)
    )
    assert state["calls"] == 2


def test_engine_without_registry_or_cache_never_reuses(tmp_path: Path) -> None:
    cache = AnalysisCache(_repository(tmp_path))
    state = {"calls": 0}

    class Counter:
        def execute(self, call: object, context: object) -> ToolResult:
            state["calls"] += 1
            return ToolResult(output={"area": 1.5})

    store = ArtifactStore(tmp_path / "data")
    # no tool registry: no registry hash, so caching must stay disabled
    engine = ExecutionEngine({_TOOL_ID: Counter()}, artifact_store=store, cache=cache)
    engine.execute(_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    engine.execute(_plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32))
    assert state["calls"] == 2


# ---------------------------------------------------------------------------
# replayed evidence identity
# ---------------------------------------------------------------------------


def _mask_evidence(observation_id: str) -> ChangeMaskEvidence:
    return ChangeMaskEvidence(
        evidence_id=f"evidence_{uuid4().hex}",
        target_class="sar_backscatter_change",
        change_kind="symmetric_change",
        temporal=TemporalPairEvidence(
            t1_observation_id=observation_id,
            t2_observation_id="obs_" + "f" * 32,
            order_source="metadata",
        ),
        mask=MaskAsset(
            asset_id="mask-asset",
            path="unused",
            sha256="0" * 64,
            width=4,
            height=4,
            source_grid_observation_id=observation_id,
        ),
        tool_id=_TOOL_ID,
        domain=DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=()),
        provenance=EvidenceProvenance(
            created_at=datetime.now(timezone.utc),
            operation_id="op",
            input_asset_id="asset",
        ),
    )


def _measurement_evidence(source: ChangeMaskEvidence) -> MeasurementEvidence:
    return MeasurementEvidence(
        evidence_id=f"evidence_{uuid4().hex}",
        source_evidence_id=source.evidence_id,
        value=1.0,
        unit="ha",
        method="test_method",
        calculation_crs="EPSG:32633",
        positive_pixel_count=1,
        valid_pixel_count=1,
        tool_id=_TOOL_ID,
        provenance=EvidenceProvenance(
            created_at=datetime.now(timezone.utc),
            operation_id="op-measure",
            input_asset_id=source.mask.asset_id,
            parent_evidence_ids=(source.evidence_id,),
        ),
    )


def _chain_plan() -> ExecutionPlan:
    return ExecutionPlan(
        steps=(
            PlanStep("a", _TOOL_ID, parameters={"step": "mask"}),
            PlanStep("b", _TOOL_ID, parameters={"unit": "ha"}, depends_on=("a",)),
        ),
        planner_version="1",
    )


def test_replayed_evidence_mints_fresh_ids_and_remaps_references(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    cache = AnalysisCache(repository)
    calls = {"mask": 0, "measure": 0}

    class Chain:
        def execute(self, call: object, context: object) -> ToolResult:
            if call.step_id == "a":  # type: ignore[attr-defined]
                calls["mask"] += 1
                return ToolResult(
                    output={"mask_step": True}, evidence=_mask_evidence("obs_" + "0" * 32)
                )
            calls["measure"] += 1
            source = call.prior_results["a"].evidence  # type: ignore[attr-defined]
            return ToolResult(output={"area": 1.0}, evidence=_measurement_evidence(source))

    def _engine() -> ExecutionEngine:
        return ExecutionEngine(
            {_TOOL_ID: Chain()},  # type: ignore[dict-item]
            artifact_store=ArtifactStore(tmp_path / "data"),
            tool_registry=load_tool_registry(),
            cache=cache,
        )

    first = _engine().execute(
        _chain_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32)
    )
    assert calls == {"mask": 1, "measure": 1}

    second = _engine().execute(
        _chain_plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32)
    )
    assert calls == {"mask": 1, "measure": 1}  # both steps replayed from cache

    first_mask, first_measurement = first[0].evidence, first[1].evidence
    second_mask, second_measurement = second[0].evidence, second[1].evidence
    assert second_mask.evidence_id != first_mask.evidence_id
    assert second_measurement.evidence_id != first_measurement.evidence_id
    assert second_measurement.source_evidence_id == second_mask.evidence_id
    assert second_measurement.provenance.parent_evidence_ids == (second_mask.evidence_id,)

    # the replayed graph must stay internally consistent (no dangling edges)
    graph = EvidenceGraph(
        nodes=(second_mask, second_measurement),
        edges=_evidence_edges((second_mask, second_measurement)),
    )
    assert graph.node_by_id[second_measurement.source_evidence_id] is second_mask


def test_unremappable_replay_reference_fails_closed_to_a_cache_miss(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    cache = AnalysisCache(repository)
    calls = {"mask": 0, "measure": 0}

    class Chain:
        def execute(self, call: object, context: object) -> ToolResult:
            if call.step_id == "a":  # type: ignore[attr-defined]
                calls["mask"] += 1
                return ToolResult(
                    output={"mask_step": True}, evidence=_mask_evidence("obs_" + "0" * 32)
                )
            calls["measure"] += 1
            source = call.prior_results["a"].evidence  # type: ignore[attr-defined]
            return ToolResult(output={"area": 1.0}, evidence=_measurement_evidence(source))

    def _engine() -> ExecutionEngine:
        return ExecutionEngine(
            {_TOOL_ID: Chain()},  # type: ignore[dict-item]
            artifact_store=ArtifactStore(tmp_path / "data"),
            tool_registry=load_tool_registry(),
            cache=cache,
        )

    _engine().execute(_chain_plan(), ExecutionContext("job_" + "a" * 32, "ana_" + "a" * 32))
    assert calls == {"mask": 1, "measure": 1}

    # Drop only the mask step's cache entry: the cached measurement still
    # references the first run's mask evidence ID, which cannot be remapped in
    # this run, so the measurement step must re-execute instead of replaying.
    with repository._db.transaction() as connection:
        cursor = connection.execute(
            "DELETE FROM cache_entries WHERE payload_json LIKE '%\"mask_step\"%'"
        )
        assert cursor.rowcount == 1

    second = _engine().execute(
        _chain_plan(), ExecutionContext("job_" + "b" * 32, "ana_" + "b" * 32)
    )
    assert calls == {"mask": 2, "measure": 2}
    assert second[1].evidence.source_evidence_id == second[0].evidence.evidence_id


def test_cached_replay_through_job_runner_completes_second_analysis(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    cache = AnalysisCache(repository)

    class MaskOnly:
        def execute(self, call: object, context: object) -> ToolResult:
            return ToolResult(
                output={"mask_step": True}, evidence=_mask_evidence("obs_" + "0" * 32)
            )

    engine = ExecutionEngine(
        {_TOOL_ID: MaskOnly()},  # type: ignore[dict-item]
        artifact_store=ArtifactStore(tmp_path / "data"),
        tool_registry=load_tool_registry(),
        cache=cache,
    )
    now = datetime.now(timezone.utc)
    payload = {
        "query": "show me where change happened",
        "intent": {"task_family": "CHANGE_LOCALIZE", "matched_rule": "test-fixture"},
        "observation_ids": ["obs_" + "0" * 32, "obs_" + "f" * 32],
    }
    for suffix in ("a" * 32, "b" * 32):
        repository.create_analysis(
            AnalysisRecord(
                analysis_id=f"ana_{suffix}",
                status=AnalysisStatus.PENDING,
                created_at=now,
                updated_at=now,
                payload=dict(payload),
            )
        )
    runner = JobRunner(repository, engine)
    runner.start()
    try:
        jobs = [runner.submit(f"ana_{suffix}", _plan()) for suffix in ("a" * 32, "b" * 32)]
        for job in jobs:
            deadline = time.time() + 5
            while time.time() < deadline:
                record = repository.get_job(job.job_id)
                if record is not None and record.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                    JobStatus.INTERRUPTED,
                }:
                    break
                time.sleep(0.01)
            assert repository.get_job(job.job_id).status is JobStatus.SUCCEEDED
    finally:
        runner.stop()

    evidence_ids: dict[str, list[str]] = {}
    for suffix in ("a" * 32, "b" * 32):
        with repository._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT evidence_id FROM evidence WHERE analysis_id = ?",
                (f"ana_{suffix}",),
            ).fetchall()
        evidence_ids[suffix] = [row["evidence_id"] for row in rows]
    assert len(evidence_ids["a" * 32]) == 1
    assert len(evidence_ids["b" * 32]) == 1
    assert evidence_ids["a" * 32] != evidence_ids["b" * 32]
