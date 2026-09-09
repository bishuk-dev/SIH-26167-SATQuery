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
from pathlib import Path

import pytest

from satquery.artifacts import ArtifactStore
from satquery.execution import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionPlan,
    PlanStep,
    ToolExecutionError,
    ToolResult,
)
from satquery.execution.cache import AnalysisCache, CacheKeyError, build_cache_key
from satquery.execution.models import ArtifactMetadata, ArtifactOutput
from satquery.persistence import Database, MetadataRepository
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
