from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from satquery.artifacts import (
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactTraversalError,
    mask_footprint_geojson,
)
from satquery.execution import ArtifactMetadata

_EVIDENCE_ID = "evidence_" + "a" * 32
_GRID = {
    "width": 2,
    "height": 3,
    "crs": "EPSG:32643",
    "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
    "bounds": [500000.0, 1999980.0, 500020.0, 2000000.0],
    "source_grid_observation_id": "obs_" + "1" * 32,
    "value_semantics": "binary_0_1",
}


def test_stage_publish_resolve_hashes_and_metadata(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("json")
    staged.path.write_text('{"ok":true}', encoding="utf-8")
    record = store.publish(staged, ArtifactMetadata(media_type="application/json"))
    loaded, path = store.resolve(record.artifact_id)
    assert loaded.sha256 == record.sha256
    assert path.read_text(encoding="utf-8") == '{"ok":true}'
    assert not staged.path.exists()


def test_resolve_rejects_tampered_artifact(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("bin")
    staged.path.write_bytes(b"safe")
    record = store.publish(staged, ArtifactMetadata())
    record.path.write_bytes(b"tampered")
    with pytest.raises(ArtifactNotFoundError, match="hash"):
        store.resolve(record.artifact_id)


def test_stage_rejects_traversal_and_unknown_suffix(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(Exception):
        store.stage("../escape")
    with pytest.raises(Exception):
        store.stage("exe")
    with pytest.raises(ArtifactNotFoundError):
        store.resolve("artifact_" + "a" * 32 + "/../metadata.json")


def test_publish_rejects_symlink(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("json")
    target = tmp_path / "outside"
    target.write_text("outside", encoding="utf-8")
    staged.path.symlink_to(target)
    with pytest.raises(Exception):
        store.publish(staged, ArtifactMetadata())


def test_summary_projection_carries_no_local_path(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("json")
    staged.path.write_text("{}", encoding="utf-8")
    record = store.publish(staged, ArtifactMetadata(evidence_id=_EVIDENCE_ID))
    summary = store.summarize(record)
    payload = json.dumps(asdict(summary), default=str)
    assert str(tmp_path) not in payload
    assert summary.artifact_id == record.artifact_id
    assert summary.evidence_id == _EVIDENCE_ID
    assert summary.sha256 == record.sha256
    assert not hasattr(summary, "path")


def test_approved_media_type_and_safe_download_filename(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("tif")
    staged.path.write_bytes(b"iiif")
    record = store.publish(staged, ArtifactMetadata(media_type="image/tiff"))
    assert store.approved_media_type(record) == "image/tiff"
    assert store.download_filename(record) == f"{record.artifact_id}.tif"

    staged_json = store.stage("json")
    staged_json.path.write_text("{}", encoding="utf-8")
    html_record = store.publish(
        staged_json, ArtifactMetadata(media_type="text/html")
    )
    with pytest.raises(ArtifactNotFoundError, match="media type"):
        store.approved_media_type(html_record)
    # The filename is derived only from the verified storage key and ID.
    assert store.download_filename(html_record) == f"{html_record.artifact_id}.json"


def test_resolve_for_evidence_resolves_only_registered_ids(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("json")
    staged.path.write_text("{}", encoding="utf-8")
    record = store.publish(staged, ArtifactMetadata(evidence_id=_EVIDENCE_ID))

    for bad_id in (
        "artifact_" + "a" * 32,
        "evidence_../escape",
        "evidence_" + "z" * 32,
        _EVIDENCE_ID + "/../metadata.json",
    ):
        with pytest.raises(ArtifactNotFoundError):
            store.resolve_for_evidence(bad_id)

    unknown = "evidence_" + "b" * 32
    with pytest.raises(ArtifactNotFoundError):
        store.resolve_for_evidence(unknown)

    resolved, path = store.resolve_for_evidence(_EVIDENCE_ID)
    assert resolved.artifact_id == record.artifact_id
    assert path.is_file()


def test_resolve_for_evidence_fails_closed_on_tamper_and_ambiguity(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("bin")
    staged.path.write_bytes(b"safe")
    record = store.publish(staged, ArtifactMetadata(evidence_id=_EVIDENCE_ID))
    record.path.write_bytes(b"tampered")
    with pytest.raises(ArtifactNotFoundError, match="hash"):
        store.resolve_for_evidence(_EVIDENCE_ID)

    second = ArtifactStore(tmp_path / "other")
    first_stage = second.stage("json")
    first_stage.path.write_text("{}", encoding="utf-8")
    second.publish(first_stage, ArtifactMetadata(evidence_id=_EVIDENCE_ID))
    second_stage = second.stage("json")
    second_stage.path.write_text("{}", encoding="utf-8")
    second.publish(second_stage, ArtifactMetadata(evidence_id=_EVIDENCE_ID))
    with pytest.raises(ArtifactStoreError, match="multiple artifacts"):
        second.resolve_for_evidence(_EVIDENCE_ID)


def test_mask_grid_provenance_parses_and_fails_closed(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("tif")
    staged.path.write_bytes(b"iiif")
    record = store.publish(
        staged, ArtifactMetadata(media_type="image/tiff", extra={"grid": dict(_GRID)})
    )
    grid = store.mask_grid(record)
    assert grid.width == 2
    assert grid.height == 3
    assert grid.crs == "EPSG:32643"
    assert grid.bounds == (500000.0, 1999980.0, 500020.0, 2000000.0)
    assert grid.source_grid_observation_id == _GRID["source_grid_observation_id"]

    for mutation in (
        {"grid": {}},
        {"grid": {**_GRID, "width": 0}},
        {"grid": {**_GRID, "transform": [1.0, 0.0, 0.0, 0.0, -1.0]}},
        {"grid": {**_GRID, "value_semantics": "grayscale"}},
        {"grid": {**_GRID, "bounds": [10.0, 0.0, 0.0, 10.0]}},
        {"grid": {**_GRID, "source_grid_observation_id": ""}},
    ):
        staged = store.stage("tif")
        staged.path.write_bytes(b"iiif")
        broken = store.publish(
            staged, ArtifactMetadata(media_type="image/tiff", extra=mutation)
        )
        with pytest.raises(ArtifactNotFoundError, match="grid"):
            store.mask_grid(broken)

    staged = store.stage("tif")
    staged.path.write_bytes(b"iiif")
    plain = store.publish(staged, ArtifactMetadata(media_type="image/tiff"))
    with pytest.raises(ArtifactNotFoundError, match="grid"):
        store.mask_grid(plain)


def test_bbox_to_polygon_conversion_uses_recorded_crs(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("tif")
    staged.path.write_bytes(b"iiif")
    record = store.publish(
        staged, ArtifactMetadata(media_type="image/tiff", extra={"grid": dict(_GRID)})
    )
    footprint = mask_footprint_geojson(record, store.mask_grid(record))
    assert footprint["type"] == "Feature"
    assert footprint["crs"] == {"type": "name", "name": "EPSG:32643"}
    ring = footprint["geometry"]["coordinates"][0]
    assert ring == [
        [500000.0, 1999980.0],
        [500020.0, 1999980.0],
        [500020.0, 2000000.0],
        [500000.0, 2000000.0],
        [500000.0, 1999980.0],
    ]
    assert footprint["properties"]["artifact_id"] == record.artifact_id
    assert footprint["properties"]["value_semantics"] == "binary_0_1"


def test_pixel_space_masks_are_labelled_without_inventing_a_crs(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    staged = store.stage("tif")
    staged.path.write_bytes(b"iiif")
    record = store.publish(
        staged,
        ArtifactMetadata(
            media_type="image/tiff",
            extra={"grid": {**_GRID, "crs": None}},
        ),
    )
    footprint = mask_footprint_geojson(record, store.mask_grid(record))
    assert footprint["crs"] == {"type": "name", "name": "PIXEL_SPACE"}
    assert footprint["properties"]["coordinate_space"] == "pixel"
    ring = footprint["geometry"]["coordinates"][0]
    assert ring == [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0], [0.0, 0.0]]
    assert "EPSG:4326" not in json.dumps(footprint)


def test_published_artifact_is_retained_when_a_later_step_fails(tmp_path: Path):
    from satquery.execution import (
        ExecutionContext,
        ExecutionEngine,
        ExecutionPlan,
        PlanStep,
        ToolExecutionError,
        ToolResult,
    )

    store = ArtifactStore(tmp_path / "data")
    published: list[str] = []

    class PublishAdapter:
        def execute(self, call, context):
            staged = store.stage("json")
            staged.path.write_text("{}", encoding="utf-8")
            record = store.publish(staged, ArtifactMetadata())
            published.append(record.artifact_id)
            return ToolResult(output="ok")

    class FailingAdapter:
        def execute(self, call, context):
            raise RuntimeError("boom")

    engine = ExecutionEngine(
        {"publish": PublishAdapter(), "fail": FailingAdapter()},
        artifact_store=store,
        timeout_seconds=2,
    )
    with pytest.raises(ToolExecutionError):
        engine.execute(
            ExecutionPlan(
                steps=(
                    PlanStep("s1", "publish"),
                    PlanStep("s2", "fail", depends_on=("s1",)),
                )
            ),
            ExecutionContext(job_id="job_" + "c" * 32, analysis_id="ana_" + "c" * 32),
        )

    assert list(store.staging_root.iterdir()) == []
    resolved, _path = store.resolve(published[0])
    assert resolved.artifact_id == published[0]
