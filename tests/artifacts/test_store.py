from __future__ import annotations

from pathlib import Path

import pytest

from satquery.artifacts import ArtifactNotFoundError, ArtifactStore, ArtifactTraversalError
from satquery.execution import ArtifactMetadata


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
