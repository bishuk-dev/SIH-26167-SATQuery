"""Private, hash-verified storage for derived execution artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from satquery.execution.models import (
    ArtifactMetadata,
    ArtifactRecord,
    StagedArtifact,
)

_ARTIFACT_ID = re.compile(r"^artifact_[0-9a-f]{32}$")
_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9._-]{0,15}$")
_ALLOWED_SUFFIXES = frozenset({"bin", "csv", "geojson", "json", "npy", "png", "tif", "tiff"})


class ArtifactStoreError(RuntimeError):
    """Base storage failure."""


class ArtifactNotFoundError(ArtifactStoreError):
    """The requested artifact is absent or invalid."""


class ArtifactTraversalError(ArtifactStoreError):
    """A storage path escaped the controlled artifact root."""


class ArtifactStore:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.artifacts_root = self.data_root / "artifacts"
        self.staging_root = self.artifacts_root / ".staging"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def _contained(self, path: Path, root: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactTraversalError("artifact path escapes controlled root") from exc
        return resolved

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        normalized = suffix.lower().lstrip(".")
        if not _SUFFIX.fullmatch(normalized) or normalized not in _ALLOWED_SUFFIXES:
            raise ArtifactStoreError("unsupported artifact suffix")
        return normalized

    def stage(self, suffix: str) -> StagedArtifact:
        normalized = self._normalize_suffix(suffix)
        artifact_id = f"artifact_{uuid4().hex}"
        path = self._contained(
            self.staging_root / f"{artifact_id}.{normalized}.tmp", self.staging_root
        )
        return StagedArtifact(artifact_id=artifact_id, path=path, suffix=normalized)

    def discard(self, staged: StagedArtifact) -> None:
        path = self._contained(staged.path, self.staging_root)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _sha256(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        return digest.hexdigest(), size

    def publish(self, staged: StagedArtifact, metadata: ArtifactMetadata) -> ArtifactRecord:
        source = self._contained(staged.path, self.staging_root)
        if not source.is_file() or source.is_symlink():
            raise ArtifactStoreError("staged artifact is not a regular file")
        if staged.suffix != self._normalize_suffix(staged.suffix):
            raise ArtifactStoreError("staged artifact suffix is invalid")
        if staged.artifact_id != source.name.split(".", 1)[0] or not _ARTIFACT_ID.fullmatch(staged.artifact_id):
            raise ArtifactStoreError("staged artifact ID is invalid")

        digest, size = self._sha256(source)
        final_dir = self._contained(self.artifacts_root / staged.artifact_id, self.artifacts_root)
        final_dir.mkdir(parents=False, exist_ok=False)
        final_path = self._contained(
            final_dir / f"artifact.{staged.suffix}", final_dir
        )
        metadata_path = final_dir / "metadata.json"
        now = datetime.now(timezone.utc)
        payload = {
            "artifact_id": staged.artifact_id,
            "storage_key": f"{staged.artifact_id}/artifact.{staged.suffix}",
            "sha256": digest,
            "size_bytes": size,
            "media_type": metadata.media_type,
            "created_at": now.isoformat(),
            "metadata": {
                "analysis_id": metadata.analysis_id,
                "evidence_id": metadata.evidence_id,
                "description": metadata.description,
                "extra": metadata.extra,
            },
        }
        temporary_metadata = self._contained(final_dir / ".metadata.tmp", final_dir)
        try:
            os.replace(source, final_path)
            with temporary_metadata.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_metadata, metadata_path)
        except BaseException:
            temporary_metadata.unlink(missing_ok=True)
            if final_path.exists():
                final_path.unlink(missing_ok=True)
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        return ArtifactRecord(
            artifact_id=staged.artifact_id,
            path=final_path,
            storage_key=payload["storage_key"],
            sha256=digest,
            size_bytes=size,
            media_type=metadata.media_type,
            created_at=now,
            metadata=metadata,
        )

    def resolve(self, artifact_id: str) -> tuple[ArtifactRecord, Path]:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ArtifactNotFoundError("unknown artifact")
        directory = self._contained(self.artifacts_root / artifact_id, self.artifacts_root)
        metadata_path = self._contained(directory / "metadata.json", directory)
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise ArtifactNotFoundError("artifact metadata is missing")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            storage_key = payload["storage_key"]
            expected_prefix = f"{artifact_id}/artifact."
            if not isinstance(storage_key, str) or not storage_key.startswith(expected_prefix):
                raise ArtifactNotFoundError("artifact storage key is invalid")
            path = self._contained(self.artifacts_root / storage_key, self.artifacts_root)
            if path.parent != directory or not path.is_file() or path.is_symlink():
                raise ArtifactNotFoundError("artifact file is missing")
            digest, size = self._sha256(path)
            if digest != payload["sha256"] or size != payload["size_bytes"]:
                raise ArtifactNotFoundError("artifact hash verification failed")
            metadata_payload = payload.get("metadata", {})
            metadata = ArtifactMetadata(
                analysis_id=metadata_payload.get("analysis_id"),
                evidence_id=metadata_payload.get("evidence_id"),
                media_type=payload["media_type"],
                description=metadata_payload.get("description"),
                extra=metadata_payload.get("extra", {}),
            )
            record = ArtifactRecord(
                artifact_id=artifact_id,
                path=path,
                storage_key=storage_key,
                sha256=digest,
                size_bytes=size,
                media_type=payload["media_type"],
                created_at=datetime.fromisoformat(payload["created_at"]),
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise ArtifactNotFoundError("artifact metadata is invalid") from exc
        return record, path

    def discard_staging(self) -> None:
        for child in self.staging_root.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)


__all__ = ["ArtifactStore", "ArtifactStoreError", "ArtifactNotFoundError", "ArtifactTraversalError"]
