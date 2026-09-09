"""Private, hash-verified storage for derived execution artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from satquery.execution.models import (
    ArtifactMetadata,
    ArtifactRecord,
    StagedArtifact,
)

_ARTIFACT_ID = re.compile(r"^artifact_[0-9a-f]{32}$")
_EVIDENCE_ID = re.compile(r"^evidence_[0-9a-f]{32}$")
_SUFFIX = re.compile(r"^[a-z0-9][a-z0-9._-]{0,15}$")
_ALLOWED_SUFFIXES = frozenset({"bin", "csv", "geojson", "json", "npy", "png", "tif", "tiff"})
# Code-owned suffix -> media-type mapping. HTTP serving may expose only
# these approved types, and only when the published metadata agrees.
_APPROVED_SUFFIX_MEDIA_TYPES = {
    "bin": "application/octet-stream",
    "csv": "text/csv",
    "geojson": "application/geo+json",
    "json": "application/json",
    "npy": "application/octet-stream",
    "png": "image/png",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    """Safe public projection of a published artifact; never carries local paths."""

    artifact_id: str
    storage_key: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: datetime
    analysis_id: str | None
    evidence_id: str | None
    description: str | None
    extra: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MaskGridProvenance:
    """Recorded grid of a published binary mask; georeferencing may be absent."""

    width: int
    height: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    source_grid_observation_id: str
    value_semantics: str


def mask_footprint_geojson(
    record: ArtifactRecord, grid: MaskGridProvenance
) -> dict[str, Any]:
    """Project a recorded mask grid as an explicitly labelled GeoJSON Feature.

    The geometry is only the recorded bounds rectangle on the recorded grid;
    a CRS is never invented. Masks without a CRS stay in pixel space and are
    labelled ``PIXEL_SPACE`` so no geographic coordinates are implied.
    """

    if grid.crs is None:
        left, bottom, right, top = 0.0, 0.0, float(grid.width), float(grid.height)
        crs_name = "PIXEL_SPACE"
        coordinate_space = "pixel"
    else:
        left, bottom, right, top = grid.bounds
        crs_name = grid.crs
        coordinate_space = "grid"
    ring = [
        [left, bottom],
        [right, bottom],
        [right, top],
        [left, top],
        [left, bottom],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "crs": {"type": "name", "name": crs_name},
        "properties": {
            "artifact_id": record.artifact_id,
            "evidence_id": record.metadata.evidence_id,
            "artifact_sha256": record.sha256,
            "coordinate_space": coordinate_space,
            "source_grid_observation_id": grid.source_grid_observation_id,
            "value_semantics": grid.value_semantics,
            "width": grid.width,
            "height": grid.height,
        },
    }


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

    def summarize(self, record: ArtifactRecord) -> ArtifactSummary:
        """Project a verified record for public API use; no local paths."""

        return ArtifactSummary(
            artifact_id=record.artifact_id,
            storage_key=record.storage_key,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            media_type=record.media_type,
            created_at=record.created_at,
            analysis_id=record.metadata.analysis_id,
            evidence_id=record.metadata.evidence_id,
            description=record.metadata.description,
            extra=dict(record.metadata.extra),
        )

    def approved_media_type(self, record: ArtifactRecord) -> str:
        """Return the suffix-derived approved media type; fail closed otherwise."""

        suffix = record.storage_key.rsplit(".", 1)[-1]
        expected = _APPROVED_SUFFIX_MEDIA_TYPES.get(suffix)
        if expected is None or record.media_type != expected:
            raise ArtifactNotFoundError("artifact media type is not approved")
        return expected

    def download_filename(self, record: ArtifactRecord) -> str:
        """Return a server-generated attachment filename with no user input."""

        suffix = record.storage_key.rsplit(".", 1)[-1]
        if suffix not in _ALLOWED_SUFFIXES:
            raise ArtifactNotFoundError("artifact suffix is not approved")
        return f"{record.artifact_id}.{suffix}"

    def resolve_for_evidence(self, evidence_id: str) -> tuple[ArtifactRecord, Path]:
        """Resolve the published artifact linked to an evidence ID, fully verified."""

        if not _EVIDENCE_ID.fullmatch(evidence_id):
            raise ArtifactNotFoundError("unknown evidence artifact")
        try:
            candidates = sorted(self.artifacts_root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ArtifactNotFoundError("artifact storage is unavailable") from exc
        matches: list[str] = []
        for directory in candidates:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file() or metadata_path.is_symlink():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and metadata.get("evidence_id") == evidence_id:
                artifact_id = payload.get("artifact_id")
                if isinstance(artifact_id, str):
                    matches.append(artifact_id)
        if not matches:
            raise ArtifactNotFoundError("no published artifact is linked to this evidence")
        if len(matches) > 1:
            raise ArtifactStoreError("multiple artifacts claim the same evidence ID")
        return self.resolve(matches[0])

    def mask_grid(self, record: ArtifactRecord) -> MaskGridProvenance:
        """Parse and validate the recorded grid provenance of a binary mask."""

        grid = record.metadata.extra.get("grid")
        if not isinstance(grid, dict):
            raise ArtifactNotFoundError("artifact has no mask grid provenance")
        try:
            width = grid["width"]
            height = grid["height"]
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("width")
            if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
                raise ValueError("height")
            if grid["value_semantics"] != "binary_0_1":
                raise ValueError("value_semantics")
            source_id = grid["source_grid_observation_id"]
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("source_grid_observation_id")
            transform_values = tuple(float(value) for value in grid["transform"])
            if len(transform_values) != 6 or not all(math.isfinite(v) for v in transform_values):
                raise ValueError("transform")
            bounds_values = tuple(float(value) for value in grid["bounds"])
            if len(bounds_values) != 4 or not all(math.isfinite(v) for v in bounds_values):
                raise ValueError("bounds")
            left, bottom, right, top = bounds_values
            if not left < right or not bottom < top:
                raise ValueError("bounds order")
            crs = grid.get("crs")
            if crs is not None and (not isinstance(crs, str) or not crs.strip()):
                raise ValueError("crs")
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactNotFoundError("artifact mask grid provenance is invalid") from exc
        return MaskGridProvenance(
            width=width,
            height=height,
            crs=crs,
            transform=transform_values,  # type: ignore[arg-type]
            bounds=(left, bottom, right, top),
            source_grid_observation_id=source_id,
            value_semantics="binary_0_1",
        )

    def discard_staging(self) -> None:
        for child in self.staging_root.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)


__all__ = [
    "ArtifactNotFoundError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactSummary",
    "ArtifactTraversalError",
    "MaskGridProvenance",
    "mask_footprint_geojson",
]
