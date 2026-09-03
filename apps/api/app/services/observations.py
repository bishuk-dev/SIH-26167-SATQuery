"""Secure upload lifecycle for observation registration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from satquery.ingestion import FilesystemObservationStore, RasterInspector
from satquery.ingestion.exceptions import (
    AssetStorageError,
    InvalidUploadError,
    RasterResourceLimitError,
)
from satquery.visualization.derivatives import VisualizationDerivativeGenerator
from satquery.visualization.models import ObservationRegistration

UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_ORIGINAL_NAME_LENGTH = 255


class ObservationIngestionService:
    def __init__(
        self,
        *,
        inspector: RasterInspector,
        store: FilesystemObservationStore,
        derivative_generator: VisualizationDerivativeGenerator,
    ) -> None:
        self._inspector = inspector
        self._store = store
        self._derivative_generator = derivative_generator

    async def ingest(self, upload: UploadFile) -> ObservationRegistration:
        quarantine_path: Path | None = None
        visualization_path: Path | None = None
        try:
            original_name = _safe_original_name(upload.filename)
            observation_id = f"obs_{uuid4().hex}"
            asset_id = f"asset_{uuid4().hex}"
            suffix = Path(original_name).suffix.lower()
            quarantine_path = self._store.create_quarantine_file(asset_id, suffix)
            await self._stream_upload(upload, quarantine_path)
            inspected = self._inspector.inspect(
                quarantine_path,
                observation_id=observation_id,
                asset_id=asset_id,
            )
            visualization_asset_id = f"asset_{uuid4().hex}"
            visualization_path = self._store.create_quarantine_file(
                visualization_asset_id, ".tif"
            )
            storage_path = f"observations/{observation_id}/visualization.tif"
            visualization = self._derivative_generator.create(
                quarantine_path,
                visualization_path,
                inspected,
                asset_id=visualization_asset_id,
                storage_path=storage_path,
            )
            registered = self._store.register(
                quarantine_path,
                inspected,
                original_name=original_name,
                visualization_path=visualization_path,
                visualization=visualization,
            )
            if not isinstance(registered, ObservationRegistration):
                raise AssetStorageError("Visualization registration was incomplete")
            return registered
        finally:
            try:
                await upload.close()
            finally:
                cleanup_error: AssetStorageError | None = None
                for path in (quarantine_path, visualization_path):
                    if path is None:
                        continue
                    try:
                        self._store.discard_quarantine(path)
                    except AssetStorageError as exc:
                        cleanup_error = exc
                if cleanup_error is not None:
                    raise cleanup_error

    async def _stream_upload(self, upload: UploadFile, destination: Path) -> None:
        bytes_written = 0
        try:
            with destination.open("wb") as file_handle:
                while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                    bytes_written += len(chunk)
                    if bytes_written > self._inspector.limits.max_file_size_bytes:
                        raise RasterResourceLimitError(
                            "Uploaded file exceeds the configured file-size limit"
                        )
                    file_handle.write(chunk)
        except OSError as exc:
            raise AssetStorageError("Could not write upload to quarantine") from exc


def _safe_original_name(filename: str | None) -> str:
    if filename is None or any(ord(character) < 32 for character in filename):
        raise InvalidUploadError("Upload must include a valid filename")
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if not basename.strip() or basename in {".", ".."}:
        raise InvalidUploadError("Upload must include a valid filename")
    if len(basename) > MAX_ORIGINAL_NAME_LENGTH:
        raise InvalidUploadError("Upload filename is too long")
    return basename
