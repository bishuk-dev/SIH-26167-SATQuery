"""Filesystem-backed registration for immutable source observations."""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path

from pydantic import ValidationError

from satquery.ingestion.exceptions import AssetStorageError, ObservationNotFoundError
from satquery.ingestion.models import ObservationState, SourceAsset
from satquery.visualization.exceptions import VisualizationAssetNotFoundError
from satquery.visualization.models import ObservationRegistration, VisualizationAsset

METADATA_FILENAME = "metadata.json"
ORIGINAL_FILENAME = "original.tif"
VISUALIZATION_FILENAME = "visualization.tif"
VISUALIZATION_METADATA_FILENAME = "visualization.json"
ASSET_ID_PATTERN = re.compile(r"^asset_[0-9a-f]{32}$")
OBSERVATION_ID_PATTERN = re.compile(r"^obs_[0-9a-f]{32}$")


class FilesystemObservationStore:
    """Promote inspected quarantine files into server-controlled storage."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.quarantine_root = self.data_root / "quarantine"
        self.observations_root = self.data_root / "observations"
        self.asset_index_root = self.data_root / "assets"

    def create_quarantine_file(self, asset_id: str, suffix: str) -> Path:
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise AssetStorageError("Refusing an untrusted asset identifier")
        self._ensure_roots()
        safe_suffix = suffix if suffix in {".tif", ".tiff"} else ".upload"
        path = self.quarantine_root / f"{asset_id}{safe_suffix}"
        try:
            path.touch(exist_ok=False)
        except OSError as exc:
            raise AssetStorageError("Could not create quarantine file") from exc
        return path

    def register(
        self,
        quarantine_path: Path,
        inspected: ObservationState,
        *,
        original_name: str,
        visualization_path: Path | None = None,
        visualization: VisualizationAsset | None = None,
    ) -> ObservationState | ObservationRegistration:
        if visualization_path is not None or visualization is not None:
            if visualization_path is None or visualization is None:
                raise AssetStorageError("Incomplete visualization registration")
            return self.register_with_visualization(
                quarantine_path,
                inspected,
                original_name=original_name,
                visualization_path=visualization_path,
                visualization=visualization,
            )
        self._assert_quarantine_path(quarantine_path)
        if OBSERVATION_ID_PATTERN.fullmatch(inspected.observation_id) is None:
            raise AssetStorageError("Refusing an untrusted observation identifier")
        if ASSET_ID_PATTERN.fullmatch(inspected.source_asset.asset_id) is None:
            raise AssetStorageError("Refusing an untrusted asset identifier")

        observation_dir = self.observations_root / inspected.observation_id
        final_path = observation_dir / ORIGINAL_FILENAME
        metadata_path = observation_dir / METADATA_FILENAME
        metadata_temp_path = observation_dir / f".{METADATA_FILENAME}.tmp"
        storage_key = (
            Path("observations") / inspected.observation_id / ORIGINAL_FILENAME
        )
        observation_dir_created = False

        try:
            observation_dir.mkdir(parents=False, exist_ok=False)
            observation_dir_created = True
            stored_asset = SourceAsset(
                asset_id=inspected.source_asset.asset_id,
                original_name=original_name,
                path=storage_key.as_posix(),
                sha256=inspected.source_asset.sha256,
            )
            registered = ObservationState.model_validate(
                {
                    **inspected.model_dump(),
                    "source_asset": stored_asset,
                }
            )
            os.replace(quarantine_path, final_path)
            metadata_temp_path.write_text(
                registered.model_dump_json(indent=2), encoding="utf-8"
            )
            os.replace(metadata_temp_path, metadata_path)
            final_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            return registered
        except (OSError, ValueError) as exc:
            if observation_dir_created:
                try:
                    self._remove_observation_dir(observation_dir)
                except OSError as cleanup_error:
                    raise AssetStorageError(
                        "Could not clean up partial observation registration"
                    ) from cleanup_error
            raise AssetStorageError("Could not register inspected observation") from exc

    def register_with_visualization(
        self,
        quarantine_path: Path,
        inspected: ObservationState,
        *,
        original_name: str,
        visualization_path: Path,
        visualization: VisualizationAsset,
    ) -> ObservationRegistration:
        self._assert_quarantine_path(quarantine_path)
        self._assert_quarantine_path(visualization_path)
        self._validate_registration_ids(inspected, visualization)

        observation_dir = self.observations_root / inspected.observation_id
        original_path = observation_dir / ORIGINAL_FILENAME
        derivative_path = observation_dir / VISUALIZATION_FILENAME
        metadata_path = observation_dir / METADATA_FILENAME
        visualization_metadata_path = observation_dir / VISUALIZATION_METADATA_FILENAME
        index_path = self.asset_index_root / f"{visualization.asset_id}.json"
        expected_derivative_key = (
            Path("observations") / inspected.observation_id / VISUALIZATION_FILENAME
        ).as_posix()
        if visualization.path != expected_derivative_key:
            raise AssetStorageError("Visualization storage key is inconsistent")

        observation_created = False
        index_created = False
        try:
            self._ensure_roots()
            if index_path.exists():
                raise AssetStorageError("Visualization asset identifier already exists")
            observation_dir.mkdir(parents=False, exist_ok=False)
            observation_created = True
            registered = self._registered_state(inspected, original_name)
            os.replace(quarantine_path, original_path)
            os.replace(visualization_path, derivative_path)
            self._write_json_atomic(metadata_path, registered.model_dump_json(indent=2))
            self._write_json_atomic(
                visualization_metadata_path,
                visualization.model_dump_json(indent=2),
            )
            self._write_json_atomic(index_path, visualization.model_dump_json(indent=2))
            index_created = True
            for immutable_path in (
                original_path,
                derivative_path,
                metadata_path,
                visualization_metadata_path,
                index_path,
            ):
                immutable_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            return ObservationRegistration(
                observation=registered,
                visualization=visualization,
            )
        except (OSError, ValueError) as exc:
            if index_created:
                self._make_writable(index_path)
                index_path.unlink(missing_ok=True)
            if observation_created:
                try:
                    self._remove_observation_dir(observation_dir)
                except OSError as cleanup_error:
                    raise AssetStorageError(
                        "Could not clean up partial observation registration"
                    ) from cleanup_error
            if isinstance(exc, AssetStorageError):
                raise
            raise AssetStorageError(
                "Could not register observation derivatives"
            ) from exc

    def resolve_visualization(
        self, asset_id: str
    ) -> tuple[VisualizationAsset, Path]:
        if ASSET_ID_PATTERN.fullmatch(asset_id) is None:
            raise VisualizationAssetNotFoundError("Visualization asset was not found")
        index_path = self.asset_index_root / f"{asset_id}.json"
        try:
            asset = VisualizationAsset.model_validate_json(
                index_path.read_text(encoding="utf-8")
            )
            expected_key = (
                Path("observations") / asset.observation_id / VISUALIZATION_FILENAME
            ).as_posix()
            if asset.asset_id != asset_id or asset.path != expected_key:
                raise ValueError("Visualization index is inconsistent")
            path = (self.data_root / asset.path).resolve(strict=True)
            path.relative_to(self.observations_root)
            if not path.is_file():
                raise OSError("Visualization path is not a file")
            return asset, path
        except (OSError, RuntimeError, ValidationError, ValueError):
            raise VisualizationAssetNotFoundError(
                "Visualization asset was not found"
            ) from None

    def load_registration(
        self, observation_id: str
    ) -> tuple[ObservationRegistration, Path]:
        if OBSERVATION_ID_PATTERN.fullmatch(observation_id) is None:
            raise ObservationNotFoundError("Observation was not found")
        observation_dir = self.observations_root / observation_id
        metadata_path = observation_dir / METADATA_FILENAME
        visualization_metadata_path = observation_dir / VISUALIZATION_METADATA_FILENAME
        try:
            observation = ObservationState.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            visualization = VisualizationAsset.model_validate_json(
                visualization_metadata_path.read_text(encoding="utf-8")
            )
            if observation.observation_id != observation_id:
                raise ValueError("Observation metadata is inconsistent")
            expected_key = (
                Path("observations") / observation_id / VISUALIZATION_FILENAME
            ).as_posix()
            if (
                visualization.observation_id != observation_id
                or visualization.parent_asset_id != observation.source_asset.asset_id
                or visualization.path != expected_key
            ):
                raise ValueError("Visualization provenance is inconsistent")
            visualization_path = (self.data_root / visualization.path).resolve(
                strict=True
            )
            visualization_path.relative_to(self.observations_root)
            if not visualization_path.is_file():
                raise OSError("Visualization path is not a file")
            return (
                ObservationRegistration(
                    observation=observation,
                    visualization=visualization,
                ),
                visualization_path,
            )
        except (OSError, RuntimeError, ValidationError, ValueError):
            raise ObservationNotFoundError("Observation was not found") from None

    def discard_quarantine(self, path: Path) -> None:
        try:
            self._assert_quarantine_path(path)
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise AssetStorageError("Could not clean up quarantine file") from exc

    def _ensure_roots(self) -> None:
        try:
            self.quarantine_root.mkdir(parents=True, exist_ok=True)
            self.observations_root.mkdir(parents=True, exist_ok=True)
            self.asset_index_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AssetStorageError("Could not initialize observation storage") from exc

    def _assert_quarantine_path(self, path: Path) -> None:
        if path.parent.resolve() != self.quarantine_root.resolve():
            raise AssetStorageError("Refusing a file outside controlled quarantine")

    @staticmethod
    def _registered_state(
        inspected: ObservationState, original_name: str
    ) -> ObservationState:
        storage_key = (
            Path("observations") / inspected.observation_id / ORIGINAL_FILENAME
        )
        stored_asset = SourceAsset(
            asset_id=inspected.source_asset.asset_id,
            original_name=original_name,
            path=storage_key.as_posix(),
            sha256=inspected.source_asset.sha256,
        )
        return ObservationState.model_validate(
            {**inspected.model_dump(), "source_asset": stored_asset}
        )

    @staticmethod
    def _validate_registration_ids(
        inspected: ObservationState, visualization: VisualizationAsset
    ) -> None:
        if OBSERVATION_ID_PATTERN.fullmatch(inspected.observation_id) is None:
            raise AssetStorageError("Refusing an untrusted observation identifier")
        if ASSET_ID_PATTERN.fullmatch(inspected.source_asset.asset_id) is None:
            raise AssetStorageError("Refusing an untrusted asset identifier")
        if visualization.observation_id != inspected.observation_id:
            raise AssetStorageError(
                "Visualization observation provenance is inconsistent"
            )
        if visualization.parent_asset_id != inspected.source_asset.asset_id:
            raise AssetStorageError("Visualization parent provenance is inconsistent")

    @staticmethod
    def _write_json_atomic(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _make_writable(path: Path) -> None:
        if path.exists():
            path.chmod(stat.S_IWRITE | stat.S_IREAD)

    @staticmethod
    def _remove_observation_dir(path: Path) -> None:
        if not path.exists():
            return
        for child in path.iterdir():
            child.chmod(stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(path)
