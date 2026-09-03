"""Read-only GeoTIFF metadata inspection without raster data reads."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine
from rasterio.errors import RasterioIOError

from satquery.ingestion.config import RasterSafetyLimits
from satquery.ingestion.exceptions import (
    InvalidRasterError,
    RasterDriverMismatchError,
    RasterResourceLimitError,
    UnsupportedRasterDriverError,
)
from satquery.ingestion.models import (
    AffineTransform,
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)

INSPECTOR_VERSION = "raster-inspector/0.1.0"
HASH_CHUNK_SIZE = 1024 * 1024


class RasterInspector:
    """Inspect local TIFF headers/tags and return a validated observation record."""

    def __init__(self, limits: RasterSafetyLimits | None = None) -> None:
        self._limits = limits or RasterSafetyLimits.from_env()

    @property
    def limits(self) -> RasterSafetyLimits:
        return self._limits

    def inspect(
        self,
        path: str | Path,
        *,
        observation_id: str,
        asset_id: str,
    ) -> ObservationState:
        raster_path = self._resolve_file(path)
        file_size = raster_path.stat().st_size
        if file_size > self._limits.max_file_size_bytes:
            self._raise_limit(
                "file size",
                file_size,
                self._limits.max_file_size_bytes,
            )

        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(raster_path, mode="r", sharing=False) as dataset:
                    self._validate_driver(dataset.driver, raster_path.suffix)
                    self._validate_shape(
                        width=dataset.width,
                        height=dataset.height,
                        band_count=dataset.count,
                    )
                    observation_fields = self._inspect_open_dataset(dataset)
        except (
            RasterResourceLimitError,
            UnsupportedRasterDriverError,
            RasterDriverMismatchError,
        ):
            raise
        except (RasterioIOError, OSError, ValueError) as exc:
            raise InvalidRasterError(
                f"Could not safely inspect raster metadata for {raster_path.name!r}"
            ) from exc

        try:
            source_asset = SourceAsset(
                asset_id=asset_id,
                original_name=raster_path.name,
                path=str(raster_path),
                sha256=_sha256(raster_path),
            )
        except OSError as exc:
            raise InvalidRasterError(
                f"Could not hash inspected raster {raster_path.name!r}"
            ) from exc
        return ObservationState(
            observation_id=observation_id,
            source_asset=source_asset,
            provenance=ObservationProvenance(
                created_at=datetime.now(timezone.utc),
                ingestion_version=INSPECTOR_VERSION,
            ),
            **observation_fields,
        )

    def _resolve_file(self, path: str | Path) -> Path:
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidRasterError(f"Raster path does not exist: {path}") from exc
        if not resolved.is_file():
            raise InvalidRasterError(f"Raster path is not a regular file: {resolved}")
        return resolved

    def _validate_driver(self, driver: str, suffix: str) -> None:
        if driver not in self._limits.allowed_drivers:
            raise UnsupportedRasterDriverError(
                f"Raster driver {driver!r} is not allowed; expected GTiff"
            )
        if suffix.lower() not in self._limits.allowed_extensions:
            raise RasterDriverMismatchError(
                f"GTiff content requires a .tif or .tiff filename, got {suffix or '<none>'!r}"
            )

    def _validate_shape(self, *, width: int, height: int, band_count: int) -> None:
        checks = (
            ("width", width, self._limits.max_width),
            ("height", height, self._limits.max_height),
            ("band count", band_count, self._limits.max_band_count),
            ("pixel count", width * height, self._limits.max_pixel_count),
        )
        for name, actual, limit in checks:
            if actual > limit:
                self._raise_limit(name, actual, limit)

    @staticmethod
    def _raise_limit(name: str, actual: int, limit: int) -> None:
        raise RasterResourceLimitError(
            f"Raster {name} {actual:,} exceeds configured limit {limit:,}"
        )

    @staticmethod
    def _inspect_open_dataset(dataset: Any) -> dict[str, Any]:
        dataset_tags = _all_tags(dataset)
        nodata = tuple(dataset.nodatavals)
        dtypes = tuple(dataset.dtypes)
        bands = tuple(
            BandMetadata(
                index=index,
                description=dataset.descriptions[index - 1] or None,
                dtype=dtypes[index - 1],
                nodata=nodata[index - 1],
                tags=_all_tags(dataset, band_index=index),
            )
            for index in dataset.indexes
        )

        has_transform = not (
            dataset.crs is None and dataset.transform == Affine.identity()
        )
        has_crs = dataset.crs is not None
        geo = _geo_metadata(dataset, has_transform=has_transform)

        warnings: list[str] = []
        if not has_crs:
            warnings.append("CRS_UNAVAILABLE")
        if not has_transform:
            warnings.append("TRANSFORM_UNAVAILABLE")

        return {
            "raster": RasterMetadata(
                driver=dataset.driver,
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                dtypes=dtypes,
                nodata=nodata,
                tags=dataset_tags,
            ),
            "sensor": SensorMetadata(
                modality=_modality(dataset_tags),
                sensor_name=_first_tag(
                    dataset_tags, "SENSOR_NAME", "SENSOR", "INSTRUMENT"
                ),
                platform=_first_tag(dataset_tags, "PLATFORM", "SATELLITE"),
                product_level=_first_tag(
                    dataset_tags, "PRODUCT_LEVEL", "PROCESSING_LEVEL"
                ),
                bands=bands,
                polarizations=_polarizations(dataset_tags),
            ),
            "geo": geo,
            "temporal": TemporalMetadata(
                acquisition_time=_acquisition_time(dataset_tags)
            ),
            "validity": ValidityMetadata(
                has_crs=has_crs,
                has_transform=has_transform,
                has_nodata=any(value is not None for value in nodata),
                warnings=tuple(warnings),
            ),
        }


def _all_tags(dataset: Any, band_index: int | None = None) -> dict[str, str]:
    args = () if band_index is None else (band_index,)
    tags = {str(key): str(value) for key, value in dataset.tags(*args).items()}
    for namespace in dataset.tag_namespaces(*args):
        if not namespace:
            continue
        namespaced = dataset.tags(*args, ns=namespace)
        tags.update(
            {f"{namespace}:{key}": str(value) for key, value in namespaced.items()}
        )
    return tags


def _geo_metadata(dataset: Any, *, has_transform: bool) -> GeoMetadata:
    if not has_transform:
        return GeoMetadata()

    transform = dataset.transform
    units: str | None = None
    if dataset.crs is not None:
        units = "degree" if dataset.crs.is_geographic else dataset.crs.linear_units

    return GeoMetadata(
        crs=dataset.crs.to_string() if dataset.crs is not None else None,
        transform=AffineTransform(
            a=transform.a,
            b=transform.b,
            c=transform.c,
            d=transform.d,
            e=transform.e,
            f=transform.f,
        ),
        bounds=GeoBounds(
            left=dataset.bounds.left,
            bottom=dataset.bounds.bottom,
            right=dataset.bounds.right,
            top=dataset.bounds.top,
        ),
        native_gsd_x=math.hypot(transform.a, transform.d),
        native_gsd_y=math.hypot(transform.b, transform.e),
        units=units,
    )


def _normalized_key(value: str) -> str:
    leaf = value.rsplit(":", maxsplit=1)[-1]
    return re.sub(r"[^A-Z0-9]", "", leaf.upper())


def _first_tag(tags: dict[str, str], *candidate_keys: str) -> str | None:
    for candidate in candidate_keys:
        normalized_candidate = _normalized_key(candidate)
        for key, value in tags.items():
            if _normalized_key(key) == normalized_candidate and value.strip():
                return value.strip()
    return None


def _modality(tags: dict[str, str]) -> Modality:
    raw = _first_tag(tags, "MODALITY")
    if raw is None:
        return Modality.UNKNOWN
    normalized = raw.strip().lower().replace("-", "_")
    aliases = {
        "optical": Modality.OPTICAL,
        "multispectral": Modality.MULTISPECTRAL,
        "multi_spectral": Modality.MULTISPECTRAL,
        "sar": Modality.SAR,
    }
    return aliases.get(normalized, Modality.UNKNOWN)


def _polarizations(tags: dict[str, str]) -> tuple[str, ...]:
    raw = _first_tag(
        tags, "POLARIZATIONS", "POLARISATIONS", "POLARIZATION", "POLARISATION"
    )
    if raw is None:
        return ()
    return tuple(value.upper() for value in re.split(r"[,;/\s]+", raw) if value)


def _acquisition_time(tags: dict[str, str]) -> datetime | None:
    raw = _first_tag(
        tags,
        "ACQUISITION_TIME",
        "ACQUISITION_DATETIME",
        "SENSING_TIME",
        "DATETIME",
        "TIFFTAG_DATETIME",
    )
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    # A timestamp without an offset cannot support a reliable UTC instant.
    return parsed if parsed.tzinfo is not None else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
