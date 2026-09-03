"""Bounded creation of display-only Cloud Optimized GeoTIFF derivatives."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import RasterioError
from rasterio.shutil import copy as raster_copy
from rasterio.windows import Window

from satquery.geo.coordinates import transform_bounds
from satquery.geo.exceptions import CoordinateTransformError
from satquery.ingestion.models import GeoBounds, Modality, ObservationState
from satquery.visualization.config import VisualizationSettings
from satquery.visualization.exceptions import (
    VisualizationGenerationError,
    VisualizationResourceLimitError,
)
from satquery.visualization.models import (
    BandStretch,
    RenderingMode,
    TileScheme,
    VisualizationAsset,
)

GENERATOR_VERSION = "visualization-cog/0.1.0"
WEB_MERCATOR_CRS = "EPSG:3857"
HASH_CHUNK_SIZE = 1024 * 1024
PERCENTILE_LOW = 2.0
PERCENTILE_HIGH = 98.0


class VisualizationDerivativeGenerator:
    """Create a display COG while retaining a one-to-one source pixel grid."""

    def __init__(self, settings: VisualizationSettings | None = None) -> None:
        self.settings = settings or VisualizationSettings.from_env()

    def create(
        self,
        source_path: str | Path,
        destination_path: str | Path,
        observation: ObservationState,
        *,
        asset_id: str,
        storage_path: str,
    ) -> VisualizationAsset:
        source = Path(source_path)
        destination = Path(destination_path)
        working = destination.with_name(f".{destination.name}.working.tif")
        completed = False
        try:
            with rasterio.open(source, "r", sharing=False) as dataset:
                selected, rendering = _select_display_bands(dataset, observation)
                stretches = self._calculate_stretches(
                    dataset, selected, observation.sensor.modality
                )
                self._write_working_derivative(
                    dataset,
                    working,
                    selected,
                    rendering,
                    stretches,
                    observation,
                )

            destination.unlink(missing_ok=True)
            raster_copy(
                working,
                destination,
                driver="COG",
                blocksize=self.settings.derivative_block_size,
                compress="DEFLATE",
                overview_resampling="average",
            )
            if destination.stat().st_size > self.settings.max_derivative_size_bytes:
                raise VisualizationResourceLimitError(
                    "Visualization derivative exceeds the configured size limit"
                )

            tile_scheme, tile_crs, tile_extent = _tile_metadata(observation)
            asset = VisualizationAsset(
                asset_id=asset_id,
                observation_id=observation.observation_id,
                parent_asset_id=observation.source_asset.asset_id,
                path=storage_path,
                sha256=_sha256(destination),
                rendering=rendering,
                source_band_indexes=selected,
                stretches=stretches,
                width=observation.raster.width,
                height=observation.raster.height,
                color_band_count=len(selected),
                alpha_band=len(selected) + 1,
                crs=observation.geo.crs,
                transform=observation.geo.transform,
                source_bounds=observation.geo.bounds,
                tile_scheme=tile_scheme,
                tile_crs=tile_crs,
                tile_extent=tile_extent,
                created_at=datetime.now(timezone.utc),
                generator_version=GENERATOR_VERSION,
            )
            completed = True
            return asset
        except VisualizationResourceLimitError:
            raise
        except (OSError, RasterioError, ValueError) as exc:
            raise VisualizationGenerationError(
                "Could not create a visualization derivative"
            ) from exc
        finally:
            working.unlink(missing_ok=True)
            if not completed:
                destination.unlink(missing_ok=True)

    def _calculate_stretches(
        self,
        dataset: rasterio.DatasetReader,
        selected: tuple[int, ...],
        modality: Modality,
    ) -> tuple[BandStretch, ...]:
        sample_height = min(dataset.height, self.settings.statistics_sample_size)
        sample_width = min(dataset.width, self.settings.statistics_sample_size)
        sample = dataset.read(
            selected,
            out_shape=(len(selected), sample_height, sample_width),
            masked=True,
            resampling=Resampling.average,
        )
        stretches = []
        for position, source_band in enumerate(selected):
            values = sample[position].compressed()
            magnitude = np.iscomplexobj(values)
            if magnitude:
                values = np.abs(values)
            values = values.astype("float64", copy=False)
            values = values[np.isfinite(values)]
            logarithmic = modality is Modality.SAR and _can_log(values)
            if magnitude and logarithmic:
                scale = "magnitude_log1p"
            elif magnitude:
                scale = "magnitude"
            elif logarithmic:
                scale = "log1p"
            else:
                scale = "linear"
            if logarithmic:
                values = np.log1p(values)
            if values.size == 0:
                lower, upper = 0.0, 1.0
            else:
                lower, upper = np.percentile(
                    values, (PERCENTILE_LOW, PERCENTILE_HIGH)
                )
                if not math.isfinite(lower) or not math.isfinite(upper):
                    lower, upper = 0.0, 1.0
            stretches.append(
                BandStretch(
                    source_band=source_band,
                    lower=float(lower),
                    upper=float(upper),
                    scale=scale,
                )
            )
        return tuple(stretches)

    def _write_working_derivative(
        self,
        source: rasterio.DatasetReader,
        destination: Path,
        selected: tuple[int, ...],
        rendering: RenderingMode,
        stretches: tuple[BandStretch, ...],
        observation: ObservationState,
    ) -> None:
        count = len(selected) + 1
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=count,
            nodata=None,
            tiled=True,
            blockxsize=self.settings.derivative_block_size,
            blockysize=self.settings.derivative_block_size,
            compress="DEFLATE",
            interleave="pixel",
        )
        with rasterio.open(destination, "w", **profile) as target:
            target.colorinterp = (
                (ColorInterp.gray, ColorInterp.alpha)
                if rendering is RenderingMode.GRAYSCALE
                else (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.alpha,
                )
            )
            target.update_tags(
                SATQUERY_ASSET_KIND="visualization",
                SATQUERY_PARENT_ASSET_ID=observation.source_asset.asset_id,
                SATQUERY_OBSERVATION_ID=observation.observation_id,
                SATQUERY_RENDERING=rendering.value,
                SATQUERY_GENERATOR_VERSION=GENERATOR_VERSION,
            )
            for window in _windows(
                source.width,
                source.height,
                self.settings.derivative_block_size,
            ):
                data = source.read(selected, window=window, masked=True)
                finite = np.isfinite(np.asarray(data)).all(axis=0)
                mask = ~np.ma.getmaskarray(data).any(axis=0) & finite
                for output_band, stretch in enumerate(stretches, start=1):
                    rendered = _render_band(data[output_band - 1], stretch)
                    target.write(rendered, output_band, window=window)
                target.write(mask.astype("uint8") * 255, count, window=window)


def _select_display_bands(
    dataset: rasterio.DatasetReader, observation: ObservationState
) -> tuple[tuple[int, ...], RenderingMode]:
    if observation.sensor.modality is Modality.SAR or dataset.count < 3:
        return (1,), RenderingMode.GRAYSCALE

    color_interpretations = tuple(dataset.colorinterp)
    by_color = {
        color: index
        for index, color in enumerate(color_interpretations, start=1)
        if color in {ColorInterp.red, ColorInterp.green, ColorInterp.blue}
    }
    expected_colors = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(color in by_color for color in expected_colors):
        return (
            by_color[ColorInterp.red],
            by_color[ColorInterp.green],
            by_color[ColorInterp.blue],
        ), RenderingMode.RGB

    by_name: dict[str, int] = {}
    color_names = {"red", "green", "blue"}
    for band in observation.sensor.bands:
        candidates: set[str] = set()
        raw_names = [str(value) for value in band.tags.values()]
        if band.description:
            raw_names.append(band.description)
        for raw_name in raw_names:
            candidates.update(
                token
                for token in re.split(r"[^a-z]+", raw_name.lower())
                if token
            )
        for color in color_names:
            if color in candidates and color not in by_name:
                by_name[color] = band.index
    if all(color in by_name for color in ("red", "green", "blue")):
        return (by_name["red"], by_name["green"], by_name["blue"]), RenderingMode.RGB
    return (1, 2, 3), RenderingMode.RGB


def _render_band(data: np.ma.MaskedArray, stretch: BandStretch) -> np.ndarray:
    values = data.filled(0)
    if stretch.scale.startswith("magnitude"):
        values = np.abs(values)
    values = values.astype("float64", copy=False)
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)
    if stretch.scale.endswith("log1p"):
        values = np.log1p(np.maximum(values, 0.0))
    if stretch.upper <= stretch.lower:
        rendered = np.full(values.shape, 128, dtype="uint8")
    else:
        scaled = (values - stretch.lower) / (stretch.upper - stretch.lower)
        rendered = np.clip(scaled * 255.0, 0.0, 255.0).astype("uint8")
    rendered[np.ma.getmaskarray(data) | ~finite] = 0
    return rendered


def _can_log(values: np.ndarray) -> bool:
    return bool(values.size and np.min(values) >= 0.0)


def _windows(width: int, height: int, size: int):
    for row in range(0, height, size):
        for column in range(0, width, size):
            yield Window(
                column,
                row,
                min(size, width - column),
                min(size, height - row),
            )


def _tile_metadata(
    observation: ObservationState,
) -> tuple[TileScheme, str | None, GeoBounds]:
    if observation.geo.crs is not None and observation.geo.bounds is not None:
        try:
            extent = transform_bounds(
                observation.geo.bounds,
                observation.geo.crs,
                WEB_MERCATOR_CRS,
            )
            return TileScheme.WEB_MERCATOR, WEB_MERCATOR_CRS, extent
        except CoordinateTransformError:
            pass
    return (
        TileScheme.PIXEL,
        None,
        GeoBounds(
            left=0,
            bottom=0,
            right=float(observation.raster.width),
            top=float(observation.raster.height),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
