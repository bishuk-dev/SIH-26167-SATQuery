"""Read-only XYZ and pixel-grid tile rendering from visualization COGs."""

from __future__ import annotations

import math
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning, RasterioError
from rasterio.io import MemoryFile
from rasterio.warp import reproject
from rasterio.windows import Window

from satquery.ingestion.storage import FilesystemObservationStore
from satquery.visualization.config import VisualizationSettings
from satquery.visualization.exceptions import (
    InvalidTileRequestError,
    TileRenderingError,
)
from satquery.visualization.models import TileScheme, VisualizationAsset

WEB_MERCATOR_HALF_WORLD = math.pi * 6_378_137.0


class RasterTileService:
    def __init__(
        self,
        store: FilesystemObservationStore,
        settings: VisualizationSettings | None = None,
    ) -> None:
        self._store = store
        self.settings = settings or VisualizationSettings.from_env()

    def render(self, asset_id: str, z: int, x: int, y: int) -> bytes:
        self._validate_coordinates(z, x, y)
        asset, path = self._store.resolve_visualization(asset_id)
        try:
            with rasterio.open(path, "r", sharing=False) as dataset:
                if asset.tile_scheme is TileScheme.WEB_MERCATOR:
                    rgba = self._render_web_mercator(dataset, asset, z, x, y)
                else:
                    rgba = self._render_pixel_grid(dataset, asset, z, x, y)
            return _encode_png(rgba)
        except InvalidTileRequestError:
            raise
        except (OSError, RasterioError, ValueError) as exc:
            raise TileRenderingError("Could not render the requested tile") from exc

    def _validate_coordinates(self, z: int, x: int, y: int) -> None:
        if z < 0 or z > self.settings.max_tile_zoom:
            raise InvalidTileRequestError("Tile zoom is outside the supported range")
        side = 1 << z
        if x < 0 or y < 0 or x >= side or y >= side:
            raise InvalidTileRequestError("Tile coordinates are outside the zoom grid")

    def _render_web_mercator(
        self,
        dataset: rasterio.DatasetReader,
        asset: VisualizationAsset,
        z: int,
        x: int,
        y: int,
    ) -> np.ndarray:
        if dataset.crs is None:
            raise TileRenderingError("Web Mercator tile source has no CRS")
        size = self.settings.tile_size
        bounds = _xyz_bounds(z, x, y)
        transform = Affine(
            (bounds[2] - bounds[0]) / size,
            0,
            bounds[0],
            0,
            -(bounds[3] - bounds[1]) / size,
            bounds[3],
        )
        rgba = np.zeros((4, size, size), dtype="uint8")
        for destination_band in range(asset.color_band_count):
            reproject(
                source=rasterio.band(dataset, destination_band + 1),
                destination=rgba[destination_band],
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=transform,
                dst_crs=asset.tile_crs,
                dst_nodata=0,
                resampling=Resampling.bilinear,
                num_threads=1,
            )
        reproject(
            source=rasterio.band(dataset, asset.alpha_band),
            destination=rgba[3],
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=transform,
            dst_crs=asset.tile_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            num_threads=1,
        )
        if asset.color_band_count == 1:
            rgba[1] = rgba[0]
            rgba[2] = rgba[0]
        return rgba

    def _render_pixel_grid(
        self,
        dataset: rasterio.DatasetReader,
        asset: VisualizationAsset,
        z: int,
        x: int,
        y: int,
    ) -> np.ndarray:
        size = self.settings.tile_size
        side = 1 << z
        window = Window(
            col_off=x * dataset.width / side,
            row_off=y * dataset.height / side,
            width=dataset.width / side,
            height=dataset.height / side,
        )
        colors = dataset.read(
            tuple(range(1, asset.color_band_count + 1)),
            window=window,
            out_shape=(asset.color_band_count, size, size),
            resampling=Resampling.bilinear,
        )
        alpha = dataset.read(
            asset.alpha_band,
            window=window,
            out_shape=(size, size),
            resampling=Resampling.nearest,
        )
        rgba = np.zeros((4, size, size), dtype="uint8")
        rgba[: asset.color_band_count] = colors
        rgba[3] = alpha
        if asset.color_band_count == 1:
            rgba[1] = colors[0]
            rgba[2] = colors[0]
        return rgba


def _xyz_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    span = (2.0 * WEB_MERCATOR_HALF_WORLD) / (1 << z)
    left = -WEB_MERCATOR_HALF_WORLD + x * span
    right = left + span
    top = WEB_MERCATOR_HALF_WORLD - y * span
    bottom = top - span
    return left, bottom, right, top


def _encode_png(rgba: np.ndarray) -> bytes:
    # XYZ PNGs are intentionally non-georeferenced; their URL defines location.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as memory:
            with memory.open(
                driver="PNG",
                width=rgba.shape[2],
                height=rgba.shape[1],
                count=4,
                dtype="uint8",
            ) as output:
                output.colorinterp = (
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.alpha,
                )
                output.write(rgba)
            return memory.read()
