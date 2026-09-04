"""Frozen, coordinate-preserving preprocessing for text-guided grounding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioError

from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.registry.models import GroundingPreprocessingProfile
from satquery.visualization.models import VisualizationAsset


@dataclass(frozen=True)
class PreparedGroundingImage:
    image: Image.Image
    source_width: int
    source_height: int

    @property
    def scale_to_source_x(self) -> float:
        return self.source_width / self.image.width

    @property
    def scale_to_source_y(self) -> float:
        return self.source_height / self.image.height


class GroundingImagePreprocessor:
    def __init__(self, profile: GroundingPreprocessingProfile) -> None:
        self.profile = profile

    def from_visualization(
        self, path: str | Path, asset: VisualizationAsset
    ) -> PreparedGroundingImage:
        try:
            with rasterio.open(path, "r", sharing=False) as dataset:
                source_width, source_height = dataset.width, dataset.height
                width, height = grounding_input_size(
                    source_width,
                    source_height,
                    self.profile.shortest_edge,
                    self.profile.longest_edge,
                )
                colors = dataset.read(
                    tuple(range(1, asset.color_band_count + 1)),
                    out_shape=(asset.color_band_count, height, width),
                    resampling=Resampling.bilinear,
                )
                alpha = dataset.read(
                    asset.alpha_band,
                    out_shape=(height, width),
                    resampling=Resampling.nearest,
                )
        except (OSError, RasterioError, ValueError) as exc:
            raise ModelInputUnsupportedError(
                "Could not read the registered visualization asset for grounding"
            ) from exc

        if asset.color_band_count == 1:
            colors = np.repeat(colors, 3, axis=0)
        array = np.moveaxis(colors, 0, -1).astype("uint8", copy=False)
        array[alpha == 0] = 0
        return PreparedGroundingImage(
            image=Image.fromarray(array),
            source_width=source_width,
            source_height=source_height,
        )

    def from_pil(self, image: Image.Image) -> PreparedGroundingImage:
        source = image.convert("RGB")
        width, height = grounding_input_size(
            source.width,
            source.height,
            self.profile.shortest_edge,
            self.profile.longest_edge,
        )
        prepared = source.resize((width, height), Image.Resampling.BILINEAR)
        return PreparedGroundingImage(prepared, source.width, source.height)


def grounding_input_size(
    width: int,
    height: int,
    shortest_edge: int,
    longest_edge: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ModelInputUnsupportedError("Grounding image dimensions must be positive")
    scale = shortest_edge / min(width, height)
    if max(width, height) * scale > longest_edge:
        scale = longest_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))
