"""Frozen image preprocessing shared by production and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioError

from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.registry.models import PreprocessingProfile
from satquery.visualization.models import VisualizationAsset


class FrozenImagePreprocessor:
    def __init__(self, profile: PreprocessingProfile) -> None:
        self.profile = profile

    def from_visualization(
        self, path: str | Path, asset: VisualizationAsset
    ) -> Image.Image:
        try:
            with rasterio.open(path, "r", sharing=False) as dataset:
                width, height = _fit_size(
                    dataset.width,
                    dataset.height,
                    self.profile.width,
                    self.profile.height,
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
                "Could not read the registered visualization asset"
            ) from exc

        if asset.color_band_count == 1:
            colors = np.repeat(colors, 3, axis=0)
        image = np.moveaxis(colors, 0, -1).astype("uint8", copy=False)
        padding = np.asarray(self.profile.padding_rgb, dtype="uint8")
        image[alpha == 0] = padding
        return _pad_image(Image.fromarray(image), self.profile)

    def from_pil(self, image: Image.Image) -> Image.Image:
        return _pad_image(image.convert("RGB"), self.profile)


def _fit_size(
    width: int, height: int, maximum_width: int, maximum_height: int
) -> tuple[int, int]:
    scale = min(maximum_width / width, maximum_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _pad_image(image: Image.Image, profile: PreprocessingProfile) -> Image.Image:
    resized = image.resize(
        _fit_size(image.width, image.height, profile.width, profile.height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (profile.width, profile.height), profile.padding_rgb)
    offset = (
        (profile.width - resized.width) // 2,
        (profile.height - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas
