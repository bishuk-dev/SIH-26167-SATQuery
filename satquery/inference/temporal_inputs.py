"""Shared temporal optical input preparation for inference specialists."""

from __future__ import annotations

import numpy as np
import rasterio

from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.ingestion.models import Modality, ObservationState
from satquery.verification.domain import require_domain


def read_aligned_rgb_pair(t1: ObservationState, t2: ObservationState) -> tuple[np.ndarray, np.ndarray]:
    """Validate and read an aligned optical RGB pair as float32 tensors."""
    require_domain(t1, supported_modalities=(Modality.OPTICAL, Modality.MULTISPECTRAL))
    require_domain(t2, supported_modalities=(Modality.OPTICAL, Modality.MULTISPECTRAL))
    if t1.geo.crs is None or t2.geo.crs is None or t1.geo.transform is None or t2.geo.transform is None:
        raise ModelInputUnsupportedError("temporal RGB models require verified georeferencing")
    if (
        t1.raster.width,
        t1.raster.height,
        t1.geo.crs,
        t1.geo.transform,
    ) != (
        t2.raster.width,
        t2.raster.height,
        t2.geo.crs,
        t2.geo.transform,
    ):
        raise ModelInputUnsupportedError("temporal RGB models require an aligned temporal pair")
    for observation in (t1, t2):
        semantics = tuple(band.description for band in observation.sensor.bands)
        if semantics != ("R", "G", "B"):
            raise ModelInputUnsupportedError("temporal RGB models require semantic R/G/B bands")

    first = _read_rgb(t1)
    second = _read_rgb(t2)
    return first, second


def _read_rgb(observation: ObservationState) -> np.ndarray:
    with rasterio.open(observation.source_asset.path) as source:
        return source.read((1, 2, 3)).astype("float32") / 255.0
