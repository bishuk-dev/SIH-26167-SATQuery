"""Sensor-semantic band role resolution for multi-sensor remote sensing.

Resolves semantic band roles (RED, GREEN, NIR, SWIR, SAR_CO_POL, SAR_CROSS_POL)
across diverse sensor platforms (Sentinel-2, Landsat, Sentinel-1, etc.) without
hardcoding sensor-specific band numbers or names.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

import numpy as np

from satquery.geo.exceptions import MissingBandError


class SemanticBandRole(StrEnum):
    """Canonical remote-sensing semantic band roles."""

    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"
    NIR = "NIR"
    SWIR = "SWIR"
    SWIR1 = "SWIR1"
    SWIR2 = "SWIR2"
    RED_EDGE = "RED_EDGE"
    SAR_CO_POL = "SAR_CO_POL"
    SAR_CROSS_POL = "SAR_CROSS_POL"


# Priority-ordered alias mappings per sensor family
SENSOR_BAND_MAPPINGS: dict[str, dict[SemanticBandRole, tuple[str, ...]]] = {
    "sentinel2": {
        SemanticBandRole.RED: ("b04", "b4", "b04_10m", "red"),
        SemanticBandRole.GREEN: ("b03", "b3", "b03_10m", "green"),
        SemanticBandRole.BLUE: ("b02", "b2", "b02_10m", "blue"),
        SemanticBandRole.NIR: ("b08", "b8", "b08_10m", "b8a", "b8a_20m", "nir"),
        SemanticBandRole.SWIR: ("b11", "b12", "b11_20m", "b12_20m", "swir"),
        SemanticBandRole.SWIR1: ("b11", "b11_20m", "swir1"),
        SemanticBandRole.SWIR2: ("b12", "b12_20m", "swir2"),
        SemanticBandRole.RED_EDGE: ("b05", "b06", "b07", "b5", "b6", "b7"),
    },
    "landsat": {
        SemanticBandRole.RED: ("sr_b4", "b4", "band4", "red"),
        SemanticBandRole.GREEN: ("sr_b3", "b3", "band3", "green"),
        SemanticBandRole.BLUE: ("sr_b2", "b2", "band2", "blue"),
        SemanticBandRole.NIR: ("sr_b5", "b5", "band5", "nir"),
        SemanticBandRole.SWIR: ("sr_b6", "sr_b7", "b6", "b7", "band6", "band7", "swir"),
        SemanticBandRole.SWIR1: ("sr_b6", "b6", "band6", "swir1"),
        SemanticBandRole.SWIR2: ("sr_b7", "b7", "band7", "swir2"),
    },
    "sentinel1": {
        SemanticBandRole.SAR_CO_POL: ("vv", "hh", "sar_vv", "sar_hh", "sigma0_vv", "gamma0_vv", "copol", "co_pol"),
        SemanticBandRole.SAR_CROSS_POL: ("vh", "hv", "sar_vh", "sar_hv", "sigma0_vh", "gamma0_vh", "crosspol", "cross_pol"),
    },
    "generic": {
        SemanticBandRole.RED: ("red", "b04", "b4", "band4", "r"),
        SemanticBandRole.GREEN: ("green", "b03", "b3", "band3", "g"),
        SemanticBandRole.BLUE: ("blue", "b02", "b2", "band2", "b"),
        SemanticBandRole.NIR: ("nir", "b08", "b8", "band8", "b8a", "b5", "band5", "near_infrared"),
        SemanticBandRole.SWIR: ("swir", "swir1", "swir2", "b11", "b12", "b6", "b7", "band11", "band12"),
        SemanticBandRole.SWIR1: ("swir1", "b11", "band11"),
        SemanticBandRole.SWIR2: ("swir2", "b12", "band12"),
        SemanticBandRole.RED_EDGE: ("red_edge", "rededge", "re", "b5", "b6", "b7"),
        SemanticBandRole.SAR_CO_POL: ("vv", "hh", "co_pol", "copol", "sar_vv", "sar_hh"),
        SemanticBandRole.SAR_CROSS_POL: ("vh", "hv", "cross_pol", "crosspol", "sar_vh", "sar_hv"),
    },
}


def _normalize_role(role: SemanticBandRole | str) -> SemanticBandRole:
    if isinstance(role, SemanticBandRole):
        return role
    try:
        return SemanticBandRole(role.upper())
    except ValueError:
        # Check case-insensitive match
        for r in SemanticBandRole:
            if r.value.lower() == role.lower():
                return r
        raise ValueError(f"Unknown semantic band role: {role!r}")


def resolve_band_for_role(
    available_band_names: list[str] | tuple[str, ...] | set[str] | Mapping[str, Any],
    role: SemanticBandRole | str,
    *,
    sensor: str | None = None,
) -> str:
    """Resolve a single band name from available bands matching the requested semantic role.

    Args:
        available_band_names: Iterable of band names or mapping of band names to data.
        role: Requested semantic role (e.g. RED, NIR, SAR_CO_POL).
        sensor: Optional sensor identifier ('sentinel2', 'landsat', 'sentinel1') to prioritize.

    Returns:
        The matched band name from available_band_names.

    Raises:
        MissingBandError: If no candidate band matches the semantic role.
    """
    sem_role = _normalize_role(role)
    keys = list(available_band_names.keys()) if isinstance(available_band_names, Mapping) else list(available_band_names)

    # Search sensor-specific mappings first if sensor is specified
    sensors_to_check: list[str] = []
    if sensor:
        sensor_clean = sensor.lower().replace("-", "").replace("_", "")
        for s_key in SENSOR_BAND_MAPPINGS:
            if s_key in sensor_clean or sensor_clean in s_key:
                sensors_to_check.append(s_key)
    sensors_to_check.append("generic")

    # 1. Exact match on alias
    for s_name in sensors_to_check:
        aliases = SENSOR_BAND_MAPPINGS[s_name].get(sem_role, ())
        for alias in aliases:
            for k in keys:
                if k.strip().lower() == alias:
                    return k

    # 2. Substring or token match on alias
    for s_name in sensors_to_check:
        aliases = SENSOR_BAND_MAPPINGS[s_name].get(sem_role, ())
        for alias in aliases:
            for k in keys:
                k_clean = k.strip().lower()
                # If alias is a distinct word or substring
                if alias in k_clean:
                    return k

    raise MissingBandError(
        f"Semantic band role '{sem_role.value}' could not be resolved from available bands: {keys}"
    )


def get_semantic_band(
    bands: Mapping[str, np.ndarray],
    role: SemanticBandRole | str,
    *,
    sensor: str | None = None,
) -> tuple[str, np.ndarray]:
    """Retrieve the band name and array for a requested semantic role.

    Returns:
        (resolved_band_name, band_array)
    """
    matched_name = resolve_band_for_role(bands, role, sensor=sensor)
    return matched_name, bands[matched_name]
