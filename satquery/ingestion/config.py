"""Configuration for bounded, metadata-only raster inspection."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

MEBIBYTE = 1024 * 1024


class RasterSafetyLimits(BaseModel):
    """Resource ceilings checked before any raster data are read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_file_size_bytes: int = Field(default=512 * MEBIBYTE, gt=0)
    max_width: int = Field(default=50_000, gt=0)
    max_height: int = Field(default=50_000, gt=0)
    max_pixel_count: int = Field(default=150_000_000, gt=0)
    max_band_count: int = Field(default=32, gt=0)
    allowed_drivers: frozenset[str] = frozenset({"GTiff"})
    allowed_extensions: frozenset[str] = frozenset({".tif", ".tiff"})

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RasterSafetyLimits:
        values = os.environ if environ is None else environ

        def read_positive_int(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                parsed = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if parsed <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return parsed

        defaults = cls()
        raw_bytes = values.get("MAX_UPLOAD_SIZE_BYTES")
        if raw_bytes is not None and raw_bytes.strip():
            try:
                max_file_size_bytes = int(raw_bytes)
            except ValueError as exc:
                raise ValueError("MAX_UPLOAD_SIZE_BYTES must be a positive integer") from exc
            if max_file_size_bytes <= 0:
                raise ValueError("MAX_UPLOAD_SIZE_BYTES must be a positive integer")
        else:
            max_mebibytes = read_positive_int(
                "MAX_UPLOAD_SIZE_MB", defaults.max_file_size_bytes // MEBIBYTE
            )
            max_file_size_bytes = max_mebibytes * MEBIBYTE
        return cls(
            max_file_size_bytes=max_file_size_bytes,
            max_width=read_positive_int("MAX_RASTER_WIDTH", defaults.max_width),
            max_height=read_positive_int("MAX_RASTER_HEIGHT", defaults.max_height),
            max_pixel_count=read_positive_int(
                "MAX_RASTER_PIXELS", defaults.max_pixel_count
            ),
            max_band_count=read_positive_int(
                "MAX_RASTER_BANDS", defaults.max_band_count
            ),
        )
