"""Resource bounds for visualization derivative and tile generation."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from satquery.ingestion.config import MEBIBYTE


class VisualizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_size: int = Field(default=256, ge=64, le=1024)
    derivative_block_size: int = Field(default=512, ge=128, le=1024)
    statistics_sample_size: int = Field(default=512, ge=64, le=2048)
    max_derivative_size_bytes: int = Field(default=512 * MEBIBYTE, gt=0)
    max_tile_zoom: int = Field(default=24, ge=0, le=30)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> VisualizationSettings:
        values = os.environ if environ is None else environ
        defaults = cls()

        def positive_int(name: str, default: int) -> int:
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

        return cls(
            tile_size=defaults.tile_size,
            derivative_block_size=defaults.derivative_block_size,
            statistics_sample_size=positive_int(
                "VISUALIZATION_SAMPLE_SIZE", defaults.statistics_sample_size
            ),
            max_derivative_size_bytes=positive_int(
                "MAX_VISUALIZATION_SIZE_MB",
                defaults.max_derivative_size_bytes // MEBIBYTE,
            )
            * MEBIBYTE,
            max_tile_zoom=positive_int(
                "MAX_TILE_ZOOM", defaults.max_tile_zoom
            ),
        )
