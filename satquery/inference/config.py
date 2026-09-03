"""Runtime settings for bounded local VQA inference."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class VqaRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_root: Path = Path("./models")
    allow_remote_network: bool = False
    device: str = "cpu"
    cpu_threads: int = Field(default=2, ge=1, le=32)
    max_question_characters: int = Field(default=500, ge=1, le=4000)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> VqaRuntimeSettings:
        values = os.environ if environ is None else environ
        remote_value = values.get("ENABLE_REMOTE_NETWORK", "false").strip().lower()
        if remote_value not in {"true", "false"}:
            raise ValueError("ENABLE_REMOTE_NETWORK must be true or false")
        try:
            cpu_threads = int(values.get("VQA_CPU_THREADS", "2"))
        except ValueError as exc:
            raise ValueError("VQA_CPU_THREADS must be a positive integer") from exc
        return cls(
            model_root=Path(values.get("MODEL_ROOT", "./models")),
            allow_remote_network=remote_value == "true",
            device=values.get("GPU_DEVICE", "cpu"),
            cpu_threads=cpu_threads,
        )
