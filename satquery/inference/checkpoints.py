"""Shared checkpoint verification for registered inference specialists."""

from __future__ import annotations

import hashlib
from pathlib import Path

from satquery.inference.exceptions import ModelUnavailableError


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_checkpoint(path: Path, expected_sha256: str, *, model_name: str) -> None:
    if not path.is_file():
        raise ModelUnavailableError(f"{model_name} checkpoint is unavailable")
    if sha256_file(path) != expected_sha256:
        raise ModelUnavailableError(f"{model_name} checkpoint hash is invalid")
