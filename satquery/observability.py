"""Small, dependency-free observability primitives for the API boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Only operational metadata is allowed into API logs. In particular, do not
# add request payloads, arbitrary extras, filesystem paths, or model prompts.
_SAFE_LOG_FIELDS = frozenset(
    {"request_id", "job_id", "analysis_id", "event", "status_code", "duration_ms"}
)


class JsonLogFormatter(logging.Formatter):
    """Format a log record as one JSON object without sensitive payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _SAFE_LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None and isinstance(value, (str, int, float, bool)):
                payload[field] = value
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def configure_json_logging() -> logging.Logger:
    """Install the API's idempotent JSON handler and return its logger."""

    logger = logging.getLogger("satquery.api")
    logger.setLevel(logging.INFO)
    if not any(getattr(handler, "_satquery_json", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._satquery_json = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    return logger


def _component(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def _check_sqlite(application: Any) -> dict[str, Any]:
    repository = getattr(application.state, "observation_repository", None)
    database = getattr(repository, "_db", None)
    if database is None:
        return _component("unavailable")
    try:
        with database.read_transaction() as connection:
            connection.execute("SELECT 1").fetchone()
    except Exception:
        return _component("unavailable")
    return _component("ready")


def _check_filesystem(application: Any) -> dict[str, Any]:
    store = getattr(application.state, "observation_store", None)
    root = getattr(store, "data_root", None)
    if not isinstance(root, Path):
        return _component("unavailable")
    try:
        ready = root.is_dir() and os.access(root, os.R_OK | os.W_OK | os.X_OK)
    except OSError:
        ready = False
    return _component("ready" if ready else "unavailable")


def _check_registries(application: Any) -> dict[str, Any]:
    tool_registry = getattr(application.state, "tool_registry", None)
    model_registry = getattr(application.state, "model_registry", None)
    capabilities = getattr(application.state, "runtime_capabilities", None)
    ready = (
        tool_registry is not None
        and model_registry is not None
        and capabilities is not None
        and hasattr(tool_registry, "tools")
        and hasattr(model_registry, "models")
    )
    if not ready:
        return _component("unavailable")
    return _component(
        "ready",
        tool_count=len(tool_registry.tools),
        model_count=len(model_registry.models),
        capability_count=len(capabilities),
    )


def _check_queue(application: Any) -> dict[str, Any]:
    runner = getattr(application.state, "job_runner", None)
    queue = getattr(runner, "_queue", None)
    if queue is None:
        return _component("unavailable")
    try:
        depth = queue.qsize()
        capacity = queue.maxsize
        ready = capacity > 0 and depth < capacity
    except (AttributeError, NotImplementedError):
        return _component("unavailable")
    return _component(
        "ready" if ready else "unavailable", depth=depth, capacity=capacity
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_models(application: Any) -> dict[str, Any]:
    """Check registered checkpoint bytes without importing a model runtime."""

    registry = getattr(application.state, "model_registry", None)
    if registry is None or not hasattr(registry, "models"):
        return _component("unavailable", checkpoint_loading=False)
    roots = getattr(application.state, "model_roots", ())
    roots = tuple(Path(root) for root in roots if root is not None)
    verified = 0
    missing = 0
    invalid = 0
    for model_id, registration in registry.models.items():
        checkpoint = None
        candidates = []
        for root in roots:
            candidates.extend(
                (
                    root / model_id / registration.checkpoint_file,
                    root
                    / "cache"
                    / ("models--" + model_id.replace("/", "--"))
                    / "snapshots"
                    / registration.revision
                    / registration.checkpoint_file,
                )
            )
        for candidate in candidates:
            if candidate.is_file():
                checkpoint = candidate
                break
        if checkpoint is None:
            missing += 1
            continue
        try:
            expected_size = getattr(registration, "checkpoint_size_bytes", None)
            valid_size = expected_size is None or checkpoint.stat().st_size == expected_size
            if valid_size and _sha256(checkpoint) == registration.checkpoint_sha256:
                verified += 1
            else:
                invalid += 1
        except (OSError, ValueError):
            invalid += 1
    ready = not missing and not invalid
    return _component(
        "ready" if ready else "unavailable",
        checkpoint_loading=False,
        registered_count=len(registry.models),
        verified_count=verified,
        missing_count=missing,
        invalid_count=invalid,
    )


def readiness_components(application: Any) -> dict[str, dict[str, Any]]:
    """Return safe component states; this function never invokes inference."""

    components = {
        "sqlite": _check_sqlite(application),
        "filesystem": _check_filesystem(application),
        "registries": _check_registries(application),
        "queue": _check_queue(application),
        "models": _check_models(application),
    }
    # Phase 5 runs in RESTRICTED_CAPABILITY mode while optional learned-model
    # checkpoints are absent. Deterministic registered tools remain runnable,
    # so missing optional checkpoints are reported but do not make the API
    # unrouteable by an orchestrator.
    components["models"]["required"] = False
    return components


def readiness_payload(application: Any) -> dict[str, Any]:
    components = readiness_components(application)
    required = ("sqlite", "filesystem", "registries", "queue")
    ready = all(components[name]["status"] == "ready" for name in required)
    return {"status": "ready" if ready else "not_ready", "components": components}


# Short aliases keep the stdlib logging surface convenient for callers while
# retaining the explicit names used by the API factory.
JsonFormatter = JsonLogFormatter
configure_logging = configure_json_logging


__all__ = [
    "JsonLogFormatter",
    "JsonFormatter",
    "configure_json_logging",
    "configure_logging",
    "readiness_components",
    "readiness_payload",
]
