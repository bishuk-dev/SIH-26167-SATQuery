"""Phase 5 Task 19 health, readiness, and structured logging contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from satquery.observability import JsonLogFormatter


def test_liveness_does_not_require_application_dependencies(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        client.app.state.observation_repository = None
        client.app.state.tool_registry = None
        client.app.state.model_registry = None
        client.app.state.runtime_capabilities = None
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_components_without_loading_checkpoints(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert {"sqlite", "filesystem", "registries", "queue", "models"} <= set(
        body["components"]
    )
    assert body["components"]["sqlite"]["status"] == "ready"
    assert body["components"]["models"]["checkpoint_loading"] is False
    assert str(tmp_path) not in response.text


def test_status_and_limits_are_safe_and_typed(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        status = client.get("/api/v1/system/status")
        limits = client.get("/api/v1/system/limits")
        legacy_limits = client.get("/limits")

    assert status.status_code == 200
    assert status.json()["components"]["models"]["checkpoint_loading"] is False
    assert limits.status_code == 200
    assert limits.json() == legacy_limits.json()
    assert limits.json()["raster"]["max_file_size_bytes"] > 0
    assert "path" not in limits.text


def test_json_formatter_keeps_correlation_ids_and_drops_sensitive_fields() -> None:
    formatter = JsonLogFormatter()
    record = logging.makeLogRecord(
        {
            "name": "satquery.test",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "msg": "job completed",
            "request_id": "req_123",
            "job_id": "job_456",
            "analysis_id": "ana_789",
            "query": "secret query text",
            "secret": "token-value",
            "path": "/private/file.tif",
            "hidden_prompt": "system prompt",
        }
    )

    payload = json.loads(formatter.format(record))

    assert payload["request_id"] == "req_123"
    assert payload["job_id"] == "job_456"
    assert payload["analysis_id"] == "ana_789"
    for sensitive in ("query", "secret", "path", "hidden_prompt"):
        assert sensitive not in payload
    assert "secret query text" not in formatter.format(record)
    assert "token-value" not in formatter.format(record)
    assert "private/file.tif" not in formatter.format(record)
    assert "system prompt" not in formatter.format(record)


def test_request_log_contains_request_correlation_id_not_request_body(
    tmp_path: Path, caplog
) -> None:
    application = create_app(data_root=tmp_path / "data")
    with TestClient(application) as client:
        with caplog.at_level(logging.INFO, logger="satquery.api"):
            response = client.get("/health/live")

    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records]
    assert messages
    assert any(
        getattr(record, "request_id", None) == response.headers["X-Request-ID"]
        for record in caplog.records
    )
    assert all("/health/live" not in message for message in messages)
