"""Phase 5 Task 1 — versioned API foundation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from apps.api.app.main import (
    add_request_id_middleware,
    create_app,
    install_v1_error_handlers,
)

REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{32}$")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(data_root=tmp_path / "data")) as c:
        yield c


def _error(response_body: dict) -> dict:
    return response_body["error"]


# ---------------------------------------------------------------------------
# liveness and request correlation
# ---------------------------------------------------------------------------


def test_liveness_is_cheap_and_alive(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_liveness_has_request_id_header(client: TestClient) -> None:
    response = client.get("/health/live")
    assert REQUEST_ID_PATTERN.match(response.headers["X-Request-ID"])


def test_request_id_is_lowercase_hex(client: TestClient) -> None:
    header = client.get("/health/live").headers["X-Request-ID"]
    assert REQUEST_ID_PATTERN.match(header)


def test_independent_requests_get_different_ids(client: TestClient) -> None:
    first = client.get("/health/live").headers["X-Request-ID"]
    second = client.get("/health/live").headers["X-Request-ID"]
    assert first != second


# ---------------------------------------------------------------------------
# system version
# ---------------------------------------------------------------------------


def test_system_version(client: TestClient) -> None:
    response = client.get("/api/v1/system/version")
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert body["application"] == "satquery"
    assert body["application_version"]
    assert body["phase5_mode"] == "RESTRICTED_CAPABILITY"


def test_system_version_has_request_id_header(client: TestClient) -> None:
    response = client.get("/api/v1/system/version")
    assert REQUEST_ID_PATTERN.match(response.headers["X-Request-ID"])


# ---------------------------------------------------------------------------
# documentation surfaces
# ---------------------------------------------------------------------------


def test_swagger_ui_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_redoc_served(client: TestClient) -> None:
    assert client.get("/redoc").status_code == 200


def test_openapi_json_served(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_openapi_contains_task1_paths(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health/live" in paths
    assert "/api/v1/system/version" in paths
    # legacy routes remain documented
    assert "/api/observations" in paths
    assert "/api/vqa" in paths
    assert "/api/grounding" in paths


def test_openapi_metadata(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["openapi"]
    assert schema["info"]["title"] == "SatQuery API"
    assert schema["info"]["version"]


def test_openapi_task1_operation_ids_are_exact(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/health/live"]["get"]["operationId"] == "get_liveness"
    assert (
        schema["paths"]["/api/v1/system/version"]["get"]["operationId"]
        == "get_system_version_v1"
    )


def test_openapi_has_no_duplicate_operation_ids(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation_ids: list[str] = []
    for operations in schema["paths"].values():
        for operation in operations.values():
            if isinstance(operation, dict) and "operationId" in operation:
                operation_ids.append(operation["operationId"])
    assert len(operation_ids) == len(set(operation_ids))


def test_openapi_task1_route_has_system_tag(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/system/version"]["get"]
    assert "System" in operation["tags"]


def test_openapi_serialization_contains_no_local_paths(
    client: TestClient, tmp_path: Path
) -> None:
    serialized = str(client.get("/openapi.json").json())
    assert str(tmp_path) not in serialized
    assert "data_root" not in serialized


# ---------------------------------------------------------------------------
# v1 error envelope — validation, 404, 405, 500
# ---------------------------------------------------------------------------


def _tiny_v1_app() -> TestClient:
    """Test-only app with the real v1 handlers and a strict-body route."""

    class StrictBody(BaseModel):
        count: int = Field(gt=0)

    application = FastAPI()
    add_request_id_middleware(application)
    install_v1_error_handlers(application)

    @application.post("/api/v1/test-only")
    def strict_route(payload: StrictBody) -> dict[str, int]:
        return {"count": payload.count}

    @application.get("/api/v1/test-only/boom")
    def boom_route() -> dict[str, str]:
        raise RuntimeError("sensitive internal failure detail")

    @application.get("/test-only/legacy-boom")
    def legacy_boom_route() -> dict[str, str]:
        raise RuntimeError("legacy internal failure")

    return TestClient(application, raise_server_exceptions=False)


def test_unexpected_legacy_exception_keeps_legacy_shape_with_header() -> None:
    client = _tiny_v1_app()
    response = client.get("/test-only/legacy-boom")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"  # legacy default shape
    assert REQUEST_ID_PATTERN.match(response.headers["X-Request-ID"])


def test_v1_validation_error_uses_frozen_envelope() -> None:
    client = _tiny_v1_app()
    response = client.post("/api/v1/test-only", json={"count": -1})
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error"}
    detail = _error(body)
    assert set(detail) == {"code", "message", "outcome", "details", "request_id"}
    assert detail["code"] == "INVALID_REQUEST"
    assert detail["outcome"] == "REQUEST_INPUT"
    assert REQUEST_ID_PATTERN.match(detail["request_id"])
    assert detail["request_id"] == response.headers["X-Request-ID"]


def test_v1_validation_details_are_safe() -> None:
    client = _tiny_v1_app()
    secret_value = "TOPSECRETVALUE"
    response = client.post("/api/v1/test-only", json={"count": secret_value})
    detail = _error(response.json())
    assert secret_value not in response.text
    assert "TOPSECRET" not in response.text
    for item in detail["details"]["validation_errors"]:
        assert set(item) <= {"field", "category"}


def test_unknown_v1_path_uses_frozen_404_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    detail = _error(response.json())
    assert detail["code"] == "NOT_FOUND"
    assert detail["outcome"] == "REJECT"
    assert detail["message"] == "The requested API resource was not found."
    assert REQUEST_ID_PATTERN.match(detail["request_id"])
    assert detail["request_id"] == response.headers["X-Request-ID"]


def test_wrong_method_on_v1_route_uses_frozen_405_envelope(
    client: TestClient,
) -> None:
    response = client.delete("/api/v1/system/version")
    assert response.status_code == 405
    detail = _error(response.json())
    assert detail["code"] == "METHOD_NOT_ALLOWED"
    assert detail["outcome"] == "REJECT"
    assert REQUEST_ID_PATTERN.match(detail["request_id"])
    assert detail["request_id"] == response.headers["X-Request-ID"]


def test_unexpected_v1_exception_returns_sanitized_500() -> None:
    client = _tiny_v1_app()
    response = client.get("/api/v1/test-only/boom")
    assert response.status_code == 500
    detail = _error(response.json())
    assert detail["code"] == "INTERNAL_ERROR"
    assert detail["outcome"] == "ABSTAIN"
    assert detail["message"] == "An unexpected internal error occurred."
    assert REQUEST_ID_PATTERN.match(detail["request_id"])
    # the header must survive the application-level exception path and match
    # the body ID even though the response bypasses the middleware return
    assert REQUEST_ID_PATTERN.match(response.headers["X-Request-ID"])
    assert response.headers["X-Request-ID"] == detail["request_id"]
    assert "sensitive internal failure detail" not in response.text
    assert "RuntimeError" not in response.text


# ---------------------------------------------------------------------------
# legacy compatibility
# ---------------------------------------------------------------------------


def test_legacy_observation_route_exists(client: TestClient) -> None:
    response = client.post("/api/observations")
    assert response.status_code != 404


def test_legacy_vqa_route_exists(client: TestClient) -> None:
    response = client.post("/api/vqa", json={})
    assert response.status_code == 422  # legacy validation, not 404


def test_legacy_grounding_route_exists(client: TestClient) -> None:
    response = client.post("/api/grounding", json={})
    assert response.status_code == 422  # legacy validation, not 404


def test_legacy_tile_route_exists(client: TestClient) -> None:
    response = client.get("/tiles/no-such-asset/0/0/0.png")
    assert response.status_code in (400, 404)


def test_legacy_invalid_observation_response_unchanged(
    client: TestClient,
) -> None:
    response = client.post("/api/observations")
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}
    detail = body["error"]
    # legacy FailureDetail shape — no request_id, no v1 envelope
    assert set(detail) == {
        "code",
        "severity",
        "outcome",
        "user_message",
        "technical_message",
        "affected_requirement",
        "recoverable",
        "required_action",
        "evidence_ids",
        "warnings",
    }
    assert detail["outcome"] == "REJECT"
    assert "request_id" not in detail


def test_legacy_invalid_vqa_response_unchanged(client: TestClient) -> None:
    response = client.post("/api/vqa", json={"observation_id": "bad", "question": ""})
    assert response.status_code == 422
    detail = response.json()["error"]
    assert detail["code"] == "INVALID_VQA_REQUEST"
    assert "request_id" not in detail


def test_legacy_invalid_grounding_response_unchanged(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/grounding", json={"observation_id": "bad", "query": "!!!"}
    )
    assert response.status_code == 422
    detail = response.json()["error"]
    assert detail["code"] == "INVALID_GROUNDING_REQUEST"
    assert "request_id" not in detail


def test_legacy_404_keeps_legacy_shape(client: TestClient) -> None:
    response = client.get("/api/not-a-legacy-route")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
