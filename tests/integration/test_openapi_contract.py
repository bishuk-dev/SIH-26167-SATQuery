"""Task 20 contract tests for the public OpenAPI document."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from apps.api.app.security import SecuritySettings
from apps.api.app.routes.v1_pairs import PairCreateRequest
from apps.api.app.routes.v1_query import QueryPlanRequest


def _client(tmp_path: Path, *, api_key: str | None = None) -> TestClient:
    return TestClient(
        create_app(
            data_root=tmp_path / "data",
            security_settings=SecuritySettings(api_key=api_key),
        )
    )


def _operations(document: dict[str, Any]):
    for path, methods in document["paths"].items():
        for method, operation in methods.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                yield path, method, operation


def test_openapi_product_metadata_and_examples(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        document = client.get("/openapi.json").json()

    operation_ids: list[str] = []
    for path, method, operation in _operations(document):
        if path.startswith("/api/v1"):
            assert operation.get("operationId"), f"{method} {path} has no operationId"
            assert operation.get("tags"), f"{method} {path} has no tags"
            assert operation.get("summary"), f"{method} {path} has no summary"
            assert operation.get("responses"), f"{method} {path} has no responses"
            for status, response in operation["responses"].items():
                assert response.get("description"), f"{method} {path} {status} lacks description"
        operation_ids.append(operation["operationId"])
    assert len(operation_ids) == len(set(operation_ids))

    # Request examples are machine-valid rather than prose placeholders.
    components = document["components"]["schemas"]
    query_example = components["QueryPlanRequest"]["examples"][0]
    pair_example = components["PairCreateRequest"]["examples"][0]
    assert QueryPlanRequest.model_validate(query_example).query
    assert PairCreateRequest.model_validate(pair_example).observation_a


def test_writes_publish_structured_non_2xx_responses_and_no_path_properties(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        document = client.get("/openapi.json").json()

    for path, method, operation in _operations(document):
        if not path.startswith("/api/v1") or method not in {"post", "put", "patch", "delete"}:
            continue
        non_2xx = {
            status: response
            for status, response in operation["responses"].items()
            if status.isdigit() and int(status) >= 400
        }
        assert non_2xx, f"{method} {path} has no documented failure response"
        assert any(
            response.get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("$ref", "")
            .endswith("/ApiErrorV1")
            for response in non_2xx.values()
        ), f"{method} {path} has no structured failure envelope"

    for schema in document["components"]["schemas"].values():
        assert "path" not in schema.get("properties", {}), schema


def test_legacy_operations_are_deprecated(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = client.get("/openapi.json").json()
    legacy = [
        (path, method, operation)
        for path, method, operation in _operations(document)
        if path in {"/api/observations", "/api/vqa", "/api/grounding", "/limits"}
    ]
    assert legacy
    assert all(operation.get("deprecated") is True for _, _, operation in legacy)
    assert all("Legacy" in operation.get("tags", []) for _, _, operation in legacy)


def test_security_is_visible_only_when_api_key_is_enabled(tmp_path: Path) -> None:
    with _client(tmp_path / "disabled") as client:
        document = client.get("/openapi.json").json()
    assert "securitySchemes" not in document.get("components", {})
    assert not any("security" in operation for _, _, operation in _operations(document))

    with _client(tmp_path / "enabled", api_key="secret") as client:
        document = client.get("/openapi.json").json()
    assert "APIKeyHeader" in document["components"]["securitySchemes"]
    assert all(
        "security" in operation
        for path, _, operation in _operations(document)
        if path not in {"/api/observations", "/api/vqa", "/api/grounding", "/tiles/{asset_id}/{z}/{x}/{y}.png"}
    )
