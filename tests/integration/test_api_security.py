from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from apps.api.app.security import SecuritySettings


def test_security_settings_are_strictly_loaded_from_environment() -> None:
    settings = SecuritySettings.from_env(
        {
            "SATQUERY_CORS_ORIGINS": "https://one.example,https://two.example",
            "SATQUERY_MAX_QUERY_BYTES": "128",
            "SATQUERY_MAX_ROI_VERTICES": "7",
            "SATQUERY_MAX_QUEUED_JOBS": "3",
            "SATQUERY_MAX_RESULT_BYTES": "4096",
            "SATQUERY_API_KEY": "secret",
        }
    )
    assert settings.cors_origins == ("https://one.example", "https://two.example")
    assert settings.max_query_bytes == 128
    assert settings.max_roi_vertices == 7
    assert settings.max_queue_size == 3
    assert settings.max_result_bytes == 4096
    assert settings.api_key == "secret"


def test_configured_api_key_is_required_and_documented(tmp_path: Path) -> None:
    with TestClient(
        create_app(
            data_root=tmp_path / "data",
            security_settings=SecuritySettings(api_key="secret"),
        )
    ) as client:
        missing = client.get("/api/v1/system/version")
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "UNAUTHORIZED"
        valid = client.get("/api/v1/system/version", headers={"X-API-Key": "secret"})
        assert valid.status_code == 200
        schema = client.get("/openapi.json").json()
        assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"


def test_disabled_api_key_has_no_auth_scheme(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        schema = client.get("/openapi.json").json()
    assert "securitySchemes" not in schema.get("components", {})


def test_query_and_roi_limits_are_rejected_before_input_loading(tmp_path: Path) -> None:
    settings = SecuritySettings(max_query_bytes=256, max_roi_vertices=2)
    with TestClient(create_app(data_root=tmp_path / "data", security_settings=settings)) as client:
        oversized = client.post(
            "/api/v1/query/plan",
            json={"query": "x" * 300},
        )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "QUERY_TOO_LARGE"

        roi = client.post(
            "/api/v1/query/plan",
            json={
                "query": "show change",
                "roi": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [2, 0]]],
                },
            },
        )
        assert roi.status_code == 413
        assert roi.json()["error"]["code"] == "ROI_TOO_LARGE"


def test_raw_resource_references_are_rejected_before_planning(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.post(
            "/api/v1/query/plan",
            json={"query": "show change", "parameters": {"model_id": "model_fake"}},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSAFE_INPUT"


def test_queue_and_result_limits_are_enforced(tmp_path: Path) -> None:
    settings = SecuritySettings(max_queue_size=1, max_result_bytes=10 * 1024 * 1024)
    application = create_app(data_root=tmp_path / "data", security_settings=settings)
    application.state.job_runner.has_queue_capacity = lambda: False
    with TestClient(application) as client:
        queued = client.post("/api/v1/query", json={"query": "what changed?"})
        assert queued.status_code == 429
        assert queued.json()["error"]["code"] == "RESOURCE_BUSY"

    tiny = create_app(
        data_root=tmp_path / "tiny-data",
        security_settings=SecuritySettings(max_result_bytes=1),
    )
    with TestClient(tiny) as client:
        result = client.get("/api/v1/system/version")
    assert result.status_code == 500
    assert result.json()["error"]["code"] == "RESULT_TOO_LARGE"


def test_cors_is_exact_and_does_not_allow_disallowed_origins(tmp_path: Path) -> None:
    settings = SecuritySettings(cors_origins=("https://allowed.example",))
    with TestClient(create_app(data_root=tmp_path / "data", security_settings=settings)) as client:
        allowed = client.get(
            "/api/v1/system/version", headers={"Origin": "https://allowed.example"}
        )
        denied = client.get(
            "/api/v1/system/version", headers={"Origin": "https://denied.example"}
        )
    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in denied.headers
