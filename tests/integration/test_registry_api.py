from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app.main import create_app


def test_registry_api_projects_safe_tool_metadata(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.get("/api/v1/tools")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["registry_hash"]
        assert {item["tool_id"] for item in body["items"]} == {
            "sar_temporal_change_v1",
            "mask_agreement_v1",
            "compute_mask_area_v1",
        }
        assert all("implementation" not in item for item in body["items"])
        assert all("path" not in item for item in body["items"])


def test_registry_routes_use_frozen_tags_and_typed_item_responses(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        schema = client.get("/openapi.json").json()

    assert schema["paths"]["/api/v1/tools"]["get"]["tags"] == ["Tools"]
    assert schema["paths"]["/api/v1/tools/{tool_id}"]["get"]["tags"] == ["Tools"]
    assert schema["paths"]["/api/v1/models"]["get"]["tags"] == ["Models"]
    assert schema["paths"]["/api/v1/models/{model_id}"]["get"]["tags"] == ["Models"]
    assert schema["paths"]["/api/v1/system/capabilities"]["get"]["tags"] == ["System"]
    assert schema["paths"]["/api/v1/tools/{tool_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("ToolV1")
    assert schema["paths"]["/api/v1/models/{model_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("ModelV1")


def test_registry_api_get_and_unknown_tool(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.get("/api/v1/tools/sar_temporal_change_v1")
        assert response.status_code == 200
        assert response.json()["tool_id"] == "sar_temporal_change_v1"

        missing = client.get("/api/v1/tools/not_registered")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_model_api_projects_safe_model_metadata(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.get("/api/v1/models")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["items"]
        assert {item["registry_id"] for item in body["items"]} >= {
            "smolvlm_256m_instruct_v1",
            "grounding_dino_tiny_v1",
        }
        assert all("checkpoint_file" not in item for item in body["items"])
        assert all("path" not in item for item in body["items"])

        model = client.get("/api/v1/models/smolvlm_256m_instruct_v1")
        assert model.status_code == 200
        assert model.json()["registry_id"] == "smolvlm_256m_instruct_v1"


def test_capability_api_keeps_blocked_specialists_unavailable(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path / "data")) as client:
        response = client.get("/api/v1/system/capabilities")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["registry_hash"]
        by_id = {item["capability_id"]: item for item in body["items"]}
        assert by_id["learned_structural_change"]["status"] == (
            "UNAVAILABLE_CONTRACT_BLOCKED"
        )
        assert by_id["learned_change_captioning"]["status"] == (
            "UNAVAILABLE_CONTRACT_BLOCKED"
        )
        assert by_id["learned_flood_segmentation"]["status"] == (
            "UNAVAILABLE_CONTRACT_BLOCKED"
        )
        assert "implementation" not in by_id["learned_structural_change"]
        assert "path" not in by_id["learned_structural_change"]
