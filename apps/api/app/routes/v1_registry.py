"""Safe v1 projections of registered tools and runtime capabilities."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.app.schemas import ApiModel

router = APIRouter()


class ToolInputV1(ApiModel):
    name: str
    kind: str
    temporal_role: str | None = None


class ToolParameterV1(ApiModel):
    type: str
    values: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_items: int | None = None
    max_items: int | None = None


class ToolEvidenceV1(ApiModel):
    type: str
    deterministic: bool = False
    requires_verified_common_grid: bool = False
    interpretation: str | None = None


class ToolV1(ApiModel):
    tool_id: str
    title: str
    kind: str
    description: str
    executor: str
    inputs: tuple[ToolInputV1, ...] = ()
    parameters: dict[str, ToolParameterV1] = {}
    evidence: ToolEvidenceV1


class CapabilityV1(ApiModel):
    capability_id: str
    status: str
    source_phase: int | None = None
    tool_ids: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    executor: str | None = None
    contract_status: str | None = None
    runtime_status: str | None = None
    promotion_status: str | None = None
    reason: str | None = None


class ModelV1(ApiModel):
    registry_id: str
    task: str
    provider: str
    model_id: str
    architecture: str
    license: str
    preprocessing_profile: str
    frozen: bool
    supported_modalities: tuple[str, ...] | None = None
    training_domain: str | None = None
    output_semantics: str | None = None


class ToolCollectionV1(ApiModel):
    registry_hash: str
    items: tuple[ToolV1, ...]


class CapabilityCollectionV1(ApiModel):
    registry_hash: str
    items: tuple[CapabilityV1, ...]


class ModelCollectionV1(ApiModel):
    items: tuple[ModelV1, ...]


def _model_projection(registry_id: str, registration: Any) -> dict[str, Any]:
    payload = registration.model_dump(mode="json")
    allowed = {
        "task",
        "provider",
        "model_id",
        "architecture",
        "license",
        "preprocessing_profile",
        "frozen",
        "supported_modalities",
        "training_domain",
        "output_semantics",
    }
    return {
        "registry_id": registry_id,
        **{key: value for key, value in payload.items() if key in allowed},
    }


@router.get(
    "/api/v1/tools",
    tags=["Tools"],
    response_model=ToolCollectionV1,
    operation_id="list_tools_v1",
    summary="List registered deterministic tools.",
    description="Returns bounded tool metadata without executable implementation paths.",
)
def list_tools_v1(request: Request) -> ToolCollectionV1:
    registry = request.app.state.tool_registry
    return ToolCollectionV1(
        registry_hash=registry.registry_hash,
        items=tuple(ToolV1.model_validate(item) for item in registry.public_projection()),
    )


@router.get(
    "/api/v1/tools/{tool_id}",
    tags=["Tools"],
    response_model=ToolV1,
    operation_id="get_tool_v1",
    summary="Get one registered deterministic tool.",
    description="Returns a safe projection of one code-owned tool registration.",
)
def get_tool_v1(request: Request, tool_id: str) -> dict[str, Any]:
    registry = request.app.state.tool_registry
    if registry.get(tool_id) is None:
        raise HTTPException(status_code=404)
    return ToolV1.model_validate(registry.public_projection(tool_id)[0])


@router.get(
    "/api/v1/models",
    tags=["Models"],
    response_model=ModelCollectionV1,
    operation_id="list_models_v1",
    summary="List registered models.",
    description="Returns safe model metadata without checkpoint paths or files.",
)
def list_models_v1(request: Request) -> ModelCollectionV1:
    registry = request.app.state.model_registry
    return ModelCollectionV1(
        items=tuple(
            ModelV1.model_validate(_model_projection(registry_id, registration))
            for registry_id, registration in registry.models.items()
        )
    )


@router.get(
    "/api/v1/models/{model_id}",
    tags=["Models"],
    response_model=ModelV1,
    operation_id="get_model_v1",
    summary="Get one registered model.",
    description="Returns safe metadata for one registered model.",
)
def get_model_v1(request: Request, model_id: str) -> dict[str, Any]:
    registry = request.app.state.model_registry
    registration = registry.models.get(model_id)
    if registration is None:
        raise HTTPException(status_code=404)
    return ModelV1.model_validate(_model_projection(model_id, registration))


@router.get(
    "/api/v1/system/capabilities",
    tags=["System"],
    response_model=CapabilityCollectionV1,
    operation_id="list_capabilities_v1",
    summary="List runtime capability states.",
    description=(
        "Reports frozen capability inventory and dynamic readiness. Blocked or "
        "unpromoted specialists are never exposed as available."
    ),
)
def list_capabilities_v1(request: Request) -> CapabilityCollectionV1:
    registry = request.app.state.tool_registry
    capabilities = request.app.state.runtime_capabilities
    return CapabilityCollectionV1(
        registry_hash=registry.registry_hash,
        items=tuple(
            CapabilityV1.model_validate(capability.model_dump(mode="json"))
            for capability in capabilities
        ),
    )
