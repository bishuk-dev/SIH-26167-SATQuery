"""Strict deterministic-tool and runtime-capability registries.

The YAML file is declarative metadata only. Execution remains code-owned: an
executor enum and implementation allow-list are required before a tool can be
considered runnable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from os import PathLike

import yaml
from pydantic import Field, field_validator, model_validator

from satquery.ingestion.models import ContractModel
from satquery.registry.models import (
    ModelRegistry,
    ToolExecutor,
    load_model_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOL_REGISTRY = Path(__file__).with_name("tools.yaml")
DEFAULT_CAPABILITY_CONTRACT = (
    PROJECT_ROOT / "experiments" / "phase5_backend" / "backend_contract.yaml"
)

_EXECUTOR_IMPLEMENTATIONS: dict[ToolExecutor, str] = {
    ToolExecutor.SAR_TEMPORAL_CHANGE: "satquery.analytics.sar.sar_temporal_change",
    ToolExecutor.MASK_AGREEMENT: (
        "satquery.analytics.reconciliation.diagnostic_mask_agreement"
    ),
    ToolExecutor.MASK_AREA: "satquery.analytics.measurement.measure_mask_area",
}
_UNSAFE_PARAMETER_NAMES = {"command", "code", "url", "path"}
_PARAMETER_TYPES = {
    "boolean",
    "number",
    "integer",
    "enum",
    "enum_list",
    "server_id",
    "roi",
    "mask",
}
_KNOWN_MODEL_TASKS = {"single_image_vqa", "text_guided_grounding"}


class CapabilityState(StrEnum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_WITH_LIMITS = "AVAILABLE_WITH_LIMITS"
    UNAVAILABLE_CONTRACT_BLOCKED = "UNAVAILABLE_CONTRACT_BLOCKED"
    UNAVAILABLE_MODEL_NOT_INSTALLED = "UNAVAILABLE_MODEL_NOT_INSTALLED"
    UNAVAILABLE_RUNTIME = "UNAVAILABLE_RUNTIME"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class ToolInput(ContractModel):
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    temporal_role: str | None = None


class ToolParameter(ContractModel):
    type: str
    values: tuple[str, ...] = ()

    @field_validator("values", mode="before")
    @classmethod
    def normalize_values(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    minimum: float | None = None
    maximum: float | None = None
    min_items: int | None = Field(default=None, ge=0)
    max_items: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ToolParameter":
        if self.type not in _PARAMETER_TYPES:
            raise ValueError(f"unsupported parameter type: {self.type}")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("parameter minimum cannot exceed maximum")
        if self.type in {"number", "integer"}:
            if self.minimum is None or self.maximum is None:
                raise ValueError("numeric parameters require minimum and maximum")
        if self.type in {"enum", "enum_list"} and not self.values:
            raise ValueError("enum parameters require non-empty values")
        if self.type == "enum_list":
            if self.min_items is None or self.max_items is None:
                raise ValueError("enum_list parameters require item bounds")
        if self.min_items is not None and self.max_items is not None:
            if self.min_items > self.max_items:
                raise ValueError("parameter min_items cannot exceed max_items")
        return self


class ToolEvidence(ContractModel):
    type: str = Field(min_length=1)
    deterministic: bool = False
    requires_verified_common_grid: bool = False
    interpretation: str | None = None


class ToolRegistration(ContractModel):
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)
    executor: ToolExecutor
    implementation: str = Field(min_length=1)
    inputs: tuple[ToolInput, ...] = ()
    parameters: dict[str, ToolParameter] = Field(default_factory=dict)
    evidence: ToolEvidence

    @field_validator("executor", mode="before")
    @classmethod
    def normalize_executor(cls, value: object) -> object:
        return ToolExecutor(value) if isinstance(value, str) else value

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_implementation(self) -> "ToolRegistration":
        expected = _EXECUTOR_IMPLEMENTATIONS.get(self.executor)
        if expected is None:
            raise ValueError(f"unknown executor: {self.executor.value}")
        if self.implementation != expected:
            raise ValueError(
                f"implementation for {self.executor.value} must be {expected}"
            )
        for name in self.parameters:
            if name.casefold() in _UNSAFE_PARAMETER_NAMES:
                raise ValueError(f"unsafe parameter name: {name}")
        return self


class ToolRegistry(ContractModel):
    schema_version: Literal[1]
    tools: dict[str, ToolRegistration]

    @model_validator(mode="after")
    def validate_tool_keys(self) -> "ToolRegistry":
        for tool_id, tool in self.tools.items():
            if tool_id != tool.tool_id:
                raise ValueError(f"tool ID does not match registry key: {tool_id}")
        return self

    @property
    def registry_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, tool_id: str) -> ToolRegistration | None:
        return self.tools.get(tool_id)

    def public_projection(self, tool_id: str | None = None) -> list[dict[str, Any]]:
        registrations = (
            [self.tools[tool_id]]
            if tool_id is not None and tool_id in self.tools
            else list(self.tools.values())
        )
        return [
            {
                "tool_id": registration.tool_id,
                "title": registration.title,
                "kind": registration.kind,
                "description": registration.description,
                "executor": registration.executor.value,
                "inputs": [item.model_dump(mode="json") for item in registration.inputs],
                "parameters": {
                    name: parameter.model_dump(mode="json")
                    for name, parameter in registration.parameters.items()
                },
                "evidence": registration.evidence.model_dump(mode="json"),
            }
            for registration in registrations
        ]


class RuntimeCapability(ContractModel):
    capability_id: str = Field(min_length=1)
    status: CapabilityState
    source_phase: int | None = None
    tool_ids: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    executor: ToolExecutor | None = None
    contract_status: str | None = None
    runtime_status: str | None = None
    promotion_status: str | None = None
    reason: str | None = None


def load_tool_registry(path: str | Path | None = None) -> ToolRegistry:
    """Load and validate the strict, code-bound tool registry."""

    registry_path = Path(path or DEFAULT_TOOL_REGISTRY)
    raw = _read_yaml_with_duplicate_detection(registry_path)
    if not isinstance(raw, dict):
        raise ValueError("tool registry must be a mapping")
    unknown_keys = set(raw) - {"schema_version", "tools"}
    if unknown_keys:
        names = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ValueError(f"unknown top-level tool registry keys: {names}")
    raw_tools = raw.get("tools")
    if not isinstance(raw_tools, dict):
        raise ValueError("tool registry must contain a tools mapping")
    prepared: dict[str, Any] = {}
    for tool_id, value in raw_tools.items():
        if not isinstance(tool_id, str) or not isinstance(value, dict):
            raise ValueError("tool registry IDs and records must be mappings")
        if tool_id in prepared:
            raise ValueError(f"duplicate tool ID: {tool_id}")
        record = dict(value)
        record["tool_id"] = tool_id
        prepared[tool_id] = record
    try:
        return ToolRegistry.model_validate(
            {"schema_version": raw.get("schema_version"), "tools": prepared}
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def build_runtime_capabilities(
    registry: ToolRegistry,
    *,
    capability_documents: Mapping[str, Mapping[str, Any]] | None = None,
    model_registry: ModelRegistry | None = None,
    readiness_probe: Callable[[ToolRegistration], bool] | None = None,
    model_readiness_probe: Callable[[str, Any], bool] | None = None,
    model_root: str | PathLike[str] | None = None,
) -> tuple[RuntimeCapability, ...]:
    """Derive runtime status from frozen capability documents and probes.

    A frozen ``AVAILABLE`` label is never sufficient by itself. Tool-backed
    capabilities additionally need every tool registration, a code-owned
    executor, and a successful readiness probe. Blocked or unpromoted learned
    capabilities remain unavailable regardless of probe results.
    """

    documents = (
        _load_capability_documents()
        if capability_documents is None
        else capability_documents
    )
    models = model_registry or load_model_registry()
    result: list[RuntimeCapability] = []
    for capability_id, document in documents.items():
        _validate_capability_document(capability_id, document)
        status = _coerce_capability_state(document.get("status", "NOT_IMPLEMENTED"))
        tool_ids = _capability_id_list(capability_id, document, "tool_ids")
        model_ids = _capability_id_list(capability_id, document, "model_ids")
        contract_status = document.get("contract_status")
        promotion_status = document.get("promotion_status")
        runtime_status = document.get("runtime_status")
        reason: str | None = None
        executor: ToolExecutor | None = None

        if status is CapabilityState.UNAVAILABLE_CONTRACT_BLOCKED or contract_status == "BLOCKED":
            status = CapabilityState.UNAVAILABLE_CONTRACT_BLOCKED
            reason = "capability contract is blocked"
        elif promotion_status is not None and promotion_status != "PROMOTED":
            status = CapabilityState.UNAVAILABLE_CONTRACT_BLOCKED
            reason = "capability is not promoted"
        elif tool_ids:
            runtime_status = "PROBED"
            registrations = [registry.get(tool_id) for tool_id in tool_ids]
            if any(registration is None for registration in registrations):
                status = CapabilityState.UNAVAILABLE_RUNTIME
                reason = "one or more capability tools are not registered"
            elif any(registration.executor not in _EXECUTOR_IMPLEMENTATIONS for registration in registrations if registration):
                status = CapabilityState.UNAVAILABLE_RUNTIME
                reason = "capability has no known executor adapter"
            else:
                first = registrations[0]
                executor = first.executor if first is not None else None
                probe = readiness_probe or _default_readiness_probe
                try:
                    ready = all(probe(registration) for registration in registrations if registration)
                except Exception:
                    ready = False
                if not ready:
                    status = CapabilityState.UNAVAILABLE_RUNTIME
                    reason = "registered executor readiness probe failed"
        elif model_ids:
            missing = [model_id for model_id in model_ids if model_id not in models.models]
            if missing:
                runtime_status = "NOT_PROBED"
                status = CapabilityState.UNAVAILABLE_MODEL_NOT_INSTALLED
                reason = "required model registration is not present"
            elif model_root is None or model_readiness_probe is None:
                runtime_status = "NOT_PROBED"
                status = CapabilityState.UNAVAILABLE_RUNTIME
                reason = "model checkpoint/runtime readiness was not probed"
            else:
                runtime_status = "PROBED"
                registrations = [models.models[model_id] for model_id in model_ids]
                if any(getattr(model, "task", None) not in _KNOWN_MODEL_TASKS for model in registrations):
                    status = CapabilityState.UNAVAILABLE_RUNTIME
                    reason = "no known runtime adapter for one or more models"
                else:
                    checkpoint_root = Path(model_root)
                    checkpoints_present = all(
                        _checkpoint_present(
                            checkpoint_root,
                            model_id,
                            models.models[model_id],
                        )
                        for model_id in model_ids
                    )
                    ready = checkpoints_present and all(
                        model_readiness_probe(model_id, models.models[model_id])
                        for model_id in model_ids
                    )
                    if not ready:
                        status = CapabilityState.UNAVAILABLE_RUNTIME
                        reason = "model checkpoint/runtime readiness probe failed"

        result.append(
            RuntimeCapability(
                capability_id=capability_id,
                status=status,
                source_phase=document.get("source_phase"),
                tool_ids=tool_ids,
                model_ids=model_ids,
                executor=executor,
                contract_status=contract_status,
                runtime_status=runtime_status,
                promotion_status=promotion_status,
                reason=reason,
            )
        )
    return tuple(result)


def load_runtime_capabilities(
    registry: ToolRegistry | None = None, **kwargs: Any
) -> tuple[RuntimeCapability, ...]:
    return build_runtime_capabilities(registry or load_tool_registry(), **kwargs)


def _coerce_capability_state(value: Any) -> CapabilityState:
    try:
        return CapabilityState(value)
    except ValueError as exc:
        raise ValueError(f"unknown capability state: {value}") from exc


def _default_readiness_probe(registration: ToolRegistration) -> bool:
    module_name, attribute = registration.implementation.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return callable(getattr(module, attribute))


def _checkpoint_present(root: Path, model_id: str, registration: Any) -> bool:
    checkpoint = root / model_id / registration.checkpoint_file
    return checkpoint.is_file()


def _validate_capability_document(
    capability_id: object, document: object
) -> None:
    if not isinstance(capability_id, str) or not capability_id:
        raise ValueError("capability IDs must be non-empty strings")
    if not isinstance(document, Mapping):
        raise ValueError(f"capability '{capability_id}' must be a mapping")
    for field in ("tool_ids", "model_ids"):
        value = document.get(field, ())
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError(
                f"capability '{capability_id}' field '{field}' must be a list"
            )
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError(
                f"capability '{capability_id}' field '{field}' must contain IDs"
            )


def _capability_id_list(
    capability_id: str, document: Mapping[str, Any], field: str
) -> tuple[str, ...]:
    value = document.get(field, ())
    _validate_capability_document(capability_id, document)
    return tuple(value)


def _load_capability_documents() -> dict[str, Mapping[str, Any]]:
    raw = _read_yaml_with_duplicate_detection(DEFAULT_CAPABILITY_CONTRACT)
    if not isinstance(raw, dict):
        raise ValueError("capability contract must be a mapping")
    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("capability contract must contain a capabilities mapping")
    for capability_id, document in capabilities.items():
        _validate_capability_document(capability_id, document)
    return dict(capabilities)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _read_yaml_with_duplicate_detection(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return yaml.load(file_handle, Loader=_UniqueKeyLoader)


__all__ = [
    "CapabilityState",
    "RuntimeCapability",
    "ToolEvidence",
    "ToolExecutor",
    "ToolInput",
    "ToolParameter",
    "ToolRegistration",
    "ToolRegistry",
    "build_runtime_capabilities",
    "load_runtime_capabilities",
    "load_tool_registry",
]
