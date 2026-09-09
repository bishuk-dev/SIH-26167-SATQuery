"""Versioned model, tool, and preprocessing registry access."""

from satquery.registry.models import (
    BifoldPreprocessingProfile,
    ModelRegistration,
    MultisensorModelRegistration,
    NativeMultisensorPreprocessingProfile,
    PreprocessingProfile,
    ToolExecutor,
    load_model_registry,
    load_preprocessing_registry,
)
from satquery.registry.tools import (
    CapabilityState,
    RuntimeCapability,
    ToolEvidence,
    ToolInput,
    ToolParameter,
    ToolRegistration,
    ToolRegistry,
    build_runtime_capabilities,
    load_runtime_capabilities,
    load_tool_registry,
)

__all__ = [
    "BifoldPreprocessingProfile",
    "ModelRegistration",
    "MultisensorModelRegistration",
    "NativeMultisensorPreprocessingProfile",
    "PreprocessingProfile",
    "ToolExecutor",
    "CapabilityState",
    "RuntimeCapability",
    "ToolEvidence",
    "ToolInput",
    "ToolParameter",
    "ToolRegistration",
    "ToolRegistry",
    "build_runtime_capabilities",
    "load_runtime_capabilities",
    "load_tool_registry",
    "load_model_registry",
    "load_preprocessing_registry",
]
