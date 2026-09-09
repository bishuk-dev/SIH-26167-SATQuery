from __future__ import annotations

from pathlib import Path

import pytest

from satquery.registry.tools import (
    CapabilityState,
    ToolExecutor,
    build_runtime_capabilities,
    load_tool_registry,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_tool_registry_rejects_unbounded_parameters(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.yaml",
        """
schema_version: 1
tools:
  bad:
    title: Bad
    kind: deterministic_analytics
    description: Bad
    executor: sar_temporal_change
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters:
      command: {type: string}
    evidence: {type: change_mask, deterministic: true}
""",
    )
    with pytest.raises(ValueError, match="command"):
        load_tool_registry(path)


def test_tool_registry_rejects_unknown_executor(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.yaml",
        """
schema_version: 1
tools:
  bad:
    title: Bad
    kind: deterministic_analytics
    description: Bad
    executor: shell
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters: {}
    evidence: {type: change_mask, deterministic: true}
""",
    )
    with pytest.raises(ValueError, match="executor"):
        load_tool_registry(path)


def test_tool_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.yaml",
        """
schema_version: 1
tools:
  duplicate:
    title: First
    kind: deterministic_analytics
    description: First
    executor: sar_temporal_change
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters: {}
    evidence: {type: change_mask, deterministic: true}
  duplicate:
    title: Second
    kind: deterministic_analytics
    description: Second
    executor: sar_temporal_change
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters: {}
    evidence: {type: change_mask, deterministic: true}
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_tool_registry(path)


def test_tool_registry_requires_implementation_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.yaml",
        """
schema_version: 1
tools:
  bad:
    title: Bad
    kind: deterministic_analytics
    description: Bad
    executor: sar_temporal_change
    inputs: []
    parameters: {}
    evidence: {type: change_mask, deterministic: true}
""",
    )
    with pytest.raises(ValueError, match="implementation"):
        load_tool_registry(path)


def test_tool_registry_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "tools.yaml",
        """
schema_version: 1
unexpected: true
tools: {}
""",
    )
    with pytest.raises(ValueError, match="unknown top-level"):
        load_tool_registry(path)


def test_tool_registry_hash_is_stable_for_mapping_order(tmp_path: Path) -> None:
    first = _write(
        tmp_path / "first.yaml",
        """
schema_version: 1
tools:
  sar_temporal_change_v1:
    title: SAR
    kind: deterministic_analytics
    description: SAR change
    executor: sar_temporal_change
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters:
      threshold: {type: number, minimum: 0, maximum: 1000000}
    evidence: {type: change_mask, deterministic: true}
  mask_agreement_v1:
    title: Agreement
    kind: deterministic_analytics
    description: Agreement
    executor: mask_agreement
    implementation: satquery.analytics.reconciliation.diagnostic_mask_agreement
    inputs: []
    parameters: {}
    evidence: {type: agreement, deterministic: true}
""",
    )
    second = _write(
        tmp_path / "second.yaml",
        """
schema_version: 1
tools:
  mask_agreement_v1:
    title: Agreement
    kind: deterministic_analytics
    description: Agreement
    executor: mask_agreement
    implementation: satquery.analytics.reconciliation.diagnostic_mask_agreement
    inputs: []
    parameters: {}
    evidence: {type: agreement, deterministic: true}
  sar_temporal_change_v1:
    title: SAR
    kind: deterministic_analytics
    description: SAR change
    executor: sar_temporal_change
    implementation: satquery.analytics.sar.sar_temporal_change
    inputs: []
    parameters:
      threshold: {type: number, minimum: 0, maximum: 1000000}
    evidence: {type: change_mask, deterministic: true}
""",
    )
    assert load_tool_registry(first).registry_hash == load_tool_registry(second).registry_hash


def test_mask_area_tool_is_registered_to_a_code_owned_executor() -> None:
    registry = load_tool_registry()
    tool = registry.get("compute_mask_area_v1")
    assert tool is not None
    assert tool.executor is ToolExecutor.MASK_AREA
    assert tool.implementation == "satquery.analytics.measurement.measure_mask_area"


def test_runtime_capability_requires_registration_and_readiness() -> None:
    registry = load_tool_registry()
    capability = build_runtime_capabilities(
        registry,
        capability_documents={
            "deterministic": {
                "status": "AVAILABLE",
                "tool_ids": ["sar_temporal_change_v1"],
            }
        },
        readiness_probe=lambda _tool: False,
    )[0]
    assert capability.status is CapabilityState.UNAVAILABLE_RUNTIME


def test_runtime_capability_is_available_after_registered_ready_probe() -> None:
    registry = load_tool_registry()
    capability = build_runtime_capabilities(
        registry,
        capability_documents={
            "deterministic": {
                "status": "AVAILABLE",
                "tool_ids": ["sar_temporal_change_v1"],
            }
        },
        readiness_probe=lambda _tool: True,
    )[0]
    assert capability.status is CapabilityState.AVAILABLE
    assert capability.executor is ToolExecutor.SAR_TEMPORAL_CHANGE


def test_blocked_capability_cannot_become_available() -> None:
    registry = load_tool_registry()
    capability = build_runtime_capabilities(
        registry,
        capability_documents={
            "blocked": {
                "status": "UNAVAILABLE_CONTRACT_BLOCKED",
                "contract_status": "BLOCKED",
                "promotion_status": "NOT_PROMOTED",
                "tool_ids": ["sar_temporal_change_v1"],
            }
        },
        readiness_probe=lambda _tool: True,
    )[0]
    assert capability.status is CapabilityState.UNAVAILABLE_CONTRACT_BLOCKED


def test_malformed_capability_document_fails_closed() -> None:
    registry = load_tool_registry()
    with pytest.raises(ValueError, match="capability 'broken'"):
        build_runtime_capabilities(
            registry,
            capability_documents={"broken": "not a mapping"},  # type: ignore[arg-type]
        )


def test_model_capability_is_not_claimed_probed_without_runtime_probe() -> None:
    registry = load_tool_registry()
    capability = build_runtime_capabilities(
        registry,
        capability_documents={
            "model": {
                "status": "AVAILABLE_WITH_LIMITS",
                "model_ids": ["smolvlm_256m_instruct_v1"],
                "contract_status": "ACCEPTED",
                "promotion_status": "PROMOTED",
            }
        },
    )[0]
    assert capability.status is CapabilityState.UNAVAILABLE_RUNTIME
    assert capability.runtime_status == "NOT_PROBED"
