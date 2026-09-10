"""Typed evidence graph representation for persisted analyses.

The graph is intentionally a small adjacency model over validated Pydantic
scientific evidence records. It stores evidence payloads and typed edges; it is
not a graph framework and does not infer scientific relationships from prose.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, computed_field, model_validator

from satquery.evidence.models import (
    AgreementEvidence,
    ChangeCaptionEvidence,
    ChangeMaskEvidence,
    FloodMaskEvidence,
    GroundingEvidence,
    MeasurementEvidence,
    VqaEvidence,
)
from satquery.ingestion.models import ContractModel


class EdgeType(StrEnum):
    DERIVED_FROM = "DERIVED_FROM"
    MEASURED_FROM = "MEASURED_FROM"
    SUPPORTS = "SUPPORTS"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    DESCRIBES = "DESCRIBES"
    LOCALIZED_BY = "LOCALIZED_BY"


EvidenceNode = Annotated[
    VqaEvidence
    | GroundingEvidence
    | ChangeMaskEvidence
    | FloodMaskEvidence
    | ChangeCaptionEvidence
    | MeasurementEvidence
    | AgreementEvidence,
    Field(discriminator="task"),
]
EVIDENCE_NODE_ADAPTER: TypeAdapter[EvidenceNode] = TypeAdapter(EvidenceNode)

_DERIVATION_EDGE_TYPES = frozenset(
    {
        EdgeType.DERIVED_FROM,
        EdgeType.MEASURED_FROM,
        EdgeType.LOCALIZED_BY,
    }
)


class EvidenceEdge(ContractModel):
    """A typed relationship between two evidence nodes.

    ``source_evidence_id -> target_evidence_id`` means the target record uses,
    measures, supports, describes, localizes, or conflicts with the source,
    depending on ``edge_type``. The graph validates IDs but verification rules
    decide whether the relationship is scientifically admissible.
    """

    source_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    target_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    edge_type: EdgeType

    @model_validator(mode="after")
    def require_distinct_nodes(self) -> "EvidenceEdge":
        if self.source_evidence_id == self.target_evidence_id:
            raise ValueError("evidence edges cannot point to themselves")
        return self


class EvidenceGraph(ContractModel):
    """Evidence nodes plus typed edges for one analysis.

    Duplicate node IDs are accepted only when their payload JSON is identical,
    allowing idempotent row loading without allowing contradictory evidence to
    share an identifier.
    """

    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...] = ()
    input_ids: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def node_by_id(self) -> dict[str, EvidenceNode]:
        unique: dict[str, EvidenceNode] = {}
        for node in self.nodes:
            unique.setdefault(evidence_id_of(node), node)
        return unique

    @model_validator(mode="after")
    def validate_graph(self) -> "EvidenceGraph":
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("input_ids must be unique")

        payloads_by_id: dict[str, str] = {}
        for node in self.nodes:
            node_id = evidence_id_of(node)
            payload = canonical_evidence_json(node)
            previous = payloads_by_id.setdefault(node_id, payload)
            if previous != payload:
                raise ValueError(
                    f"duplicate evidence ID {node_id} has different payloads"
                )

        node_ids = set(payloads_by_id)
        for edge in self.edges:
            if (
                edge.source_evidence_id not in node_ids
                or edge.target_evidence_id not in node_ids
            ):
                raise ValueError(
                    "dangling evidence edge references an ID absent from graph nodes"
                )

        _reject_derivation_cycles(self.edges)
        return self

    def to_evidence_rows(self, *, analysis_id: str) -> tuple[dict[str, Any], ...]:
        """Serialize unique nodes as rows compatible with the SQLite schema."""

        rows: list[dict[str, Any]] = []
        for evidence_id, node in self.node_by_id.items():
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "analysis_id": analysis_id,
                    "created_at": node.provenance.created_at.isoformat().replace("+00:00", "Z"),
                    "payload_json": canonical_evidence_json(node),
                }
            )
        return tuple(rows)

    def to_edge_rows(self, *, analysis_id: str) -> tuple[dict[str, Any], ...]:
        """Serialize graph edges as rows compatible with the SQLite schema."""

        return tuple(
            {
                "analysis_id": analysis_id,
                "source_evidence_id": edge.source_evidence_id,
                "target_evidence_id": edge.target_evidence_id,
                "edge_type": edge.edge_type.value,
            }
            for edge in self.edges
        )

    @classmethod
    def from_rows(
        cls,
        evidence_rows: tuple[Any, ...] | list[Any],
        edge_rows: tuple[Any, ...] | list[Any],
        *,
        input_ids: tuple[str, ...] = (),
    ) -> "EvidenceGraph":
        """Load evidence JSON and edge rows from dict- or sqlite.Row-like objects."""

        nodes: list[EvidenceNode] = []
        for row in evidence_rows:
            payload = _row_value(row, "payload_json")
            if isinstance(payload, str):
                nodes.append(EVIDENCE_NODE_ADAPTER.validate_json(payload))
            else:
                nodes.append(EVIDENCE_NODE_ADAPTER.validate_python(payload))

        edges = tuple(
            EvidenceEdge(
                source_evidence_id=str(_row_value(row, "source_evidence_id")),
                target_evidence_id=str(_row_value(row, "target_evidence_id")),
                edge_type=EdgeType(str(_row_value(row, "edge_type"))),
            )
            for row in edge_rows
        )
        return cls(nodes=tuple(nodes), edges=edges, input_ids=input_ids)


def evidence_id_of(node: EvidenceNode) -> str:
    return node.evidence_id


def canonical_evidence_json(node: EvidenceNode) -> str:
    return json.dumps(
        node.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    return row[key]


def _reject_derivation_cycles(edges: tuple[EvidenceEdge, ...]) -> None:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge.edge_type in _DERIVATION_EDGE_TYPES:
            adjacency.setdefault(edge.source_evidence_id, []).append(edge.target_evidence_id)

    visited: set[str] = set()
    active: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in active:
            raise ValueError("cycle detected in evidence derivation edges")
        if node_id in visited:
            return
        active.add(node_id)
        for child_id in adjacency.get(node_id, ()):
            visit(child_id)
        active.remove(node_id)
        visited.add(node_id)

    for node_id in tuple(adjacency):
        visit(node_id)


__all__ = [
    "EVIDENCE_NODE_ADAPTER",
    "EdgeType",
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceNode",
    "canonical_evidence_json",
    "evidence_id_of",
]
