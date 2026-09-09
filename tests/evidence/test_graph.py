"""Evidence graph structural contract tests for Phase 5 Task 11."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from satquery.evidence.graph import EdgeType, EvidenceEdge, EvidenceGraph
from satquery.evidence.models import (
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    MaskAsset,
    MeasurementEvidence,
    TemporalPairEvidence,
    VqaEvidence,
    VqaPrediction,
)
from satquery.ingestion.models import AffineTransform, GeoBounds, Modality


def _eid(index: int) -> str:
    return f"evidence_{index:032x}"


def _provenance(*parents: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        operation_id="test_operation",
        input_asset_id="asset_1",
        parent_evidence_ids=tuple(parents),
    )


def _domain() -> DomainAssessment:
    return DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=())


def _write_mask(path, values) -> str:
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(32643),
        transform=transform,
    ) as dataset:
        dataset.write(values.astype("uint8"), 1)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_asset(tmp_path, *, asset_id: str = "mask_1") -> MaskAsset:
    import numpy as np

    path = tmp_path / f"{asset_id}.tif"
    sha256 = _write_mask(path, np.array([[1, 0], [1, 1]], dtype="uint8"))
    return MaskAsset(
        asset_id=asset_id,
        path=str(path),
        sha256=sha256,
        width=2,
        height=2,
        crs="EPSG:32643",
        transform=AffineTransform(a=10.0, b=0.0, c=0.0, d=0.0, e=-10.0, f=20.0),
        bounds=GeoBounds(left=0.0, bottom=0.0, right=20.0, top=20.0),
        source_grid_observation_id="obs_t1",
    )


def _change_mask(tmp_path, *, evidence_id: str = _eid(1)) -> ChangeMaskEvidence:
    return ChangeMaskEvidence(
        evidence_id=evidence_id,
        target_class="vegetation",
        change_kind="gain",
        temporal=TemporalPairEvidence(
            t1_observation_id="obs_t1",
            t2_observation_id="obs_t2",
            order_source="metadata",
        ),
        mask=_mask_asset(tmp_path),
        tool_id="temporal_difference_v1",
        domain=_domain(),
        provenance=_provenance(),
    )


def _measurement(source_id: str, *, evidence_id: str = _eid(2), value: float = 300.0) -> MeasurementEvidence:
    return MeasurementEvidence(
        evidence_id=evidence_id,
        source_evidence_id=source_id,
        value=value,
        unit="m2",
        method="projected_affine_determinant",
        calculation_crs="EPSG:32643",
        positive_pixel_count=3,
        valid_pixel_count=4,
        tool_id="compute_mask_area_v1",
        provenance=_provenance(source_id),
    )


def _vqa(*, evidence_id: str = _eid(3), answer: str = "yes") -> VqaEvidence:
    return VqaEvidence(
        evidence_id=evidence_id,
        prediction=VqaPrediction(answer=answer, raw_score=0.5),
        source_observations=("obs_t1",),
        source_modalities=(Modality.OPTICAL,),
        model={
            "registry_id": "registry_v1",
            "model_id": "org/model",
            "revision": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "preprocessing_profile": "profile_v1",
            "preprocessing_version": "1.0.0",
        },
        domain=_domain(),
        provenance=_provenance(),
    )


def test_graph_rejects_dangling_edges(tmp_path) -> None:
    mask = _change_mask(tmp_path)

    with pytest.raises(ValueError, match="dangling"):
        EvidenceGraph(
            nodes=(mask,),
            edges=(
                EvidenceEdge(
                    source_evidence_id=mask.evidence_id,
                    target_evidence_id=_eid(99),
                    edge_type=EdgeType.SUPPORTS,
                ),
            ),
        )


def test_graph_rejects_duplicate_ids_with_different_payloads(tmp_path) -> None:
    mask = _change_mask(tmp_path, evidence_id=_eid(1))
    vqa = _vqa(evidence_id=_eid(1))

    with pytest.raises(ValueError, match="duplicate evidence ID"):
        EvidenceGraph(nodes=(mask, vqa), edges=())


def test_graph_reuses_duplicate_ids_with_identical_payloads(tmp_path) -> None:
    mask = _change_mask(tmp_path, evidence_id=_eid(1))

    graph = EvidenceGraph(nodes=(mask, mask), edges=())

    assert tuple(graph.node_by_id) == (mask.evidence_id,)


def test_graph_rejects_cycles_only_for_derivation_edges(tmp_path) -> None:
    mask = _change_mask(tmp_path, evidence_id=_eid(1))
    measurement = _measurement(mask.evidence_id, evidence_id=_eid(2))

    with pytest.raises(ValueError, match="cycle"):
        EvidenceGraph(
            nodes=(mask, measurement),
            edges=(
                EvidenceEdge(
                    source_evidence_id=mask.evidence_id,
                    target_evidence_id=measurement.evidence_id,
                    edge_type=EdgeType.DERIVED_FROM,
                ),
                EvidenceEdge(
                    source_evidence_id=measurement.evidence_id,
                    target_evidence_id=mask.evidence_id,
                    edge_type=EdgeType.DERIVED_FROM,
                ),
            ),
        )

    graph = EvidenceGraph(
        nodes=(mask, measurement),
        edges=(
            EvidenceEdge(
                source_evidence_id=mask.evidence_id,
                target_evidence_id=measurement.evidence_id,
                edge_type=EdgeType.SUPPORTS,
            ),
            EvidenceEdge(
                source_evidence_id=measurement.evidence_id,
                target_evidence_id=mask.evidence_id,
                edge_type=EdgeType.CONFLICTS_WITH,
            ),
        ),
    )
    assert len(graph.edges) == 2


def test_graph_round_trips_discriminated_persistence_rows(tmp_path) -> None:
    mask = _change_mask(tmp_path, evidence_id=_eid(1))
    measurement = _measurement(mask.evidence_id, evidence_id=_eid(2))
    edge = EvidenceEdge(
        source_evidence_id=mask.evidence_id,
        target_evidence_id=measurement.evidence_id,
        edge_type=EdgeType.MEASURED_FROM,
    )
    graph = EvidenceGraph(nodes=(mask, measurement), edges=(edge,), input_ids=("obs_t1", "obs_t2"))

    evidence_rows = graph.to_evidence_rows(analysis_id="ana_" + "0" * 32)
    edge_rows = graph.to_edge_rows(analysis_id="ana_" + "0" * 32)
    assert json.loads(evidence_rows[0]["payload_json"])["task"] == "change_localize"

    loaded = EvidenceGraph.from_rows(evidence_rows, edge_rows, input_ids=("obs_t1", "obs_t2"))

    assert loaded.node_by_id.keys() == graph.node_by_id.keys()
    assert loaded.edges == graph.edges
