from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.artifacts import ArtifactStore
from satquery.execution import ExecutionContext, ExecutionEngine, ExecutionPlan, PlanStep
from satquery.execution.adapters import (
    AdapterInputError,
    build_registered_adapters,
)
from satquery.execution.models import ToolCall
from satquery.ingestion.models import (
    AffineTransform,
    BandMetadata,
    GeoBounds,
    GeoMetadata,
    Modality,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)
from satquery.persistence import Database, MetadataRepository, ObservationRecord
from satquery.registry.models import ToolExecutor
from satquery.registry.tools import load_tool_registry


_T1_ID = "obs_" + "1" * 32
_T2_ID = "obs_" + "2" * 32
_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_TRANSFORM = Affine(10, 0, 500000, 0, -10, 2000000)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raster(path: Path, data: np.ndarray, *, crs: str = "EPSG:32643") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=1 if data.ndim == 2 else data.shape[0],
        dtype=str(data.dtype),
        crs=crs,
        transform=_TRANSFORM,
    ) as dataset:
        if data.ndim == 2:
            dataset.write(data, 1)
        else:
            dataset.write(data)


def _observation(
    observation_id: str,
    path: Path,
    *,
    modality: Modality = Modality.SAR,
    acquisition_time: datetime = _NOW,
    radiometric_domain: str | None = "backscatter_db",
) -> ObservationState:
    tags = {"calibration": "sigma0"}
    if radiometric_domain is not None:
        tags["radiometric_domain"] = radiometric_domain
    bands = (
        BandMetadata(index=1, description="VV", dtype="float32", tags=tags),
    )
    return ObservationState(
        observation_id=observation_id,
        source_asset=SourceAsset(
            asset_id="asset_" + observation_id.removeprefix("obs_"),
            original_name=f"{observation_id}.tif",
            path=str(path),
            sha256=_sha256(path),
        ),
        raster=RasterMetadata(
            driver="GTiff",
            width=2,
            height=2,
            band_count=1,
            dtypes=("float32",),
            nodata=(None,),
        ),
        sensor=SensorMetadata(
            modality=modality,
            sensor_name="Sentinel-1" if modality is Modality.SAR else "Sentinel-2",
            product_level="GRD" if modality is Modality.SAR else "L2A",
            bands=bands,
            polarizations=("VV",) if modality is Modality.SAR else (),
        ),
        geo=GeoMetadata(
            crs="EPSG:32643",
            transform=AffineTransform(a=10, b=0, c=500000, d=0, e=-10, f=2000000),
            bounds=GeoBounds(left=500000, bottom=1999980, right=500020, top=2000000),
            native_gsd_x=10,
            native_gsd_y=10,
        ),
        temporal=TemporalMetadata(acquisition_time=acquisition_time),
        validity=ValidityMetadata(has_crs=True, has_transform=True, has_nodata=False),
        provenance=ObservationProvenance(created_at=_NOW, ingestion_version="test"),
    )


def _repo(tmp_path: Path, *observations: ObservationState) -> MetadataRepository:
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    repo = MetadataRepository(db)
    for observation in observations:
        repo.create_observation(
            ObservationRecord(
                observation_id=observation.observation_id,
                created_at=_NOW,
                payload=observation.model_dump(mode="json"),
            )
        )
    return repo


def _context(repo: MetadataRepository, *, verified: bool = True) -> ExecutionContext:
    metadata = {}
    if verified:
        metadata["verified_common_grid_pairs"] = ((_T1_ID, _T2_ID),)
    return ExecutionContext(
        job_id="job_" + "a" * 32,
        analysis_id="ana_" + "a" * 32,
        repository=repo,
        metadata=metadata,
    )


def _sar_parameters() -> dict[str, object]:
    return {
        "radiometric_domain": "backscatter_db",
        "polarizations": ["VV"],
        "threshold": 3.0,
    }


def test_registered_adapter_builder_covers_every_tool_and_rejects_missing_mapping(tmp_path: Path) -> None:
    registry = load_tool_registry()
    adapters = build_registered_adapters(registry, artifact_store=ArtifactStore(tmp_path))
    assert set(adapters) == set(registry.tools)

    with pytest.raises(ValueError, match="no adapter constructor"):
        build_registered_adapters(
            registry,
            artifact_store=ArtifactStore(tmp_path / "other"),
            constructors={ToolExecutor.MASK_AREA: lambda *_args, **_kwargs: object()},
        )


def test_sar_temporal_change_adapter_emits_change_mask_evidence_and_published_artifact(tmp_path: Path) -> None:
    _write_raster(tmp_path / "t1.tif", np.array([[1, 1], [1, 1]], dtype="float32"))
    _write_raster(tmp_path / "t2.tif", np.array([[4, 1], [0, 5]], dtype="float32"))
    repo = _repo(
        tmp_path,
        _observation(_T1_ID, tmp_path / "t1.tif"),
        _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
    )
    store = ArtifactStore(tmp_path / "data")
    adapters = build_registered_adapters(load_tool_registry(), artifact_store=store)
    engine = ExecutionEngine(adapters, artifact_store=store, timeout_seconds=2)

    (result,) = engine.execute(
        ExecutionPlan(
            steps=(
                PlanStep(
                    "step_sar",
                    "sar_temporal_change_v1",
                    input_bindings={"t1": _T1_ID, "t2": _T2_ID},
                    parameters=_sar_parameters(),
                ),
            )
        ),
        _context(repo),
    )

    evidence = result.evidence
    assert evidence.task == "change_localize"
    assert evidence.tool_id == "sar_temporal_change_v1"
    assert evidence.temporal.t1_observation_id == _T1_ID
    assert evidence.temporal.t2_observation_id == _T2_ID
    assert evidence.temporal.order_source == "metadata"
    assert evidence.mask.asset_id.startswith("artifact_")
    assert evidence.mask.sha256 == _sha256(Path(evidence.mask.path))
    assert evidence.mask.source_grid_observation_id == _T1_ID
    assert result.artifacts[0].metadata.extra["source_hashes"] == {
        _T1_ID: repo.get_observation(_T1_ID).payload["source_asset"]["sha256"],
        _T2_ID: repo.get_observation(_T2_ID).payload["source_asset"]["sha256"],
    }
    with rasterio.open(evidence.mask.path) as dataset:
        assert dataset.read(1).tolist() == [[1, 0], [0, 1]]


def test_sar_adapter_fails_closed_for_wrong_modality_missing_semantics_order_and_unverified_grid(tmp_path: Path) -> None:
    _write_raster(tmp_path / "t1.tif", np.ones((2, 2), dtype="float32"))
    _write_raster(tmp_path / "t2.tif", np.ones((2, 2), dtype="float32") * 2)
    registry = load_tool_registry()
    adapter = build_registered_adapters(registry, artifact_store=ArtifactStore(tmp_path / "data"))["sar_temporal_change_v1"]
    call = ToolCall(
        step_id="step_sar",
        tool_id="sar_temporal_change_v1",
        input_bindings={"t1": _T1_ID, "t2": _T2_ID},
        parameters=_sar_parameters(),
    )

    with pytest.raises(AdapterInputError, match="SAR modality"):
        adapter.execute(
            call,
            _context(
                _repo(
                    tmp_path / "wrong_modality",
                    _observation(_T1_ID, tmp_path / "t1.tif", modality=Modality.MULTISPECTRAL),
                    _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
                )
            ),
        )

    with pytest.raises(AdapterInputError, match="radiometric_domain"):
        adapter.execute(
            call,
            _context(
                _repo(
                    tmp_path / "missing_semantics",
                    _observation(_T1_ID, tmp_path / "t1.tif", radiometric_domain=None),
                    _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
                )
            ),
        )

    with pytest.raises(AdapterInputError, match="temporal order"):
        adapter.execute(
            call,
            _context(
                _repo(
                    tmp_path / "wrong_order",
                    _observation(_T1_ID, tmp_path / "t1.tif", acquisition_time=_NOW.replace(day=3)),
                    _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
                )
            ),
        )

    with pytest.raises(AdapterInputError, match="verified common grid"):
        adapter.execute(
            call,
            _context(
                _repo(
                    tmp_path / "unverified",
                    _observation(_T1_ID, tmp_path / "t1.tif"),
                    _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
                ),
                verified=False,
            ),
        )


def test_mask_area_adapter_measures_only_from_prior_mask_evidence(tmp_path: Path) -> None:
    _write_raster(tmp_path / "t1.tif", np.array([[1, 1], [1, 1]], dtype="float32"))
    _write_raster(tmp_path / "t2.tif", np.array([[4, 1], [0, 5]], dtype="float32"))
    repo = _repo(
        tmp_path,
        _observation(_T1_ID, tmp_path / "t1.tif"),
        _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
    )
    store = ArtifactStore(tmp_path / "data")
    engine = ExecutionEngine(
        build_registered_adapters(load_tool_registry(), artifact_store=store),
        artifact_store=store,
        timeout_seconds=2,
    )

    _, measurement_result = engine.execute(
        ExecutionPlan(
            steps=(
                PlanStep(
                    "step_sar",
                    "sar_temporal_change_v1",
                    input_bindings={"t1": _T1_ID, "t2": _T2_ID},
                    parameters=_sar_parameters(),
                ),
                PlanStep(
                    "step_area",
                    "compute_mask_area_v1",
                    input_bindings={"mask": "step_sar"},
                    parameters={"unit": "m2"},
                    depends_on=("step_sar",),
                ),
            )
        ),
        _context(repo),
    )

    evidence = measurement_result.evidence
    assert evidence.task == "measurement"
    assert evidence.tool_id == "compute_mask_area_v1"
    assert evidence.source_evidence_id.startswith("evidence_")
    assert evidence.value == 200.0
    assert evidence.positive_pixel_count == 2
    assert evidence.valid_pixel_count == 4
    assert evidence.provenance.parent_evidence_ids == (evidence.source_evidence_id,)


def test_mask_agreement_adapter_emits_diagnostic_agreement(tmp_path: Path) -> None:
    _write_raster(tmp_path / "t1.tif", np.array([[1, 1], [1, 1]], dtype="float32"))
    _write_raster(tmp_path / "t2.tif", np.array([[4, 1], [0, 5]], dtype="float32"))
    repo = _repo(
        tmp_path,
        _observation(_T1_ID, tmp_path / "t1.tif"),
        _observation(_T2_ID, tmp_path / "t2.tif", acquisition_time=_NOW.replace(day=2)),
    )
    store = ArtifactStore(tmp_path / "data")
    adapters = build_registered_adapters(load_tool_registry(), artifact_store=store)
    engine = ExecutionEngine(adapters, artifact_store=store, timeout_seconds=2)
    lower_threshold = _sar_parameters()
    lower_threshold["threshold"] = 1.0

    first, second, agreement = engine.execute(
        ExecutionPlan(
            steps=(
                PlanStep(
                    "step_first",
                    "sar_temporal_change_v1",
                    input_bindings={"t1": _T1_ID, "t2": _T2_ID},
                    parameters=_sar_parameters(),
                ),
                PlanStep(
                    "step_second",
                    "sar_temporal_change_v1",
                    input_bindings={"t1": _T1_ID, "t2": _T2_ID},
                    parameters=lower_threshold,
                ),
                PlanStep(
                    "step_agreement",
                    "mask_agreement_v1",
                    input_bindings={"first": "step_first", "second": "step_second"},
                    depends_on=("step_first", "step_second"),
                ),
            )
        ),
        _context(repo),
    )

    assert first.evidence.mask.sha256 == _sha256(Path(first.evidence.mask.path))
    assert second.evidence.mask.sha256 == _sha256(Path(second.evidence.mask.path))
    assert agreement.evidence.task == "mask_agreement"
    assert agreement.evidence.tool_id == "mask_agreement_v1"
    assert agreement.evidence.first_evidence_id == first.evidence.evidence_id
    assert agreement.evidence.second_evidence_id == second.evidence.evidence_id
    assert agreement.evidence.interpretation == "agreement_not_accuracy"
    assert agreement.evidence.intersection_pixel_count == 2
    assert agreement.evidence.union_pixel_count == 3
    assert agreement.evidence.value == pytest.approx(2 / 3)
