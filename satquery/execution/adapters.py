"""Typed adapters between registered execution tools and scientific code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import rasterio
from affine import Affine

from satquery.analytics.measurement import measure_mask_area
from satquery.analytics.reconciliation import diagnostic_mask_agreement
from satquery.analytics.sar import SarInputContract, sar_temporal_change
from satquery.artifacts import ArtifactStore
from satquery.evidence.models import (
    ChangeMaskEvidence,
    DomainAssessment,
    DomainStatus,
    EvidenceProvenance,
    FloodMaskEvidence,
    MaskAsset,
    MeasurementEvidence,
    TemporalPairEvidence,
)
from satquery.execution.models import (
    ArtifactMetadata,
    ArtifactOutput,
    ExecutionContext,
    ToolAdapter,
    ToolCall,
    ToolResult,
)
from satquery.ingestion.models import AffineTransform, Modality, ObservationState
from satquery.registry.models import ToolExecutor
from satquery.registry.tools import ToolRegistration, ToolRegistry


class AdapterInputError(ValueError):
    """A registered adapter refused unsupported or unsafe scientific input."""


MaskEvidence = ChangeMaskEvidence | FloodMaskEvidence
AdapterConstructor = Callable[[ToolRegistration, ArtifactStore | None], ToolAdapter]


def _file_sha256(path: Path) -> str:
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def _affine(transform: AffineTransform) -> Affine:
    return Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def _observation(context: ExecutionContext, observation_id: str) -> ObservationState:
    if context.repository is None:
        raise AdapterInputError("observation lookup requires a metadata repository")
    record = context.repository.get_observation(observation_id)
    if record is None:
        raise AdapterInputError(f"observation is not registered: {observation_id}")
    try:
        return ObservationState.model_validate_json(json.dumps(record.payload))
    except Exception as exc:
        raise AdapterInputError(f"observation metadata is invalid: {observation_id}") from exc


def _verify_source_asset(observation: ObservationState) -> Path:
    path = Path(observation.source_asset.path)
    if not path.is_file() or path.is_symlink():
        raise AdapterInputError("registered source asset is not a regular file")
    digest = _file_sha256(path)
    if digest != observation.source_asset.sha256:
        raise AdapterInputError("registered source asset hash mismatch")
    return path


def _require_parameters(call: ToolCall, required: Sequence[str]) -> dict[str, Any]:
    missing = [name for name in required if name not in call.parameters]
    if missing:
        raise AdapterInputError(f"missing required tool parameters: {', '.join(missing)}")
    return dict(call.parameters)


def _as_polarizations(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise AdapterInputError("polarizations must be an explicit bounded list")
    polarizations = tuple(value)
    if not polarizations or any(not isinstance(item, str) for item in polarizations):
        raise AdapterInputError("polarizations must contain semantic names")
    return polarizations


def _shared_band_tag(
    observations: Sequence[ObservationState], polarizations: Sequence[str], tag: str
) -> str:
    values: set[str] = set()
    for observation in observations:
        polarization_to_band = dict(zip(observation.sensor.polarizations, observation.sensor.bands, strict=False))
        for polarization in polarizations:
            band = polarization_to_band.get(polarization)
            if band is None:
                raise AdapterInputError(f"observation lacks requested SAR polarization: {polarization}")
            value = band.tags.get(tag)
            if not value:
                raise AdapterInputError(f"SAR input is missing required {tag} semantics")
            values.add(value)
    if len(values) != 1:
        raise AdapterInputError(f"SAR {tag} semantics disagree across the temporal pair")
    return values.pop()


def _verified_common_grid(context: ExecutionContext, t1_id: str, t2_id: str) -> bool:
    pairs = context.metadata.get("verified_common_grid_pairs", ())
    for pair in pairs:
        if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)):
            if tuple(pair) == (t1_id, t2_id):
                return True
    return False


def _require_ordered_sar_pair(
    context: ExecutionContext, t1_id: str, t2_id: str
) -> tuple[ObservationState, ObservationState]:
    if t1_id == t2_id:
        raise AdapterInputError("temporal tools require distinct observations")
    t1 = _observation(context, t1_id)
    t2 = _observation(context, t2_id)
    if t1.sensor.modality is not Modality.SAR or t2.sensor.modality is not Modality.SAR:
        raise AdapterInputError("SAR temporal change requires SAR modality for both observations")
    if t1.temporal.acquisition_time is None or t2.temporal.acquisition_time is None:
        raise AdapterInputError("temporal order requires acquisition timestamps")
    if t1.temporal.acquisition_time >= t2.temporal.acquisition_time:
        raise AdapterInputError("temporal order must be T1 before T2")
    if not _verified_common_grid(context, t1_id, t2_id):
        raise AdapterInputError("SAR temporal change requires a verified common grid")
    if (t1.raster.width, t1.raster.height) != (t2.raster.width, t2.raster.height):
        raise AdapterInputError("verified common grid metadata has inconsistent dimensions")
    if t1.geo.crs is None or t1.geo.transform is None or t1.geo.bounds is None:
        raise AdapterInputError("SAR change-mask artifacts require georeferenced T1 metadata")
    if t2.geo.crs != t1.geo.crs or t2.geo.transform != t1.geo.transform:
        raise AdapterInputError("verified common grid metadata is inconsistent")
    if not t1.sensor.sensor_name or not t2.sensor.sensor_name:
        raise AdapterInputError("SAR input must preserve sensor identity")
    if t1.sensor.sensor_name != t2.sensor.sensor_name:
        raise AdapterInputError("SAR temporal pair sensors disagree")
    return t1, t2


def _read_sar_arrays(
    observation: ObservationState, polarizations: Sequence[str]
) -> dict[str, np.ndarray]:
    path = _verify_source_asset(observation)
    band_by_polarization = {
        polarization: index + 1 for index, polarization in enumerate(observation.sensor.polarizations)
    }
    arrays: dict[str, np.ndarray] = {}
    with rasterio.open(path) as dataset:
        if dataset.width != observation.raster.width or dataset.height != observation.raster.height:
            raise AdapterInputError("source raster dimensions disagree with registered metadata")
        for polarization in polarizations:
            band_index = band_by_polarization.get(polarization)
            if band_index is None or band_index > dataset.count:
                raise AdapterInputError(f"source raster lacks polarization band: {polarization}")
            arrays[polarization] = dataset.read(band_index)
    return arrays


def _final_artifact_path(store: ArtifactStore, artifact_id: str, suffix: str) -> Path:
    return store.artifacts_root / artifact_id / f"artifact.{suffix}"


def _mask_asset(
    *,
    store: ArtifactStore,
    artifact_id: str,
    suffix: str,
    digest: str,
    source_grid: ObservationState,
) -> MaskAsset:
    assert source_grid.geo.transform is not None
    assert source_grid.geo.bounds is not None
    return MaskAsset(
        asset_id=artifact_id,
        path=str(_final_artifact_path(store, artifact_id, suffix)),
        sha256=digest,
        width=source_grid.raster.width,
        height=source_grid.raster.height,
        crs=source_grid.geo.crs,
        transform=source_grid.geo.transform,
        bounds=source_grid.geo.bounds,
        source_grid_observation_id=source_grid.observation_id,
    )


def _resolve_prior_evidence(call: ToolCall, input_name: str, context: ExecutionContext) -> Any:
    binding = call.input_bindings.get(input_name)
    if not binding:
        raise AdapterInputError(f"missing required input binding: {input_name}")
    if binding in call.prior_results:
        return call.prior_results[binding].evidence
    evidence_map = context.metadata.get("evidence", {})
    if isinstance(evidence_map, Mapping) and binding in evidence_map:
        return evidence_map[binding]
    raise AdapterInputError(f"input binding does not resolve to prior evidence: {input_name}")


def _resolve_mask_evidence(call: ToolCall, input_name: str, context: ExecutionContext) -> MaskEvidence:
    evidence = _resolve_prior_evidence(call, input_name, context)
    if not isinstance(evidence, (ChangeMaskEvidence, FloodMaskEvidence)):
        raise AdapterInputError("mask measurement/agreement requires mask evidence")
    return evidence


def _load_mask(mask: MaskAsset) -> np.ndarray:
    path = Path(mask.path)
    if not path.is_file() or path.is_symlink():
        raise AdapterInputError("mask asset is not available as an immutable file")
    if _file_sha256(path) != mask.sha256:
        raise AdapterInputError("mask asset hash mismatch")
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise AdapterInputError("mask asset must be single-band")
        values = dataset.read(1)
    if not np.isin(values, (0, 1)).all():
        raise AdapterInputError("mask asset must be strictly binary 0/1")
    return values.astype(bool)


class SarTemporalChangeAdapter:
    def __init__(self, registration: ToolRegistration, artifact_store: ArtifactStore | None) -> None:
        self.registration = registration
        self.artifact_store = artifact_store

    def execute(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        if self.artifact_store is None:
            raise AdapterInputError("SAR temporal change requires an artifact store")
        t1_id = call.input_bindings.get("t1")
        t2_id = call.input_bindings.get("t2")
        if not t1_id or not t2_id:
            raise AdapterInputError("SAR temporal change requires T1 and T2 inputs")
        parameters = _require_parameters(call, ("radiometric_domain", "polarizations", "threshold"))
        radiometric_domain = parameters["radiometric_domain"]
        if not isinstance(radiometric_domain, str):
            raise AdapterInputError("radiometric_domain must be explicit text")
        polarizations = _as_polarizations(parameters["polarizations"])
        threshold = parameters["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise AdapterInputError("threshold must be numeric")

        t1, t2 = _require_ordered_sar_pair(context, t1_id, t2_id)
        registered_domain = _shared_band_tag((t1, t2), polarizations, "radiometric_domain")
        if registered_domain != radiometric_domain:
            raise AdapterInputError("registered SAR radiometric_domain does not match the requested contract")
        calibration = _shared_band_tag((t1, t2), polarizations, "calibration")
        contract = SarInputContract(
            polarizations=polarizations,
            radiometric_domain=radiometric_domain,
            sensor=t1.sensor.sensor_name or "",
            calibration=calibration,
        )
        result = sar_temporal_change(
            _read_sar_arrays(t1, polarizations),
            _read_sar_arrays(t2, polarizations),
            contract,
            threshold=float(threshold),
        )

        staged = self.artifact_store.stage("tif")
        with rasterio.open(
            staged.path,
            "w",
            driver="GTiff",
            width=result.mask.shape[1],
            height=result.mask.shape[0],
            count=1,
            dtype="uint8",
            crs=t1.geo.crs,
            transform=_affine(t1.geo.transform),  # type: ignore[arg-type]
        ) as dataset:
            dataset.write(result.mask.astype("uint8"), 1)
        digest = _file_sha256(staged.path)
        evidence_id = f"evidence_{uuid4().hex}"
        evidence = ChangeMaskEvidence(
            evidence_id=evidence_id,
            target_class="sar_backscatter_change",
            change_kind="symmetric_change",
            temporal=TemporalPairEvidence(
                t1_observation_id=t1.observation_id,
                t2_observation_id=t2.observation_id,
                order_source="metadata",
            ),
            mask=_mask_asset(
                store=self.artifact_store,
                artifact_id=staged.artifact_id,
                suffix=staged.suffix,
                digest=digest,
                source_grid=t1,
            ),
            tool_id=call.tool_id,
            domain=DomainAssessment(
                status=DomainStatus.IN_DOMAIN,
                reasons=("EXPLICIT_SAR_BACKSCATTER_CONTRACT", "VERIFIED_COMMON_GRID"),
            ),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id=call.step_id,
                input_asset_id=t2.source_asset.asset_id,
            ),
        )
        source_hashes = {
            t1.observation_id: t1.source_asset.sha256,
            t2.observation_id: t2.source_asset.sha256,
        }
        return ToolResult(
            output={
                "mask_artifact_id": staged.artifact_id,
                "positive_pixel_count": int(result.mask.sum()),
                "valid_pixel_count": int(result.valid.sum()),
                "method": result.method,
                "source_hashes": source_hashes,
            },
            artifacts=(
                ArtifactOutput(
                    staged=staged,
                    metadata=ArtifactMetadata(
                        analysis_id=context.analysis_id,
                        evidence_id=evidence_id,
                        media_type="image/tiff",
                        description="Deterministic SAR temporal change binary mask",
                        extra={
                            "tool_id": call.tool_id,
                            "source_observation_ids": [t1.observation_id, t2.observation_id],
                            "source_hashes": source_hashes,
                            "radiometric_domain": radiometric_domain,
                            "polarizations": list(polarizations),
                        },
                    ),
                ),
            ),
            evidence=evidence,
        )


class MaskAreaAdapter:
    def __init__(self, registration: ToolRegistration, artifact_store: ArtifactStore | None) -> None:
        self.registration = registration
        del artifact_store

    def execute(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        parameters = _require_parameters(call, ("unit",))
        unit = parameters["unit"]
        if unit not in {"m2", "ha", "km2"}:
            raise AdapterInputError("area unit is not registered")
        source = _resolve_mask_evidence(call, "mask", context)
        if source.mask.crs is None or source.mask.transform is None:
            raise AdapterInputError("mask area measurement requires mask CRS and transform")
        values = _load_mask(source.mask)
        measured = measure_mask_area(values, source.mask.transform, source.mask.crs, unit=unit)  # type: ignore[arg-type]
        evidence = MeasurementEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            source_evidence_id=source.evidence_id,
            value=measured.value,
            unit=measured.unit,  # type: ignore[arg-type]
            method=measured.method,
            calculation_crs=measured.calculation_crs,
            positive_pixel_count=measured.positive_pixel_count,
            valid_pixel_count=measured.valid_pixel_count,
            tool_id=call.tool_id,
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id=call.step_id,
                input_asset_id=source.mask.asset_id,
                parent_evidence_ids=(source.evidence_id,),
            ),
        )
        return ToolResult(output=measured, evidence=evidence)


class MaskAgreementAdapter:
    def __init__(self, registration: ToolRegistration, artifact_store: ArtifactStore | None) -> None:
        self.registration = registration
        del artifact_store

    def execute(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        first = _resolve_mask_evidence(call, "first", context)
        second = _resolve_mask_evidence(call, "second", context)
        valid = call.parameters.get("valid")
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
        evidence = diagnostic_mask_agreement(
            first,
            second,
            valid,
            tool_id=call.tool_id,
            operation_id=call.step_id,
        )
        return ToolResult(output={"iou": evidence.value}, evidence=evidence)


_DEFAULT_CONSTRUCTORS: dict[ToolExecutor, AdapterConstructor] = {
    ToolExecutor.SAR_TEMPORAL_CHANGE: SarTemporalChangeAdapter,
    ToolExecutor.MASK_AREA: MaskAreaAdapter,
    ToolExecutor.MASK_AGREEMENT: MaskAgreementAdapter,
}


def build_registered_adapters(
    registry: ToolRegistry,
    *,
    artifact_store: ArtifactStore | None = None,
    constructors: Mapping[ToolExecutor, AdapterConstructor] | None = None,
) -> dict[str, ToolAdapter]:
    """Instantiate one adapter for every registered tool.

    The registry's ``executor`` value is a code-owned key. Startup fails closed
    when a registered executor has no constructor mapping; no import path from
    YAML is executed dynamically.
    """

    constructor_map = dict(_DEFAULT_CONSTRUCTORS if constructors is None else constructors)
    required = {registration.executor for registration in registry.tools.values()}
    missing = sorted(executor.value for executor in required if executor not in constructor_map)
    if missing:
        raise ValueError(f"no adapter constructor for registered executors: {', '.join(missing)}")
    adapters: dict[str, ToolAdapter] = {}
    for tool_id, registration in registry.tools.items():
        if tool_id in adapters:
            raise ValueError(f"duplicate adapter registration for tool: {tool_id}")
        adapters[tool_id] = constructor_map[registration.executor](registration, artifact_store)
    return adapters


__all__ = [
    "AdapterInputError",
    "MaskAgreementAdapter",
    "MaskAreaAdapter",
    "SarTemporalChangeAdapter",
    "build_registered_adapters",
]
