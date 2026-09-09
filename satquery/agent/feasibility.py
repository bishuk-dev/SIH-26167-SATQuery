"""Centralized, fail-closed feasibility checks for bounded agent workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from satquery.agent.models import (
    FailureOutcome,
    FeasibilityCheck,
    FeasibilityIssue,
    FeasibilityResult,
    QueryIntent,
)
from satquery.geo.models import PairCompatibility
from satquery.ingestion.models import Modality, ObservationState
from satquery.registry.tools import CapabilityState, RuntimeCapability


_TEMPORAL_TASKS = {
    "CHANGE_VQA",
    "CHANGE_LOCALIZE",
    "CHANGE_MEASURE",
    "CHANGE_DESCRIPTION",
    "CROSS_MODAL_VQA",
}
_PIXELWISE_TASKS = {"MEASURE", "CHANGE_MEASURE", "CHANGE_LOCALIZE"}
_AVAILABLE_STATES = {
    CapabilityState.AVAILABLE.value,
    CapabilityState.AVAILABLE_WITH_LIMITS.value,
}
_OUTCOME_RANK: dict[FailureOutcome, int] = {
    "ALLOW": 0,
    "ALLOW_WITH_WARNING": 1,
    "ABSTAIN": 2,
    "REQUEST_INPUT": 3,
    "REJECT": 4,
}


@dataclass(frozen=True, slots=True)
class _Capability:
    capability_id: str
    status: str


def _capability(value: RuntimeCapability | Mapping[str, Any]) -> _Capability:
    if isinstance(value, RuntimeCapability):
        return _Capability(value.capability_id, value.status.value)
    if not isinstance(value, Mapping):
        raise TypeError("capabilities must contain RuntimeCapability records or mappings")
    capability_id = value.get("capability_id") or value.get("id")
    status = value.get("status")
    if hasattr(status, "value"):
        status = status.value
    if not isinstance(capability_id, str) or not isinstance(status, str):
        raise TypeError("capability records require capability_id and status")
    return _Capability(capability_id, status)


def _semantic_band_names(observation: ObservationState) -> set[str]:
    names: set[str] = set()
    for band in observation.sensor.bands:
        values = [band.description or "", *band.tags.keys(), *band.tags.values()]
        for value in values:
            normalized = value.casefold().replace("_", " ").replace("-", " ")
            names.add(" ".join(normalized.split()))
    return names


def _has_band(observation: ObservationState, role: str) -> bool:
    aliases = {
        "NIR": {"nir", "near infrared", "near-infrared", "b8", "band 8"},
        "RED": {"red", "b4", "band 4"},
        "GREEN": {"green", "b3", "band 3"},
        "SWIR1": {"swir1", "swir 1", "short wave infrared 1", "b11", "band 11"},
    }[role]
    names = _semantic_band_names(observation)
    return any(alias in names for alias in aliases)


def _required_capability_ids(intent: QueryIntent) -> tuple[str, ...]:
    family = intent.task_family
    if family == "SINGLE_VQA":
        return ("single_image_vqa",)
    if family == "GROUND_OBJECT":
        return ("text_guided_grounding",)
    if family == "CHANGE_DESCRIPTION":
        return ("change_captioning",)
    if family == "CROSS_MODAL_VQA":
        return ("multisensor_land_cover_joint",)
    if intent.requested_measurement in {"ndvi", "ndwi", "mndwi"}:
        return (f"spectral_index_{intent.requested_measurement}",)
    if intent.requested_measurement == "area":
        return ("mask_area_measurement",)
    if family in {"CHANGE_VQA", "CHANGE_LOCALIZE", "CHANGE_MEASURE"}:
        return ("temporal_difference",)
    return ()


def _is_sar_semantics_unknown(observation: ObservationState) -> bool:
    if observation.sensor.modality is not Modality.SAR:
        return False
    if not observation.sensor.polarizations:
        return True
    tags = {
        key.casefold()
        for band in observation.sensor.bands
        for key in band.tags
    }
    return not tags.intersection(
        {"radiometric_domain", "radiometric_units", "calibration", "backscatter_domain"}
    ) and observation.sensor.sensor_name is None


def _is_jpeg(observation: ObservationState) -> bool:
    driver = observation.raster.driver.casefold()
    name = observation.source_asset.original_name.casefold()
    return driver in {"jpeg", "jpg"} or name.endswith((".jpg", ".jpeg"))


def _check(
    check_id: str,
    *,
    passed: bool,
    outcome: FailureOutcome = "ALLOW",
    code: str | None = None,
    message: str | None = None,
    severity: str = "INFO",
) -> FeasibilityCheck:
    return FeasibilityCheck(
        check_id=check_id,
        passed=passed,
        severity=severity,  # type: ignore[arg-type]
        outcome=outcome,
        code=code,
        message=message,
    )


class FeasibilityValidator:
    """Run every pure scientific gate and then apply frozen precedence."""

    def validate(
        self,
        intent: QueryIntent,
        observations: Sequence[ObservationState],
        pair: PairCompatibility | None,
        roi: object | None,
        capabilities: Sequence[RuntimeCapability | Mapping[str, Any]],
    ) -> FeasibilityResult:
        del roi  # ROI shape/limits are transport concerns until planning.
        observations = tuple(observations)
        capability_records = tuple(_capability(value) for value in capabilities)
        required_ids = _required_capability_ids(intent)
        checks = (
            self._check_temporal_pair(intent, observations, pair),
            self._check_temporal_order(intent, pair),
            self._check_overlap(intent, pair),
            self._check_grid(intent, pair),
            self._check_required_bands(intent, observations),
            self._check_measurement_crs(intent, observations),
            self._check_sar_semantics(intent, observations),
            self._check_image_quality(intent, observations),
            self._check_capabilities(required_ids, capability_records),
        )

        failures: list[FeasibilityIssue] = []
        warnings: list[FeasibilityIssue] = []
        for check in checks:
            if check.passed or check.code is None or check.message is None:
                continue
            issue = FeasibilityIssue(
                code=check.code,
                severity=check.severity,
                outcome=check.outcome,
                message=check.message,
            )
            if check.outcome == "ALLOW_WITH_WARNING":
                warnings.append(issue)
            else:
                failures.append(issue)

        outcome = max(
            (check.outcome for check in checks if not check.passed),
            key=lambda value: _OUTCOME_RANK[value],
            default="ALLOW",
        )
        allowed = tuple(
            record.capability_id
            for record in capability_records
            if record.status in _AVAILABLE_STATES
            and (not required_ids or record.capability_id in required_ids)
        )
        return FeasibilityResult(
            checks=checks,
            outcome=outcome,
            failures=tuple(failures),
            warnings=tuple(warnings),
            allowed_capability_ids=allowed,
        )

    @staticmethod
    def _check_temporal_pair(
        intent: QueryIntent,
        observations: Sequence[ObservationState],
        pair: PairCompatibility | None,
    ) -> FeasibilityCheck:
        if intent.task_family not in _TEMPORAL_TASKS:
            return _check("temporal_pair", passed=True)
        if len(observations) >= 2 and pair is not None:
            return _check("temporal_pair", passed=True)
        return _check(
            "temporal_pair",
            passed=False,
            outcome="REQUEST_INPUT",
            code="MISSING_TEMPORAL_PAIR",
            message="Change analysis requires a second observation of the same area from another time.",
            severity="ERROR",
        )

    @staticmethod
    def _check_temporal_order(
        intent: QueryIntent, pair: PairCompatibility | None
    ) -> FeasibilityCheck:
        if intent.task_family not in _TEMPORAL_TASKS or pair is None:
            return _check("temporal_order", passed=True)
        if pair.temporal.order_known or intent.temporal_direction == "UNKNOWN":
            return _check("temporal_order", passed=True)
        return _check(
            "temporal_order",
            passed=False,
            outcome="REQUEST_INPUT",
            code="TEMPORAL_ORDER_UNKNOWN",
            message="Temporal order is required for this request.",
            severity="ERROR",
        )

    @staticmethod
    def _check_overlap(intent: QueryIntent, pair: PairCompatibility | None) -> FeasibilityCheck:
        if intent.task_family not in _TEMPORAL_TASKS or pair is None:
            return _check("spatial_overlap", passed=True)
        if pair.overlap.known and pair.overlap.sufficient is False:
            return _check(
                "spatial_overlap",
                passed=False,
                outcome="REJECT",
                code="NO_SPATIAL_OVERLAP",
                message="The observations do not cover a sufficient common area.",
                severity="ERROR",
            )
        return _check("spatial_overlap", passed=True)

    @staticmethod
    def _check_grid(intent: QueryIntent, pair: PairCompatibility | None) -> FeasibilityCheck:
        if intent.task_family not in _PIXELWISE_TASKS or pair is None:
            return _check("pixel_grid", passed=True)
        if pair.grid.aligned is True:
            return _check("pixel_grid", passed=True)
        return _check(
            "pixel_grid",
            passed=False,
            outcome="REJECT",
            code="PAIR_ALIGNMENT_INVALID",
            message="Pixelwise analysis requires a verified common grid or an explicit alignment workflow.",
            severity="ERROR",
        )

    @staticmethod
    def _check_required_bands(
        intent: QueryIntent, observations: Sequence[ObservationState]
    ) -> FeasibilityCheck:
        requirements = {
            "ndvi": ("RED", "NIR"),
            "ndwi": ("GREEN", "NIR"),
            "mndwi": ("GREEN", "SWIR1"),
        }
        required = requirements.get(intent.requested_measurement or "")
        if required is None or not observations:
            return _check("required_bands", passed=True)
        missing = sorted(
            role
            for role in required
            if any(not _has_band(observation, role) for observation in observations)
        )
        if not missing:
            return _check("required_bands", passed=True)
        return _check(
            "required_bands",
            passed=False,
            outcome="REJECT",
            code="MISSING_REQUIRED_BAND",
            message=f"This operation requires semantic band roles: {', '.join(missing)}.",
            severity="ERROR",
        )

    @staticmethod
    def _check_measurement_crs(
        intent: QueryIntent, observations: Sequence[ObservationState]
    ) -> FeasibilityCheck:
        if intent.requested_measurement != "area" or not observations:
            return _check("measurement_crs", passed=True)
        if all(observation.geo.crs is not None for observation in observations):
            return _check("measurement_crs", passed=True)
        return _check(
            "measurement_crs",
            passed=False,
            outcome="REJECT",
            code="CRS_REQUIRED_FOR_MEASUREMENT",
            message="A CRS is required to calculate geographic area.",
            severity="ERROR",
        )

    @staticmethod
    def _check_sar_semantics(
        intent: QueryIntent, observations: Sequence[ObservationState]
    ) -> FeasibilityCheck:
        sar_requested = intent.target_semantic == "sar" or intent.task_family == "CROSS_MODAL_VQA"
        if not sar_requested or not any(_is_sar_semantics_unknown(item) for item in observations):
            return _check("sar_semantics", passed=True)
        return _check(
            "sar_semantics",
            passed=False,
            outcome="REQUEST_INPUT",
            code="UNKNOWN_SAR_POLARIZATION",
            message="SAR polarization and radiometric semantics must be supplied before specialist execution.",
            severity="ERROR",
        )

    @staticmethod
    def _check_image_quality(
        intent: QueryIntent, observations: Sequence[ObservationState]
    ) -> FeasibilityCheck:
        if intent.task_family != "SINGLE_VQA" or not any(_is_jpeg(item) for item in observations):
            return _check("image_quality", passed=True)
        return _check(
            "image_quality",
            passed=False,
            outcome="ALLOW_WITH_WARNING",
            code="GEOREFERENCE_UNAVAILABLE",
            message="JPEG VQA is limited to pixel-space interpretation when georeferencing is unavailable.",
            severity="WARNING",
        )

    @staticmethod
    def _check_capabilities(
        required_ids: Sequence[str], capabilities: Sequence[_Capability]
    ) -> FeasibilityCheck:
        if not required_ids or not capabilities:
            return _check("capabilities", passed=True)
        matching = [item for item in capabilities if item.capability_id in required_ids]
        if matching and all(item.status in _AVAILABLE_STATES for item in matching):
            return _check("capabilities", passed=True)
        return _check(
            "capabilities",
            passed=False,
            outcome="ABSTAIN",
            code="MODEL_UNAVAILABLE",
            message="No available registered capability can execute this request.",
            severity="ERROR",
        )


__all__ = ["FeasibilityValidator"]
