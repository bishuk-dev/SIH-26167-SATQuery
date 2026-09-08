"""Universal fail-closed model input domain gate."""

from __future__ import annotations

from collections.abc import Iterable

from satquery.evidence.models import DomainAssessment, DomainStatus
from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.ingestion.models import Modality, ObservationState


def require_domain(
    observation: ObservationState,
    *,
    supported_modalities: Iterable[Modality],
    required_sensor_names: Iterable[str] = (),
    required_polarizations: tuple[str, ...] | None = None,
    required_radiometric_domain: str | None = None,
) -> DomainAssessment:
    """Validate metadata required by a specialist before model execution."""

    reasons: list[str] = []
    supported = tuple(supported_modalities)
    if observation.sensor.modality is Modality.UNKNOWN:
        reasons.append("UNKNOWN_MODALITY")
    elif observation.sensor.modality not in supported:
        reasons.append("UNSUPPORTED_MODALITY")
    if observation.sensor.sensor_name is None:
        reasons.append("UNKNOWN_SENSOR")
    if required_sensor_names and observation.sensor.sensor_name not in tuple(required_sensor_names):
        reasons.append("UNSUPPORTED_SENSOR")
    if required_polarizations is not None and observation.sensor.polarizations != required_polarizations:
        reasons.append("UNKNOWN_OR_UNSUPPORTED_POLARIZATION")
    if required_radiometric_domain is not None and observation.raster.tags.get("RADIOMETRIC_DOMAIN") != required_radiometric_domain:
        reasons.append("UNKNOWN_OR_UNSUPPORTED_RADIOMETRIC_DOMAIN")
    if reasons:
        raise ModelInputUnsupportedError("domain gate rejected input: " + ", ".join(reasons))
    return DomainAssessment(status=DomainStatus.IN_DOMAIN, reasons=())
