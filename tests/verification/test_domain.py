from __future__ import annotations

from types import SimpleNamespace

import pytest

from satquery.evidence.models import DomainStatus
from satquery.inference.exceptions import ModelInputUnsupportedError
from satquery.ingestion.models import Modality
from satquery.verification.domain import require_domain


def _observation(*, modality=Modality.OPTICAL, sensor_name="Sentinel-2", polarizations=(), tags=None):
    return SimpleNamespace(
        sensor=SimpleNamespace(modality=modality, sensor_name=sensor_name, polarizations=polarizations),
        raster=SimpleNamespace(tags=tags or {}),
    )


def test_known_sensor_and_modality_are_in_domain() -> None:
    result = require_domain(_observation(), supported_modalities=(Modality.OPTICAL,))
    assert result.status is DomainStatus.IN_DOMAIN


@pytest.mark.parametrize(
    "observation",
    [
        _observation(modality=Modality.UNKNOWN),
        _observation(sensor_name=None),
        _observation(modality=Modality.SAR),
    ],
)
def test_unknown_or_unsupported_inputs_fail_closed(observation) -> None:
    with pytest.raises(ModelInputUnsupportedError):
        require_domain(observation, supported_modalities=(Modality.OPTICAL,))


def test_sar_contract_requires_explicit_semantics() -> None:
    observation = _observation(
        modality=Modality.SAR,
        sensor_name="Sentinel-1",
        polarizations=("VV", "VH"),
        tags={"RADIOMETRIC_DOMAIN": "backscatter_db"},
    )
    result = require_domain(
        observation,
        supported_modalities=(Modality.SAR,),
        required_polarizations=("VV", "VH"),
        required_radiometric_domain="backscatter_db",
    )
    assert result.status is DomainStatus.IN_DOMAIN
