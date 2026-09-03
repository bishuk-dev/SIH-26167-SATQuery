from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from satquery.ingestion.models import (
    BandMetadata,
    GeoMetadata,
    ObservationProvenance,
    ObservationState,
    RasterMetadata,
    SensorMetadata,
    SourceAsset,
    TemporalMetadata,
    ValidityMetadata,
)


def _observation() -> ObservationState:
    return ObservationState(
        observation_id="obs-1",
        source_asset=SourceAsset(
            asset_id="asset-1",
            original_name="example.tif",
            path="/data/example.tif",
            sha256="0" * 64,
        ),
        raster=RasterMetadata(
            driver="GTiff",
            width=4,
            height=3,
            band_count=1,
            dtypes=("uint16",),
            nodata=(0,),
        ),
        sensor=SensorMetadata(
            bands=(BandMetadata(index=1, dtype="uint16", nodata=0),)
        ),
        geo=GeoMetadata(),
        temporal=TemporalMetadata(),
        validity=ValidityMetadata(
            has_crs=False,
            has_transform=False,
            has_nodata=True,
        ),
        provenance=ObservationProvenance(
            created_at=datetime.now(timezone.utc),
            ingestion_version="test/1",
        ),
    )


def test_observation_contract_round_trips_and_is_immutable() -> None:
    observation = _observation()

    restored = ObservationState.model_validate_json(observation.model_dump_json())

    assert restored == observation
    with pytest.raises(ValidationError):
        observation.observation_id = "changed"  # type: ignore[misc]


def test_observation_rejects_band_count_mismatch() -> None:
    payload = _observation().model_dump()
    payload["raster"]["band_count"] = 2
    payload["raster"]["dtypes"] = ("uint16", "uint16")
    payload["raster"]["nodata"] = (0, 0)

    with pytest.raises(ValidationError, match="sensor bands must match"):
        ObservationState.model_validate(payload)


def test_temporal_metadata_rejects_timezone_free_timestamp() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        TemporalMetadata(acquisition_time=datetime(2026, 9, 3, 12, 0))


def test_observation_rejects_inconsistent_validity_flags() -> None:
    payload = _observation().model_dump()
    payload["validity"]["has_crs"] = True

    with pytest.raises(ValidationError, match="validity flags must match"):
        ObservationState.model_validate(payload)
