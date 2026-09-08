from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from satquery.evidence.models import EvidenceModelProvenance
from satquery.inference.change_captioning import ChangeCaptionService
from satquery.inference.exceptions import ModelExecutionError, ModelInputUnsupportedError, TemporalOrderUnknownError
from satquery.ingestion.models import Modality, TemporalMetadata

from tests.inference.test_change_detection import _observation, _write_image


class FakeCaptionBackend:
    def caption(self, t1_rgb, t2_rgb) -> str:
        return "A building appears."


class EmptyCaptionBackend:
    def caption(self, t1_rgb, t2_rgb) -> str:
        return "  "


def _service(tmp_path: Path, backend=None) -> ChangeCaptionService:
    model = EvidenceModelProvenance(
        registry_id="chg2cap_fixture",
        model_id="fixture/model",
        revision="0" * 40,
        checkpoint_sha256="0" * 64,
        preprocessing_profile="chg2cap_fixture_v1",
        preprocessing_version="1.0.0",
    )
    return ChangeCaptionService(backend or FakeCaptionBackend(), model)


def test_caption_service_preserves_pair_order_and_returns_text_only(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    pair = (
        _observation(image, "t1"),
        _observation(image, "t2").model_copy(
            update={"temporal": TemporalMetadata(acquisition_time=datetime(2020, 1, 2, tzinfo=timezone.utc))}
        ),
    )

    evidence = _service(tmp_path).describe(*pair)

    assert evidence.temporal.t1_observation_id == pair[0].observation_id
    assert not hasattr(evidence, "measurement")


def test_caption_service_rejects_unknown_temporal_order(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1").model_copy(update={"temporal": TemporalMetadata()})
    t2 = _observation(image, "t2").model_copy(update={"temporal": TemporalMetadata()})

    with pytest.raises(TemporalOrderUnknownError, match="temporal order"):
        _service(tmp_path).describe(t1, t2)


def test_caption_service_rejects_wrong_modality_and_empty_output(tmp_path: Path) -> None:
    image = tmp_path / "image.tif"
    _write_image(image)
    t1 = _observation(image, "t1")
    t2 = _observation(image, "t2").model_copy(
        update={"temporal": TemporalMetadata(acquisition_time=datetime(2020, 1, 2, tzinfo=timezone.utc))}
    )
    sar = t1.model_copy(update={"sensor": t1.sensor.model_copy(update={"modality": Modality.SAR})})
    with pytest.raises(ModelInputUnsupportedError):
        _service(tmp_path).describe(sar, t2)
    with pytest.raises(ModelExecutionError, match="empty"):
        _service(tmp_path, EmptyCaptionBackend()).describe(t1, t2)
