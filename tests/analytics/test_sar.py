from __future__ import annotations

import numpy as np
import pytest

from satquery.analytics.sar import (
    SarInputContract,
    UnknownSarSemanticsError,
    mask_agreement,
    sar_temporal_change,
)


def test_db_contract_uses_difference_not_log_ratio() -> None:
    contract = SarInputContract(
        polarizations=("VV", "VH"),
        radiometric_domain="backscatter_db",
        sensor="Sentinel-1",
        calibration="calibrated",
    )
    t1 = np.array([["-10", "-5"]], dtype="float32")
    t2 = np.array([["-7", "-8"]], dtype="float32")

    result = sar_temporal_change(t1, t2, contract, threshold=3.0)

    np.testing.assert_array_equal(result.score, np.abs(t2 - t1))


def test_linear_contract_uses_safe_log_ratio() -> None:
    contract = SarInputContract(
        polarizations=("VV", "VH"),
        radiometric_domain="backscatter_linear",
        sensor="Sentinel-1",
        calibration="calibrated",
    )
    result = sar_temporal_change(
        np.array([[1.0, 2.0]]), np.array([[2.0, 1.0]]), contract, threshold=0.5
    )

    assert np.isfinite(result.score[result.valid]).all()


def test_unknown_radiometric_domain_or_polarization_rejects() -> None:
    contract = SarInputContract(
        polarizations=("HH",),
        radiometric_domain="unknown",
        sensor="RISAT",
        calibration="unknown",
    )

    with pytest.raises(UnknownSarSemanticsError):
        sar_temporal_change(np.ones((1, 1)), np.ones((1, 1)), contract, threshold=1.0)


def test_agreement_is_iou_not_confidence() -> None:
    result = mask_agreement(
        np.array([[True, False], [True, False]]),
        np.array([[True, True], [False, False]]),
        np.ones((2, 2), dtype=bool),
    )

    assert result.metric == "mask_iou"
    assert result.interpretation == "agreement_not_accuracy"
    assert result.iou == pytest.approx(1 / 3)
