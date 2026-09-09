"""Deterministic SAR temporal-change and diagnostic mask-agreement tests."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.analytics.exceptions import (
    AnalyticsError,
    InvalidRasterArrayError,
)
from satquery.analytics.sar import (
    SarInputContract,
    mask_agreement,
    sar_temporal_change,
)

_DB_CONTRACT = SarInputContract(
    polarizations=("VV",),
    radiometric_domain="backscatter_db",
    sensor="Sentinel-1",
    calibration="gamma0",
)

_LINEAR_CONTRACT = SarInputContract(
    polarizations=("VV",),
    radiometric_domain="backscatter_linear",
    sensor="Sentinel-1",
    calibration="gamma0",
)


def _t():
    return np.array([[1.0, 4.0], [np.nan, 2.0]])


def test_db_contract_uses_difference_not_log_ratio():
    t1 = _t()
    t2 = np.array([[2.0, 4.0], [1.0, 5.0]])
    result = sar_temporal_change(
        {"VV": t1}, {"VV": t2}, _DB_CONTRACT, threshold=3.0
    )
    expected_valid = np.isfinite(t1) & np.isfinite(t2)
    np.testing.assert_array_equal(
        result.score[expected_valid], np.abs(t2 - t1)[expected_valid]
    )
    # change mask is thresholded on the score; invalid pixels never change
    np.testing.assert_array_equal(
        result.mask, np.array([[False, False], [False, True]])
    )


def test_db_change_mask_thresholds_exactly():
    t1 = np.array([[0.0, 0.0]])
    t2 = np.array([[3.0, 2.999]])
    result = sar_temporal_change(
        {"VV": t1}, {"VV": t2}, _DB_CONTRACT, threshold=3.0
    )
    assert result.mask[0, 0] and not result.mask[0, 1]


def test_linear_contract_uses_safe_log_ratio():
    t1 = np.array([[0.0, 1.0, 1e-9]])
    t2 = np.array([[0.0, 10.0, 2.0]])
    result = sar_temporal_change(
        {"VV": t1}, {"VV": t2}, _LINEAR_CONTRACT, threshold=0.5
    )
    valid = result.valid
    assert valid[0, 2]  # epsilon keeps the 1e-9 cell computable
    assert np.isfinite(result.score[valid]).all()
    np.testing.assert_allclose(
        result.score[0, 1], abs(np.log((10.0 + 1e-6) / (1.0 + 1e-6))), rtol=1e-12
    )


def test_linear_negative_power_is_invalid_not_an_error():
    t1 = np.array([[-1.0, 1.0]])
    t2 = np.array([[1.0, 1.0]])
    result = sar_temporal_change(
        {"VV": t1}, {"VV": t2}, _LINEAR_CONTRACT, threshold=0.5
    )
    assert not result.valid[0, 0]
    assert result.valid[0, 1]


def test_unknown_radiometric_domain_rejects():
    with pytest.raises(AnalyticsError, match="radiometric"):
        contract = SarInputContract(
            polarizations=("VV",),
            radiometric_domain="amplitude",
            sensor="Sentinel-1",
            calibration="unknown",
        )
        sar_temporal_change(
            {"VV": _t()}, {"VV": _t()}, contract, threshold=1.0
        )


def test_unknown_polarization_name_rejects():
    with pytest.raises(AnalyticsError, match="polarization"):
        SarInputContract(
            polarizations=("XX",),
            radiometric_domain="backscatter_db",
            sensor="Sentinel-1",
            calibration="gamma0",
        )


def test_missing_or_extra_polarization_channels_reject():
    t = _t()
    with pytest.raises(AnalyticsError, match="VV"):
        sar_temporal_change({}, {"VV": t}, _DB_CONTRACT, threshold=1.0)
    dual = SarInputContract(
        polarizations=("VV", "VH"),
        radiometric_domain="backscatter_db",
        sensor="Sentinel-1",
        calibration="gamma0",
    )
    with pytest.raises(AnalyticsError, match="VH"):
        sar_temporal_change({"VV": t}, {"VV": t}, dual, threshold=1.0)
    with pytest.raises(AnalyticsError, match="unexpected"):
        sar_temporal_change(
            {"VV": t, "VH": t}, {"VV": t, "VH": t}, _DB_CONTRACT, threshold=1.0
        )


def test_polarization_is_matched_by_name_not_position():
    # channel order in the input mapping must not matter
    t1 = {"VV": _t(), "VH": np.full((2, 2), 3.0)}
    t2 = {"VH": np.full((2, 2), 6.0), "VV": _t()}
    dual = SarInputContract(
        polarizations=("VV", "VH"),
        radiometric_domain="backscatter_db",
        sensor="Sentinel-1",
        calibration="gamma0",
    )
    result = sar_temporal_change(t1, t2, dual, threshold=1.0)
    # combined score is the elementwise max across contracted polarizations
    np.testing.assert_allclose(result.score[0, 0], 3.0)  # VH: |6 - 3|


def test_score_is_max_across_polarizations():
    t1 = {"VV": np.array([[1.0]]), "VH": np.array([[0.0]])}
    t2 = {"VV": np.array([[2.0]]), "VH": np.array([[9.0]])}
    dual = SarInputContract(
        polarizations=("VV", "VH"),
        radiometric_domain="backscatter_db",
        sensor="Sentinel-1",
        calibration="gamma0",
    )
    result = sar_temporal_change(t1, t2, dual, threshold=1.0)
    np.testing.assert_allclose(result.score, [[9.0]])
    assert set(result.scores) == {"VV", "VH"}


def test_shape_mismatch_rejects():
    with pytest.raises(InvalidRasterArrayError):
        sar_temporal_change(
            {"VV": np.ones((2, 2))},
            {"VV": np.ones((3, 2))},
            _DB_CONTRACT,
            threshold=1.0,
        )


def test_negative_threshold_rejects():
    with pytest.raises(ValueError, match="threshold"):
        sar_temporal_change({"VV": _t()}, {"VV": _t()}, _DB_CONTRACT, threshold=-1.0)


# ---------------------------------------------------------------------------
# mask agreement — diagnostic only
# ---------------------------------------------------------------------------


def test_agreement_is_iou_not_confidence():
    first = np.array([[1, 1], [0, 0]], dtype=bool)
    second = np.array([[1, 0], [0, 1]], dtype=bool)
    valid = np.ones((2, 2), dtype=bool)
    result = mask_agreement(first, second, valid)
    assert result.metric == "mask_iou"
    assert result.interpretation == "agreement_not_accuracy"
    assert result.intersection == 1
    assert result.union == 3
    assert result.iou == pytest.approx(1 / 3)
    assert result.valid_pixel_count == 4


def test_agreement_empty_union_yields_none():
    result = mask_agreement(
        np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)
    )
    assert result.intersection == 0
    assert result.union == 0
    assert result.iou is None


def test_agreement_respects_valid_mask():
    first = np.array([[1, 1]], dtype=bool)
    second = np.array([[1, 1]], dtype=bool)
    valid = np.array([[True, False]])
    result = mask_agreement(first, second, valid)
    assert result.intersection == 1
    assert result.union == 1
    assert result.iou == 1.0
    assert result.valid_pixel_count == 1


def test_agreement_rejects_non_binary_masks():
    with pytest.raises(InvalidRasterArrayError):
        mask_agreement(
            np.array([[0.5, 1.0]]), np.zeros((1, 2)), np.ones((1, 2), dtype=bool)
        )
