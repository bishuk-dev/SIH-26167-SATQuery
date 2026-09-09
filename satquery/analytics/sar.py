"""Deterministic SAR temporal-change scoring and diagnostic mask agreement.

Only two explicitly contracted radiometric domains are supported — absolute
decibel difference and safe log-ratio of linear power. Unknown radiometric
semantics (amplitude, complex, raw DN, gamma0-vs-sigma0 confusion) reject
instead of being guessed. Polarizations are matched strictly by semantic
name, never by channel position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from satquery.analytics.exceptions import (
    AnalyticsError,
    InvalidRasterArrayError,
)

RadiometricDomain = Literal["backscatter_db", "backscatter_linear"]

KNOWN_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})
SUPPORTED_RADIOMETRIC_DOMAINS = frozenset({"backscatter_db", "backscatter_linear"})

# frozen epsilon for linear-power log-ratio; keeps zero-power pixels finite
LINEAR_LOG_RATIO_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class SarInputContract:
    """Explicit semantics for a SAR temporal pair; nothing is inferred."""

    polarizations: tuple[str, ...]
    radiometric_domain: str
    sensor: str
    calibration: str

    def __post_init__(self) -> None:
        polarizations = tuple(self.polarizations)
        if not polarizations:
            raise AnalyticsError("SAR contract must name at least one polarization")
        unknown = [p for p in polarizations if p not in KNOWN_POLARIZATIONS]
        if unknown:
            raise AnalyticsError(
                f"unknown SAR polarization semantics: {', '.join(sorted(unknown))}; "
                f"known names are {', '.join(sorted(KNOWN_POLARIZATIONS))}"
            )
        if len(set(polarizations)) != len(polarizations):
            raise AnalyticsError("SAR polarization contract must not repeat a channel")
        if self.radiometric_domain not in SUPPORTED_RADIOMETRIC_DOMAINS:
            raise AnalyticsError(
                f"unsupported SAR radiometric domain {self.radiometric_domain!r}; "
                "amplitude, complex, and raw-DN inputs cannot be scored without "
                "an explicit conversion contract"
            )
        if not self.sensor or not self.calibration:
            raise AnalyticsError("SAR contract must declare sensor and calibration")


@dataclass(frozen=True, slots=True)
class SarChangeResult:
    score: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    scores: dict[str, np.ndarray]
    contract: SarInputContract
    threshold: float
    method: str


def _validate_pair(
    t1: Mapping[str, np.ndarray],
    t2: Mapping[str, np.ndarray],
    contract: SarInputContract,
) -> tuple[int, int]:
    expected = set(contract.polarizations)
    shape: tuple[int, int] | None = None
    for role, mapping in (("T1", t1), ("T2", t2)):
        keys = set(mapping)
        missing = sorted(expected - keys)
        if missing:
            raise AnalyticsError(
                f"{role} is missing contracted polarizations: {', '.join(missing)}"
            )
        unexpected = sorted(keys - expected)
        if unexpected:
            raise AnalyticsError(
                f"{role} has unexpected polarizations outside the contract: "
                f"{', '.join(unexpected)}"
            )
        for name, array in mapping.items():
            values = np.asarray(array)
            if values.ndim != 2:
                raise InvalidRasterArrayError(
                    f"{role} polarization {name!r} must be two-dimensional"
                )
            if shape is None:
                shape = (values.shape[0], values.shape[1])
            elif values.shape != shape:
                raise InvalidRasterArrayError(
                    f"{role} polarization {name!r} shape {values.shape} does not "
                    f"match the pair grid {shape}"
                )
    assert shape is not None  # missing-key checks above guarantee non-empty
    return shape


def _polarization_score(
    first: np.ndarray,
    second: np.ndarray,
    contract: SarInputContract,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (score, valid) for one polarization in the contracted domain."""

    first = np.asarray(first, dtype="float64")
    second = np.asarray(second, dtype="float64")
    if contract.radiometric_domain == "backscatter_db":
        # dB is already logarithmic: change magnitude is the absolute difference
        invalid = ~np.isfinite(first) | ~np.isfinite(second)
        score = np.abs(second - first)
        method = "absolute_db_difference"
    else:
        invalid = ~np.isfinite(first) | ~np.isfinite(second) | (first < 0) | (second < 0)
        epsilon = LINEAR_LOG_RATIO_EPSILON
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.abs(np.log((second + epsilon) / (first + epsilon)))
        method = "absolute_log_ratio_linear_power"
    valid = ~invalid & np.isfinite(score)
    return score, valid, method


def sar_temporal_change(
    t1: Mapping[str, np.ndarray],
    t2: Mapping[str, np.ndarray],
    contract: SarInputContract,
    *,
    threshold: float,
) -> SarChangeResult:
    """Score a SAR temporal pair under an explicit semantics contract.

    The combined ``score`` is the elementwise maximum across the contracted
    polarizations (any-channel change), and ``mask`` thresholds that score at
    the supplied threshold. Invalid pixels are never flagged as change.
    """

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    shape = _validate_pair(t1, t2, contract)

    scores: dict[str, np.ndarray] = {}
    valid = np.ones(shape, dtype=bool)
    methods: set[str] = set()
    for name in contract.polarizations:
        score, pol_valid, method = _polarization_score(
            t1[name], t2[name], contract
        )
        scores[name] = score
        valid &= pol_valid
        methods.add(method)

    combined = np.zeros(shape, dtype="float64")
    for score in scores.values():
        np.maximum(combined, score, out=combined, where=valid)
    mask = valid & (combined >= threshold)

    return SarChangeResult(
        score=combined,
        mask=mask,
        valid=valid,
        scores=scores,
        contract=contract,
        threshold=threshold,
        method="+".join(sorted(methods)),
    )


@dataclass(frozen=True, slots=True)
class AgreementResult:
    metric: Literal["mask_iou"]
    interpretation: Literal["agreement_not_accuracy"]
    intersection: int
    union: int
    iou: float | None
    valid_pixel_count: int


def _as_binary(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.dtype == bool:
        return values
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise InvalidRasterArrayError("mask must not contain non-finite values")
    if not np.isin(values, (0, 1)).all():
        raise InvalidRasterArrayError("mask must be boolean or strictly binary 0/1")
    return values.astype(bool)


def mask_agreement(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> AgreementResult:
    """Diagnostic IoU between two binary masks over valid pixels.

    Agreement is not accuracy: low IoU must never be interpreted as either
    mask being wrong, and Phase 4 maps no decision policy onto IoU values.
    """

    first_binary = _as_binary(first)
    second_binary = _as_binary(second)
    valid_array = np.asarray(valid, dtype=bool)
    if not (first_binary.shape == second_binary.shape == valid_array.shape):
        raise InvalidRasterArrayError("masks and valid mask must share one shape")

    intersection = int((first_binary & second_binary & valid_array).sum())
    union = int(((first_binary | second_binary) & valid_array).sum())
    return AgreementResult(
        metric="mask_iou",
        interpretation="agreement_not_accuracy",
        intersection=intersection,
        union=union,
        iou=(intersection / union) if union else None,
        valid_pixel_count=int(valid_array.sum()),
    )
