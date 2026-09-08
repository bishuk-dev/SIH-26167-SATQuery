"""Radiometric-domain-aware deterministic SAR change analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from satquery.ingestion.models import ContractModel


class UnknownSarSemanticsError(ValueError):
    """Raised when SAR polarization or radiometric units are not explicit."""


class SarInputContract(ContractModel):
    polarizations: tuple[str, ...]
    radiometric_domain: Literal["backscatter_db", "backscatter_linear", "unknown"]
    sensor: str = Field(min_length=1)
    calibration: Literal["calibrated", "unknown"]


@dataclass(frozen=True, slots=True)
class SarChangeResult:
    score: np.ndarray
    valid: np.ndarray
    mask: np.ndarray
    threshold: float


@dataclass(frozen=True, slots=True)
class AgreementResult:
    metric: Literal["mask_iou"]
    interpretation: Literal["agreement_not_accuracy"]
    intersection: int
    union: int
    iou: float | None
    valid_pixel_count: int


def sar_temporal_change(
    t1: np.ndarray,
    t2: np.ndarray,
    contract: SarInputContract,
    *,
    threshold: float,
) -> SarChangeResult:
    if threshold < 0:
        raise ValueError("SAR threshold cannot be negative")
    if contract.radiometric_domain not in {"backscatter_db", "backscatter_linear"}:
        raise UnknownSarSemanticsError("SAR radiometric domain is unknown")
    if not contract.polarizations or any(not item.strip() for item in contract.polarizations):
        raise UnknownSarSemanticsError("SAR polarization semantics are unknown")
    first, second = np.asarray(t1, dtype="float64"), np.asarray(t2, dtype="float64")
    if first.shape != second.shape:
        raise ValueError("SAR temporal arrays must have equal shapes")
    finite = np.isfinite(first) & np.isfinite(second)
    if contract.radiometric_domain == "backscatter_db":
        score = np.abs(second - first)
        valid = finite
    else:
        epsilon = 1e-6
        valid = finite & (first > 0) & (second > 0)
        score = np.zeros(first.shape, dtype="float64")
        np.divide(second + epsilon, first + epsilon, out=score, where=valid)
        score[valid] = np.abs(np.log(score[valid]))
    return SarChangeResult(score=score, valid=valid, mask=valid & (score > threshold), threshold=threshold)


def mask_agreement(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> AgreementResult:
    first_array, second_array, valid_array = (
        np.asarray(first, dtype=bool),
        np.asarray(second, dtype=bool),
        np.asarray(valid, dtype=bool),
    )
    if first_array.shape != second_array.shape or first_array.shape != valid_array.shape:
        raise ValueError("agreement masks and valid mask must have equal shapes")
    intersection = int(np.count_nonzero(first_array & second_array & valid_array))
    union = int(np.count_nonzero((first_array | second_array) & valid_array))
    return AgreementResult(
        metric="mask_iou",
        interpretation="agreement_not_accuracy",
        intersection=intersection,
        union=union,
        iou=intersection / union if union else None,
        valid_pixel_count=int(valid_array.sum()),
    )
