"""Versioned (v1) transport schemas for the canonical API error contract.

The envelope is frozen in experiments/phase5_backend/api_contract.md:

    {"error": {"code", "message", "outcome", "details", "request_id"}}

Scientific outcomes are independent of HTTP status codes; an HTTP 422 may
carry outcome REQUEST_INPUT, for example.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from satquery.geo.models import ModalityPairType

from apps.api.app.schemas import ApiModel


class FailureOutcomeV1(str, Enum):
    """Frozen scientific failure outcomes; never an HTTP status."""

    ALLOW = "ALLOW"
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"
    REQUEST_INPUT = "REQUEST_INPUT"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"


class FailureDetailV1(ApiModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    outcome: FailureOutcomeV1
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(pattern=r"^req_[0-9a-f]{32}$")


class ApiErrorV1(ApiModel):
    error: FailureDetailV1


class PairTypeV1(str, Enum):
    """Supported pair purposes at the API boundary."""

    TEMPORAL = "temporal"
    TEMPORAL_SAME_MODALITY = ModalityPairType.TEMPORAL_SAME_MODALITY.value
    TEMPORAL_CROSS_MODAL = ModalityPairType.TEMPORAL_CROSS_MODAL.value
    OPTICAL_SAR = ModalityPairType.OPTICAL_SAR.value


class PairCreateRequest(ApiModel):
    observation_a: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    observation_b: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    pair_type: PairTypeV1
    explicit_t1_id: str | None = Field(
        default=None, pattern=r"^obs_[0-9a-f]{32}$"
    )

    @model_validator(mode="after")
    def validate_explicit_temporal_observation(self) -> "PairCreateRequest":
        if self.explicit_t1_id is not None and self.explicit_t1_id not in {
            self.observation_a,
            self.observation_b,
        }:
            raise ValueError("explicit_t1_id must identify one pair observation")
        return self


class PairTemporalOrder(ApiModel):
    t1_observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    t2_observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")


class PairValidationResponse(ApiModel):
    pair_id: str | None = Field(default=None, pattern=r"^pair_[0-9a-f]{32}$")
    observation_a: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    observation_b: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    pair_type: PairTypeV1
    explicit_t1_id: str | None = None
    temporal_order: PairTemporalOrder | None = None
    validation: dict[str, Any]
    outcome: FailureOutcomeV1
    warnings: tuple[str, ...] = ()
    created_at: datetime | None = None


class PairPage(ApiModel):
    items: tuple[PairValidationResponse, ...]
    next_cursor: str | None = None
