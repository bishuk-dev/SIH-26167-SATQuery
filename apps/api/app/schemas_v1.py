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

from pydantic import ConfigDict, Field, model_validator

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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "code": "INVALID_REQUEST",
                    "message": "The request body failed validation.",
                    "outcome": "REJECT",
                    "details": {},
                    "request_id": "req_00000000000000000000000000000000",
                }
            ]
        }
    )

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    outcome: FailureOutcomeV1
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(pattern=r"^req_[0-9a-f]{32}$")


class ApiErrorV1(ApiModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "The request body failed validation.",
                        "outcome": "REJECT",
                        "details": {},
                        "request_id": "req_00000000000000000000000000000000",
                    }
                }
            ]
        }
    )

    error: FailureDetailV1


class PairTypeV1(str, Enum):
    """Supported pair purposes at the API boundary."""

    TEMPORAL = "temporal"
    TEMPORAL_SAME_MODALITY = ModalityPairType.TEMPORAL_SAME_MODALITY.value
    TEMPORAL_CROSS_MODAL = ModalityPairType.TEMPORAL_CROSS_MODAL.value
    OPTICAL_SAR = ModalityPairType.OPTICAL_SAR.value


class PairCreateRequest(ApiModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "observation_a": "obs_00000000000000000000000000000001",
                    "observation_b": "obs_00000000000000000000000000000002",
                    "pair_type": "temporal",
                }
            ]
        }
    )

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


class AnalysisSummaryV1(ApiModel):
    """History projection of one persisted analysis submission."""

    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    status: str
    intent: str
    created_at: datetime
    updated_at: datetime
    rerun_of: str | None = Field(default=None, pattern=r"^ana_[0-9a-f]{32}$")


class AnalysisDetailV1(AnalysisSummaryV1):
    observation_ids: tuple[str, ...] = ()
    plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    registry_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AnalysisPageV1(ApiModel):
    items: tuple[AnalysisSummaryV1, ...]
    next_cursor: str | None = None


class EvidenceEdgeV1(ApiModel):
    source_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    target_evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    edge_type: str = Field(min_length=1)


class EvidenceItemV1(ApiModel):
    """Persisted evidence payload with local filesystem paths removed."""

    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    created_at: datetime
    evidence: dict[str, Any]


class AnalysisEvidenceV1(ApiModel):
    items: tuple[EvidenceItemV1, ...]
    edges: tuple[EvidenceEdgeV1, ...]


class TraceEventV1(ApiModel):
    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    sequence: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    created_at: datetime
    payload: dict[str, Any]


class AnalysisTraceV1(ApiModel):
    items: tuple[TraceEventV1, ...]


class AnalysisArtifactV1(ApiModel):
    artifact_id: str = Field(pattern=r"^artifact_[0-9a-f]{32}$")
    evidence_id: str | None = Field(default=None, pattern=r"^evidence_[0-9a-f]{32}$")
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class AnalysisArtifactsV1(ApiModel):
    items: tuple[AnalysisArtifactV1, ...]


class AnalysisReproducibilityV1(ApiModel):
    """Exact frozen identities needed to reproduce one analysis."""

    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    status: str
    plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    registry_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observation_ids: tuple[str, ...] = ()
    input_hashes: dict[str, str]
    steps: tuple[dict[str, Any], ...]
    artifacts: tuple[AnalysisArtifactV1, ...]


class AnalysisRerunResponseV1(ApiModel):
    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    status: str
    rerun_of: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    registry_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
