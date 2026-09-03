"""Structured evidence returned by deterministic inference adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from satquery.ingestion.models import ContractModel, Modality


class DomainStatus(StrEnum):
    IN_DOMAIN = "in_domain"
    SHIFTED = "shifted"
    UNKNOWN = "unknown"


class VqaPrediction(ContractModel):
    answer: str = Field(min_length=1)
    raw_score: float | None = None


class EvidenceModelProvenance(ContractModel):
    registry_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_profile: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)


class DomainAssessment(ContractModel):
    status: DomainStatus
    reasons: tuple[str, ...] = ()


class EvidenceProvenance(ContractModel):
    created_at: datetime
    operation_id: str = Field(min_length=1)
    input_asset_id: str = Field(min_length=1)
    parent_evidence_ids: tuple[str, ...] = ()

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        return value


class VqaEvidence(ContractModel):
    evidence_id: str = Field(pattern=r"^evidence_[0-9a-f]{32}$")
    task: Literal["single_image_vqa"] = "single_image_vqa"
    prediction: VqaPrediction
    source_observations: tuple[str, ...]
    source_modalities: tuple[Modality, ...]
    model: EvidenceModelProvenance
    domain: DomainAssessment
    warnings: tuple[str, ...] = ()
    provenance: EvidenceProvenance
