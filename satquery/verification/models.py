"""Typed models for independent verification checks and reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

from pydantic import Field

from satquery.ingestion.models import ContractModel


class VerificationStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CheckResult(ContractModel):
    """Result of an individual verification check."""

    rule_id: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    status: VerificationStatus
    message: str = Field(min_length=1)
    details: dict[str, str] = Field(default_factory=dict)


class VerificationReport(ContractModel):
    """Aggregated verification report covering geometric, temporal, physical, and provenance checks."""

    overall_status: VerificationStatus
    checks: tuple[CheckResult, ...]
    passed_count: int = Field(ge=0)
    warn_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)

    @property
    def is_valid(self) -> bool:
        """Return True if no verification check produced a FAIL status."""
        return self.overall_status != VerificationStatus.FAIL

    @classmethod
    def from_checks(cls, checks: list[CheckResult] | tuple[CheckResult, ...]) -> VerificationReport:
        passed = sum(1 for c in checks if c.status == VerificationStatus.PASS)
        warned = sum(1 for c in checks if c.status == VerificationStatus.WARN)
        failed = sum(1 for c in checks if c.status == VerificationStatus.FAIL)

        if failed > 0:
            overall = VerificationStatus.FAIL
        elif warned > 0:
            overall = VerificationStatus.WARN
        else:
            overall = VerificationStatus.PASS

        return cls(
            overall_status=overall,
            checks=tuple(checks),
            passed_count=passed,
            warn_count=warned,
            failed_count=failed,
        )
