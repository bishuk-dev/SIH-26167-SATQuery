"""Versioned (v1) transport schemas for the canonical API error contract.

The envelope is frozen in experiments/phase5_backend/api_contract.md:

    {"error": {"code", "message", "outcome", "details", "request_id"}}

Scientific outcomes are independent of HTTP status codes; an HTTP 422 may
carry outcome REQUEST_INPUT, for example.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

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
