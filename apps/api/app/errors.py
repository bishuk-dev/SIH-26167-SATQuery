"""V1 transport error mapping: request IDs, the frozen envelope, and HTTP
translation for the canonical ``/api/v1`` namespace.

Scientific domain exceptions in ``satquery`` are never repurposed as HTTP
errors here; translation happens at this transport boundary only.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from apps.api.app.schemas_v1 import FailureOutcomeV1

INTERNAL_ERROR_CODE = "INTERNAL_ERROR"
INTERNAL_ERROR_MESSAGE = "An unexpected internal error occurred."

# validation detail is capped so pathological payloads cannot inflate responses
_MAX_VALIDATION_ITEMS = 20


def new_request_id() -> str:
    """Server-side correlation ID; incoming client IDs are never trusted."""

    return f"req_{uuid.uuid4().hex}"


def request_id_from(request: Request) -> str:
    """Resolve the one correlation ID for this exchange.

    Middleware already set ``request.state.request_id`` for requests that
    reached routing; the eager-default ``getattr`` antipattern is avoided so
    a fallback ID is minted only when genuinely absent.
    """

    request_id = getattr(request.state, "request_id", None)
    return request_id if request_id is not None else new_request_id()


def with_request_id(response: JSONResponse, request_id: str) -> JSONResponse:
    """Attach the correlation ID header to a handler-created response."""

    response.headers["X-Request-ID"] = request_id
    return response


def failure_response(
    *,
    code: str,
    message: str,
    outcome: FailureOutcomeV1,
    status_code: int,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Render the frozen v1 error envelope; request_id is mandatory."""

    payload = {
        "error": {
            "code": code,
            "message": message,
            "outcome": outcome.value,
            "details": details or {},
            "request_id": request_id,
        }
    }
    # the header travels with the response itself so the body/header
    # invariant holds even when the middleware return path is bypassed
    # (application-level exception handlers)
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={"X-Request-ID": request_id},
    )


def internal_error_response(request_id: str) -> JSONResponse:
    """Sanitized 500: never returns exception text, repr, or tracebacks."""

    return failure_response(
        code=INTERNAL_ERROR_CODE,
        message=INTERNAL_ERROR_MESSAGE,
        outcome=FailureOutcomeV1.ABSTAIN,
        status_code=500,
        request_id=request_id,
    )


def safe_validation_details(error: ValidationError) -> dict[str, Any]:
    """Extract field locations and stable categories only.

    Arbitrary user input values are never echoed: Pydantic's ``input`` and
    free-form message fields are dropped, keeping the detail safe to return.
    """

    items = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "category": item["type"],
        }
        for item in error.errors()[:_MAX_VALIDATION_ITEMS]
    ]
    return {"validation_errors": items}
