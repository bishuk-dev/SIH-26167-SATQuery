"""Strict, environment-backed API and resource-abuse boundaries."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import Response

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.schemas_v1 import FailureOutcomeV1


class SecuritySettings(BaseModel):
    """Deploy-time security ceilings and authentication settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cors_origins: tuple[str, ...] = ()
    cors_allow_credentials: bool = False
    max_query_bytes: int = Field(default=1_048_576, gt=0)
    max_roi_vertices: int = Field(default=10_000, gt=0)
    max_queue_size: int = Field(default=32, gt=0)
    max_result_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    api_key: str | None = None

    @model_validator(mode="after")
    def validate_origins(self) -> "SecuritySettings":
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if origin == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("CORS origins must be exact HTTP(S) origins")
            if parsed.path or parsed.query or parsed.fragment:
                raise ValueError("CORS origins must not contain a path or query")
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError("wildcard CORS origins cannot be used with credentials")
        if self.api_key == "":
            raise ValueError("api_key must be omitted when API-key authentication is disabled")
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SecuritySettings":
        values = os.environ if environ is None else environ

        def get(*names: str) -> str | None:
            for name in names:
                if name in values:
                    return values[name]
            return None

        def positive(name: str, *aliases: str, default: int) -> int:
            raw = get(name, *aliases)
            if raw is None or not raw.strip():
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return value

        raw_origins = get("SATQUERY_CORS_ORIGINS", "CORS_ORIGINS") or ""
        origins = tuple(part.strip() for part in raw_origins.split(",") if part.strip())
        raw_credentials = get("SATQUERY_CORS_ALLOW_CREDENTIALS", "CORS_ALLOW_CREDENTIALS")
        credentials = False
        if raw_credentials is not None and raw_credentials.strip():
            if raw_credentials.strip().casefold() not in {"true", "false"}:
                raise ValueError("SATQUERY_CORS_ALLOW_CREDENTIALS must be true or false")
            credentials = raw_credentials.strip().casefold() == "true"
        raw_key = get("SATQUERY_API_KEY", "API_KEY")
        return cls(
            cors_origins=origins,
            cors_allow_credentials=credentials,
            max_query_bytes=positive(
                "SATQUERY_MAX_QUERY_BYTES", "MAX_QUERY_BYTES", default=1_048_576
            ),
            max_roi_vertices=positive(
                "SATQUERY_MAX_ROI_VERTICES", "MAX_ROI_VERTICES", default=10_000
            ),
            max_queue_size=positive(
                "SATQUERY_MAX_QUEUED_JOBS", "SATQUERY_MAX_QUEUE_SIZE", "MAX_QUEUE_SIZE", default=32
            ),
            max_result_bytes=positive(
                "SATQUERY_MAX_RESULT_BYTES", "MAX_RESULT_BYTES", default=10 * 1024 * 1024
            ),
            api_key=raw_key if raw_key else None,
        )


# Kept as a descriptive compatibility alias for callers that use the API name.
ApiSecuritySettings = SecuritySettings


def _error(request: Request, code: str, message: str, status_code: int) -> Response:
    return failure_response(
        code=code,
        message=message,
        outcome=FailureOutcomeV1.REJECT,
        status_code=status_code,
        request_id=request_id_from(request),
    )


def _vertex_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return 1
        return sum(_vertex_count(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_vertex_count(item) for key, item in value.items() if key in {"coordinates", "geometry"})
    return 0


_UNSAFE_KEYS = {"path", "url", "uri", "model", "model_id", "checkpoint", "implementation", "executor"}


def _contains_unsafe_input(value: Any, *, key: str | None = None) -> bool:
    if key is not None and key.casefold() in _UNSAFE_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_unsafe_input(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsafe_input(item) for item in value)
    if isinstance(value, str) and key not in {"query"}:
        parsed = urlsplit(value)
        return value.startswith(("/", "\\")) or parsed.scheme in {"file", "http", "https"}
    return False


async def _limited_body(request: Request, limit: int) -> bytes | None:
    """Read one request body with a hard cap and replay it to Starlette."""

    receive = request._receive  # type: ignore[attr-defined]
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        message_type = message.get("type")
        if message_type == "http.disconnect":
            # A disconnected client must terminate the receive loop. Uvicorn
            # may continue returning disconnect notifications, so continuing
            # here would leak a request task in a hot loop.
            return None
        if message_type != "http.request":
            continue
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    body = b"".join(chunks)
    sent = False

    async def replay() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = replay  # type: ignore[attr-defined]
    request._body = body  # type: ignore[attr-defined]
    return body


async def security_middleware(request: Request, call_next: Callable[..., Any], settings: SecuritySettings) -> Response:
    is_query = request.url.path.startswith("/api/v1/query") and request.method in {"POST", "PUT", "PATCH"}
    if is_query:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > settings.max_query_bytes:
                    return _error(request, "QUERY_TOO_LARGE", "The query request exceeds the configured size limit.", 413)
            except ValueError:
                return _error(request, "INVALID_CONTENT_LENGTH", "The request content length is invalid.", 400)
        body = await _limited_body(request, settings.max_query_bytes)
        if body is None:
            return _error(request, "QUERY_TOO_LARGE", "The query request exceeds the configured size limit.", 413)
        try:
            payload = json.loads(body) if body else None
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            if _vertex_count(payload.get("roi")) > settings.max_roi_vertices:
                return _error(request, "ROI_TOO_LARGE", "The ROI exceeds the configured vertex limit.", 413)
            if _contains_unsafe_input(payload):
                return _error(request, "UNSAFE_INPUT", "The request contains an unsupported raw resource reference.", 422)

    response = await call_next(request)
    if not request.url.path.startswith("/api/v1"):
        return response
    content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
    if content_type == "text/event-stream":
        return response
    # FileResponse is represented as a streaming response after call_next.
    # ArtifactStore and the tile service enforce their own bounded binary
    # outputs; do not consume those files into memory in this middleware.
    binary_response = content_type.startswith("image/") or content_type in {
        "application/octet-stream",
    }
    if binary_response:
        return response

    limit = settings.max_result_bytes
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                return _error(request, "RESULT_TOO_LARGE", "The response exceeds the configured result size limit.", 500)
        except ValueError:
            # A malformed server-generated length must not disable the cap.
            return _error(request, "INVALID_CONTENT_LENGTH", "The response content length is invalid.", 500)

    body = getattr(response, "body", None)
    if body is None and hasattr(response, "body_iterator"):
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.body_iterator:
            total += len(chunk)
            if total > limit:
                return _error(request, "RESULT_TOO_LARGE", "The response exceeds the configured result size limit.", 500)
            chunks.append(chunk)
        body = b"".join(chunks)
    if body is not None and len(body) > limit:
        return _error(request, "RESULT_TOO_LARGE", "The response exceeds the configured result size limit.", 500)
    if body is not None:
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )
    return response


def api_key_dependency(settings: SecuritySettings) -> Any:
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def require_api_key(api_key: str | None = Security(header)) -> None:
        if api_key is None or settings.api_key is None or not secrets.compare_digest(api_key, settings.api_key):
            raise HTTPException(status_code=401, detail="authentication required", headers={"WWW-Authenticate": "ApiKey"})

    return require_api_key


def auth_dependencies(settings: SecuritySettings) -> list[Any]:
    # Use Security at the router boundary so FastAPI emits the API-key
    # requirement in OpenAPI. With no configured key, returning no dependency
    # keeps both runtime and schema security-free.
    return [Security(api_key_dependency(settings))] if settings.api_key is not None else []


__all__ = [
    "ApiSecuritySettings",
    "SecuritySettings",
    "api_key_dependency",
    "auth_dependencies",
    "security_middleware",
]
