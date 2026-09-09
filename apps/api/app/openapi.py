"""FastAPI/OpenAPI product metadata for the canonical API.

This module owns the public API identity (title, description, version) and
the ordered tag catalogue covering the frozen route families in
experiments/phase5_backend/api_contract.md. Tags for families that have no
implemented routes yet are declared here but only appear in the schema once
routes use them.
"""

from __future__ import annotations

from typing import Any

from apps.api.app.schemas_v1 import ApiErrorV1

API_TITLE = "SatQuery API"
API_VERSION = "0.1.0"
API_SUMMARY = (
    "Evidence-grounded, sensor-aware remote-sensing analysis with a "
    "natural-language interface."
)
API_DESCRIPTION = """\
SatQuery AI turns registered observations and bounded natural-language \
requests into persistent, verified, reproducible scientific evidence.

The canonical API lives under `/api/v1`. Pre-existing legacy routes \
(`/api/observations`, `/api/vqa`, `/api/grounding`, `/tiles`) remain \
backward-compatible during Phase 5 and are tagged `Legacy`.

Numeric and spatial claims are always derived from deterministic model or \
GIS outputs; language-model text never creates scientific evidence.
"""

DOCS_URL = "/docs"
REDOC_URL = "/redoc"
OPENAPI_URL = "/openapi.json"

# FastAPI's built-in documentation UIs remain package-free and use their
# documented CDN assets. These options improve interactive use without
# affecting the offline OpenAPI JSON contract.
SWAGGER_UI_PARAMETERS: dict[str, Any] = {
    "displayRequestDuration": True,
    "filter": True,
    "persistAuthorization": True,
    "tryItOutEnabled": True,
}

_ERROR_DESCRIPTIONS = {
    400: "The request is malformed or contains an invalid parameter.",
    401: "Authentication is required when API-key security is enabled.",
    404: "The requested resource was not found.",
    409: "The request conflicts with current resource state.",
    413: "The request exceeds a configured resource limit.",
    415: "The uploaded media type is unsupported.",
    422: "The request failed validation or scientific feasibility checks.",
    429: "The service cannot accept the request because capacity is exhausted.",
    500: "The server could not complete the request.",
    501: "The requested capability is not implemented or enabled.",
    503: "A required dependency is unavailable or temporarily busy.",
}


def error_responses(*statuses: int) -> dict[int, dict[str, Any]]:
    """Return structured v1 error responses for a route decorator.

    Keeping this mapping in one module prevents one route from accidentally
    documenting FastAPI's internal validation model while the runtime emits
    the frozen ``ApiErrorV1`` envelope.
    """

    return {
        status: {
            "model": ApiErrorV1,
            "description": _ERROR_DESCRIPTIONS.get(
                status, "The request could not be processed."
            ),
        }
        for status in statuses
    }


# Ordered tag catalogue for the frozen route families. Route families
# without implemented routes simply do not appear in the schema yet.
TAGS = [
    {"name": "System", "description": "Service version, capabilities, health, and limits."},
    {"name": "Observations", "description": "Immutable observation upload and metadata inspection."},
    {"name": "Pairs", "description": "Validated temporal observation pairs."},
    {"name": "Query Agent", "description": "Natural-language analysis planning and submission."},
    {"name": "Analyses", "description": "Analysis history, trace, and reproducibility."},
    {"name": "Jobs", "description": "Durable execution jobs, progress, and cancellation."},
    {"name": "Tools", "description": "Registered deterministic tool introspection."},
    {"name": "Models", "description": "Registered model capability introspection."},
    {"name": "Evidence", "description": "Evidence graph and spatial evidence views."},
    {"name": "Artifacts", "description": "Immutable derived artifacts and downloads."},
    {"name": "Reports", "description": "Deterministic JSON/HTML analysis reports."},
    {"name": "Tiles", "description": "Raster tile rendering over registered assets."},
    {
        "name": "Legacy",
        "description": (
            "Pre-v1 routes retained for backward compatibility; the canonical "
            "contract lives under /api/v1."
        ),
    },
]
