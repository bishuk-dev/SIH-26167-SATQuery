"""FastAPI/OpenAPI product metadata for the canonical API.

This module owns the public API identity (title, description, version) and
the ordered tag catalogue covering the frozen route families in
experiments/phase5_backend/api_contract.md. Tags for families that have no
implemented routes yet are declared here but only appear in the schema once
routes use them.
"""

from __future__ import annotations

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
