"""Deterministic JSON/HTML analysis report endpoints.

Reports are read-only projections of persisted analyses. PDF rendering stays
disabled until a separately audited renderer is approved: the PDF route always
answers ``501 PDF_REPORT_DISABLED``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from apps.api.app.errors import failure_response, request_id_from
from apps.api.app.openapi import error_responses
from apps.api.app.schemas_v1 import ApiErrorV1, FailureOutcomeV1
from satquery.reporting import AnalysisReport, render_html

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


def _report_or_404(request: Request, analysis_id: str) -> AnalysisReport:
    record = request.app.state.observation_repository.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404)
    return request.app.state.report_builder.build(record)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisReport,
    operation_id="get_analysis_report_v1",
    summary="Return the deterministic JSON report for one analysis.",
    description="Builds a path-free JSON report from persisted analysis and evidence records.",
    responses=error_responses(401, 404),
)
def get_analysis_report_v1(request: Request, analysis_id: str) -> AnalysisReport:
    return _report_or_404(request, analysis_id)


@router.get(
    "/{analysis_id}/html",
    operation_id="render_analysis_report_html_v1",
    response_class=HTMLResponse,
    summary="Render the deterministic HTML report for one analysis.",
    description="Renders the same persisted report as deterministic escaped HTML.",
    responses={
        200: {"description": "The deterministic HTML report."},
        **error_responses(401, 404),
    },
)
def render_analysis_report_html_v1(request: Request, analysis_id: str) -> HTMLResponse:
    report = _report_or_404(request, analysis_id)
    return HTMLResponse(content=render_html(report), media_type="text/html")


@router.get(
    "/{analysis_id}/pdf",
    status_code=501,
    operation_id="get_analysis_report_pdf_v1",
    summary="PDF reports are disabled.",
    description=(
        "PDF rendering is disabled until a separately audited renderer is "
        "approved; no PDF is ever produced."
    ),
    responses={
        501: {"model": ApiErrorV1, "description": "PDF rendering is disabled."},
        **error_responses(401, 404),
    },
)
def get_analysis_report_pdf_v1(request: Request, analysis_id: str) -> JSONResponse:
    _report_or_404(request, analysis_id)
    return failure_response(
        code="PDF_REPORT_DISABLED",
        message="PDF reports are disabled; no audited PDF renderer is approved.",
        outcome=FailureOutcomeV1.REJECT,
        status_code=501,
        request_id=request_id_from(request),
    )


__all__ = ["router"]
