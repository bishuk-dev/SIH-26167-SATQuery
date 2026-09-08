"""Analysis orchestration service."""

from __future__ import annotations

from apps.api.app.services.open_model import OpenModelAdapter
from apps.api.app.services.query_router import classify_task
from satquery.ingestion.exceptions import ObservationNotFoundError


TASK_ROUTES = {
    "VQA": "single_image_vqa",
    "CAPTION": "single_image_caption",
    "GROUNDING": "text_guided_grounding",
    "CHANGE_ANALYSIS": "bi_temporal_change",
    "OPTICAL_SAR_ANALYSIS": "optical_sar_analysis",
    "MULTI_IMAGE_ANALYSIS": "multi_image_analysis",
}


class AnalysisService:

    def __init__(
        self,
        *,
        observation_store,
        open_model: OpenModelAdapter,
    ) -> None:
        self._observation_store = observation_store
        self._open_model = open_model
def analyze(
    self,
    observation_ids: list[str],
    query: str,
) -> dict[str, object]:

        for observation_id in observation_ids:
            self._observation_store.load_registration(observation_id)

        task = classify_task(
            query,
            len(observation_ids),
        )

        route = TASK_ROUTES.get(
            task,
            "unknown",
        )

        execution_trace = {
            "selected_task": task,
            "route": route,
            "input_count": len(observation_ids),
            "query_received": True,
            "execution_status": "NOT_EXECUTED",
        }

        return {
            "task": task,
            "observation_ids": observation_ids,
            "query": query,
            "status": "ACCEPTED",
            "execution_trace": execution_trace,
            "evidence": None,
        }