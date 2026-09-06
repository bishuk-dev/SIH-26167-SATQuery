"""Temporal Change-VQA orchestration tool enforcing pair validation and evidence lineage."""

from __future__ import annotations

from typing import Any
import uuid

from PIL import Image

from satquery.analytics.temporal import TemporalAnalytics
from satquery.core.contracts.temporal import ChangeVQAResult
from satquery.inference.config import VqaRuntimeSettings
from satquery.ingestion.models import ObservationState
from satquery.models.change_vqa.baseline import load_change_vqa_model
from satquery.verification.models import VerificationStatus


class TemporalVqaTool:
    """Orchestration tool for bi-temporal visual question answering."""

    @classmethod
    def execute(
        cls,
        observation_pre: ObservationState,
        observation_post: ObservationState,
        query: str,
        image_t1: Image.Image | None = None,
        image_t2: Image.Image | None = None,
        *,
        supporting_evidence_ids: tuple[str, ...] = (),
        settings: VqaRuntimeSettings | None = None,
    ) -> ChangeVQAResult:
        """Validate observation pair and execute bi-temporal Change-VQA."""
        # 1. Validation gate before processing
        pair, verification = TemporalAnalytics.build_temporal_pair(observation_pre, observation_post)

        if not verification.is_valid:
            failed_reasons = [c.message for c in verification.checks if c.status == VerificationStatus.FAIL]
            refusal_reason = "; ".join(failed_reasons)
            return ChangeVQAResult(
                query=query,
                answer=f"Temporal Change-VQA rejected due to invalid observation pair: {refusal_reason}",
                confidence=0.0,
                pair_id=pair.pair_id,
                supporting_evidence_ids=supporting_evidence_ids,
                model_provenance=None,
                limitations=(
                    "Validation gate failed prior to model execution.",
                    f"Failures: {refusal_reason}",
                ),
            )

        if image_t1 is None or image_t2 is None:
            return ChangeVQAResult(
                query=query,
                answer="Missing visual inputs; cannot perform learned bi-temporal inference.",
                confidence=0.0,
                pair_id=pair.pair_id,
                supporting_evidence_ids=supporting_evidence_ids,
                model_provenance=None,
                limitations=(
                    "Missing actual visual input for learned bi-temporal inference.",
                    "Both image_t1 and image_t2 are required.",
                ),
            )

        # 2. Run learned Change-VQA specialist
        backend = load_change_vqa_model(settings=settings)
        return backend.answer_change_vqa(
            image_t1=image_t1,
            image_t2=image_t2,
            question=query,
            pair_id=pair.pair_id,
            supporting_evidence_ids=supporting_evidence_ids,
        )


def answer_temporal_change_query(
    observation_pre: ObservationState,
    observation_post: ObservationState,
    query: str,
    image_t1: Image.Image | None = None,
    image_t2: Image.Image | None = None,
    *,
    supporting_evidence_ids: tuple[str, ...] = (),
    settings: VqaRuntimeSettings | None = None,
) -> ChangeVQAResult:
    """Convenience wrapper for TemporalVqaTool.execute."""
    return TemporalVqaTool.execute(
        observation_pre=observation_pre,
        observation_post=observation_post,
        query=query,
        image_t1=image_t1,
        image_t2=image_t2,
        supporting_evidence_ids=supporting_evidence_ids,
        settings=settings,
    )
