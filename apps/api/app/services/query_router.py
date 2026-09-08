"""Query-to-task routing for SatQuery."""

from __future__ import annotations


def classify_task(query: str, observation_count: int) -> str:
    q = query.strip().lower()

    # Two-image workflows
    if observation_count == 2:
        if any(word in q for word in [
            "change",
            "changed",
            "increase",
            "decrease",
            "before",
            "after",
        ]):
            return "CHANGE_ANALYSIS"

        if "optical" in q and "sar" in q:
            return "OPTICAL_SAR_ANALYSIS"

        if any(word in q for word in [
            "compare",
            "comparison",
            "both images",
            "both sensors",
        ]):
            return "MULTI_IMAGE_ANALYSIS"

    # Single-image workflows
    if any(word in q for word in [
        "highlight",
        "locate",
        "where is",
        "where are",
        "show me",
    ]):
        return "GROUNDING"

    if any(word in q for word in [
        "describe",
        "caption",
        "scene",
        "land-cover",
    ]):
        return "CAPTION"

    return "VQA"