"""Separate open-model adapter for SatQuery backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpenModelResult:
    answer: str
    model_name: str
    confidence: float | None = None


class OpenModelAdapter:
    """
    Adapter for a separate open remote-sensing model.

    This class is intentionally isolated from the existing SatQuery model.
    """

    def __init__(self, model_name: str = "open-model-placeholder") -> None:
        self.model_name = model_name

    def answer(
        self,
        image_path: str,
        query: str,
    ) -> OpenModelResult:

        return OpenModelResult(
            answer="Open model is not connected yet.",
            model_name=self.model_name,
            confidence=None,
        )