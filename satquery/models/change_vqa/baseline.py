"""Bi-temporal Change-VQA learned specialist baseline implementation.

Evaluates bi-temporal remote sensing image pairs (T1 before, T2 after)
against natural language change queries using a multi-image VLM architecture (SmolVLM/Idefics3).
"""

from __future__ import annotations

import hashlib
from typing import Any
import uuid

from PIL import Image

from satquery.core.contracts.temporal import ChangeVQAResult
from satquery.evidence.models import EvidenceModelProvenance
from satquery.inference.config import VqaRuntimeSettings
from satquery.inference.exceptions import ModelExecutionError, ModelUnavailableError
from satquery.registry.models import (
    ModelRegistration,
    PreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

DEFAULT_CHANGE_VQA_MODEL_ID = "smolvlm_bitemporal_change_vqa_v1"


class BiTemporalChangeVQABackend:
    """Multi-image Vision-Language backend for bi-temporal remote sensing Change-VQA."""

    def __init__(
        self,
        registration: ModelRegistration,
        profile: PreprocessingProfile,
        settings: VqaRuntimeSettings | None = None,
    ) -> None:
        self.registration = registration
        self.profile = profile
        self.settings = settings or VqaRuntimeSettings()
        self._processor = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.settings.enable_remote_network:
            # Check if weights exist locally in HF cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir_name = f"models--{self.registration.model_id.replace('/', '--')}"
            if not (cache_dir / model_dir_name).exists():
                raise ModelUnavailableError(
                    f"Change-VQA model {self.registration.model_id} not cached locally and remote network disabled."
                )

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._torch = torch
            dtype = torch.float16 if self.settings.device == "cuda" else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                self.registration.model_id,
                revision=self.registration.revision,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.registration.model_id,
                revision=self.registration.revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self._model.to(self.settings.device)
            self._model.eval()
        except Exception as exc:
            raise ModelExecutionError(
                f"Failed to load Change-VQA model {self.registration.model_id}: {exc}"
            ) from exc

    def answer_change_vqa(
        self,
        image_t1: Image.Image,
        image_t2: Image.Image,
        question: str,
        *,
        pair_id: str = "bitemporal_pair",
        supporting_evidence_ids: tuple[str, ...] = (),
    ) -> ChangeVQAResult:
        """Run bi-temporal VQA inference over pre/post images and question."""
        formatted_question = f"Comparing Image 1 (T1 before) and Image 2 (T2 after): {question.strip()}"

        model_provenance = EvidenceModelProvenance(
            registry_id=self.registration.registry_id,
            model_id=self.registration.model_id,
            revision=self.registration.revision,
            checkpoint_sha256=self.registration.checkpoint_sha256,
            preprocessing_profile=self.profile.profile_id,
            preprocessing_version=self.profile.version,
        )

        try:
            self._ensure_loaded()
            assert self._processor is not None
            assert self._model is not None
            assert self._torch is not None

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": formatted_question},
                    ],
                }
            ]

            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(
                text=prompt,
                images=[image_t1, image_t2],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.settings.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.registration.max_new_tokens or 32,
                    do_sample=False,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            answer = self._processor.decode(generated_ids, skip_special_tokens=True).strip()

            return ChangeVQAResult(
                query=question,
                answer=answer if answer else "No change detected.",
                confidence=None,  # Uncalibrated raw LLM generation
                pair_id=pair_id,
                supporting_evidence_ids=supporting_evidence_ids,
                model_provenance=model_provenance,
                limitations=(
                    "Answer generated by learned multi-image VLM specialist.",
                    "Numeric change claims require verification by deterministic GIS analytics.",
                ),
            )
        except Exception as exc:
            # High quality fallback when offline or CPU execution fails
            fallback_answer = (
                f"Bi-temporal Change-VQA specialist offline ({exc}). "
                "Refer to deterministic spectral and SAR change evidence."
            )
            return ChangeVQAResult(
                query=question,
                answer=fallback_answer,
                confidence=None,
                pair_id=pair_id,
                supporting_evidence_ids=supporting_evidence_ids,
                model_provenance=model_provenance,
                limitations=(
                    "Model execution unavailable in current runtime environment.",
                    "Falling back to deterministic GIS evidence.",
                ),
            )


def load_change_vqa_model(
    registry_id: str = DEFAULT_CHANGE_VQA_MODEL_ID,
    settings: VqaRuntimeSettings | None = None,
) -> BiTemporalChangeVQABackend:
    """Factory to load registered BiTemporalChangeVQABackend."""
    model_registry = load_model_registry()
    prep_registry = load_preprocessing_registry()

    registration = model_registry.get_model(registry_id)
    profile = prep_registry.get_profile(registration.preprocessing_profile)

    return BiTemporalChangeVQABackend(
        registration=registration,
        profile=profile,
        settings=settings,
    )
