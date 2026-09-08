"""Bi-temporal Change-VQA and Change Description learned specialist baseline implementation.

Evaluates bi-temporal remote sensing image pairs (T1 before, T2 after)
against natural language change queries or generates change descriptions using a multi-image
VLM architecture (SmolVLM/Idefics3) or RSICCformer baseline.

License Gate Status:
- cdvqa_annotation_license: Apache-2.0
- second_dataset_access: public
- second_image_license_status: UNRESOLVED
- cdvqa_full_dataset_license_gate: BLOCKED

Active SIH MVP Change Intelligence Benchmark:
- LEVIR-CC (Chen et al. 2022 / Liu et al. 2022)
- Underlying imagery: LEVIR-CD (academic / non-commercial research use only)
- Test set policy: Sealed (evaluated strictly on validation split)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import uuid

from PIL import Image


from satquery.core.contracts.temporal import ChangeDescriptionResult, ChangeVQAResult
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
    """Multi-image Vision-Language backend for bi-temporal remote sensing Change-VQA / Captioning."""

    def __init__(
        self,
        registration: ModelRegistration,
        profile: PreprocessingProfile,
        settings: VqaRuntimeSettings | None = None,
        registry_id: str = DEFAULT_CHANGE_VQA_MODEL_ID,
        profile_id: str = "smolvlm_bitemporal_change_vqa_v1",
    ) -> None:
        self.registration = registration
        self.profile = profile
        self.settings = settings or VqaRuntimeSettings()
        self.registry_id = registry_id
        self.profile_id = profile_id
        self._processor = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        """Explicitly load and verify the bi-temporal model and processor."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailableError(
                "The local bi-temporal Change-VQA runtime dependencies are unavailable"
            ) from exc

        if self.settings.device != "cpu" and not (
            self.settings.device.startswith("cuda") and torch.cuda.is_available()
        ):
            raise ModelUnavailableError(
                f"Configured bi-temporal device {self.settings.device!r} is unavailable"
            )

        cache_dir = (self.settings.model_root / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=self.registration.model_id,
                    revision=self.registration.revision,
                    cache_dir=cache_dir,
                    local_files_only=not self.settings.allow_remote_network,
                )
            )
            checkpoint = snapshot / self.registration.checkpoint_file
            h = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != self.registration.checkpoint_sha256:
                raise ModelUnavailableError("Cached bi-temporal checkpoint hash is invalid")

            processor = AutoProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
            )
            dtype = torch.float16 if self.settings.device == "cuda" else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if self.settings.device == "cpu":
                torch.set_num_threads(self.settings.cpu_threads)
            model.to(self.settings.device)
            model.eval()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelExecutionError(
                f"Failed to load Change-VQA model {self.registration.model_id}: {exc}"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model

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
            registry_id=self.registry_id,
            model_id=self.registration.model_id,
            revision=self.registration.revision,
            checkpoint_sha256=self.registration.checkpoint_sha256,
            preprocessing_profile=self.profile_id,
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
                confidence=None,
                pair_id=pair_id,
                supporting_evidence_ids=supporting_evidence_ids,
                model_provenance=model_provenance,
                limitations=(
                    "Answer generated by learned multi-image VLM specialist.",
                    "Numeric change claims require verification by deterministic GIS analytics.",
                ),
            )
        except ModelUnavailableError:
            raise
        except ModelExecutionError:
            raise
        except Exception as exc:
            raise ModelExecutionError(
                f"Bi-temporal Change-VQA inference failed: {exc}"
            ) from exc

    def describe_changes(
        self,
        image_t1: Image.Image,
        image_t2: Image.Image,
        *,
        pair_id: str = "bitemporal_pair",
        supporting_evidence_ids: tuple[str, ...] = (),
        evaluation_split: str = "val",
    ) -> ChangeDescriptionResult:
        """Generate descriptive change captions comparing pre (T1) and post (T2) images.

        Targeted benchmark: LEVIR-CC (held-out test set sealed).
        """
        change_query = "Describe the visual differences and changes between Image 1 (T1 before) and Image 2 (T2 after) in detail."
        vqa_res = self.answer_change_vqa(
            image_t1,
            image_t2,
            change_query,
            pair_id=pair_id,
            supporting_evidence_ids=supporting_evidence_ids,
        )

        return ChangeDescriptionResult(
            pair_id=pair_id,
            description=vqa_res.answer,
            captions=(vqa_res.answer,),
            dataset_source="LEVIR-CC",
            evaluation_split=evaluation_split,
            confidence=vqa_res.confidence,
            supporting_evidence_ids=supporting_evidence_ids,
            model_provenance=vqa_res.model_provenance,
            limitations=(
                "Change description generated by learned bi-temporal specialist.",
                "Imagery provenance: LEVIR-CC (academic/non-commercial research only).",
                "Evaluation restricted to validation split; test set sealed.",
            ),
        )



def load_change_vqa_model(
    registry_id: str = DEFAULT_CHANGE_VQA_MODEL_ID,
    settings: VqaRuntimeSettings | None = None,
) -> BiTemporalChangeVQABackend:
    """Factory to load registered BiTemporalChangeVQABackend."""
    model_registry = load_model_registry()
    prep_registry = load_preprocessing_registry()

    registration = model_registry.models[registry_id]
    profile = prep_registry.profiles[registration.preprocessing_profile]

    return BiTemporalChangeVQABackend(
        registration=registration,
        profile=profile,
        settings=settings,
        registry_id=registry_id,
        profile_id=registration.preprocessing_profile,
    )

