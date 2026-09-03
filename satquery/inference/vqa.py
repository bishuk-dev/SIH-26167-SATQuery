"""Registered frozen single-image VQA inference."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image

from satquery.evidence.models import (
    DomainAssessment,
    DomainStatus,
    EvidenceModelProvenance,
    EvidenceProvenance,
    VqaEvidence,
    VqaPrediction,
)
from satquery.inference.config import VqaRuntimeSettings
from satquery.inference.exceptions import (
    ModelExecutionError,
    ModelUnavailableError,
)
from satquery.inference.preprocessing import FrozenImagePreprocessor
from satquery.ingestion.models import Modality
from satquery.ingestion.storage import FilesystemObservationStore
from satquery.registry.models import (
    ModelRegistration,
    PreprocessingProfile,
    load_model_registry,
    load_preprocessing_registry,
)

DEFAULT_MODEL_REGISTRY_ID = "smolvlm_256m_instruct_v1"


class VqaBackend(Protocol):
    def answer(self, image: Image.Image, question: str) -> str: ...


class SmolVlmBackend:
    """Lazy local Transformers backend for the frozen SmolVLM checkpoint."""

    def __init__(
        self,
        registration: ModelRegistration,
        profile: PreprocessingProfile,
        settings: VqaRuntimeSettings,
    ) -> None:
        self.registration = registration
        self.profile = profile
        self.settings = settings
        self._processor = None
        self._model = None
        self._torch = None

    def answer(self, image: Image.Image, question: str) -> str:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        prompt_text = self.profile.prompt_template.format(question=question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            prompt = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
                do_resize=False,
            ).to(self.settings.device)
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.registration.max_new_tokens,
                )
            input_length = inputs["input_ids"].shape[-1]
            answer = self._processor.decode(
                generated[0, input_length:],
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            raise ModelExecutionError("Frozen VQA model execution failed") from exc
        if not answer:
            raise ModelExecutionError("Frozen VQA model returned an empty answer")
        return answer

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailableError(
                "The local VQA runtime dependencies are unavailable"
            ) from exc

        if self.settings.device != "cpu" and not (
            self.settings.device.startswith("cuda") and torch.cuda.is_available()
        ):
            raise ModelUnavailableError(
                f"Configured VQA device {self.settings.device!r} is unavailable"
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
            if _sha256(checkpoint) != self.registration.checkpoint_sha256:
                raise ModelUnavailableError("Cached VQA checkpoint hash is invalid")
            processor = AutoProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=self.registration.allow_remote_code,
                dtype=torch.float32,
            )
            torch.set_num_threads(self.settings.cpu_threads)
            model.to(self.settings.device)
            model.eval()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "The registered VQA checkpoint is not available locally"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model


class SingleImageVqaService:
    """Map a registered observation and question to structured VQA evidence."""

    def __init__(
        self,
        store: FilesystemObservationStore,
        *,
        settings: VqaRuntimeSettings | None = None,
        backend: VqaBackend | None = None,
        model_registry_id: str = DEFAULT_MODEL_REGISTRY_ID,
    ) -> None:
        self.settings = settings or VqaRuntimeSettings.from_env()
        models = load_model_registry()
        profiles = load_preprocessing_registry()
        try:
            self.registration = models.models[model_registry_id]
            self.profile = profiles.profiles[
                self.registration.preprocessing_profile
            ]
        except KeyError as exc:
            raise ValueError("VQA model registry references an unknown entry") from exc
        self.model_registry_id = model_registry_id
        self._store = store
        self._preprocessor = FrozenImagePreprocessor(self.profile)
        self._backend = backend or SmolVlmBackend(
            self.registration,
            self.profile,
            self.settings,
        )

    def answer(self, observation_id: str, question: str) -> VqaEvidence:
        clean_question = question.strip()
        if not clean_question or (
            len(clean_question) > self.settings.max_question_characters
        ):
            raise ModelExecutionError("Question is empty or exceeds the input limit")

        registered, visualization_path = self._store.load_registration(
            observation_id
        )
        image = self._preprocessor.from_visualization(
            visualization_path,
            registered.visualization,
        )
        answer = self._backend.answer(image, clean_question).strip()
        if not answer:
            raise ModelExecutionError("Frozen VQA model returned an empty answer")

        observation = registered.observation
        domain_reasons = ["MODEL_NOT_REMOTE_SENSING_ADAPTED"]
        warnings = [
            "FROZEN_GENERIC_VLM_BASELINE",
            "DISPLAY_DERIVATIVE_USED_FOR_GENERIC_VLM",
        ]
        if observation.sensor.modality is Modality.SAR:
            domain_reasons.append("GENERIC_VLM_SAR_INPUT")
            warnings.append("SAR_INTERPRETATION_NOT_SPECIALIZED")
        if observation.sensor.sensor_name is None:
            domain_reasons.append("SENSOR_UNKNOWN")

        return VqaEvidence(
            evidence_id=f"evidence_{uuid4().hex}",
            prediction=VqaPrediction(answer=answer, raw_score=None),
            source_observations=(observation.observation_id,),
            source_modalities=(observation.sensor.modality,),
            model=EvidenceModelProvenance(
                registry_id=self.model_registry_id,
                model_id=self.registration.model_id,
                revision=self.registration.revision,
                checkpoint_sha256=self.registration.checkpoint_sha256,
                preprocessing_profile=self.registration.preprocessing_profile,
                preprocessing_version=self.profile.version,
            ),
            domain=DomainAssessment(
                status=DomainStatus.SHIFTED,
                reasons=tuple(domain_reasons),
            ),
            warnings=tuple(warnings),
            provenance=EvidenceProvenance(
                created_at=datetime.now(timezone.utc),
                operation_id="single_image_vqa",
                input_asset_id=registered.visualization.asset_id,
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
