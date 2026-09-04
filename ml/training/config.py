"""Validated configuration for Phase 2B LoRA adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.training.sampling import SamplingStrategy


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    run_name: str = Field(min_length=1)
    model_registry_id: str = Field(min_length=1)
    preprocessing_profile: str = Field(min_length=1)
    manifest: Path
    train_split: Literal["train"] = "train"
    validation_split: Literal["validation"] = "validation"
    seed: int = Field(ge=0)
    epochs: float = Field(gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_train_batch_size: int = Field(gt=0)
    per_device_eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    gradient_checkpointing: bool
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    lora_target_modules: tuple[str, ...] = Field(min_length=1)
    logging_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0, le=64)
    evaluation_max_samples: int = Field(gt=0, le=1000)
    training_sampling: SamplingStrategy = "all"

    @model_validator(mode="after")
    def validate_lora_rank(self) -> TrainingConfig:
        if self.lora_alpha < self.lora_rank:
            raise ValueError("lora_alpha must be greater than or equal to lora_rank")
        return self


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if isinstance(values, dict) and isinstance(values.get("lora_target_modules"), list):
        values["lora_target_modules"] = tuple(values["lora_target_modules"])
    return TrainingConfig.model_validate(values)
