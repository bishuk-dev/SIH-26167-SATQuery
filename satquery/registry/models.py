"""Validated model and preprocessing registry loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator

from satquery.ingestion.models import ContractModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "models" / "registry.yaml"
DEFAULT_PREPROCESSING_REGISTRY = Path(__file__).with_name("preprocessing.yaml")


class ModelRegistration(ContractModel):
    task: Literal["single_image_vqa"]
    provider: Literal["huggingface"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_file: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture: str = Field(min_length=1)
    license: str = Field(min_length=1)
    preprocessing_profile: str = Field(min_length=1)
    frozen: Literal[True]
    allow_remote_code: Literal[False]
    max_new_tokens: int = Field(gt=0, le=64)


class GroundingModelRegistration(ContractModel):
    task: Literal["text_guided_grounding"]
    provider: Literal["huggingface"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_file: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture: str = Field(min_length=1)
    license: str = Field(min_length=1)
    preprocessing_profile: str = Field(min_length=1)
    frozen: Literal[True]
    allow_remote_code: Literal[False]


class MultisensorModelRegistration(ContractModel):
    task: Literal["multilabel_land_cover"]
    provider: Literal["huggingface"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_file: Literal["model.safetensors"]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_size_bytes: int = Field(gt=0)
    architecture: Literal["ConfigILM ResNet-50"]
    license: Literal["MIT"]
    preprocessing_profile: str = Field(min_length=1)
    input_channels: int = Field(ge=2, le=12)
    output_classes: Literal[19]
    frozen: Literal[True]
    allow_remote_code: Literal[False]


class ModelRegistry(ContractModel):
    schema_version: Literal[1]
    models: dict[
        str,
        Annotated[
            ModelRegistration
            | GroundingModelRegistration
            | MultisensorModelRegistration,
            Field(discriminator="task"),
        ],
    ]


class PreprocessingProfile(ContractModel):
    task: Literal["single_image_vqa"]
    version: str = Field(min_length=1)
    input_asset_kind: Literal["visualization"]
    image_mode: Literal["RGB"]
    resize: Literal["fit_pad"]
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    resampling: Literal["bilinear"]
    padding_rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ]
    nodata_policy: Literal["alpha_to_padding"]
    processor_source: Literal["checkpoint"]
    processor_resize: Literal["disabled"]
    prompt_template: str = Field(min_length=1)

    @field_validator("padding_rgb", mode="before")
    @classmethod
    def normalize_yaml_color(cls, value: object) -> object:
        # YAML has no tuple syntax; normalize its sequence before strict validation.
        if isinstance(value, list):
            return tuple(value)
        return value


class GroundingPreprocessingProfile(ContractModel):
    task: Literal["text_guided_grounding"]
    version: str = Field(min_length=1)
    input_asset_kind: Literal["visualization"]
    image_mode: Literal["RGB"]
    resize: Literal["shortest_edge_with_longest_cap"]
    shortest_edge: int = Field(gt=0, le=4096)
    longest_edge: int = Field(gt=0, le=4096)
    resampling: Literal["bilinear"]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float = Field(gt=0)
    nodata_policy: Literal["alpha_to_black"]
    processor_source: Literal["checkpoint"]
    processor_resize: Literal["disabled"]
    query_format: Literal["lowercase_period"]
    box_threshold: float = Field(ge=0, le=1)
    text_threshold: float = Field(ge=0, le=1)
    max_normalized_box_area: float | None = Field(default=None, gt=0, le=1)

    @field_validator("image_mean", "image_std", mode="before")
    @classmethod
    def normalize_yaml_triplet(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class NativeMultisensorPreprocessingProfile(ContractModel):
    task: Literal["multisensor_native"]
    version: str = Field(min_length=1)
    dataset: Literal["BigEarthNet v2.0.0"]
    input_asset_kind: Literal["native_geotiff_bands"]
    immutable: Literal[True]
    s1_band_order: tuple[Literal["VV", "VH"], Literal["VV", "VH"]]
    s2_band_order: tuple[str, ...]
    nodata_policy: Literal["preserve_native_mask_fail_closed_for_model_input"]

    @field_validator("s1_band_order", "s2_band_order", mode="before")
    @classmethod
    def normalize_yaml_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("s1_band_order")
    @classmethod
    def require_s1_order(cls, value: tuple[str, str]) -> tuple[str, str]:
        if value != ("VV", "VH"):
            raise ValueError("native Sentinel-1 order must be VV, VH")
        return value

    @field_validator("s2_band_order")
    @classmethod
    def require_s2_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = (
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        )
        if value != expected:
            raise ValueError("native Sentinel-2 semantic order changed")
        return value


class BifoldPreprocessingProfile(ContractModel):
    task: Literal["multilabel_land_cover"]
    version: Literal["0.2.0"]
    input_asset_kind: Literal["native_geotiff_bands"]
    band_order: tuple[str, ...]
    excluded_native_bands: tuple[str, ...]
    width: Literal[120]
    height: Literal[120]
    continuous_resampling: Literal["nearest"]
    mask_resampling: Literal["nearest"]
    input_dtype: Literal["float32"]
    scaling_before_normalization: Literal["none"]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    statistics_split: Literal["official_train"]
    statistics_timing: Literal["after_120_nearest_resampling"]
    nodata_policy: Literal["reject_nonfinite_or_masked_required_pixels"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @field_validator(
        "band_order", "excluded_native_bands", "means", "stds", mode="before"
    )
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("stds")
    @classmethod
    def require_positive_stds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item <= 0 for item in value):
            raise ValueError("normalization standard deviations must be positive")
        return value

    def model_post_init(self, __context: object) -> None:
        if len(self.band_order) != len(self.means) or len(self.means) != len(self.stds):
            raise ValueError("band order and normalization statistics must align")


class PreprocessingRegistry(ContractModel):
    schema_version: Literal[1]
    profiles: dict[
        str,
        Annotated[
            PreprocessingProfile
            | GroundingPreprocessingProfile
            | NativeMultisensorPreprocessingProfile
            | BifoldPreprocessingProfile,
            Field(discriminator="task"),
        ],
    ]


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    return ModelRegistry.model_validate(_read_yaml(path or DEFAULT_MODEL_REGISTRY))


def load_preprocessing_registry(
    path: str | Path | None = None,
) -> PreprocessingRegistry:
    return PreprocessingRegistry.model_validate(
        _read_yaml(path or DEFAULT_PREPROCESSING_REGISTRY)
    )


def _read_yaml(path: str | Path) -> object:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)
