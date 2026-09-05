from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from satquery.inference.multisensor_preprocessing import (
    MultisensorPreprocessingError,
    SealedTestAccessError,
    materialized_band_paths,
    preprocess_bifold_bands,
)
from satquery.registry import load_model_registry, load_preprocessing_registry
from satquery.registry.models import BifoldPreprocessingProfile


def _profile(name: str) -> BifoldPreprocessingProfile:
    profile = load_preprocessing_registry().profiles[name]
    assert isinstance(profile, BifoldPreprocessingProfile)
    return profile


def _bands() -> dict[str, np.ndarray]:
    return {
        "VV": np.full((120, 120), -12.0, dtype=np.float32),
        "VH": np.full((120, 120), -19.0, dtype=np.float32),
        "B01": np.full((20, 20), 101.0, dtype=np.float32),
        "B02": np.full((120, 120), 102.0, dtype=np.float32),
        "B03": np.full((120, 120), 103.0, dtype=np.float32),
        "B04": np.full((120, 120), 104.0, dtype=np.float32),
        "B05": np.full((60, 60), 105.0, dtype=np.float32),
        "B06": np.full((60, 60), 106.0, dtype=np.float32),
        "B07": np.full((60, 60), 107.0, dtype=np.float32),
        "B08": np.full((120, 120), 108.0, dtype=np.float32),
        "B8A": np.full((60, 60), 109.0, dtype=np.float32),
        "B09": np.full((20, 20), 110.0, dtype=np.float32),
        "B11": np.full((60, 60), 111.0, dtype=np.float32),
        "B12": np.full((60, 60), 112.0, dtype=np.float32),
    }


@pytest.mark.parametrize(
    ("profile_name", "channels"),
    [
        ("bifold_resnet50_s1_v020", 2),
        ("bifold_resnet50_s2_v020", 10),
        ("bifold_resnet50_all_v020", 12),
    ],
)
def test_frozen_profiles_produce_expected_channel_shapes(
    profile_name: str, channels: int
) -> None:
    result = preprocess_bifold_bands(_bands(), _profile(profile_name))
    assert result.shape == (channels, 120, 120)
    assert result.dtype == torch.float32


def test_semantic_order_normalization_and_exclusions_are_exact() -> None:
    profile = _profile("bifold_resnet50_all_v020")
    bands = _bands()
    first = preprocess_bifold_bands(bands, profile)
    bands["B01"].fill(9999)
    bands["B09"].fill(9999)
    second = preprocess_bifold_bands(bands, profile)

    assert profile.band_order == (
        "VV",
        "VH",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )
    assert profile.excluded_native_bands == ("B01", "B09")
    assert torch.equal(first, second)
    expected_b02 = (102.0 - profile.means[2]) / profile.stds[2]
    assert first[2, 0, 0].item() == pytest.approx(expected_b02)


def test_nearest_resize_is_deterministic() -> None:
    profile = _profile("bifold_resnet50_s2_v020")
    bands = _bands()
    bands["B05"] = np.arange(4, dtype=np.float32).reshape(2, 2)
    first = preprocess_bifold_bands(bands, profile)
    second = preprocess_bifold_bands(bands, profile)

    assert torch.equal(first, second)
    b05 = first[3] * profile.stds[3] + profile.means[3]
    assert b05[0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert b05[-1, -1].item() == pytest.approx(3.0, abs=1e-4)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -9999.0])
def test_invalid_or_nodata_required_pixels_fail_closed(invalid: float) -> None:
    profile = _profile("bifold_resnet50_s1_v020")
    bands = _bands()
    bands["VV"][0, 0] = invalid
    nodata = {"VV": -9999.0} if invalid == -9999.0 else None
    with pytest.raises(MultisensorPreprocessingError, match="NoData"):
        preprocess_bifold_bands(bands, profile, nodata_values=nodata)


def test_sealed_test_paths_require_explicit_final_evaluation_access() -> None:
    sample = {"official_split": "test", "s1_name": "s1", "patch_id": "s2"}
    with pytest.raises(SealedTestAccessError):
        materialized_band_paths(Path("dataset"), sample, split="test")

    paths = materialized_band_paths(
        Path("dataset"), sample, split="test", allow_sealed_test=True
    )
    assert all("sealed_test" in path.parts for path in paths.values())


def test_bifold_models_are_pinned_to_matching_profiles() -> None:
    models = load_model_registry().models
    profiles = load_preprocessing_registry().profiles
    for suffix, channels in (("s1", 2), ("s2", 10), ("all", 12)):
        key = f"bifold_resnet50_{suffix}_v020"
        model = models[key]
        profile = profiles[key]
        assert model.preprocessing_profile == key
        assert model.input_channels == channels
        assert isinstance(profile, BifoldPreprocessingProfile)
        assert len(profile.band_order) == channels
