from __future__ import annotations

import importlib.util

import pytest
import numpy as np
import torch

from ml.evaluation.phase4e_bifold_baseline import BIGEARTHNET_19_CLASS_ORDER
from ml.training import phase4f_s2_head
from satquery.inference.multisensor_preprocessing import SealedTestAccessError


def test_phase4f_training_module_exists() -> None:
    assert importlib.util.find_spec("ml.training.phase4f_s2_head") is not None


class _SyntheticEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(4, 4)
        self.fc = torch.nn.Linear(4, 19)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.stem(inputs))


class _SyntheticInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = _SyntheticEncoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(inputs)


class _SyntheticCheckpoint(torch.nn.Module):
    def __init__(self, class_order: tuple[str, ...] = tuple(map(str, range(19)))) -> None:
        super().__init__()
        self.model = _SyntheticInner()
        self.config = type(
            "Config",
            (),
            {
                "classes": 19,
                "class_names": list(class_order),
                "timm_model_name": "resnet50",
            },
        )()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_phase4f_allows_only_training_and_validation_splits(split: str) -> None:
    phase4f_s2_head.require_phase4f_split(split)


def test_phase4f_keeps_test_sealed() -> None:
    with pytest.raises(SealedTestAccessError, match="sealed"):
        phase4f_s2_head.require_phase4f_split("test")


def test_phase4f_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="train or validation"):
        phase4f_s2_head.require_phase4f_split("development")


def test_head_only_freeze_preserves_head_weights_and_freezes_backbone() -> None:
    model = _SyntheticCheckpoint()
    original_weight = model.model.vision_encoder.fc.weight.detach().clone()

    summary = phase4f_s2_head.freeze_s2_classifier_head(model)

    assert summary.trainable_parameter_names == (
        "model.vision_encoder.fc.weight",
        "model.vision_encoder.fc.bias",
    )
    assert summary.trainable_parameter_count == 95
    assert summary.frozen_parameter_count == 20
    assert torch.equal(model.model.vision_encoder.fc.weight, original_weight)
    assert all(
        parameter.requires_grad
        == name.startswith("model.vision_encoder.fc.")
        for name, parameter in model.named_parameters()
    )


def test_head_only_freeze_rejects_checkpoint_class_order_mismatch() -> None:
    wrong_order = tuple(reversed(BIGEARTHNET_19_CLASS_ORDER))

    with pytest.raises(ValueError, match="class order"):
        phase4f_s2_head.freeze_s2_classifier_head(
            _SyntheticCheckpoint(), official_class_order=wrong_order
        )


def test_phase4e_baseline_linkage_rejects_changed_metric() -> None:
    linked = {"s2": dict(phase4f_s2_head.FROZEN_PHASE4E_S2_BASELINE)}
    phase4f_s2_head.validate_phase4e_baseline_linkage(linked)
    linked["s2"]["macro_average_precision"] = 0.75

    with pytest.raises(ValueError, match="baseline linkage"):
        phase4f_s2_head.validate_phase4e_baseline_linkage(linked)


def test_phase4f_provenance_requires_runtime_and_preserves_test_sealing() -> None:
    base = {
        "git_sha": "a" * 40,
        "dirty_worktree": False,
        "kaggle_experiment": "phase4f-bifold-s2-head-adaptation",
        "kaggle_kernel": "satquery-phase4f-bifold-s2-head-adaptation",
        "runtime_seconds": 12.5,
        "device": "Tesla T4",
        "cuda_version": "12.6",
        "torch_version": "2.8.0+cu126",
        "python_version": "3.12.0",
        "peak_gpu_memory_bytes": 1234,
        "model_id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
        "model_revision": "c" * 40,
        "base_checkpoint_sha256": "d" * 64,
        "materialized_package_sha256": "e" * 64,
        "checkpoint_output_sha256": "b" * 64,
        "manifest_sha256": "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6",
        "preprocessing_profile": "bifold_resnet50_s2_v020",
        "test_accessed": False,
    }
    provenance = phase4f_s2_head.Phase4FProvenance.model_validate(base)
    assert provenance.runtime_seconds == 12.5
    assert provenance.model_revision == "c" * 40

    with pytest.raises(ValueError):
        phase4f_s2_head.Phase4FProvenance.model_validate({**base, "test_accessed": True})


def test_head_only_training_changes_classifier_but_not_backbone() -> None:
    torch.manual_seed(7)
    model = _SyntheticCheckpoint()
    phase4f_s2_head.freeze_s2_classifier_head(model)
    original_stem = model.model.vision_encoder.stem.weight.detach().clone()
    original_head = model.model.vision_encoder.fc.weight.detach().clone()
    inputs = torch.randn(8, 4)
    targets = (torch.rand(8, 19) > 0.5).float()
    batches = [
        (("a", "b", "c", "d"), inputs[:4], targets[:4]),
        (("e", "f", "g", "h"), inputs[4:], targets[4:]),
    ]
    config = phase4f_s2_head.Phase4FTrainingConfig(
        batch_size=4,
        max_epochs=1,
        early_stopping_patience=0,
    )

    result = phase4f_s2_head.fit_head_only(
        model,
        train_batches=batches,
        validation_batches=batches,
        config=config,
        device="cpu",
    )

    assert result.epochs_completed == 1
    assert len(result.train_loss) == 1
    assert len(result.validation_loss) == 1
    assert len(result.validation_macro_average_precision) == 1
    assert torch.equal(model.model.vision_encoder.stem.weight, original_stem)
    assert not torch.equal(model.model.vision_encoder.fc.weight, original_head)


def test_phase4f_dataset_uses_frozen_s2_profile_and_manifest_labels(tmp_path) -> None:
    manifest = {
        "samples": [
            {
                "sample_id": "pair-1",
                "official_split": "train",
                "patch_id": "s2-patch",
                "s1_name": "s1-patch",
                "labels": ["Arable land"],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    dataset = phase4f_s2_head.Phase4FS2Dataset(
        manifest_path=manifest_path,
        dataset_root=tmp_path,
        split="train",
        band_loader=lambda _path: np.ones((120, 120), dtype=np.uint16),
    )
    sample_id, inputs, target = dataset[0]

    assert sample_id == "pair-1"
    assert inputs.shape == (10, 120, 120)
    assert target.dtype == torch.float32
    assert torch.where(target)[0].tolist() == [1]
