from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

import ml.evaluation.prepare_phase2b_rsvqa as phase2b_preparation
from ml.evaluation.prepare_phase2b_rsvqa import (
    DATASET_ID,
    DATASET_REVISION,
    prepare_dataset,
)
from ml.evaluation.run_phase2b_comparison import _scene_balanced_subset
from ml.training.config import load_training_config
from ml.training.phase2b import VqaSample, hardware_report
from ml.training.precision import choose_precision_name, select_precision
from ml.training.stability import StabilityMonitorCallback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE2A_MANIFEST = (
    PROJECT_ROOT / "experiments/phase2a_smolvlm_rsvqa_lr/split_manifest.json"
)
PHASE2B_MANIFEST = (
    PROJECT_ROOT / "experiments/phase2b_smolvlm_rsvqa_lr/split_manifest.json"
)


def test_phase2b_config_pins_lora_and_non_test_training_splits() -> None:
    config = load_training_config(
        PROJECT_ROOT / "ml/configs/phase2b_smolvlm_lora.yaml"
    )

    assert config.model_registry_id == "smolvlm_256m_instruct_v1"
    assert config.train_split == "train"
    assert config.validation_split == "validation"
    assert config.lora_rank == 8
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.per_device_train_batch_size == 1


def test_phase2b_manifest_is_grouped_and_excludes_phase2a() -> None:
    phase2a = json.loads(PHASE2A_MANIFEST.read_text(encoding="utf-8"))
    phase2b = json.loads(PHASE2B_MANIFEST.read_text(encoding="utf-8"))
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for sample in phase2b["samples"]:
        scene_splits[sample["scene_id"]].add(sample["split"])

    assert len(phase2b["samples"]) == 1767
    assert len(scene_splits) == 88
    assert all(len(splits) == 1 for splits in scene_splits.values())
    assert set(scene_splits).isdisjoint(phase2a["scene_assignments"])
    assert "created_at" not in phase2b
    assert Counter(sample["split"] for sample in phase2b["samples"]) == {
        "train": 1378,
        "validation": 180,
        "test": 209,
    }


def test_kaggle_notebook_is_thin_and_invokes_repository_entrypoint() -> None:
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks/kaggle_phase2b.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "ml.evaluation.prepare_phase2b_rsvqa" in source
    assert "ml.training.phase2b" in source
    assert "--stability-smoke" in source
    assert "pip', 'uninstall', '-y', 'torchao'" in source
    assert "'torch==2.8.0'" in source
    assert "https://download.pytorch.org/whl/cu126" in source
    assert source.index("'torchao'") < source.index("import torch")
    assert "from peft" not in source
    assert "Trainer(" not in source


def test_comparison_subset_samples_across_scenes() -> None:
    samples = [
        VqaSample(
            sample_id=f"{scene}-{index}",
            scene_id=scene,
            image_path=Path(f"{scene}.jpg"),
            question="question",
            answer="answer",
        )
        for scene in ("a", "b", "c")
        for index in range(4)
    ]

    selected = _scene_balanced_subset(samples, 5)

    assert [sample.sample_id for sample in selected] == [
        "a-0",
        "b-0",
        "c-0",
        "a-1",
        "b-1",
    ]


def test_pascal_uses_fp16_even_when_runtime_claims_bf16() -> None:
    assert choose_precision_name(
        cuda_available=True,
        compute_capability=(6, 0),
        bf16_runtime_reported=True,
    ) == "fp16"


def test_ampere_uses_bf16_only_when_runtime_supports_it() -> None:
    assert choose_precision_name(
        cuda_available=True,
        compute_capability=(8, 0),
        bf16_runtime_reported=True,
    ) == "bf16"
    assert choose_precision_name(
        cuda_available=True,
        compute_capability=(8, 0),
        bf16_runtime_reported=False,
    ) == "fp16"


def test_cpu_uses_fp32() -> None:
    assert choose_precision_name(
        cuda_available=False,
        compute_capability=None,
        bf16_runtime_reported=True,
    ) == "fp32"


def test_explicit_fp32_override_on_pascal() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (6, 0)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

    assert select_precision(FakeTorch(), force_fp32=True).name == "fp32"


def test_frozen_manifest_preparation_preserves_exact_bytes(tmp_path: Path) -> None:
    phase2a_path = tmp_path / "phase2a.json"
    phase2a_path.write_text('{"scene_assignments": {}}\n', encoding="utf-8")
    data_root = tmp_path / "images"
    data_root.mkdir()
    samples = []
    assignments = {}
    counts = {}
    for index, split in enumerate(("train", "validation", "test")):
        image_bytes = f"scene-{index}".encode()
        scene_id = hashlib.sha256(image_bytes).hexdigest()
        (data_root / f"{scene_id}.jpg").write_bytes(image_bytes)
        assignments[scene_id] = split
        counts[split] = {"scenes": 1, "questions": 1}
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "scene_id": scene_id,
                "source_row": index,
                "image_path": f"unused/{scene_id}.jpg",
                "question": "question",
                "answer": "answer",
                "split": split,
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "source_row_limit": 3,
        },
        "excluded_phase2a_scene_ids": [],
        "counts": counts,
        "scene_assignments": assignments,
        "samples": samples,
    }
    manifest_path = tmp_path / "phase2b.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=4) + "\n", encoding="utf-8"
    )
    original_bytes = manifest_path.read_bytes()

    prepared = prepare_dataset(manifest_path, data_root, phase2a_path)

    assert prepared == manifest
    assert manifest_path.read_bytes() == original_bytes


def test_explicit_regeneration_is_byte_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    phase2a_path = tmp_path / "phase2a.json"
    phase2a_path.write_text('{"scene_assignments": {}}\n', encoding="utf-8")
    rows = [
        {
            "row_idx": index,
            "question": f"question-{index}",
            "answer": "answer",
            "image_url": f"https://example.test/{index}.jpg",
        }
        for index in range(20)
    ]
    monkeypatch.setattr(phase2b_preparation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(phase2b_preparation, "_fetch_rows", lambda _limit: rows)
    monkeypatch.setattr(
        phase2b_preparation,
        "_download_image",
        lambda url: url.encode("utf-8"),
    )
    manifest_path = tmp_path / "experiment" / "split_manifest.json"
    data_root = tmp_path / "data"

    phase2b_preparation.build_manifest(
        manifest_path, data_root, phase2a_path, source_row_limit=20
    )
    first_bytes = manifest_path.read_bytes()
    phase2b_preparation.build_manifest(
        manifest_path, data_root, phase2a_path, source_row_limit=20
    )

    assert manifest_path.read_bytes() == first_bytes
    assert b"created_at" not in first_bytes


def test_stability_monitor_checks_gradients_and_parameter_update() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    monitor = StabilityMonitorCallback(torch, model, "fp16")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).sum()
    monitor.record_loss(loss)
    loss.backward()
    state = type("State", (), {"global_step": 0})()

    monitor.on_pre_optimizer_step(None, state, None, model=model)
    optimizer.step()
    monitor.on_optimizer_step(None, state, None, model=model)
    changed = monitor.verify_parameters_changed(model)
    report = monitor.report(parameters_changed=changed)
    monitor.close()

    assert report["optimizer_steps_checked"] == 1
    assert report["microbatches_checked"] == 1
    assert report["trainable_parameters_changed"] is True
    assert report["gradient_norms"][0] > 0


def test_hardware_report_records_selected_p100_precision() -> None:
    class FakeProperties:
        name = "Tesla P100-PCIE-16GB"
        total_memory = 16 * 1024**3

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (6, 0)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

        @staticmethod
        def get_device_properties(_index: int) -> FakeProperties:
            return FakeProperties()

    class FakeVersion:
        cuda = "12.6"

    class FakeTorch:
        __version__ = "2.8.0+cu126"
        cuda = FakeCuda()
        version = FakeVersion()
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    precision = select_precision(FakeTorch())
    report = hardware_report(FakeTorch(), precision)

    assert precision.name == "fp16"
    assert precision.torch_dtype(FakeTorch()) == "float16"
    assert report["selected_precision"] == "fp16"
    assert report["compute_capability"] == [6, 0]
    assert report["bf16_runtime_reported"] is True
    assert report["bf16_selected"] is False
