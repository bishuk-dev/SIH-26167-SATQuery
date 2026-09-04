from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ml.evaluation.phase2c_diagnostics import build_diagnostic, question_type
from ml.evaluation.run_phase2c_validation import _score_adapter
from ml.training.config import load_training_config
from ml.training.phase2b import VqaSample
from ml.training.sampling import normalize_text, select_training_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "experiments/phase2b_smolvlm_rsvqa_lr/split_manifest.json"


def _sample(sample_id: str, scene: str, question: str, answer: str) -> VqaSample:
    return VqaSample(sample_id, scene, Path(f"{scene}.jpg"), question, answer)


def test_visual_contrast_sampling_is_deterministic_and_answer_balanced() -> None:
    samples = [
        _sample("a", "scene-a", "Is a road present?", "yes"),
        _sample("b", "scene-b", "Is a road present?", "yes"),
        _sample("c", "scene-c", "Is a road present?", "yes"),
        _sample("d", "scene-d", "Is a road present?", "no"),
        _sample("e", "scene-e", "How many roads?", "2"),
    ]

    selected, report = select_training_samples(
        samples, strategy="visual_contrast_balanced", seed=42
    )
    repeated, repeated_report = select_training_samples(
        samples, strategy="visual_contrast_balanced", seed=42
    )

    assert [sample.sample_id for sample in selected] == [
        sample.sample_id for sample in repeated
    ]
    assert Counter(normalize_text(sample.answer) for sample in selected) == {
        "yes": 3,
        "no": 3,
    }
    assert {sample.question for sample in selected} == {"Is a road present?"}
    assert report == repeated_report
    assert report.source_samples == 5
    assert report.selected_samples == 6
    assert report.contrast_question_groups == 1


def test_phase2c_diagnostic_uses_only_train_and_validation() -> None:
    diagnostic = build_diagnostic(MANIFEST)

    assert diagnostic["scope"]["test_samples_analyzed"] == 0
    assert diagnostic["samples"] == {"train": 1378, "validation": 180}
    assert diagnostic["normalized_question_overlap"]["shared_unique"] == 104
    assert (
        diagnostic["question_only_validation"]["normalized_exact_match"]
        == 0.561111
    )
    conditioning = diagnostic["train_question_conditioning"]
    assert conditioning["repeated_questions_with_multiple_answers"] == 100
    assert conditioning["balanced_training_samples"] == 464
    assert conditioning["balanced_training_scenes"] == 70


def test_phase2c_config_changes_sampling_not_model_or_lora() -> None:
    baseline = load_training_config(
        PROJECT_ROOT / "ml/configs/phase2b_smolvlm_lora.yaml"
    )
    candidate = load_training_config(
        PROJECT_ROOT / "ml/configs/phase2c_smolvlm_visual_contrast.yaml"
    )

    assert baseline.training_sampling == "all"
    assert candidate.training_sampling == "visual_contrast_balanced"
    for field in (
        "model_registry_id",
        "preprocessing_profile",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "gradient_accumulation_steps",
    ):
        assert getattr(candidate, field) == getattr(baseline, field)


def test_validation_scoring_measures_visual_gap_and_differences() -> None:
    samples = [
        {"question": "Is a road present?", "answer": "yes"},
        {"question": "Are there more roads?", "answer": "no"},
        {"question": "How many roads?", "answer": "2"},
        {"question": "Is it a rural or an urban area?", "answer": "rural"},
    ]
    result = _score_adapter(
        {
            "correct": ["yes", "no", "2", "urban"],
            "blank": ["yes", "no", "0", "urban"],
            "shuffled": ["no", "yes", "0", "urban"],
        },
        samples,
    )

    assert result["normalized_exact_match"] == {
        "correct": 0.75,
        "blank": 0.5,
        "shuffled": 0.0,
    }
    assert result["visual_dependence_gap"] == 0.25
    assert result["correct_vs_shuffled_predictions_differ"] == 3
    assert result["correct_image_only_wins"] == 3
    assert result["shuffled_image_only_wins"] == 0


def test_question_type_classification_matches_rsvqa_templates() -> None:
    assert question_type("Is a road present?") == "presence"
    assert question_type("Are there more roads than buildings?") == "comparison"
    assert question_type("What is the number of roads?") == "count"
    assert question_type("Is it a rural or an urban area?") == "rural_urban"


def test_committed_diagnostic_matches_generator() -> None:
    committed = json.loads(
        (
            PROJECT_ROOT / "experiments/phase2c_visual_contrast/diagnostic.json"
        ).read_text(encoding="utf-8")
    )

    assert committed == build_diagnostic(MANIFEST)
