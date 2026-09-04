"""Train/validation-only shortcut diagnostics for Phase 2C."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.training.sampling import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments/phase2b_smolvlm_rsvqa_lr/split_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments/phase2c_visual_contrast/diagnostic.json"
)
QUESTION_TYPES = ("presence", "comparison", "count", "rural_urban")


def question_type(question: str) -> str:
    normalized = normalize_text(question)
    if "rural or an urban" in normalized:
        return "rural_urban"
    if normalized.startswith(
        ("how many ", "what is the amount ", "what is the number ")
    ):
        return "count"
    if (
        normalized.startswith(("are there more ", "are there less "))
        or normalized.startswith("is the number ")
        and " equal to " in normalized
    ):
        return "comparison"
    if normalized.startswith("is there ") or " present" in normalized:
        return "presence"
    return "other"


def question_only_predictions(
    train_samples: list[dict[str, Any]],
    evaluation_samples: list[dict[str, Any]],
) -> list[str]:
    answers_by_question: dict[str, Counter[str]] = defaultdict(Counter)
    global_answers: Counter[str] = Counter()
    for sample in train_samples:
        answer = normalize_text(str(sample["answer"]))
        answers_by_question[normalize_text(str(sample["question"]))][answer] += 1
        global_answers[answer] += 1
    fallback = global_answers.most_common(1)[0][0]
    return [
        (
            answers_by_question[normalize_text(str(sample["question"]))]
            .most_common(1)[0][0]
            if answers_by_question[normalize_text(str(sample["question"]))]
            else fallback
        )
        for sample in evaluation_samples
    ]


def accuracy(predictions: list[str], samples: list[dict[str, Any]]) -> float:
    return sum(
        normalize_text(prediction) == normalize_text(str(sample["answer"]))
        for prediction, sample in zip(predictions, samples, strict=True)
    ) / len(samples)


def per_type_accuracy(
    predictions: list[str], samples: list[dict[str, Any]]
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for category in QUESTION_TYPES:
        indexes = [
            index
            for index, sample in enumerate(samples)
            if question_type(str(sample["question"])) == category
        ]
        result[category] = {
            "samples": len(indexes),
            "normalized_exact_match": round(
                sum(
                    normalize_text(predictions[index])
                    == normalize_text(str(samples[index]["answer"]))
                    for index in indexes
                )
                / len(indexes),
                6,
            )
            if indexes
            else 0.0,
        }
    return result


def build_diagnostic(manifest_path: Path) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    train = [sample for sample in manifest["samples"] if sample["split"] == "train"]
    validation = [
        sample for sample in manifest["samples"] if sample["split"] == "validation"
    ]
    train_questions = _answers_by_question(train)
    validation_questions = {
        normalize_text(str(sample["question"])) for sample in validation
    }
    overlapping_questions = set(train_questions) & validation_questions
    ambiguous_questions = {
        question
        for question, answers in train_questions.items()
        if len(answers) >= 2
    }
    contrast_profile = _contrast_profile(train)
    question_only = question_only_predictions(train, validation)

    return {
        "schema_version": 1,
        "scope": {
            "splits_analyzed": ["train", "validation"],
            "test_samples_analyzed": 0,
            "phase2b_comparison_predictions_available": False,
        },
        "source": {
            "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "samples": {"train": len(train), "validation": len(validation)},
        "normalized_question_overlap": {
            "train_unique": len(train_questions),
            "validation_unique": len(validation_questions),
            "shared_unique": len(overlapping_questions),
            "validation_unique_share": round(
                len(overlapping_questions) / len(validation_questions), 6
            ),
            "validation_samples_with_seen_question": sum(
                normalize_text(str(sample["question"])) in train_questions
                for sample in validation
            ),
            "validation_sample_share": round(
                sum(
                    normalize_text(str(sample["question"])) in train_questions
                    for sample in validation
                )
                / len(validation),
                6,
            ),
        },
        "answer_distribution": {
            "train": _answer_distribution(train),
            "validation": _answer_distribution(validation),
        },
        "question_types": {
            "train": _type_distribution(train),
            "validation": _type_distribution(validation),
        },
        "train_question_conditioning": {
            "repeated_questions_with_multiple_answers": len(ambiguous_questions),
            "samples_in_visual_contrast_groups": sum(
                1
                for sample in train
                if normalize_text(str(sample["question"])) in ambiguous_questions
            ),
            "validation_samples_in_visual_contrast_groups": sum(
                1
                for sample in validation
                if normalize_text(str(sample["question"])) in ambiguous_questions
            ),
            "balanced_training_samples": contrast_profile["samples"],
            "balanced_training_scenes": contrast_profile["scenes"],
        },
        "repeated_question_answer_patterns": _repeated_question_patterns(train),
        "question_only_validation": {
            "method": "train exact-question majority; global train majority fallback",
            "normalized_exact_match": round(accuracy(question_only, validation), 6),
            "per_question_type": per_type_accuracy(question_only, validation),
        },
        "visual_prediction_diagnostics": {
            "status": "requires_phase2b_validation_predictions",
            "reason": (
                "No Phase 2B comparison predictions are present locally; the supplied "
                "aggregate scores are from the frozen test subset and are not reused."
            ),
        },
        "selected_intervention": {
            "name": "visual_contrast_balanced_sampling",
            "rationale": (
                "Train only on real cross-scene groups where the same normalized "
                "question has multiple answers, balancing each answer within its "
                "question. The question alone is therefore insufficient."
            ),
            "test_samples_used": 0,
        },
    }


def _answers_by_question(
    samples: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        result[normalize_text(str(sample["question"]))][
            normalize_text(str(sample["answer"]))
        ] += 1
    return result


def _answer_distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    answers = Counter(normalize_text(str(sample["answer"])) for sample in samples)
    yes_no = answers["yes"] + answers["no"]
    return {
        "total": len(samples),
        "yes": answers["yes"],
        "no": answers["no"],
        "yes_no_total": yes_no,
        "yes_no_share": round(yes_no / len(samples), 6),
        "other_total": len(samples) - yes_no,
        "top_answers": dict(answers.most_common(10)),
    }


def _contrast_profile(samples: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        grouped[normalize_text(str(sample["question"]))][
            normalize_text(str(sample["answer"]))
        ].append(sample)
    balanced_count = 0
    scenes = set()
    for by_answer in grouped.values():
        group_scenes = {
            str(sample["scene_id"])
            for answer_samples in by_answer.values()
            for sample in answer_samples
        }
        if len(by_answer) < 2 or len(group_scenes) < 2:
            continue
        balanced_count += max(map(len, by_answer.values())) * len(by_answer)
        scenes.update(group_scenes)
    return {"samples": balanced_count, "scenes": len(scenes)}


def _repeated_question_patterns(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = _answers_by_question(samples)
    repeated = {
        question: answers
        for question, answers in grouped.items()
        if sum(answers.values()) >= 2
    }
    ranked = sorted(
        repeated.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )
    by_type = {}
    for category in QUESTION_TYPES:
        groups = [
            answers
            for question, answers in repeated.items()
            if question_type(question) == category
        ]
        by_type[category] = {
            "repeated_groups": len(groups),
            "multiple_answer_groups": sum(len(answers) >= 2 for answers in groups),
        }
    return {
        "total_repeated_groups": len(repeated),
        "single_answer_groups": sum(
            len(answers) == 1 for answers in repeated.values()
        ),
        "multiple_answer_groups": sum(
            len(answers) >= 2 for answers in repeated.values()
        ),
        "by_question_type": by_type,
        "most_frequent": [
            {
                "question": question,
                "samples": sum(answers.values()),
                "answers": dict(answers.most_common()),
            }
            for question, answers in ranked[:15]
        ],
    }


def _type_distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for category in QUESTION_TYPES:
        selected = [
            sample
            for sample in samples
            if question_type(str(sample["question"])) == category
        ]
        answers = Counter(
            normalize_text(str(sample["answer"])) for sample in selected
        )
        result[category] = {
            "samples": len(selected),
            "share": round(len(selected) / len(samples), 6),
            "top_answers": dict(answers.most_common(5)),
            "majority_share": round(
                answers.most_common(1)[0][1] / len(selected), 6
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    diagnostic = build_diagnostic(args.manifest.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
