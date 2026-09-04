"""Deterministic training-sample selection for VQA adaptation experiments."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar

SamplingStrategy = Literal["all", "visual_contrast_balanced"]
SampleT = TypeVar("SampleT")


@dataclass(frozen=True)
class SamplingReport:
    strategy: SamplingStrategy
    source_samples: int
    selected_samples: int
    selected_scenes: int
    contrast_question_groups: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def select_training_samples(
    samples: Sequence[SampleT],
    *,
    strategy: SamplingStrategy,
    seed: int,
) -> tuple[list[SampleT], SamplingReport]:
    """Select real samples; visual contrast requires one question, multiple answers."""
    if strategy == "all":
        selected = list(samples)
        return selected, _report(strategy, samples, selected, 0)
    if strategy != "visual_contrast_balanced":
        raise ValueError(f"Unsupported training sampling strategy: {strategy}")

    grouped: dict[str, dict[str, list[SampleT]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        question = normalize_text(str(getattr(sample, "question")))
        answer = normalize_text(str(getattr(sample, "answer")))
        grouped[question][answer].append(sample)

    selected: list[SampleT] = []
    contrast_groups = 0
    for question in sorted(grouped):
        by_answer = grouped[question]
        scene_ids = {
            str(getattr(sample, "scene_id"))
            for answer_samples in by_answer.values()
            for sample in answer_samples
        }
        if len(by_answer) < 2 or len(scene_ids) < 2:
            continue
        contrast_groups += 1
        largest_bucket = max(len(answer_samples) for answer_samples in by_answer.values())
        for answer in sorted(by_answer):
            answer_samples = sorted(
                by_answer[answer], key=lambda sample: str(getattr(sample, "sample_id"))
            )
            selected.extend(
                answer_samples[index % len(answer_samples)]
                for index in range(largest_bucket)
            )

    if not selected:
        raise ValueError(
            "Visual-contrast sampling found no repeated questions with different answers"
        )
    random.Random(seed).shuffle(selected)
    return selected, _report(strategy, samples, selected, contrast_groups)


def _report(
    strategy: SamplingStrategy,
    source: Sequence[SampleT],
    selected: Sequence[SampleT],
    contrast_groups: int,
) -> SamplingReport:
    return SamplingReport(
        strategy=strategy,
        source_samples=len(source),
        selected_samples=len(selected),
        selected_scenes=len(
            {str(getattr(sample, "scene_id")) for sample in selected}
        ),
        contrast_question_groups=contrast_groups,
    )
