"""Authoritative RSICC caption evaluation metrics package.

Vendored from Chen-Yang-Liu/RSICC (commit d1505e514c450c3728782ca723e82761e70bafd3).
Based on MS-COCO Caption Evaluation (Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, 2015).
License: Permissive BSD/MIT (see bleu_LICENSE).

Provides corpus-level evaluation for BLEU-1..4, ROUGE-L, and CIDEr.
"""

from __future__ import annotations

import re
from typing import Any

from .bleu import Bleu
from .cider import Cider
from .rouge import Rouge

RSICC_PROVENANCE = {
    "source_repository": "https://github.com/Chen-Yang-Liu/RSICC",
    "commit": "d1505e514c450c3728782ca723e82761e70bafd3",
    "upstream_project": "MS-COCO Caption Evaluation (Chen et al., 2015)",
    "license": "BSD-3-Clause / MIT permissive",
    "aggregation": "corpus_level",
}


def _clean_token_string(text: str) -> str:
    """Normalize tokens using standard whitespace splitting matching RSICC."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower().strip())
    tokens = [t for t in cleaned.split() if t]
    return " ".join(tokens)


def compute_rsicc_caption_metrics(
    references: list[list[str]],
    hypotheses: list[str],
) -> dict[str, Any]:
    """Compute corpus-level BLEU-1..4, ROUGE-L, and CIDEr using official RSICC modules.

    Args:
        references: List of reference lists, one list of reference strings per image.
        hypotheses: List of candidate/generated caption strings, one per image.

    Returns:
        Dictionary containing corpus-level scores and evaluation provenance.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"References count ({len(references)}) does not match hypotheses count ({len(hypotheses)})"
        )
    if not hypotheses:
        return {
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "rouge_l": 0.0,
            "cider": 0.0,
            "provenance": RSICC_PROVENANCE,
        }

    # Format into gts (dict/list of lists) and res (dict/list of lists containing 1 hypothesis)
    gts = []
    res = []
    for i in range(len(hypotheses)):
        hyp_str = _clean_token_string(hypotheses[i])
        res.append([hyp_str if hyp_str else "empty"])
        ref_strs = [_clean_token_string(r) for r in references[i] if r.strip()]
        if not ref_strs:
            ref_strs = ["empty"]
        gts.append(ref_strs)

    # 1. BLEU
    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(gts, res)

    # 2. ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)

    # 3. CIDEr
    cider_scorer = Cider(n=4, sigma=6.0)
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return {
        "bleu_1": round(float(bleu_score[0]), 6),
        "bleu_2": round(float(bleu_score[1]), 6),
        "bleu_3": round(float(bleu_score[2]), 6),
        "bleu_4": round(float(bleu_score[3]), 6),
        "rouge_l": round(float(rouge_score), 6),
        "cider": round(float(cider_score), 6),
        "provenance": RSICC_PROVENANCE,
    }


__all__ = [
    "Bleu",
    "Cider",
    "Rouge",
    "compute_rsicc_caption_metrics",
    "RSICC_PROVENANCE",
]
