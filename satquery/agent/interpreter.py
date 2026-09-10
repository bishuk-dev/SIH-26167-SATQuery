"""Deterministic, non-executable query interpretation rules."""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from satquery.agent.models import QueryIntent


class QueryInterpreter(Protocol):
    """Protocol implemented by query interpreters used by the planner."""

    def interpret(self, query: str) -> QueryIntent:
        ...


_WORD_RE = lambda *terms: re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"
)

# Keep these rules code-owned and finite.  A query can select data, never a tool.
_INJECTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous\s+instructions?"
    r"|disregard\s+(?:the\s+)?system\s+prompt"
    r"|jailbreak|developer\s+message|system\s+message"
    r"|(?:run|execute|invoke|call)\s+(?:a\s+)?(?:tool|function|command)"
    r"|<\s*(?:system|developer|tool)\b)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|never|without|avoid|not)\b",
    re.IGNORECASE,
)
_BANK_RE = _WORD_RE("bank")
_CHANGE_RE = _WORD_RE(
    "change", "changed", "changes", "difference", "differ", "before", "after",
    "temporal", "between", "over time",
)
_WATER_RE = _WORD_RE("water", "flood", "flooding", "wetland")
_GAIN_RE = _WORD_RE("gain", "gained", "expand", "expanded", "increase", "increased", "appear", "appeared")
_LOSS_RE = _WORD_RE("loss", "lost", "shrink", "shrunk", "decrease", "decreased", "disappear", "disappeared")
_GROUND_RE = _WORD_RE("where", "locate", "location", "find", "map", "outline", "object", "region")
_METADATA_RE = _WORD_RE(
    "metadata", "crs", "projection", "resolution", "gsd", "sensor", "platform",
    "acquisition", "timestamp", "date", "bands", "polarization",
)
_CAPABILITY_RE = re.compile(
    r"\b(?:what|which|can|does)\b[^?\n]{0,80}\b(?:capabilit(?:y|ies)|support(?:ed)?|available|tools?|models?)\b"
    r"|\b(?:list|show)\b[^?\n]{0,40}\b(?:capabilit(?:y|ies)|tools?|models?)\b",
    re.IGNORECASE,
)
_SAR_RE = _WORD_RE("sar", "radar", "sentinel-1", "sentinel 1", "vv", "vh", "hh", "hv")
_OPTICAL_RE = _WORD_RE("optical", "multispectral", "rgb", "sentinel-2", "sentinel 2")
_DESCRIPTION_RE = re.compile(
    r"\b(?:describe|description|caption|summari[sz]e)\b.*\b(?:change|changed|difference|before|after)\b"
    r"|\bwhat\s+(?:has\s+)?changed\b",
    re.IGNORECASE,
)
_VQA_RE = _WORD_RE(
    "what", "which", "is", "are", "does", "do", "how", "identify", "tell", "answer",
)
_AREA_RE = re.compile(
    r"\b(?:area|hectares?|ha|square\s+(?:meters?|metres?|kilometers?|kilometres?)|m2|km2)\b",
    re.IGNORECASE,
)
_INDEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ndvi", _WORD_RE("ndvi")),
    ("ndwi", _WORD_RE("ndwi")),
    ("mndwi", _WORD_RE("mndwi")),
)


def _direction(text: str, *, temporal: bool) -> str | None:
    if not temporal:
        return None
    if re.search(r"\b(?:t2|second|after|post)\b.{0,20}\b(?:to|vs|versus)\b.{0,20}\b(?:t1|first|before|pre)\b", text):
        return "T2_TO_T1"
    if re.search(r"\b(?:t1|first|before|pre)\b.{0,20}\b(?:to|vs|versus)\b.{0,20}\b(?:t2|second|after|post)\b", text):
        return "T1_TO_T2"
    if _LOSS_RE.search(text):
        return "T2_TO_T1"
    if _GAIN_RE.search(text):
        return "T1_TO_T2"
    return "UNKNOWN"


def _target(text: str) -> str | None:
    if _SAR_RE.search(text) and _OPTICAL_RE.search(text):
        return "sar"
    if _WATER_RE.search(text):
        return "water"
    for name, pattern in _INDEX_PATTERNS:
        if pattern.search(text):
            return name
    if _SAR_RE.search(text) and not _OPTICAL_RE.search(text):
        return "sar"
    if _OPTICAL_RE.search(text) and not _SAR_RE.search(text):
        return "optical"
    for name in ("building", "road", "vehicle", "forest", "vegetation"):
        if _WORD_RE(name, name + "s").search(text):
            return name
    return None


class DeterministicQueryInterpreter:
    """Interpret a bounded query using ordered, compiled lexical rules."""

    def __init__(
        self,
        *,
        max_query_characters: int | None = None,
        settings: object | None = None,
    ) -> None:
        configured_limit = None
        if settings is not None:
            configured_limit = getattr(settings, "max_query_characters", None)
            if configured_limit is None:
                configured_limit = getattr(settings, "max_question_characters", None)
        if max_query_characters is not None:
            configured_limit = max_query_characters
        self.max_query_characters = configured_limit if configured_limit is not None else 500
        if not isinstance(self.max_query_characters, int) or isinstance(self.max_query_characters, bool) or self.max_query_characters < 1:
            raise ValueError("max_query_characters must be a positive integer")

    def interpret(self, query: str) -> QueryIntent:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        text = unicodedata.normalize("NFKC", query).casefold().strip()
        if not text:
            return QueryIntent(
                task_family="AMBIGUOUS",
                matched_rule="empty_query",
                ambiguities=("query is empty",),
            )
        if len(text) > self.max_query_characters:
            return QueryIntent(
                task_family="AMBIGUOUS",
                matched_rule="query_length_limit",
                ambiguities=("query exceeds the maximum allowed length",),
            )
        if _INJECTION_RE.search(text) or re.search(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b", text):
            return QueryIntent(
                task_family="AMBIGUOUS",
                matched_rule="prompt_injection",
                ambiguities=("prompt-injection-like instruction",),
            )
        if _NEGATION_RE.search(text):
            return QueryIntent(
                task_family="AMBIGUOUS",
                target_semantic=_target(text),
                matched_rule="negation",
                ambiguities=("negated requests are not executable analysis intents",),
            )
        if _BANK_RE.search(text):
            return QueryIntent(
                task_family="AMBIGUOUS",
                target_semantic="bank",
                spatial_request=bool(_GROUND_RE.search(text)),
                matched_rule="ambiguous_bank",
                ambiguities=("bank may mean a riverbank or a financial institution",),
            )

        temporal = bool(
            _CHANGE_RE.search(text)
            or _GAIN_RE.search(text)
            or _LOSS_RE.search(text)
            or re.search(r"\b(?:t1|t2|first|second)\b", text)
        )
        measurement: str | None = None
        measurement_rule: str | None = None
        for name, pattern in _INDEX_PATTERNS:
            if pattern.search(text):
                measurement, measurement_rule = name, f"index_{name}"
                break
        if measurement is None and _AREA_RE.search(text):
            measurement, measurement_rule = "area", "area_measurement"

        # A single bounded intent must not silently choose between incompatible
        # workflows.  Measurement plus a caption/metadata/capability request is
        # a conflict, while measurement plus temporal language is one workflow.
        if measurement is not None and (
            _DESCRIPTION_RE.search(text)
            or _METADATA_RE.search(text)
            or _CAPABILITY_RE.search(text)
        ):
            return QueryIntent(
                task_family="AMBIGUOUS",
                target_semantic=_target(text),
                requested_measurement=measurement,
                temporal_direction=_direction(text, temporal=temporal),
                spatial_request=True,
                matched_rule="conflicting_intents",
                ambiguities=("measurement conflicts with another requested workflow",),
            )

        if measurement is not None:
            target = _target(text)
            return QueryIntent(
                task_family="CHANGE_MEASURE" if temporal else "MEASURE",
                target_semantic=target or ("vegetation" if measurement in {"ndvi", "ndwi", "mndwi"} else None),
                requested_measurement=measurement,
                temporal_direction=_direction(text, temporal=temporal),
                spatial_request=True,
                matched_rule=measurement_rule or "measurement",
            )

        if _METADATA_RE.search(text):
            return QueryIntent(
                task_family="METADATA_QUERY",
                target_semantic=_target(text),
                matched_rule="metadata_question",
            )
        if _SAR_RE.search(text) and _OPTICAL_RE.search(text):
            return QueryIntent(
                task_family="CROSS_MODAL_VQA",
                target_semantic=_target(text),
                temporal_direction=_direction(text, temporal=temporal),
                matched_rule="optical_sar_comparison",
            )
        if _CAPABILITY_RE.search(text):
            return QueryIntent(task_family="CAPABILITY_QUERY", matched_rule="capability_question")

        if temporal and _DESCRIPTION_RE.search(text) and not re.search(r"^what\s+(?:has\s+)?changed\b", text):
            return QueryIntent(
                task_family="CHANGE_DESCRIPTION",
                target_semantic=_target(text),
                temporal_direction=_direction(text, temporal=True),
                matched_rule="change_description",
            )
        if temporal and _WATER_RE.search(text) and (_GAIN_RE.search(text) or _LOSS_RE.search(text)):
            return QueryIntent(
                task_family="CHANGE_LOCALIZE",
                target_semantic="water",
                temporal_direction=_direction(text, temporal=True),
                spatial_request=True,
                matched_rule="water_change_localization",
            )
        if temporal:
            return QueryIntent(
                task_family="CHANGE_VQA",
                target_semantic=_target(text),
                temporal_direction=_direction(text, temporal=True),
                spatial_request=bool(_GROUND_RE.search(text)),
                matched_rule="temporal_change_question",
            )
        if not temporal and _GROUND_RE.search(text):
            return QueryIntent(
                task_family="GROUND_OBJECT",
                target_semantic=_target(text),
                spatial_request=True,
                matched_rule="grounding_request",
            )
        if _VQA_RE.search(text):
            return QueryIntent(
                task_family="SINGLE_VQA",
                target_semantic=_target(text),
                matched_rule="single_image_question",
            )
        return QueryIntent(
            task_family="AMBIGUOUS",
            matched_rule="no_matching_rule",
            ambiguities=("query does not identify a supported analysis intent",),
        )


__all__ = ["DeterministicQueryInterpreter", "QueryInterpreter"]
