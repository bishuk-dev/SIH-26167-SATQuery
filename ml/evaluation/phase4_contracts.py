"""Strict loaders for the Phase 4 source and model audit records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

AUDIT_FILES = (
    "dataset_contracts.yaml",
    "model_contracts.yaml",
    "experiment_plan.yaml",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    status: Literal["PASS", "BLOCKED"]
    role: str = Field(min_length=1)
    blockers: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.status == "PASS" and not self.blockers


class DatasetSemantics(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    modalities: tuple[str, ...]
    sensors: tuple[str, ...]
    bands_or_polarizations: tuple[str, ...]
    radiometric_domain: str
    pair_order: str
    labels: str

    @field_validator("modalities", "sensors", "bands_or_polarizations", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class DatasetContract(AuditEntry):
    authority: dict[str, Any] | None = None
    license: dict[str, Any] | None = None
    transport: dict[str, Any] | None = None
    contract: DatasetSemantics | None = None
    splits: dict[str, Any] | None = None
    references: tuple[dict[str, Any], ...] = ()


class AuditFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    audited_at: str | None = None
    status: str = Field(min_length=1)
    entries: dict[str, AuditEntry]


class Phase4ContractSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: dict[str, DatasetContract]
    models: dict[str, AuditEntry]
    experiment_plan: dict[str, Any]


def load_phase4_contracts(root: str | Path) -> Phase4ContractSet:
    """Load and validate Phase 4 audit files without authorizing blocked entries."""

    root_path = Path(root)
    payloads = {
        name: _read_yaml(root_path / name)
        for name in AUDIT_FILES
    }
    dataset_file = _parse_audit_file(
        payloads["dataset_contracts.yaml"], "datasets", DatasetContract
    )
    model_file = _parse_audit_file(payloads["model_contracts.yaml"], "models")
    experiment_plan = _parse_experiment_plan(payloads["experiment_plan.yaml"])

    for name, entry in dataset_file.entries.items():
        _validate_entry(name, entry, payloads["dataset_contracts.yaml"], kind="dataset")
    for name, entry in model_file.entries.items():
        _validate_entry(name, entry, payloads["model_contracts.yaml"], kind="model")

    return Phase4ContractSet(
        datasets=dataset_file.entries,
        models=model_file.entries,
        experiment_plan=experiment_plan,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read Phase 4 audit file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 4 audit file must contain an object: {path}")
    return payload


def _parse_audit_file(
    payload: dict[str, Any],
    field: str,
    entry_type: type[AuditEntry] = AuditEntry,
) -> AuditFile:
    if field not in payload:
        raise ValueError(f"Phase 4 audit file missing {field}")
    entries = payload[field]
    if not isinstance(entries, dict):
        raise ValueError(f"Phase 4 audit field {field} must be a mapping")
    try:
        normalized = {key: value for key, value in payload.items() if key != field}
        normalized["entries"] = entries
        parsed = AuditFile.model_validate(normalized)
        typed_entries = {
            key: entry_type.model_validate(value) for key, value in entries.items()
        }
        return parsed.model_copy(update={"entries": typed_entries})
    except ValidationError as exc:
        raise ValueError(f"Invalid Phase 4 {field} audit: {exc}") from exc


def _parse_experiment_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"schema_version", "experiment_family", "status", "shared_policy", "experiments", "promotion"}:
        raise ValueError("Invalid Phase 4 experiment plan keys")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("experiments"), dict):
        raise ValueError("Invalid Phase 4 experiment plan")
    return payload


def _validate_entry(
    name: str,
    entry: AuditEntry,
    payload: dict[str, Any],
    *,
    kind: Literal["dataset", "model"],
) -> None:
    raw = payload["datasets" if kind == "dataset" else "models"][name]
    if entry.status == "BLOCKED":
        if not entry.blockers:
            raise ValueError(f"Blocked {kind} {name} must list blockers")
        return

    required_sections = ("authority", "license", "transport", "contract", "splits")
    if kind == "model":
        required_sections = (
            "source",
            "checkpoint",
            "contract",
            "dependencies",
            "official_evaluation",
        )
    missing = [section for section in required_sections if not isinstance(raw.get(section), dict)]
    if missing:
        raise ValueError(f"PASS {kind} {name} missing sections: {', '.join(missing)}")
    if entry.blockers:
        raise ValueError(f"PASS {kind} {name} cannot list blockers")

    if kind == "model":
        source = raw["source"]
        checkpoint = raw["checkpoint"]
        if not source.get("license"):
            raise ValueError(f"PASS model {name} requires a license")
        digest = checkpoint.get("locally_verified_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"PASS model {name} requires a verified SHA-256")
        size = checkpoint.get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"PASS model {name} requires a verified checkpoint size")
