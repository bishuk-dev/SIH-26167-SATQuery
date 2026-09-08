"""Strict loaders for the Phase 4 source, dataset, and model audit records.

Task 2 scope: validate the three audit YAML files and the temporal registry
schemas. Closeout validation belongs exclusively to Task 13 and must not be
added here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)

AUDIT_FILES = (
    "dataset_contracts.yaml",
    "model_contracts.yaml",
    "experiment_plan.yaml",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_ROOT = PROJECT_ROOT / "experiments" / "phase4_temporal_analytics"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA_LIKE = re.compile(r"\b[0-9a-f]{40,64}\b")
_EXPERIMENT_ID = re.compile(r"^P4-E0[1-4]$")
# Any numeric comparison threshold inside the agreement policy would turn a
# diagnostic metric into a routing decision, which Phase 4 forbids.
_NUMERIC_THRESHOLD = re.compile(r"(?:<|>|<=|>=)\s*\d*(?:\.\d+)?")

DatasetRole = Literal[
    "primary_benchmark",
    "operational_demonstration_only",
    "robustness_source_optional",
    "independent_external_holdout",
    "excluded_from_core",
]
ModelRole = Literal["primary", "optional_challenger"]
TemporalOrder = Literal[
    "T1_then_T2",
    "A_pre_then_B_post",
    "single_flood_observation",
]
FloodPolarizationOrder = Literal[("VV", "VH")]


def _as_tuple(value: object) -> object:
    # YAML has no tuple syntax; normalize sequences before strict validation.
    return tuple(value) if isinstance(value, list) else value


StrTuple = Annotated[tuple[str, ...], BeforeValidator(_as_tuple)]
IntTuple = Annotated[tuple[int, ...], BeforeValidator(_as_tuple)]
FloatTuple = Annotated[tuple[float, ...], BeforeValidator(_as_tuple)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AuditEntry(_Strict):
    status: Literal["PASS", "BLOCKED"]
    blockers: StrTuple = ()

    @property
    def runnable(self) -> bool:
        return self.status == "PASS" and not self.blockers

    @property
    def promotable(self) -> bool:
        return self.runnable


class AuthorityRecord(_Strict):
    title: str = Field(min_length=1)
    landing_url: str = Field(min_length=1)
    repository_url: str | None = None
    doi: str | None = None
    concept_doi: str | None = None
    doi_relationship: str | None = None
    version: str | None = None


class LicenseRecord(_Strict):
    # dataset package/annotation license and upstream imagery terms are
    # intentionally separate; one must never imply the other.
    dataset_license: str | None = None
    dataset_or_annotation_license: str | None = None
    imagery_terms: str | None = None
    redistribution_allowed: bool | str | None = None


class TransportRecord(_Strict):
    url: str | None = None
    mirror_role: str | None = None
    expected_files: StrTuple = ()
    expected_size_bytes: int | None = Field(default=None, gt=0)
    publisher_hashes: dict[str, str] | str | None = None
    locally_verified_sha256: str | None = None


class DatasetSemantics(_Strict):
    modalities: StrTuple
    sensors: StrTuple
    bands_or_polarizations: StrTuple
    radiometric_domain: str = Field(min_length=1)
    spatial_resolution_m: float | int | str | list[int | float] | None = None
    pair_order: str = Field(min_length=1)
    labels: str = Field(min_length=1)
    label_limitations: StrTuple = ()
    registration_claim: str | None = None
    patch_size_px: IntTuple | None = None
    chip_size_px: IntTuple | None = None
    tile_size_px: IntTuple | None = None
    pair_count: int | None = Field(default=None, gt=0)
    sentence_count: int | None = Field(default=None, gt=0)
    label_values: StrTuple | None = None
    label_instance_count: int | None = None
    label_semantics: str | None = None
    label_layer: str | None = None
    label_dtype: str | None = None
    band_order: StrTuple | None = None
    crs: str | None = None
    file_naming: str | None = None
    geospatial_supplement: str | None = None
    caption_schema: str | None = None
    image_normalization_reference: dict[str, Any] | None = None
    tile_counts: dict[str, int] | None = None
    layout: LayoutRecord | None = None
    splits: dict[str, Any] | None = None


class LayoutRecord(_Strict):
    note: str | None = None
    materialization_rule: str | None = None
    dataset_root: str | None = None
    sentinel1_composites: str | None = None
    sentinel1_masks: str | None = None
    sentinel1_metadata: str | None = None
    sentinel2_metadata: str | None = None


class CompatibilityGate(_Strict):
    rule: str = Field(min_length=1)


class SplitPolicy(_Strict):
    authority_defined: bool
    grouping_unit: str = Field(min_length=1)
    train: str | None = None
    validation: str | None = None
    test: str | None = None
    test_sealed: bool


class ReferenceRecord(_Strict):
    url: str = Field(min_length=1)
    supports: str = Field(min_length=1)


class DatasetContract(AuditEntry):
    role: DatasetRole
    authority: AuthorityRecord
    license: LicenseRecord
    transport: TransportRecord
    contract: DatasetSemantics
    splits: SplitPolicy
    layout: LayoutRecord | None = None
    compatibility_gate: CompatibilityGate | None = None
    decoupling_rule: str | None = None
    references: tuple[ReferenceRecord, ...] = ()

    @field_validator("references", mode="before")
    @classmethod
    def _normalize_references(cls, value: object) -> object:
        return _as_tuple(value)

    @property
    def promotable(self) -> bool:
        return self.runnable and self.role == "primary_benchmark"


class ModelSourceRecord(_Strict):
    repository_url: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    revision_verified: str | None = None
    license: str | None = None


class LicenseSeparationRecord(_Strict):
    repository_source_code_license: str | None = None
    checkpoint_provenance_and_license: str | None = None
    dataset_imagery_terms: str | None = None


class CheckpointRecord(_Strict):
    authority_url: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    size_bytes: int | None = Field(default=None, gt=0)
    publisher_hashes: dict[str, str] | None = None
    locally_verified_sha256: str | None = None
    repository_revision: str | None = None
    authority_relationship: str | None = None
    hash_provenance: str | None = None
    concept_doi: str | None = None
    doi_relationship: str | None = None
    inner_weights_path: str | None = None
    license: str | None = None
    md5_provenance: str | None = None


class ModelSemantics(_Strict):
    task: str = Field(min_length=1)
    training_domain: str = Field(min_length=1)
    sensor_domain: StrTuple
    band_order: StrTuple | str
    radiometric_domain: str = Field(min_length=1)
    input_shape: IntTuple | str | None = None
    output_semantics: str = Field(min_length=1)
    threshold_or_decoder: str = Field(min_length=1)
    temporal_order: str = Field(min_length=1)
    source_config: str = Field(min_length=1)
    source_config_revision: str | None = None
    data_preprocessor: dict[str, Any] | None = None
    test_transform: str | None = None
    config_chain: StrTuple | None = None
    feature_extractor: dict[str, Any] | None = None
    image_preprocessing: str | None = None
    vocabulary: str | None = None
    max_caption_length: int | None = None
    allow_unknown_token: int | None = None
    architecture: str | None = None
    preprocessing_pipeline: str | None = None
    normalization: str | None = None


class DependencyRecord(_Strict):
    python: str = Field(min_length=1)
    pytorch: str = Field(min_length=1)
    packages: StrTuple = ()
    operating_system: StrTuple | str = ()


class OfficialEvaluationRecord(_Strict):
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    command: str = Field(min_length=1)
    metrics: list[str] | str
    metric_implementations: str | None = None


class ModelContract(AuditEntry):
    role: ModelRole
    source: ModelSourceRecord
    checkpoint: CheckpointRecord
    contract: ModelSemantics
    dependencies: DependencyRecord
    official_evaluation: OfficialEvaluationRecord
    license_separation: LicenseSeparationRecord | None = None
    not_authorized_for_core_execution: bool | None = None
    isolation_rule: str | None = None
    dataset_interface: dict[str, Any] | None = None

    @property
    def promotable(self) -> bool:
        # an optional challenger never becomes a core production model
        return self.runnable and self.role == "primary"


class SampleSelection(_Strict):
    source: str = Field(min_length=1)
    development_split: str = Field(min_length=1)
    robustness_split: str | None = None
    sealed_splits: StrTuple = ()
    seed: int | None = None
    demonstration_samples: str | None = None


class LaneMethod(_Strict):
    id: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    authority_basis: str = Field(min_length=1)
    threshold_rule: str = Field(min_length=1)


class ValidationLane(_Strict):
    purpose: str = Field(min_length=1)
    metrics: StrTuple = ()
    benchmark_use: str | None = None
    sanities: StrTuple | None = None
    method: LaneMethod | None = None
    removal_condition: str | None = None


class PrimaryGate(_Strict):
    dataset: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class RobustnessLane(_Strict):
    dataset: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    status_reporting: str = Field(min_length=1)
    gate_effect: str = Field(min_length=1)
    tuning_rule: str = Field(min_length=1)


class ExperimentContract(_Strict):
    status: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    datasets: StrTuple
    model_or_tools: StrTuple
    sample_selection: SampleSelection
    preprocessing_profile_ids: StrTuple
    metrics: StrTuple
    sanities: StrTuple
    output_files: StrTuple
    stop_conditions: StrTuple
    blockers: StrTuple = ()
    blockers_primary: StrTuple = ()
    blockers_robustness: StrTuple = ()
    validation_lanes: dict[str, ValidationLane] | None = None
    primary_gate: PrimaryGate | None = None
    robustness_lane: RobustnessLane | None = None
    metric_scope_note: str | None = None
    metric_implementation_rule: str | None = None
    pass_condition: str | None = None
    pass_condition_primary: str | None = None
    pass_condition_robustness: str | None = None
    external_holdout_condition: str | None = None

    @field_validator(
        "metrics", "sanities", "output_files", "stop_conditions", mode="before"
    )
    @classmethod
    def _normalize_sequences(cls, value: object) -> object:
        return _as_tuple(value)


class SharedPolicy(_Strict):
    source_git_sha: str = Field(min_length=1)
    grouping_unit: str = Field(min_length=1)
    development_splits: StrTuple
    sealed_splits: StrTuple
    test_access: str = Field(min_length=1)
    threshold_tuning: str = Field(min_length=1)
    metric_rule: str = Field(min_length=1)
    zero_sample_rule: str = Field(min_length=1)
    output_rule: str = Field(min_length=1)
    agreement_policy: str = Field(min_length=1)


class ExperimentPlan(_Strict):
    schema_version: Literal[1]
    experiment_family: str = Field(min_length=1)
    status: str = Field(min_length=1)
    shared_policy: SharedPolicy
    experiments: dict[str, ExperimentContract]
    promotion: dict[str, str]


class DatasetAuditFile(_Strict):
    schema_version: Literal[1]
    audited_at: str = Field(min_length=1)
    status: str = Field(min_length=1)
    datasets: dict[str, DatasetContract]


class ModelAuditFile(_Strict):
    schema_version: Literal[1]
    audited_at: str = Field(min_length=1)
    status: str = Field(min_length=1)
    models: dict[str, ModelContract]


class Phase4ContractSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: dict[str, DatasetContract]
    models: dict[str, ModelContract]
    experiment_plan: ExperimentPlan


def load_phase4_contracts(
    root: str | Path = DEFAULT_AUDIT_ROOT,
) -> Phase4ContractSet:
    """Load and validate the three Phase 4 audit files without authorizing
    blocked entries."""
    root_path = Path(root)
    payloads = {name: _read_yaml(root_path / name) for name in AUDIT_FILES}

    dataset_file = DatasetAuditFile.model_validate(payloads["dataset_contracts.yaml"])
    model_file = ModelAuditFile.model_validate(payloads["model_contracts.yaml"])
    experiment_plan = ExperimentPlan.model_validate(payloads["experiment_plan.yaml"])

    _validate_status_invariants(dataset_file, model_file)
    _validate_cross_references(dataset_file, model_file, experiment_plan)

    return Phase4ContractSet(
        datasets=dataset_file.datasets,
        models=model_file.models,
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


def _validate_status_invariants(
    dataset_file: DatasetAuditFile, model_file: ModelAuditFile
) -> None:
    for name, entry in dataset_file.datasets.items():
        if entry.status == "BLOCKED" and not entry.blockers:
            raise ValueError(f"BLOCKED dataset {name} must list blockers")
        if entry.role == "excluded_from_core" and entry.status == "PASS":
            raise ValueError(f"excluded_from_core dataset {name} cannot be PASS")
        if entry.status == "PASS":
            _validate_pass_dataset(name, entry)
    for name, entry in model_file.models.items():
        if entry.status == "BLOCKED" and not entry.blockers:
            raise ValueError(f"BLOCKED model {name} must list blockers")
        if entry.status == "PASS":
            _validate_pass_model(name, entry)


def _validate_pass_dataset(name: str, entry: DatasetContract) -> None:
    package_license = (
        entry.license.dataset_license or entry.license.dataset_or_annotation_license
    )
    if not package_license or not entry.license.imagery_terms:
        raise ValueError(
            f"PASS dataset {name} requires a dataset package license and separate "
            "imagery terms"
        )
    if entry.authority.doi is None and entry.authority.version is None:
        raise ValueError(f"PASS dataset {name} requires versioned provenance")
    if entry.transport.url is None or (
        not entry.transport.expected_files
        and entry.transport.expected_size_bytes is None
    ):
        raise ValueError(f"PASS dataset {name} requires transport identity")
    if not entry.splits.authority_defined:
        raise ValueError(f"PASS dataset {name} requires an authoritative split policy")
    if not entry.contract.modalities or not entry.contract.sensors:
        raise ValueError(f"PASS dataset {name} requires modality/sensor semantics")


def _validate_pass_model(name: str, entry: ModelContract) -> None:
    revision = entry.source.revision
    if not _SHA_LIKE.search(revision):
        raise ValueError(f"PASS model {name} requires an immutable source revision")
    if not entry.source.license:
        raise ValueError(f"PASS model {name} requires a source-code license")
    checkpoint = entry.checkpoint
    if not _SHA256.fullmatch(checkpoint.locally_verified_sha256 or ""):
        raise ValueError(
            f"PASS model {name} requires a locally verified 64-hex SHA-256"
        )
    # a checkpoint transported separately from the source repository needs its
    # own resolved license; the source-code license never covers the checkpoint
    # bytes, and a publisher hash proves identity but not licensing
    if checkpoint.authority_url != entry.source.repository_url and not (
        checkpoint.license
    ):
        raise ValueError(
            f"PASS model {name} requires a resolved checkpoint license or "
            "publisher-hash provenance"
        )
    if entry.contract.input_shape is None or "unresolved" in str(
        entry.contract.input_shape
    ):
        raise ValueError(f"PASS model {name} requires a frozen input contract")
    if not any(
        (
            entry.contract.data_preprocessor,
            entry.contract.preprocessing_pipeline,
            entry.contract.image_preprocessing,
            entry.contract.normalization,
        )
    ):
        raise ValueError(f"PASS model {name} requires a preprocessing contract")
    if not entry.dependencies.packages and entry.dependencies.pytorch in (
        "unresolved",
        "",
    ):
        raise ValueError(f"PASS model {name} requires a dependency contract")
    if not entry.official_evaluation.metrics:
        raise ValueError(f"PASS model {name} requires official evaluation metrics")


def _validate_cross_references(
    dataset_file: DatasetAuditFile,
    model_file: ModelAuditFile,
    plan: ExperimentPlan,
) -> None:
    datasets = dataset_file.datasets
    models = model_file.models

    policy = plan.shared_policy
    if _NUMERIC_THRESHOLD.search(policy.agreement_policy):
        raise ValueError(
            "agreement policy must stay diagnostic: numeric ALLOW/WARN/ABSTAIN "
            "thresholds are forbidden in Phase 4"
        )
    if "test" not in policy.sealed_splits:
        raise ValueError("sealed split policy must keep the test split sealed")

    challenger_ids = {
        name for name, model in models.items() if model.role == "optional_challenger"
    }
    for experiment_id, experiment in plan.experiments.items():
        if not _EXPERIMENT_ID.match(experiment_id):
            raise ValueError(f"unknown experiment identity: {experiment_id}")
        if not experiment.stop_conditions or not experiment.output_files:
            raise ValueError(f"{experiment_id} needs stop conditions and output files")
        if not experiment.metrics or not experiment.sanities:
            raise ValueError(f"{experiment_id} needs metrics and sanities")

        for dataset_id in experiment.datasets:
            if dataset_id not in datasets:
                raise ValueError(
                    f"{experiment_id} references unknown dataset {dataset_id}"
                )
            dataset = datasets[dataset_id]
            if dataset.role == "excluded_from_core":
                raise ValueError(
                    f"{experiment_id} references excluded_from_core dataset "
                    f"{dataset_id}"
                )
            if not dataset.splits.test_sealed:
                raise ValueError(
                    f"{experiment_id} references dataset {dataset_id} whose test "
                    "split is not sealed"
                )

        for reference in experiment.model_or_tools:
            if reference in models:
                if reference in challenger_ids:
                    raise ValueError(
                        f"{experiment_id} promotes optional_challenger model "
                        f"{reference} into a core experiment"
                    )
            elif not reference.endswith("_v1"):
                raise ValueError(
                    f"{experiment_id} references unknown model/tool {reference}"
                )

        primary_roles = {"primary_benchmark", "operational_demonstration_only"}
        restricted_roles = {"robustness_source_optional", "independent_external_holdout"}
        experiment_roles = {datasets[d].role for d in experiment.datasets}
        if restricted_roles & experiment_roles and not primary_roles & experiment_roles:
            raise ValueError(
                f"{experiment_id} has no authoritative development source: "
                "robustness/holdout data cannot become the development source"
            )

        for dataset_id in experiment.datasets:
            if (
                datasets[dataset_id].role in restricted_roles
                and dataset_id in experiment.sample_selection.development_split
            ):
                raise ValueError(
                    f"{experiment_id} development split derives from restricted "
                    f"source {dataset_id}"
                )

        if experiment.primary_gate is not None:
            gate = experiment.primary_gate
            if gate.dataset not in datasets:
                raise ValueError(
                    f"{experiment_id} primary gate references unknown dataset "
                    f"{gate.dataset}"
                )
            if datasets[gate.dataset].role != "primary_benchmark":
                raise ValueError(
                    f"{experiment_id} primary gate must use a primary_benchmark "
                    "dataset"
                )
        if experiment.robustness_lane is not None:
            lane = experiment.robustness_lane
            if lane.dataset not in datasets:
                raise ValueError(
                    f"{experiment_id} robustness lane references unknown dataset "
                    f"{lane.dataset}"
                )
            if datasets[lane.dataset].role != "robustness_source_optional":
                raise ValueError(
                    f"{experiment_id} robustness lane must use a "
                    "robustness_source_optional dataset"
                )

        if experiment_id == "P4-E01":
            _validate_e01(experiment)
        elif experiment_id == "P4-E02":
            _validate_e02(experiment)
        elif experiment_id == "P4-E03":
            _validate_e03(experiment)
        elif experiment_id == "P4-E04":
            _validate_e04(experiment)


def _validate_e01(experiment: ExperimentContract) -> None:
    lanes = experiment.validation_lanes
    if not lanes:
        raise ValueError("P4-E01 must separate formula/grid/area validation lanes")
    for lane_id in (
        "A_spectral_formula_correctness",
        "B_temporal_grid_correctness",
        "C_deterministic_area_correctness",
    ):
        if lane_id not in lanes:
            raise ValueError(f"P4-E01 missing required validation lane {lane_id}")
    if experiment.metric_scope_note is None:
        raise ValueError(
            "P4-E01 must scope segmentation metrics away from index differences"
        )


def _validate_e02(experiment: ExperimentContract) -> None:
    if experiment.primary_gate is None or experiment.robustness_lane is None:
        raise ValueError(
            "P4-E02 must declare the LEVIR-CD primary gate and the separate, "
            "non-gating S2Looking robustness lane"
        )
    if not experiment.pass_condition_primary:
        raise ValueError("P4-E02 must declare the primary pass condition")


def _validate_e03(experiment: ExperimentContract) -> None:
    metrics = set(experiment.metrics)
    required = {"BLEU_4", "METEOR", "ROUGE_L", "CIDEr"}
    if not required.issubset(metrics):
        raise ValueError(
            f"P4-E03 must declare the official caption metrics {sorted(required)}"
        )


def _validate_e04(experiment: ExperimentContract) -> None:
    if experiment.external_holdout_condition != (
        "conditional_on_audited_input_compatibility"
    ):
        raise ValueError(
            "P4-E04 external holdout must remain conditional on audited input "
            "compatibility"
        )
