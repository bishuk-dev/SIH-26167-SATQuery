"""Prepare the immutable Phase 4 BigEarthNet v2 metadata-only subset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_ROOT = PROJECT_ROOT / "data/metadata/bigearthnet_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/phase4_bigearthnet_multisensor"
DEFAULT_CLEAN_METADATA = DEFAULT_METADATA_ROOT / "metadata.parquet"
DEFAULT_EXCLUDED_METADATA = (
    DEFAULT_METADATA_ROOT / "metadata_for_patches_with_snow_cloud_or_shadow.parquet"
)

DATASET_NAME = "BigEarthNet v2"
DATASET_VERSION = "2.0.0"
DATASET_DOI = "10.5281/zenodo.10891137"
DATASET_LICENSE = "CDLA-Permissive-1.0"
EXPECTED_TOTAL_PAIRS = 549_488

CLEAN_METADATA_SHA256 = (
    "408911df2da7092da9ecc72071972a808ec486ba09f6cb048f7716793d14ded6"
)
EXCLUDED_METADATA_SHA256 = (
    "b6842b35359dfb5281dd92c674211fd4882f7865f0b442ebfec92daea6371c4e"
)
CLEAN_METADATA_MD5 = "55687065e77b6d0b0f1ff604a6e7b49c"
EXCLUDED_METADATA_MD5 = "fe31856f4986d446c9468b59d6387c91"

OFFICIAL_SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = (
    "patch_id",
    "labels",
    "split",
    "country",
    "s1_name",
    "s2v1_name",
    "contains_seasonal_snow",
    "contains_cloud_or_shadow",
)
OFFICIAL_LABELS = (
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
)
S1_BANDS = ("VV", "VH")
S2_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

GEOGRAPHIC_SUFFIX = re.compile(r"_(T\d{2}[A-Z]{3})_(\d+)_(\d+)$")
SCORE_SCALE = 1_000_000_000


class Phase4PreparationError(RuntimeError):
    """Raised when source integrity or frozen selection constraints fail."""


@dataclass(frozen=True, slots=True)
class MetadataRow:
    """Validated metadata for one official S1/S2 pair."""

    patch_id: str
    labels: tuple[str, ...]
    split: str
    country: str
    s1_name: str
    s2v1_name: str
    contains_seasonal_snow: bool
    contains_cloud_or_shadow: bool
    geographic_group_id: str


@dataclass(frozen=True, slots=True)
class GeographicGroup:
    """Indivisible repeat acquisitions of a 1200 m geographic patch cell."""

    group_id: str
    split: str
    country: str
    rows: tuple[MetadataRow, ...]
    label_counts: tuple[tuple[str, int], ...]

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Frozen, deterministic subset selection configuration."""

    seed_material: str = "satquery-phase4-bigearthnet-v2-croma-v1"
    algorithm_version: str = "ben-v2-grouped-multilabel-greedy-v1"
    train_target: int = 12_000
    validation_target: int = 3_000
    test_target: int = 3_000
    train_label_floor: int = 25
    validation_label_floor: int = 10
    test_label_floor: int = 10
    label_dimension_weight: int = 1
    country_dimension_weight: int = 1

    def target_for(self, split: str) -> int:
        return {
            "train": self.train_target,
            "validation": self.validation_target,
            "test": self.test_target,
        }[split]

    def label_floor_for(self, split: str) -> int:
        return {
            "train": self.train_label_floor,
            "validation": self.validation_label_floor,
            "test": self.test_label_floor,
        }[split]

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "country_dimension_weight": self.country_dimension_weight,
            "label_dimension_weight": self.label_dimension_weight,
            "label_floors": {
                "test": self.test_label_floor,
                "train": self.train_label_floor,
                "validation": self.validation_label_floor,
            },
            "score_scale": SCORE_SCALE,
            "seed_material": self.seed_material,
            "targets": {
                "test": self.test_target,
                "train": self.train_target,
                "validation": self.validation_target,
            },
        }


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """Paths and identities emitted by a successful preparation run."""

    manifest_path: Path
    materialization_plan_path: Path
    report_path: Path
    manifest_sha256: str
    selection_config_sha256: str
    counts: Mapping[str, Mapping[str, int]]


def file_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize stable UTF-8 JSON used by manifest and resume identities."""
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_metadata(
    path: Path,
    expected_sha256: str,
    *,
    expected_clean: bool,
) -> list[MetadataRow]:
    """Checksum, read, and strictly validate one official metadata Parquet."""
    _validate_sha256_argument(expected_sha256)
    if not path.is_file():
        raise Phase4PreparationError(f"Metadata file does not exist: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise Phase4PreparationError(
            f"Metadata SHA-256 mismatch for {path.name}: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency message only
        raise Phase4PreparationError(
            "PyArrow is required; install the project with the multisensor extra"
        ) from exc

    try:
        parquet_file = parquet.ParquetFile(path)
    except Exception as exc:
        raise Phase4PreparationError(
            f"Cannot read metadata Parquet {path.name}: {exc}"
        ) from exc
    columns = set(parquet_file.schema_arrow.names)
    missing = sorted(set(REQUIRED_COLUMNS) - columns)
    if missing:
        raise Phase4PreparationError(
            f"Metadata file {path.name} is missing required columns: {missing}"
        )

    table = parquet_file.read(columns=list(REQUIRED_COLUMNS))
    return _validate_rows(
        table.to_pylist(),
        source_name=path.name,
        expected_clean=expected_clean,
    )


def prepare_phase4_bigearthnet(
    *,
    clean_metadata_path: Path,
    excluded_metadata_path: Path,
    clean_sha256: str,
    excluded_sha256: str,
    output_dir: Path,
    config: SelectionConfig = SelectionConfig(),
    expected_total_pairs: int | None = EXPECTED_TOTAL_PAIRS,
) -> PreparationResult:
    """Create immutable selection, materialization, and audit artifacts."""
    clean_rows = load_metadata(
        clean_metadata_path,
        clean_sha256,
        expected_clean=True,
    )
    excluded_rows = load_metadata(
        excluded_metadata_path,
        excluded_sha256,
        expected_clean=False,
    )
    _validate_source_sets(clean_rows, excluded_rows, expected_total_pairs)

    groups = build_geographic_groups(clean_rows)
    selected: dict[str, list[MetadataRow]] = {}
    selected_groups: dict[str, list[GeographicGroup]] = {}
    selection_diagnostics: dict[str, dict[str, object]] = {}
    for split in OFFICIAL_SPLITS:
        split_groups = [group for group in groups if group.split == split]
        chosen_groups, diagnostics = select_groups(
            split_groups,
            target=config.target_for(split),
            label_floor=config.label_floor_for(split),
            seed_material=config.seed_material,
            label_dimension_weight=config.label_dimension_weight,
            country_dimension_weight=config.country_dimension_weight,
        )
        selected_groups[split] = chosen_groups
        selected[split] = sorted(
            (row for group in chosen_groups for row in group.rows),
            key=lambda row: row.patch_id,
        )
        selection_diagnostics[split] = diagnostics

    _validate_selected_rows(selected)
    config_bytes = canonical_json_bytes(config.as_dict())
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    manifest = _build_manifest(
        selected,
        selected_groups,
        clean_sha256=clean_sha256.lower(),
        excluded_sha256=excluded_sha256.lower(),
        config=config,
        config_sha256=config_sha256,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    materialization_plan = _build_materialization_plan(
        selected,
        manifest_sha256=manifest_sha256,
    )
    report = _build_report(
        clean_rows,
        excluded_rows,
        selected,
        selected_groups,
        clean_sha256=clean_sha256.lower(),
        excluded_sha256=excluded_sha256.lower(),
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        config=config,
        selection_diagnostics=selection_diagnostics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    _write_immutable_manifest(manifest_path, manifest_bytes)
    materialization_path = output_dir / "materialization_plan.json"
    report_path = output_dir / "preparation_report.json"
    materialization_path.write_bytes(canonical_json_bytes(materialization_plan))
    report_path.write_bytes(canonical_json_bytes(report))

    return PreparationResult(
        manifest_path=manifest_path,
        materialization_plan_path=materialization_path,
        report_path=report_path,
        manifest_sha256=manifest_sha256,
        selection_config_sha256=config_sha256,
        counts=manifest["counts"],
    )


def build_geographic_groups(rows: Sequence[MetadataRow]) -> list[GeographicGroup]:
    """Build and validate split-safe geographic groups."""
    grouped: dict[str, list[MetadataRow]] = defaultdict(list)
    for row in rows:
        grouped[row.geographic_group_id].append(row)

    result: list[GeographicGroup] = []
    for group_id in sorted(grouped):
        members = sorted(grouped[group_id], key=lambda row: row.patch_id)
        splits = {row.split for row in members}
        countries = {row.country for row in members}
        if len(splits) != 1:
            raise Phase4PreparationError(
                f"Geographic group crosses official splits: {group_id}"
            )
        if len(countries) != 1:
            raise Phase4PreparationError(
                f"Geographic group crosses countries: {group_id}"
            )
        label_counts = Counter(label for member in members for label in member.labels)
        result.append(
            GeographicGroup(
                group_id=group_id,
                split=next(iter(splits)),
                country=next(iter(countries)),
                rows=tuple(members),
                label_counts=tuple(sorted(label_counts.items())),
            )
        )
    return result


def select_groups(
    groups: Sequence[GeographicGroup],
    *,
    target: int,
    label_floor: int,
    seed_material: str,
    label_dimension_weight: int = 1,
    country_dimension_weight: int = 1,
) -> tuple[list[GeographicGroup], dict[str, object]]:
    """Select indivisible groups with deterministic lazy-greedy coverage."""
    if target < 1:
        raise ValueError("Selection target must be positive")
    if label_floor < 0:
        raise ValueError("Label floor cannot be negative")
    if not groups:
        raise Phase4PreparationError("No eligible geographic groups are available")
    split_values = {group.split for group in groups}
    if len(split_values) != 1:
        raise Phase4PreparationError("Selector input mixes official splits")
    split = next(iter(split_values))
    source_rows = sum(group.size for group in groups)
    if source_rows < target:
        raise Phase4PreparationError(
            f"Official {split} split has {source_rows} eligible rows, below target {target}"
        )

    source_label_counts: Counter[str] = Counter()
    source_country_counts: Counter[str] = Counter()
    for group in groups:
        source_label_counts.update(dict(group.label_counts))
        source_country_counts[group.country] += group.size
    for label in OFFICIAL_LABELS:
        source_label_counts.setdefault(label, 0)
    missing_labels = [
        label for label, count in source_label_counts.items() if count == 0
    ]
    if missing_labels:
        raise Phase4PreparationError(
            f"Official {split} split is missing expected labels: {missing_labels}"
        )

    label_targets = {
        label: max(
            label_floor,
            _proportional_target(count, target, source_rows),
        )
        for label, count in source_label_counts.items()
    }
    impossible = {
        label: requested
        for label, requested in label_targets.items()
        if requested > source_label_counts[label]
    }
    if impossible:
        raise Phase4PreparationError(
            f"Label coverage targets exceed source availability in {split}: {impossible}"
        )
    country_targets = {
        country: max(1, _proportional_target(count, target, source_rows))
        for country, count in source_country_counts.items()
    }
    label_weights = {
        label: max(1, SCORE_SCALE // count)
        for label, count in source_label_counts.items()
    }
    country_weights = {
        country: max(1, SCORE_SCALE // count)
        for country, count in source_country_counts.items()
    }

    selected_label_counts: Counter[str] = Counter()
    selected_country_counts: Counter[str] = Counter()
    selected: list[GeographicGroup] = []
    selected_rows = 0
    heap: list[tuple[int, str, str, GeographicGroup]] = []

    for group in groups:
        score = _coverage_score(
            group,
            selected_label_counts,
            selected_country_counts,
            label_targets,
            country_targets,
            label_weights,
            country_weights,
            label_dimension_weight,
            country_dimension_weight,
        )
        tie_break = _group_hash(seed_material, split, group.group_id)
        heapq.heappush(heap, (-score, tie_break, group.group_id, group))

    while selected_rows < target:
        if not heap:
            raise Phase4PreparationError(
                f"Exhausted {split} groups before reaching target {target}"
            )
        _, tie_break, group_id, group = heapq.heappop(heap)
        current_score = _coverage_score(
            group,
            selected_label_counts,
            selected_country_counts,
            label_targets,
            country_targets,
            label_weights,
            country_weights,
            label_dimension_weight,
            country_dimension_weight,
        )
        current_priority = (-current_score, tie_break, group_id)
        if heap and current_priority > heap[0][:3]:
            heapq.heappush(heap, (*current_priority, group))
            continue

        selected.append(group)
        selected_rows += group.size
        selected_country_counts[group.country] += group.size
        selected_label_counts.update(dict(group.label_counts))

    underrepresented = {
        label: {
            "actual": selected_label_counts[label],
            "minimum": label_floor,
        }
        for label in OFFICIAL_LABELS
        if selected_label_counts[label] < label_floor
    }
    missing_countries = sorted(
        country
        for country in source_country_counts
        if selected_country_counts[country] == 0
    )
    if underrepresented or missing_countries:
        raise Phase4PreparationError(
            f"Coverage constraints failed in {split}: "
            f"labels={underrepresented}, countries={missing_countries}"
        )

    selected.sort(key=lambda group: group.group_id)
    return selected, {
        "actual_pairs": selected_rows,
        "country_targets": dict(sorted(country_targets.items())),
        "label_targets": dict(sorted(label_targets.items())),
        "requested_pairs": target,
        "source_pairs": source_rows,
    }


def _validate_rows(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
    expected_clean: bool,
) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    patch_ids: set[str] = set()
    s1_names: set[str] = set()
    for index, raw in enumerate(raw_rows):
        context = f"{source_name} row {index}"
        patch_id = _required_text(raw.get("patch_id"), "patch_id", context)
        s1_name = _required_text(raw.get("s1_name"), "s1_name", context)
        s2v1_name = _required_text(raw.get("s2v1_name"), "s2v1_name", context)
        split = _required_text(raw.get("split"), "split", context)
        country = _required_text(raw.get("country"), "country", context)
        if split not in OFFICIAL_SPLITS:
            raise Phase4PreparationError(
                f"{context} has unsupported official split {split!r}"
            )
        if patch_id in patch_ids:
            raise Phase4PreparationError(
                f"Duplicate patch_id in {source_name}: {patch_id}"
            )
        if s1_name in s1_names:
            raise Phase4PreparationError(
                f"Duplicate s1_name pairing in {source_name}: {s1_name}"
            )
        patch_ids.add(patch_id)
        s1_names.add(s1_name)

        raw_labels = raw.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise Phase4PreparationError(f"{context} has no class labels")
        if not all(isinstance(label, str) and label.strip() for label in raw_labels):
            raise Phase4PreparationError(f"{context} has invalid class labels")
        labels = tuple(sorted(set(raw_labels)))
        unknown_labels = sorted(set(labels) - set(OFFICIAL_LABELS))
        if unknown_labels:
            raise Phase4PreparationError(
                f"{context} contains unknown labels: {unknown_labels}"
            )
        seasonal_snow = raw.get("contains_seasonal_snow")
        cloud_or_shadow = raw.get("contains_cloud_or_shadow")
        if not isinstance(seasonal_snow, bool) or not isinstance(cloud_or_shadow, bool):
            raise Phase4PreparationError(f"{context} has invalid quality flags")
        if expected_clean and (seasonal_snow or cloud_or_shadow):
            raise Phase4PreparationError(
                f"Clean metadata contains a flagged patch: {patch_id}"
            )
        if not expected_clean and not (seasonal_snow or cloud_or_shadow):
            raise Phase4PreparationError(
                f"Excluded metadata contains an unflagged patch: {patch_id}"
            )

        match = GEOGRAPHIC_SUFFIX.search(patch_id)
        if match is None:
            raise Phase4PreparationError(
                f"Cannot parse geographic group from patch_id: {patch_id}"
            )
        geographic_group_id = "_".join(match.groups())
        rows.append(
            MetadataRow(
                patch_id=patch_id,
                labels=labels,
                split=split,
                country=country,
                s1_name=s1_name,
                s2v1_name=s2v1_name,
                contains_seasonal_snow=seasonal_snow,
                contains_cloud_or_shadow=cloud_or_shadow,
                geographic_group_id=geographic_group_id,
            )
        )
    if not rows:
        raise Phase4PreparationError(f"Metadata file {source_name} contains no rows")
    return rows


def _validate_source_sets(
    clean_rows: Sequence[MetadataRow],
    excluded_rows: Sequence[MetadataRow],
    expected_total_pairs: int | None,
) -> None:
    clean_ids = {row.patch_id for row in clean_rows}
    excluded_ids = {row.patch_id for row in excluded_rows}
    overlap = clean_ids & excluded_ids
    if overlap:
        raise Phase4PreparationError(
            f"Clean and excluded metadata overlap for {len(overlap)} patch IDs"
        )
    clean_s1 = {row.s1_name for row in clean_rows}
    excluded_s1 = {row.s1_name for row in excluded_rows}
    if clean_s1 & excluded_s1:
        raise Phase4PreparationError("S1 pair identifiers overlap metadata files")
    actual_total = len(clean_rows) + len(excluded_rows)
    if expected_total_pairs is not None and actual_total != expected_total_pairs:
        raise Phase4PreparationError(
            f"BigEarthNet metadata has {actual_total} rows; "
            f"expected {expected_total_pairs} for version {DATASET_VERSION}"
        )
    observed_labels = {label for row in clean_rows for label in row.labels}
    if observed_labels != set(OFFICIAL_LABELS):
        raise Phase4PreparationError(
            "Clean metadata does not contain the frozen 19-class label set"
        )


def _validate_selected_rows(
    selected: Mapping[str, Sequence[MetadataRow]],
) -> None:
    split_by_group: dict[str, str] = {}
    selected_patch_ids: set[str] = set()
    for split in OFFICIAL_SPLITS:
        rows = selected.get(split, ())
        if not rows:
            raise Phase4PreparationError(f"No rows selected for {split}")
        for row in rows:
            if row.split != split:
                raise Phase4PreparationError(
                    f"Official split changed for patch {row.patch_id}"
                )
            prior_split = split_by_group.setdefault(row.geographic_group_id, split)
            if prior_split != split:
                raise Phase4PreparationError(
                    f"Selected geographic group crosses splits: {row.geographic_group_id}"
                )
            if row.patch_id in selected_patch_ids:
                raise Phase4PreparationError(
                    f"Selected patch appears more than once: {row.patch_id}"
                )
            selected_patch_ids.add(row.patch_id)


def _coverage_score(
    group: GeographicGroup,
    current_labels: Mapping[str, int],
    current_countries: Mapping[str, int],
    label_targets: Mapping[str, int],
    country_targets: Mapping[str, int],
    label_weights: Mapping[str, int],
    country_weights: Mapping[str, int],
    label_dimension_weight: int,
    country_dimension_weight: int,
) -> int:
    score = 0
    for label, contribution in group.label_counts:
        deficit = max(0, label_targets[label] - current_labels.get(label, 0))
        reduction = deficit * deficit - max(0, deficit - contribution) ** 2
        score += label_dimension_weight * label_weights[label] * reduction
    country_deficit = max(
        0,
        country_targets[group.country] - current_countries.get(group.country, 0),
    )
    country_reduction = (
        country_deficit * country_deficit - max(0, country_deficit - group.size) ** 2
    )
    score += (
        country_dimension_weight * country_weights[group.country] * country_reduction
    )
    return score


def _proportional_target(source_count: int, target: int, total: int) -> int:
    return (source_count * target + total // 2) // total


def _group_hash(seed_material: str, split: str, group_id: str) -> str:
    value = f"{seed_material}\0{split}\0{group_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _required_text(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase4PreparationError(f"{context} has invalid {field}")
    return value.strip()


def _validate_sha256_argument(value: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise Phase4PreparationError(f"Invalid expected SHA-256 value: {value!r}")


def _build_manifest(
    selected: Mapping[str, Sequence[MetadataRow]],
    selected_groups: Mapping[str, Sequence[GeographicGroup]],
    *,
    clean_sha256: str,
    excluded_sha256: str,
    config: SelectionConfig,
    config_sha256: str,
) -> dict[str, Any]:
    samples = []
    for split in OFFICIAL_SPLITS:
        for row in selected[split]:
            pair_digest = hashlib.sha256(
                f"{row.patch_id}\0{row.s1_name}".encode("utf-8")
            ).hexdigest()
            samples.append(
                {
                    "clean_status": {
                        "contains_cloud_or_shadow": False,
                        "contains_seasonal_snow": False,
                        "source": "official_clean_metadata",
                    },
                    "country": row.country,
                    "geographic_group_id": row.geographic_group_id,
                    "labels": list(row.labels),
                    "official_split": split,
                    "patch_id": row.patch_id,
                    "s1_archive_member_directory": (f"BigEarthNet-S1/{row.s1_name}"),
                    "s1_name": row.s1_name,
                    "s2_archive_member_directory": (f"BigEarthNet-S2/{row.patch_id}"),
                    "s2_patch_id": row.patch_id,
                    "s2v1_name": row.s2v1_name,
                    "sample_id": f"ben2_pair_{pair_digest[:24]}",
                }
            )
    samples.sort(key=lambda sample: (sample["official_split"], sample["patch_id"]))

    geographic_groups = [
        {
            "geographic_group_id": group.group_id,
            "official_split": split,
            "pair_count": group.size,
        }
        for split in OFFICIAL_SPLITS
        for group in selected_groups[split]
    ]
    geographic_groups.sort(
        key=lambda item: (item["official_split"], item["geographic_group_id"])
    )
    return {
        "counts": {
            split: {
                "geographic_groups": len(selected_groups[split]),
                "pairs": len(selected[split]),
            }
            for split in OFFICIAL_SPLITS
        },
        "country_counts": _frequency_by_split(selected, field="country"),
        "dataset": {
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "name": DATASET_NAME,
            "version": DATASET_VERSION,
        },
        "geographic_groups": geographic_groups,
        "label_counts": _frequency_by_split(selected, field="labels"),
        "manifest_id": "phase4_bigearthnet_v2_multisensor_subset_v1",
        "samples": samples,
        "schema_version": 1,
        "selection_policy": {
            **config.as_dict(),
            "config_sha256": config_sha256,
            "official_outer_split_preserved": True,
            "test_performance_used": False,
        },
        "sensor_schema_id": "bigearthnet_v2_native_geotiff_v1",
        "source_files": [
            {
                "name": "metadata.parquet",
                "publisher_md5": CLEAN_METADATA_MD5,
                "role": "selection_source_official_clean",
                "sha256": clean_sha256,
            },
            {
                "name": "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
                "publisher_md5": EXCLUDED_METADATA_MD5,
                "role": "audit_only_official_exclusions",
                "sha256": excluded_sha256,
            },
        ],
        "task": {
            "label_count": len(OFFICIAL_LABELS),
            "labels": list(OFFICIAL_LABELS),
            "type": "multi_label_land_cover_classification",
        },
    }


def _build_materialization_plan(
    selected: Mapping[str, Sequence[MetadataRow]],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    selected_rows = [row for split in OFFICIAL_SPLITS for row in selected[split]]
    selected_assets = [
        {
            "patch_id": row.patch_id,
            "s1_archive_member_directory": f"BigEarthNet-S1/{row.s1_name}",
            "s2_archive_member_directory": f"BigEarthNet-S2/{row.patch_id}",
            "sample_id": "ben2_pair_"
            + hashlib.sha256(
                f"{row.patch_id}\0{row.s1_name}".encode("utf-8")
            ).hexdigest()[:24],
        }
        for row in selected_rows
    ]
    selected_assets.sort(key=lambda item: item["patch_id"])
    full_archive_bytes = 54_439_153_171 + 63_251_710_377
    average_bytes_per_pair = full_archive_bytes / EXPECTED_TOTAL_PAIRS
    selected_estimate = round(average_bytes_per_pair * len(selected_rows))
    return {
        "archive_member_contract": {
            "s1_bands": list(S1_BANDS),
            "s1_file_template": "{s1_directory}/{s1_name}_{band}.tif",
            "s2_bands": list(S2_BANDS),
            "s2_file_template": "{s2_directory}/{patch_id}_{band}.tif",
        },
        "canonical_archives": [
            {
                "downloaded": False,
                "md5": "a55eaa2cdf6a917e296bd6601ec1e348",
                "name": "BigEarthNet-S1.tar.zst",
                "size_bytes": 54_439_153_171,
            },
            {
                "downloaded": False,
                "md5": "2245ed2d1a93f6ce637d839bc856396e",
                "name": "BigEarthNet-S2.tar.zst",
                "size_bytes": 63_251_710_377,
            },
        ],
        "estimated_selected_compressed_bytes": selected_estimate,
        "manifest_sha256": manifest_sha256,
        "materialization_state": "not_materialized",
        "required_free_disk": {
            "minimum_bytes": 8 * 1024**3,
            "recommended_bytes": 12 * 1024**3,
        },
        "schema_version": 1,
        "selected_assets": selected_assets,
        "selected_pair_count": len(selected_rows),
        "selective_access": {
            "status": "unresolved",
            "warning": (
                "The canonical imagery is published as monolithic tar.zst archives; "
                "no official per-patch selective-access path is frozen."
            ),
        },
    }


def _build_report(
    clean_rows: Sequence[MetadataRow],
    excluded_rows: Sequence[MetadataRow],
    selected: Mapping[str, Sequence[MetadataRow]],
    selected_groups: Mapping[str, Sequence[GeographicGroup]],
    *,
    clean_sha256: str,
    excluded_sha256: str,
    manifest_sha256: str,
    config_sha256: str,
    config: SelectionConfig,
    selection_diagnostics: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    selected_label_counts = _frequency_by_split(selected, field="labels")
    underrepresented = {
        split: [
            {
                "count": selected_label_counts[split].get(label, 0),
                "label": label,
                "minimum": config.label_floor_for(split),
            }
            for label in OFFICIAL_LABELS
            if selected_label_counts[split].get(label, 0)
            < config.label_floor_for(split)
        ]
        for split in OFFICIAL_SPLITS
    }
    return {
        "dataset": {
            "doi": DATASET_DOI,
            "name": DATASET_NAME,
            "version": DATASET_VERSION,
        },
        "manifest_sha256": manifest_sha256,
        "official_metadata": {
            "clean": _metadata_summary(clean_rows),
            "excluded": _metadata_summary(excluded_rows),
            "row_count_total": len(clean_rows) + len(excluded_rows),
            "sha256": {
                "metadata.parquet": clean_sha256,
                "metadata_for_patches_with_snow_cloud_or_shadow.parquet": excluded_sha256,
            },
        },
        "schema_version": 1,
        "selected": {
            "class_frequencies": selected_label_counts,
            "country_frequencies": _frequency_by_split(selected, field="country"),
            "counts": {
                split: {
                    "geographic_groups": len(selected_groups[split]),
                    "pairs": len(selected[split]),
                }
                for split in OFFICIAL_SPLITS
            },
            "selection_diagnostics": selection_diagnostics,
            "underrepresented_classes": underrepresented,
        },
        "selection_config": config.as_dict(),
        "selection_config_sha256": config_sha256,
    }


def _metadata_summary(rows: Sequence[MetadataRow]) -> dict[str, Any]:
    by_split = {
        split: [row for row in rows if row.split == split] for split in OFFICIAL_SPLITS
    }
    return {
        "class_frequencies": _frequency_by_split(by_split, field="labels"),
        "country_frequencies": _frequency_by_split(by_split, field="country"),
        "paired_s1_s2": {
            "available": sum(bool(row.patch_id and row.s1_name) for row in rows),
            "missing": sum(not (row.patch_id and row.s1_name) for row in rows),
        },
        "quality_flags": {
            "cloud_or_shadow": sum(row.contains_cloud_or_shadow for row in rows),
            "seasonal_snow": sum(row.contains_seasonal_snow for row in rows),
        },
        "row_count": len(rows),
        "split_counts": {split: len(by_split[split]) for split in OFFICIAL_SPLITS},
    }


def _frequency_by_split(
    rows_by_split: Mapping[str, Sequence[MetadataRow]],
    *,
    field: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in OFFICIAL_SPLITS:
        counter: Counter[str] = Counter()
        for row in rows_by_split[split]:
            value = getattr(row, field)
            if isinstance(value, tuple):
                counter.update(value)
            else:
                counter[str(value)] += 1
        result[split] = dict(sorted(counter.items()))
    return result


def _write_immutable_manifest(path: Path, content: bytes) -> None:
    if path.is_file():
        existing = path.read_bytes()
        if existing != content:
            raise Phase4PreparationError(
                f"Frozen manifest already exists with different bytes: {path}"
            )
        return
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the metadata-only BigEarthNet v2 Phase 4 subset"
    )
    parser.add_argument("--clean-metadata", type=Path, default=DEFAULT_CLEAN_METADATA)
    parser.add_argument(
        "--excluded-metadata", type=Path, default=DEFAULT_EXCLUDED_METADATA
    )
    parser.add_argument("--clean-sha256", default=CLEAN_METADATA_SHA256)
    parser.add_argument("--excluded-sha256", default=EXCLUDED_METADATA_SHA256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = prepare_phase4_bigearthnet(
        clean_metadata_path=args.clean_metadata.resolve(),
        excluded_metadata_path=args.excluded_metadata.resolve(),
        clean_sha256=args.clean_sha256,
        excluded_sha256=args.excluded_sha256,
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "counts": result.counts,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "materialization_plan": str(result.materialization_plan_path),
                "preparation_report": str(result.report_path),
                "selection_config_sha256": result.selection_config_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
