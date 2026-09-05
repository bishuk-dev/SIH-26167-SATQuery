from __future__ import annotations

import json
from pathlib import Path

import pytest

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")

from ml.evaluation.prepare_phase4_bigearthnet import (  # noqa: E402
    OFFICIAL_LABELS,
    Phase4PreparationError,
    SelectionConfig,
    build_geographic_groups,
    file_sha256,
    load_metadata,
    prepare_phase4_bigearthnet,
    select_groups,
)


def _row(
    split: str,
    index: int,
    *,
    tile: str | None = None,
    horizontal: int | None = None,
    vertical: int | None = None,
    flagged: bool = False,
    s1_name: str | None = None,
) -> dict[str, object]:
    split_offset = {"train": 10, "validation": 30, "test": 50}[split]
    horizontal = horizontal if horizontal is not None else split_offset + index
    vertical = vertical if vertical is not None else split_offset + index
    tile = tile or {"train": "T31AAA", "validation": "T32BBB", "test": "T33CCC"}[split]
    patch_id = (
        f"S2A_MSIL2A_201706{index + 10:02d}T101031_"
        f"N9999_R022_{tile}_{horizontal}_{vertical}"
    )
    return {
        "contains_cloud_or_shadow": flagged,
        "contains_seasonal_snow": False,
        "country": "Finland" if index % 2 == 0 else "Portugal",
        "labels": list(OFFICIAL_LABELS),
        "patch_id": patch_id,
        "s1_name": s1_name or f"S1_pair_{split}_{index}_{horizontal}_{vertical}",
        "s2v1_name": f"S2_v1_{split}_{index}_{horizontal}_{vertical}",
        "split": split,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> str:
    parquet.write_table(pyarrow.Table.from_pylist(rows), path)
    return file_sha256(path)


def _source_files(tmp_path: Path) -> tuple[Path, str, Path, str]:
    clean_rows = [
        _row(split, index)
        for split in ("train", "validation", "test")
        for index in range(5)
    ]
    excluded_rows = [
        _row(split, 90 + index, flagged=True)
        for index, split in enumerate(("train", "validation", "test"))
    ]
    clean_path = tmp_path / "metadata.parquet"
    excluded_path = tmp_path / "excluded.parquet"
    return (
        clean_path,
        _write_parquet(clean_path, clean_rows),
        excluded_path,
        _write_parquet(excluded_path, excluded_rows),
    )


def _small_config() -> SelectionConfig:
    return SelectionConfig(
        train_target=3,
        validation_target=3,
        test_target=3,
        train_label_floor=1,
        validation_label_floor=1,
        test_label_floor=1,
    )


def _prepare(tmp_path: Path, output_name: str = "output"):
    clean, clean_hash, excluded, excluded_hash = _source_files(tmp_path)
    result = prepare_phase4_bigearthnet(
        clean_metadata_path=clean,
        excluded_metadata_path=excluded,
        clean_sha256=clean_hash,
        excluded_sha256=excluded_hash,
        output_dir=tmp_path / output_name,
        config=_small_config(),
        expected_total_pairs=None,
    )
    return result, clean, clean_hash, excluded, excluded_hash


def test_checksum_failure_is_rejected(tmp_path: Path) -> None:
    clean, _, _, _ = _source_files(tmp_path)

    with pytest.raises(Phase4PreparationError, match="SHA-256 mismatch"):
        load_metadata(clean, "0" * 64, expected_clean=True)


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    row = _row("train", 0)
    del row["s1_name"]
    path = tmp_path / "missing-column.parquet"
    checksum = _write_parquet(path, [row])

    with pytest.raises(Phase4PreparationError, match="missing required columns"):
        load_metadata(path, checksum, expected_clean=True)


def test_unpaired_s1_row_is_rejected(tmp_path: Path) -> None:
    row = _row("train", 0)
    row["s1_name"] = ""
    path = tmp_path / "unpaired.parquet"
    checksum = _write_parquet(path, [row])

    with pytest.raises(Phase4PreparationError, match="invalid s1_name"):
        load_metadata(path, checksum, expected_clean=True)


def test_official_quality_flags_control_exclusion(tmp_path: Path) -> None:
    flagged_path = tmp_path / "flagged-clean.parquet"
    flagged_hash = _write_parquet(flagged_path, [_row("train", 0, flagged=True)])
    with pytest.raises(Phase4PreparationError, match="flagged patch"):
        load_metadata(flagged_path, flagged_hash, expected_clean=True)

    result, _, _, _, _ = _prepare(tmp_path, "prepared")
    manifest = json.loads(result.manifest_path.read_bytes())
    assert all(
        sample["clean_status"]["source"] == "official_clean_metadata"
        and not sample["clean_status"]["contains_seasonal_snow"]
        and not sample["clean_status"]["contains_cloud_or_shadow"]
        for sample in manifest["samples"]
    )


def test_geographic_group_cannot_cross_official_splits(tmp_path: Path) -> None:
    rows = [
        _row("train", 0, tile="T31AAA", horizontal=7, vertical=8),
        _row("test", 1, tile="T31AAA", horizontal=7, vertical=8),
    ]
    path = tmp_path / "cross-split.parquet"
    checksum = _write_parquet(path, rows)
    parsed = load_metadata(path, checksum, expected_clean=True)

    with pytest.raises(Phase4PreparationError, match="crosses official splits"):
        build_geographic_groups(parsed)


def test_selection_is_deterministic_and_preserves_split(tmp_path: Path) -> None:
    clean, clean_hash, _, _ = _source_files(tmp_path)
    rows = load_metadata(clean, clean_hash, expected_clean=True)
    groups = [
        group for group in build_geographic_groups(rows) if group.split == "train"
    ]

    first, _ = select_groups(
        groups,
        target=3,
        label_floor=1,
        seed_material="fixed",
    )
    second, _ = select_groups(
        list(reversed(groups)),
        target=3,
        label_floor=1,
        seed_material="fixed",
    )

    assert [group.group_id for group in first] == [group.group_id for group in second]
    assert {group.split for group in first} == {"train"}


def test_manifest_is_byte_reproducible_and_sidecar_independent(
    tmp_path: Path,
) -> None:
    clean, clean_hash, excluded, excluded_hash = _source_files(tmp_path)
    common = {
        "clean_metadata_path": clean,
        "excluded_metadata_path": excluded,
        "clean_sha256": clean_hash,
        "excluded_sha256": excluded_hash,
        "config": _small_config(),
        "expected_total_pairs": None,
    }
    first = prepare_phase4_bigearthnet(
        **common,
        output_dir=tmp_path / "first",
    )
    second = prepare_phase4_bigearthnet(
        **common,
        output_dir=tmp_path / "second",
    )
    expected_manifest = first.manifest_path.read_bytes()
    assert expected_manifest == second.manifest_path.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256

    first.materialization_plan_path.write_text("storage policy changed")
    repeated = prepare_phase4_bigearthnet(
        **common,
        output_dir=tmp_path / "first",
    )
    assert repeated.manifest_path.read_bytes() == expected_manifest
    assert repeated.manifest_sha256 == first.manifest_sha256


def test_test_metadata_does_not_influence_train_or_validation_selection(
    tmp_path: Path,
) -> None:
    clean, clean_hash, _, _ = _source_files(tmp_path)
    original = load_metadata(clean, clean_hash, expected_clean=True)
    original_groups = build_geographic_groups(original)
    original_ids = {}
    for split in ("train", "validation"):
        chosen, _ = select_groups(
            [group for group in original_groups if group.split == split],
            target=3,
            label_floor=1,
            seed_material="fixed",
        )
        original_ids[split] = [group.group_id for group in chosen]

    changed_rows = [
        _row(split, index + (100 if split == "test" else 0))
        for split in ("train", "validation", "test")
        for index in range(5)
    ]
    changed_path = tmp_path / "changed-test.parquet"
    changed_hash = _write_parquet(changed_path, changed_rows)
    changed = load_metadata(changed_path, changed_hash, expected_clean=True)
    changed_groups = build_geographic_groups(changed)
    for split in ("train", "validation"):
        chosen, _ = select_groups(
            [group for group in changed_groups if group.split == split],
            target=3,
            label_floor=1,
            seed_material="fixed",
        )
        assert [group.group_id for group in chosen] == original_ids[split]
