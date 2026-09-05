from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from ml.evaluation.phase4_native_audit import (
    FROZEN_SAMPLE_IDS,
    NativeAuditPreparationError,
    build_audit_member_allowlists,
    selective_package_extraction,
    verify_package_for_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/phase4_bigearthnet_multisensor"
FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
)


def _compressed_tar(members: list[tuple[str, bytes]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return zstandard.ZstdCompressor().compress(raw.getvalue())


def test_frozen_train_samples_produce_exactly_42_native_members() -> None:
    manifest = json.loads((EXPERIMENT_DIR / "split_manifest.json").read_bytes())

    allowlists = build_audit_member_allowlists(manifest)

    assert FROZEN_SAMPLE_IDS == (
        "ben2_pair_54d6cd2db80bec3b88bd1bde",
        "ben2_pair_ceb2b2ff7000d820e6d70028",
        "ben2_pair_90d4a6de879ee97d50bf2c2d",
    )
    assert len(allowlists["s1"]) == 6
    assert len(allowlists["s2"]) == 36
    assert len(allowlists["s1"]) + len(allowlists["s2"]) == 42
    assert all(path.startswith("s1/train/") for path in allowlists["s1"])
    assert all(path.startswith("s2/train/") for path in allowlists["s2"])


@pytest.mark.parametrize("wrong_split", ["validation", "test"])
def test_native_audit_refuses_non_train_frozen_sample(wrong_split: str) -> None:
    manifest = json.loads((EXPERIMENT_DIR / "split_manifest.json").read_bytes())
    selected = [
        dict(sample)
        for sample in manifest["samples"]
        if sample["sample_id"] in FROZEN_SAMPLE_IDS
    ]
    selected[0]["official_split"] = wrong_split

    with pytest.raises(NativeAuditPreparationError, match="TRAIN"):
        build_audit_member_allowlists({"samples": selected})


def test_package_sha_mismatch_is_rejected_before_archive_open(tmp_path: Path) -> None:
    package = tmp_path / "phase4_s1_selected.tar.zst"
    package.write_bytes(b"not the verified package")
    package_manifest = tmp_path / "package_manifest.json"
    package_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frozen_split_manifest_sha256": FROZEN_MANIFEST_SHA256,
                "packages": {
                    "s1": {
                        "modality": "s1",
                        "package_file": package.name,
                        "file_count": 36_002,
                        "manifest_sha256": FROZEN_MANIFEST_SHA256,
                        "package_sha256": "0" * 64,
                        "package_size_bytes": package.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NativeAuditPreparationError, match="SHA-256 mismatch"):
        verify_package_for_audit(
            package,
            package_manifest,
            modality="s1",
            frozen_manifest_sha256=FROZEN_MANIFEST_SHA256,
        )


def test_selective_extraction_skips_everything_else_and_cleans_up(
    tmp_path: Path,
) -> None:
    allowed = "s1/train/scene/scene_VV.tif"
    package = tmp_path / "phase4_s1_selected.tar.zst"
    package.write_bytes(
        _compressed_tar(
            [
                (allowed, b"train"),
                ("s1/train/other/other_VV.tif", b"other train"),
                ("s1/validation/scene/scene_VV.tif", b"validation"),
                ("s1/sealed_test/scene/scene_VV.tif", b"test"),
            ]
        )
    )
    transient_parent = tmp_path / "transient-parent"

    with selective_package_extraction(
        {"s1": package},
        {"s1": (allowed,)},
        transient_parent,
    ) as extracted_root:
        files = [path for path in extracted_root.rglob("*") if path.is_file()]
        assert [path.relative_to(extracted_root).as_posix() for path in files] == [
            allowed
        ]
        assert files[0].read_bytes() == b"train"

    assert not extracted_root.exists()
    assert transient_parent.is_dir()
