from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from ml.evaluation.materialize_phase4_bigearthnet import (
    ArchiveSpec,
    FROZEN_MANIFEST_SHA256,
    MaterializationError,
    build_allowlist,
    load_frozen_manifest,
    materialize_archive_stream,
    package_materialized_modality,
    validate_tar_member,
)


def _compressed_tar(members: list[tuple[str, bytes]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return zstandard.ZstdCompressor().compress(raw.getvalue())


def _multiframe_compressed_tar(members: list[tuple[str, bytes]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    payload = raw.getvalue()
    midpoint = len(payload) // 2
    compressor = zstandard.ZstdCompressor()
    return compressor.compress(payload[:midpoint]) + compressor.compress(payload[midpoint:])


def _spec(content: bytes, *, md5: str | None = None) -> ArchiveSpec:
    return ArchiveSpec(
        modality="s1",
        url="https://zenodo.org/example.tar.zst",
        expected_content_length=len(content),
        expected_md5=md5 or hashlib.md5(content).hexdigest(),  # noqa: S324
        bands=("VV",),
    )


def test_stream_materialization_promotes_only_after_complete_md5(
    tmp_path: Path,
) -> None:
    member = "BigEarthNet-S1/scene/patch/patch_VV.tif"
    compressed = _compressed_tar(
        [("BigEarthNet-S1/unselected/file.tif", b"skip"), (member, b"selected")]
    )
    output = tmp_path / "dataset"

    result = materialize_archive_stream(
        io.BytesIO(compressed),
        spec=_spec(compressed),
        allowlist={member: Path("train/patch/patch_VV.tif")},
        output_root=output,
        observed_content_length=len(compressed),
    )

    assert result["completion_state"] == "complete"
    assert result["selected_members_found"] == 1
    assert (output / "s1/train/patch/patch_VV.tif").read_bytes() == b"selected"
    assert not (output / "s1/unselected").exists()
    marker = json.loads((output / "s1/_materialization.json").read_text())
    assert marker["manifest_sha256"] == FROZEN_MANIFEST_SHA256


def test_checksum_failure_discards_quarantine_and_never_marks_complete(
    tmp_path: Path,
) -> None:
    member = "BigEarthNet-S1/scene/patch/patch_VV.tif"
    compressed = _compressed_tar([(member, b"selected")])
    with pytest.raises(MaterializationError, match="MD5 mismatch"):
        materialize_archive_stream(
            io.BytesIO(compressed),
            spec=_spec(compressed, md5="0" * 32),
            allowlist={member: Path("train/patch/patch_VV.tif")},
            output_root=tmp_path / "dataset",
            observed_content_length=len(compressed),
        )

    assert not (tmp_path / "dataset/s1").exists()
    assert not list((tmp_path / "dataset/.quarantine").glob("s1-*"))


def test_concatenated_zstd_frames_are_fully_consumed_and_verified(
    tmp_path: Path,
) -> None:
    member = "BigEarthNet-S1/scene/patch/patch_VV.tif"
    compressed = _multiframe_compressed_tar([(member, b"selected")])

    result = materialize_archive_stream(
        io.BytesIO(compressed),
        spec=_spec(compressed),
        allowlist={member: Path("train/patch/patch_VV.tif")},
        output_root=tmp_path / "dataset",
        observed_content_length=len(compressed),
    )

    assert result["compressed_bytes_streamed"] == len(compressed)
    assert result["observed_md5"] == hashlib.md5(compressed).hexdigest()  # noqa: S324
    assert result["selected_members_found"] == 1


def test_package_is_deterministic_verified_and_removes_loose_files(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "dataset"
    modality_root = output_root / "s1"
    files = {
        Path("train/a/a_VV.tif"): b"train",
        Path("sealed_test/b/b_VV.tif"): b"sealed",
    }
    for relative, content in files.items():
        path = modality_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    marker = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "modality": "s1",
        "archive_md5": "publisher-md5",
        "compressed_bytes_streamed": 123,
        "selected_members": len(files),
        "extracted_bytes": sum(map(len, files.values())),
    }
    (modality_root / "_materialization.json").write_text(json.dumps(marker))
    spec = ArchiveSpec("s1", "https://example", 123, "publisher-md5", ("VV",))
    allowlist = {
        "archive/a": Path("train/a/a_VV.tif"),
        "archive/b": Path("sealed_test/b/b_VV.tif"),
    }
    package_path = tmp_path / "phase4_s1_selected.tar.zst"

    first = package_materialized_modality(
        output_root=output_root,
        package_path=package_path,
        spec=spec,
        allowlist=allowlist,
        delete_loose=True,
    )
    first_bytes = package_path.read_bytes()

    assert first["file_count"] == 2
    assert first["extracted_logical_bytes"] == 11
    assert first["sealed_test_paths"] == ["s1/sealed_test/b/b_VV.tif"]
    assert first["package_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert not modality_root.exists()

    # Recreate identical logical input. Deterministic metadata and order must
    # produce the same package bytes.
    for relative, content in files.items():
        path = modality_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (modality_root / "_materialization.json").write_text(json.dumps(marker))
    second = package_materialized_modality(
        output_root=output_root,
        package_path=package_path,
        spec=spec,
        allowlist=allowlist,
        delete_loose=False,
    )

    assert package_path.read_bytes() == first_bytes
    assert second["package_sha256"] == first["package_sha256"]


def test_package_verification_failure_preserves_loose_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.evaluation.materialize_phase4_bigearthnet as materializer

    output_root = tmp_path / "dataset"
    modality_root = output_root / "s1"
    selected = modality_root / "train/a/a_VV.tif"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"train")
    marker = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "modality": "s1",
        "archive_md5": "publisher-md5",
        "compressed_bytes_streamed": 123,
        "selected_members": 1,
        "extracted_bytes": 5,
    }
    (modality_root / "_materialization.json").write_text(json.dumps(marker))
    spec = ArchiveSpec("s1", "https://example", 123, "publisher-md5", ("VV",))
    monkeypatch.setattr(
        materializer,
        "_verify_package",
        lambda *args, **kwargs: (_ for _ in ()).throw(MaterializationError("bad package")),
    )

    with pytest.raises(MaterializationError, match="bad package"):
        package_materialized_modality(
            output_root=output_root,
            package_path=tmp_path / "phase4_s1_selected.tar.zst",
            spec=spec,
            allowlist={"archive/a": Path("train/a/a_VV.tif")},
            delete_loose=True,
        )

    assert selected.read_bytes() == b"train"


@pytest.mark.parametrize(
    "info",
    [
        tarfile.TarInfo("../../escape.tif"),
        tarfile.TarInfo("/absolute.tif"),
        tarfile.TarInfo("folder\\escape.tif"),
    ],
)
def test_unsafe_archive_paths_are_rejected(info: tarfile.TarInfo) -> None:
    assert validate_tar_member(info) is not None


def test_links_and_devices_are_rejected() -> None:
    link = tarfile.TarInfo("safe-looking")
    link.type = tarfile.SYMTYPE
    assert validate_tar_member(link) is not None


def test_manifest_mismatch_fails_before_materialization(tmp_path: Path) -> None:
    altered = tmp_path / "split_manifest.json"
    altered.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="SHA-256 mismatch"):
        load_frozen_manifest(altered)


def test_real_manifest_allowlists_exact_native_member_counts() -> None:
    manifest = load_frozen_manifest()
    from ml.evaluation.materialize_phase4_bigearthnet import ARCHIVES

    assert len(build_allowlist(manifest, ARCHIVES["s1"])) == 36_002
    assert len(build_allowlist(manifest, ARCHIVES["s2"])) == 216_012
