"""Fail-closed preparation for the three-sample Phase 4D native raster audit."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import zstandard

from ml.evaluation.inspect_phase4_native_rasters import FROZEN_SAMPLE_IDS
from ml.evaluation.materialize_phase4_bigearthnet import validate_tar_member
from satquery.inference.multisensor_preprocessing import materialized_band_paths

FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
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
PACKAGE_MEMBER_COUNTS = {"s1": 36_002, "s2": 216_012}
PACKAGE_NAMES = {
    "s1": "phase4_s1_selected.tar.zst",
    "s2": "phase4_s2_selected.tar.zst",
}


class NativeAuditPreparationError(RuntimeError):
    """Raised before pixel access when audit package constraints are not met."""


def build_audit_member_allowlists(
    manifest: Mapping[str, Any],
    *,
    sample_ids: tuple[str, ...] = FROZEN_SAMPLE_IDS,
) -> dict[str, tuple[str, ...]]:
    """Build the exact 6 S1 + 36 S2 package paths for the frozen TRAIN samples."""

    if sample_ids != FROZEN_SAMPLE_IDS:
        raise NativeAuditPreparationError(
            "Native audit accepts exactly the three Phase 4C-frozen sample IDs"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise NativeAuditPreparationError("Frozen manifest samples are missing")
    by_id: dict[str, Mapping[str, object]] = {}
    for sample in samples:
        if isinstance(sample, Mapping) and sample.get("sample_id") in sample_ids:
            sample_id = sample["sample_id"]
            if not isinstance(sample_id, str) or sample_id in by_id:
                raise NativeAuditPreparationError(
                    "Frozen native-audit sample IDs must be unique"
                )
            by_id[sample_id] = sample

    allowlists: dict[str, list[str]] = {"s1": [], "s2": []}
    for sample_id in sample_ids:
        sample = by_id.get(sample_id)
        if sample is None:
            raise NativeAuditPreparationError(
                f"Frozen native-audit sample is missing: {sample_id}"
            )
        if sample.get("official_split") != "train":
            raise NativeAuditPreparationError(
                f"Frozen native-audit sample is not in TRAIN: {sample_id}"
            )
        paths = materialized_band_paths(Path(), sample, split="train")
        allowlists["s1"].extend(paths[band].as_posix() for band in S1_BANDS)
        allowlists["s2"].extend(paths[band].as_posix() for band in S2_BANDS)

    result = {key: tuple(values) for key, values in allowlists.items()}
    if (
        len(result["s1"]) != 6
        or len(result["s2"]) != 36
        or any(len(set(values)) != len(values) for values in result.values())
    ):
        raise NativeAuditPreparationError(
            "Frozen native-audit allowlist must contain 42 unique rasters"
        )
    return result


def verify_package_for_audit(
    package_path: Path,
    package_manifest_path: Path,
    *,
    modality: str,
    frozen_manifest_sha256: str = FROZEN_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify package metadata and bytes fully before any archive member is opened."""

    if modality not in PACKAGE_NAMES:
        raise NativeAuditPreparationError(f"Unsupported modality: {modality}")
    manifest = _read_json(package_manifest_path)
    packages = manifest.get("packages")
    record = packages.get(modality) if isinstance(packages, Mapping) else None
    if not isinstance(record, Mapping):
        raise NativeAuditPreparationError(
            f"Package manifest has no {modality.upper()} record"
        )
    expected_values = {
        "modality": modality,
        "package_file": PACKAGE_NAMES[modality],
        "file_count": PACKAGE_MEMBER_COUNTS[modality],
        "manifest_sha256": frozen_manifest_sha256,
    }
    if (
        manifest.get("frozen_split_manifest_sha256") != frozen_manifest_sha256
        or any(record.get(key) != value for key, value in expected_values.items())
    ):
        raise NativeAuditPreparationError(
            f"{modality.upper()} package provenance does not match the frozen contract"
        )
    package_sha = record.get("package_sha256")
    package_size = record.get("package_size_bytes")
    if (
        not isinstance(package_sha, str)
        or len(package_sha) != 64
        or any(character not in "0123456789abcdef" for character in package_sha)
        or not isinstance(package_size, int)
        or package_size <= 0
        or package_path.name != PACKAGE_NAMES[modality]
    ):
        raise NativeAuditPreparationError(
            f"{modality.upper()} package metadata is incomplete"
        )
    try:
        actual_size = package_path.stat().st_size
    except OSError as exc:
        raise NativeAuditPreparationError(
            f"Cannot access {modality.upper()} package"
        ) from exc
    if actual_size != package_size:
        raise NativeAuditPreparationError(
            f"{modality.upper()} package size mismatch"
        )
    if _sha256(package_path) != package_sha:
        raise NativeAuditPreparationError(
            f"{modality.upper()} package SHA-256 mismatch"
        )
    return dict(record)


@contextmanager
def selective_package_extraction(
    package_paths: Mapping[str, Path],
    allowlists: Mapping[str, tuple[str, ...]],
    transient_parent: Path,
) -> Iterable[Path]:
    """Extract only allowlisted members and always remove the created audit tree."""

    transient_parent.mkdir(parents=True, exist_ok=True)
    extracted_root = Path(
        tempfile.mkdtemp(prefix="phase4d-native-audit-", dir=transient_parent)
    )
    try:
        for modality, package_path in package_paths.items():
            allowed = allowlists.get(modality)
            if not allowed:
                raise NativeAuditPreparationError(
                    f"No audit allowlist was provided for {modality.upper()}"
                )
            _extract_one_package(
                package_path,
                modality=modality,
                allowed_members=allowed,
                output_root=extracted_root,
            )
        yield extracted_root
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def _extract_one_package(
    package_path: Path,
    *,
    modality: str,
    allowed_members: tuple[str, ...],
    output_root: Path,
) -> None:
    allowed = set(allowed_members)
    if len(allowed) != len(allowed_members) or any(
        not name.startswith(f"{modality}/train/") for name in allowed
    ):
        raise NativeAuditPreparationError(
            f"{modality.upper()} audit allowlist is invalid or not TRAIN-only"
        )
    found: set[str] = set()
    try:
        with package_path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(
                compressed, read_across_frames=True
            ) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        rejection = validate_tar_member(member)
                        if rejection is not None:
                            raise NativeAuditPreparationError(
                                f"Unsafe packaged member {member.name!r}: {rejection}"
                            )
                        if member.name not in allowed:
                            continue
                        if not member.isfile() or member.name in found:
                            raise NativeAuditPreparationError(
                                f"Duplicate or non-file audit member: {member.name}"
                            )
                        source = archive.extractfile(member)
                        if source is None:
                            raise NativeAuditPreparationError(
                                f"Cannot read audit member: {member.name}"
                            )
                        target = output_root.joinpath(*PurePosixPath(member.name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                        found.add(member.name)
    except (OSError, tarfile.TarError, zstandard.ZstdError) as exc:
        raise NativeAuditPreparationError(
            f"Cannot scan verified {modality.upper()} package"
        ) from exc
    missing = sorted(allowed - found)
    if missing:
        raise NativeAuditPreparationError(
            f"Verified {modality.upper()} package is missing {len(missing)} audit members"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeAuditPreparationError(f"Cannot read package manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise NativeAuditPreparationError("Package manifest must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise NativeAuditPreparationError(f"Cannot hash package: {path}") from exc
    return digest.hexdigest()

