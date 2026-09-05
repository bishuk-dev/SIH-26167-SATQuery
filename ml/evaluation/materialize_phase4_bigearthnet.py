"""Stream only the frozen BigEarthNet v2 subset from canonical archives.

Plan mode is safe and performs no network access. The transfer path requires
``--confirm-full-stream-transfer`` because selective extraction reduces local
storage, not the 109.61 GiB network transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO, Mapping, cast
from ml.evaluation.prepare_phase4_bigearthnet import S1_BANDS, S2_BANDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/phase4_bigearthnet_multisensor"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "split_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/phase4_bigearthnet_v2"
DEFAULT_REPORT = EXPERIMENT_DIR / "results/materialization_report.json"
FROZEN_MANIFEST_SHA256 = (
    "615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6"
)
SPLIT_COUNTS = {"train": 12_000, "validation": 3_000, "test": 3_001}
BUFFER_SIZE = 1024 * 1024
MAX_SELECTED_MEMBER_BYTES = 256 * 1024 * 1024
RECOMMENDED_FREE_DISK_BYTES = 8 * 1024**3
ESTIMATED_SELECTED_DISK_BYTES = 3_856_128_477
PACKAGE_COMPRESSION_LEVEL = 9
PACKAGE_PROGRESS_INTERVAL_SECONDS = 30.0


class MaterializationError(RuntimeError):
    """Raised when transfer or archive integrity fails closed."""

    def __init__(
        self, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    modality: str
    url: str
    expected_content_length: int
    expected_md5: str
    bands: tuple[str, ...]


ARCHIVES = {
    "s1": ArchiveSpec(
        modality="s1",
        url="https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1",
        expected_content_length=54_439_153_171,
        expected_md5="a55eaa2cdf6a917e296bd6601ec1e348",
        bands=S1_BANDS,
    ),
    "s2": ArchiveSpec(
        modality="s2",
        url="https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1",
        expected_content_length=63_251_710_377,
        expected_md5="2245ed2d1a93f6ce637d839bc856396e",
        bands=S2_BANDS,
    ),
}


class DigestingReader(io.RawIOBase):
    """Count, hash, and print throttled transfer/extraction progress."""

    def __init__(
        self,
        source: BinaryIO,
        *,
        total_size: int,
        modality: str,
        selected_total: int,
        progress_interval_seconds: float = PACKAGE_PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        self.source = source
        self.digest = hashlib.md5()
        self.bytes_read = 0
        self.total_size = total_size
        self.modality = modality.upper()
        self.selected_total = selected_total
        self.selected_found = 0
        self.progress_interval_seconds = progress_interval_seconds
        self._last_progress = time.monotonic()

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        if data:
            self.digest.update(data)
            self.bytes_read += len(data)
            self._print_progress()
        return data

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer)
        data = self.read(len(view))
        count = len(data)
        view[:count] = data
        return count

    @property
    def hexdigest(self) -> str:
        self._print_progress(force=True)
        return self.digest.hexdigest()

    def mark_selected(self, count: int) -> None:
        self.selected_found = count
        self._print_progress()

    def _print_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < self.progress_interval_seconds:
            return
        gib = self.bytes_read / 1024**3
        total_gib = self.total_size / 1024**3
        percent = 100.0 * self.bytes_read / self.total_size
        print(
            f"{self.modality} | {gib:.2f} / {total_gib:.2f} GiB | "
            f"{percent:.1f}% | selected {self.selected_found:,} / "
            f"{self.selected_total:,}",
            file=sys.stdout,
            flush=True,
        )
        self._last_progress = now

def load_frozen_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the exact frozen experiment definition or refuse materialization."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise MaterializationError(f"Cannot read split manifest: {path}") from exc
    observed = hashlib.sha256(content).hexdigest()
    if observed != FROZEN_MANIFEST_SHA256:
        raise MaterializationError(
            f"Frozen manifest SHA-256 mismatch: expected {FROZEN_MANIFEST_SHA256}, "
            f"got {observed}"
        )
    payload = json.loads(content)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise MaterializationError("Frozen manifest has no sample list")
    counts = {split: 0 for split in SPLIT_COUNTS}
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("official_split") not in counts:
            raise MaterializationError("Frozen manifest contains an invalid sample")
        counts[str(sample["official_split"])] += 1
    if counts != SPLIT_COUNTS:
        raise MaterializationError(
            f"Frozen manifest split counts changed: expected {SPLIT_COUNTS}, got {counts}"
        )
    return payload


def build_allowlist(manifest: Mapping[str, Any], spec: ArchiveSpec) -> dict[str, Path]:
    """Map exact canonical member names to safe relative destinations."""

    samples = manifest["samples"]
    assert isinstance(samples, list)
    result: dict[str, Path] = {}
    destinations: set[Path] = set()
    for raw_sample in samples:
        assert isinstance(raw_sample, dict)
        split = str(raw_sample["official_split"])
        storage_split = "sealed_test" if split == "test" else split
        if spec.modality == "s1":
            identity = _required_text(raw_sample, "s1_name")
            archive_parent = _strip_geographic_suffix(identity)
            archive_root = PurePosixPath("BigEarthNet-S1", archive_parent, identity)
        else:
            identity = _required_text(raw_sample, "patch_id")
            archive_parent = identity.rsplit("_", 2)[0]
            archive_root = PurePosixPath("BigEarthNet-S2", archive_parent, identity)
        for band in spec.bands:
            member = str(archive_root / f"{identity}_{band}.tif")
            destination = Path(storage_split) / identity / f"{identity}_{band}.tif"
            if member in result or destination in destinations:
                raise MaterializationError(
                    f"Manifest creates duplicate {spec.modality} member {member}"
                )
            result[member] = destination
            destinations.add(destination)
    return result


def build_plan(
    manifest: Mapping[str, Any], output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    """Describe the unavoidable transfer and bounded output without networking."""

    selected_compressed_estimate = ESTIMATED_SELECTED_DISK_BYTES
    recommended_free = max(
        RECOMMENDED_FREE_DISK_BYTES, selected_compressed_estimate * 2
    )
    samples = manifest["samples"]
    assert isinstance(samples, list)
    archive_plans = {}
    for key, spec in ARCHIVES.items():
        archive_plans[key] = {
            "canonical_url": spec.url,
            "expected_content_length_bytes": spec.expected_content_length,
            "expected_md5": spec.expected_md5,
            "selected_members_expected": len(samples) * len(spec.bands),
            "output_path": str((output_root / spec.modality).resolve()),
        }
    disk_probe = output_root.resolve()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free = shutil.disk_usage(disk_probe).free
    return {
        "schema_version": 1,
        "mode": "plan",
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "archives": archive_plans,
        "expected_network_transfer_bytes": sum(
            spec.expected_content_length for spec in ARCHIVES.values()
        ),
        "estimated_selected_disk_bytes": selected_compressed_estimate,
        "recommended_free_disk_bytes": recommended_free,
        "observed_free_disk_bytes": free,
        "free_disk_satisfies_recommendation": free >= recommended_free,
        "output_root": str(output_root.resolve()),
        "split_storage": {
            "train": "<output>/<modality>/train",
            "validation": "<output>/<modality>/validation",
            "test": "<output>/<modality>/sealed_test",
        },
        "sealed_test_policy": (
            "Extract for one-pass transfer integrity only; ordinary loaders refuse "
            "test access unless an explicit future final-evaluation action opts in."
        ),
        "resume_policy": (
            "Resume only at a completed modality whose full compressed stream MD5 "
            "already passed; a partial tar.zst stream restarts from byte zero."
        ),
        "transfer_requires_flag": "--confirm-full-stream-transfer",
    }


def materialize_archive_stream(
    source: BinaryIO,
    *,
    spec: ArchiveSpec,
    allowlist: Mapping[str, Path],
    output_root: Path,
    observed_content_length: int | None,
) -> dict[str, Any]:
    """Validate one complete compressed stream and atomically promote its subset."""

    if (
        observed_content_length is not None
        and observed_content_length != spec.expected_content_length
    ):
        raise MaterializationError(
            f"{spec.modality} Content-Length mismatch: expected "
            f"{spec.expected_content_length}, got {observed_content_length}"
        )
    final_root = output_root / spec.modality
    if final_root.exists():
        return _validated_completed_modality(final_root, spec, len(allowlist))

    quarantine_parent = output_root / ".quarantine"
    quarantine = quarantine_parent / f"{spec.modality}-{uuid.uuid4().hex}"
    payload_root = quarantine / "payload"
    payload_root.mkdir(parents=True, exist_ok=False)
    found: set[str] = set()
    duplicates: list[str] = []
    rejected: list[dict[str, str]] = []
    extracted_bytes = 0
    started = time.monotonic()
    digesting = DigestingReader(
        source,
        total_size=spec.expected_content_length,
        modality=spec.modality,
        selected_total=len(allowlist),
    )

    try:
        try:
            import zstandard  # type: ignore[import-untyped]
        except ImportError as exc:
            raise MaterializationError(
                "zstandard is required; install the project with the multisensor extra"
            ) from exc

        with zstandard.ZstdDecompressor().stream_reader(
            cast(BinaryIO, digesting), read_across_frames=True, closefd=False
        ) as decompressed:
            archive = tarfile.open(fileobj=decompressed, mode="r|")
            try:
                for member in archive:
                    reason = validate_tar_member(member)
                    if reason is not None:
                        rejected.append({"member": member.name, "reason": reason})
                        raise MaterializationError(
                            f"Unsafe or unsupported archive member {member.name!r}: {reason}"
                        )
                    destination = allowlist.get(member.name)
                    if destination is None or member.isdir():
                        continue
                    if member.name in found:
                        duplicates.append(member.name)
                        raise MaterializationError(
                            f"Duplicate selected archive member: {member.name}"
                        )
                    if member.size > MAX_SELECTED_MEMBER_BYTES:
                        raise MaterializationError(
                            f"Selected archive member is implausibly large: {member.name}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise MaterializationError(
                            f"Could not read selected archive member: {member.name}"
                        )
                    target = _safe_destination(payload_root, destination)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    written = _copy_exact(extracted, target, member.size)
                    extracted_bytes += written
                    found.add(member.name)
                    digesting.mark_selected(len(found))
            finally:
                archive.close()
            for _ in iter(lambda: decompressed.read(BUFFER_SIZE), b""):
                pass

        observed_md5 = digesting.hexdigest
        if digesting.bytes_read != spec.expected_content_length:
            raise MaterializationError(
                f"{spec.modality} compressed stream length mismatch: expected "
                f"{spec.expected_content_length}, got {digesting.bytes_read}"
            )
        if observed_md5 != spec.expected_md5:
            raise MaterializationError(
                f"{spec.modality} MD5 mismatch: expected {spec.expected_md5}, "
                f"got {observed_md5}"
            )
        missing = sorted(set(allowlist) - found)
        if missing:
            raise MaterializationError(
                f"{spec.modality} archive is missing {len(missing)} selected members"
            )

        completion = {
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "modality": spec.modality,
            "archive_md5": observed_md5,
            "compressed_bytes_streamed": digesting.bytes_read,
            "selected_members": len(found),
            "extracted_bytes": extracted_bytes,
        }
        _write_json(payload_root / "_materialization.json", completion)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(payload_root, final_root)
        return {
            "completion_state": "complete",
            "expected_content_length": spec.expected_content_length,
            "observed_content_length": observed_content_length,
            "expected_md5": spec.expected_md5,
            "observed_md5": observed_md5,
            "compressed_bytes_streamed": digesting.bytes_read,
            "selected_members_expected": len(allowlist),
            "selected_members_found": len(found),
            "selected_members_missing": [],
            "duplicate_members": duplicates,
            "rejected_unsafe_archive_members": rejected,
            "extracted_bytes": extracted_bytes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        if isinstance(exc, MaterializationError) and not exc.details:
            exc.details.update(
                {
                    "completion_state": "failed",
                    "expected_content_length": spec.expected_content_length,
                    "observed_content_length": observed_content_length,
                    "expected_md5": spec.expected_md5,
                    "observed_md5": digesting.hexdigest,
                    "compressed_bytes_streamed": digesting.bytes_read,
                    "selected_members_expected": len(allowlist),
                    "selected_members_found": len(found),
                    "selected_members_missing": sorted(set(allowlist) - found),
                    "duplicate_members": duplicates,
                    "rejected_unsafe_archive_members": rejected,
                    "extracted_bytes": extracted_bytes,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": str(exc),
                }
            )
        shutil.rmtree(quarantine, ignore_errors=True)
        raise
    finally:
        if quarantine.exists():
            shutil.rmtree(quarantine, ignore_errors=True)


def validate_tar_member(member: tarfile.TarInfo) -> str | None:
    """Return a rejection reason for an unsafe path or unsupported member type."""

    if "\x00" in member.name or "\\" in member.name:
        return "invalid path separator or NUL"
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0].endswith(":")
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        return "absolute, empty, or traversing path"
    if not (member.isfile() or member.isdir()):
        return "links, devices, FIFOs, and special members are forbidden"
    return None


def materialize_from_http(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    report_path: Path,
    modalities: tuple[str, ...] = ("s1", "s2"),
    timeout_seconds: float = 120.0,
    package_output_dir: Path | None = None,
    delete_loose_after_package: bool = False,
) -> dict[str, Any]:
    """Materialize requested modalities sequentially and persist honest progress."""

    disk_probe = output_root.resolve()
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    available = shutil.disk_usage(disk_probe).free
    if available < RECOMMENDED_FREE_DISK_BYTES:
        raise MaterializationError(
            f"Insufficient free disk for bounded extraction: require at least "
            f"{RECOMMENDED_FREE_DISK_BYTES} bytes, found {available}"
        )
    print(
        f"Storage | free {available / 1024**3:.2f} GiB | "
        f"required {RECOMMENDED_FREE_DISK_BYTES / 1024**3:.2f} GiB",
        flush=True,
    )
    report = _base_report()
    for modality in modalities:
        spec = ARCHIVES[modality]
        allowlist = build_allowlist(manifest, spec)
        final_root = output_root / modality
        if final_root.exists():
            result = _validated_completed_modality(final_root, spec, len(allowlist))
        else:
            request = urllib.request.Request(
                spec.url,
                headers={"User-Agent": "SatQuery-Phase4D-Materializer/1.0"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    final_url = response.url
                    if not final_url.startswith("https://zenodo.org/"):
                        raise MaterializationError(
                            f"Unexpected publisher redirect for {modality}: {final_url}"
                        )
                    header = response.headers.get("Content-Length")
                    observed_length = int(header) if header is not None else None
                    result = materialize_archive_stream(
                        response,
                        spec=spec,
                        allowlist=allowlist,
                        output_root=output_root,
                        observed_content_length=observed_length,
                    )
            except MaterializationError as exc:
                report["archives"][modality] = {
                    **report["archives"][modality],
                    "canonical_url": spec.url,
                    **exc.details,
                    "completion_state": "failed",
                    "error": str(exc),
                }
                _write_json(report_path, report)
                raise
            except Exception as exc:
                failure = MaterializationError(
                    f"Canonical {modality} archive transfer failed"
                )
                report["archives"][modality] = {
                    **report["archives"][modality],
                    "canonical_url": spec.url,
                    "completion_state": "failed",
                    "error": str(failure),
                }
                _write_json(report_path, report)
                raise failure from exc
        report["archives"][modality] = {
            "canonical_url": spec.url,
            **result,
            "materialized_pair_counts": {
                "train": SPLIT_COUNTS["train"],
                "validation": SPLIT_COUNTS["validation"],
                "sealed_test": SPLIT_COUNTS["test"],
            },
        }
        if package_output_dir is not None:
            package_path = package_output_dir / f"phase4_{modality}_selected.tar.zst"
            package = package_materialized_modality(
                output_root=output_root,
                package_path=package_path,
                spec=spec,
                allowlist=allowlist,
                delete_loose=delete_loose_after_package,
            )
            report["packages"][modality] = package
            _write_json(
                package_output_dir / "package_manifest.json",
                _package_manifest(report["packages"]),
            )
        _write_json(report_path, report)
    if all(
        report["archives"][key]["completion_state"] in ("complete", "complete_reused")
        for key in ARCHIVES
    ):
        report["materialized_pair_counts"] = {
            "train": SPLIT_COUNTS["train"],
            "validation": SPLIT_COUNTS["validation"],
            "sealed_test": SPLIT_COUNTS["test"],
        }
        _write_json(report_path, report)
    return report


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "archives": {
            key: {
                "canonical_url": spec.url,
                "completion_state": "not_started",
                "expected_content_length": spec.expected_content_length,
                "observed_content_length": None,
                "expected_md5": spec.expected_md5,
                "observed_md5": None,
                "compressed_bytes_streamed": 0,
                "selected_members_expected": None,
                "selected_members_found": 0,
                "selected_members_missing": [],
                "duplicate_members": [],
                "rejected_unsafe_archive_members": [],
                "extracted_bytes": 0,
                "elapsed_seconds": 0.0,
            }
            for key, spec in ARCHIVES.items()
        },
        "packages": {},
        "materialized_pair_counts": {
            "train": 0,
            "validation": 0,
            "sealed_test": 0,
        },
        "test_pixel_accessed": False,
    }


def package_materialized_modality(
    *,
    output_root: Path,
    package_path: Path,
    spec: ArchiveSpec,
    allowlist: Mapping[str, Path],
    delete_loose: bool,
) -> dict[str, Any]:
    """Create and verify one deterministic selected-data package."""

    modality_root = output_root / spec.modality
    _validated_completed_modality(modality_root, spec, len(allowlist))
    relative_paths = sorted(set(allowlist.values()), key=lambda path: path.as_posix())
    if len(relative_paths) != len(allowlist):
        raise MaterializationError(
            f"{spec.modality} package allowlist has duplicate destinations"
        )

    expected = {
        f"{spec.modality}/{relative.as_posix()}": (modality_root / relative).stat().st_size
        for relative in relative_paths
        if (modality_root / relative).is_file()
    }
    if len(expected) != len(relative_paths):
        missing = [
            relative.as_posix()
            for relative in relative_paths
            if not (modality_root / relative).is_file()
        ]
        raise MaterializationError(
            f"Cannot package {spec.modality}; {len(missing)} selected files are missing"
        )
    actual_files = {
        path.relative_to(modality_root).as_posix()
        for path in modality_root.rglob("*")
        if path.is_file() and path.name != "_materialization.json"
    }
    expected_files = {relative.as_posix() for relative in relative_paths}
    if actual_files != expected_files:
        raise MaterializationError(
            f"Cannot package {spec.modality}; materialized tree differs from allowlist"
        )

    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = package_path.with_name(f".{package_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            import zstandard  # type: ignore[import-untyped]
        except ImportError as exc:
            raise MaterializationError(
                "zstandard is required; install the project with the multisensor extra"
            ) from exc
        with temporary.open("xb") as compressed_handle:
            compressor = zstandard.ZstdCompressor(
                level=PACKAGE_COMPRESSION_LEVEL,
                threads=0,
                write_checksum=True,
                write_content_size=False,
            )
            with compressor.stream_writer(compressed_handle, closefd=False) as zstd_stream:
                with tarfile.open(fileobj=zstd_stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for relative in relative_paths:
                        source = modality_root / relative
                        archive_name = f"{spec.modality}/{relative.as_posix()}"
                        info = tarfile.TarInfo(archive_name)
                        info.size = source.stat().st_size
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        with source.open("rb") as source_handle:
                            archive.addfile(info, source_handle)

        _verify_package(temporary, expected)
        package_sha256 = _file_sha256(temporary)
        os.replace(temporary, package_path)
        result = {
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
            "modality": spec.modality,
            "source_archive_url": spec.url,
            "source_archive_md5": spec.expected_md5,
            "source_compressed_bytes": spec.expected_content_length,
            "file_count": len(expected),
            "package_file": package_path.name,
            "package_sha256": package_sha256,
            "package_size_bytes": package_path.stat().st_size,
            "extracted_logical_bytes": sum(expected.values()),
            "sealed_test_paths": sorted(
                name for name in expected if name.startswith(f"{spec.modality}/sealed_test/")
            ),
            "package_creation_parameters": {
                "archive_format": "pax_tar",
                "member_order": "lexicographic_posix_path",
                "member_mtime": 0,
                "member_uid": 0,
                "member_gid": 0,
                "member_mode": "0644",
                "compression": "zstandard",
                "compression_level": PACKAGE_COMPRESSION_LEVEL,
                "compression_threads": 0,
                "content_size_flag": False,
                "frame_checksum": True,
                "byte_reproducible": True,
            },
        }
        if delete_loose:
            _validated_completed_modality(modality_root, spec, len(allowlist))
            shutil.rmtree(modality_root)
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_package(package_path: Path, expected: Mapping[str, int]) -> None:
    try:
        import zstandard  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MaterializationError("zstandard is required to verify package") from exc
    observed: list[tuple[str, int]] = []
    with package_path.open("rb") as compressed_handle:
        with zstandard.ZstdDecompressor().stream_reader(
            compressed_handle, read_across_frames=True
        ) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                for member in archive:
                    reason = validate_tar_member(member)
                    if reason is not None or not member.isfile():
                        raise MaterializationError(
                            f"Package contains unsafe or unsupported member {member.name!r}"
                        )
                    observed.append((member.name, member.size))
    expected_items = list(expected.items())
    if observed != expected_items:
        raise MaterializationError("Package member identity, order, or size verification failed")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_manifest(packages: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "frozen_split_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "packages": dict(packages),
        "sealed_test_policy": (
            "Paths are retained for future one-time final evaluation; packaging does "
            "not decode, inspect, summarize, visualize, or preprocess raster pixels."
        ),
    }


def _validated_completed_modality(
    root: Path, spec: ArchiveSpec, selected_members: int
) -> dict[str, Any]:
    marker_path = root / "_materialization.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(
            f"Existing {spec.modality} output has no valid completion marker"
        ) from exc
    required = {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "modality": spec.modality,
        "archive_md5": spec.expected_md5,
        "compressed_bytes_streamed": spec.expected_content_length,
        "selected_members": selected_members,
    }
    if any(marker.get(key) != value for key, value in required.items()):
        raise MaterializationError(
            f"Existing {spec.modality} completion marker does not match the frozen run"
        )
    return {
        "completion_state": "complete_reused",
        "expected_content_length": spec.expected_content_length,
        "observed_content_length": spec.expected_content_length,
        "expected_md5": spec.expected_md5,
        "observed_md5": spec.expected_md5,
        "compressed_bytes_streamed": spec.expected_content_length,
        "selected_members_expected": selected_members,
        "selected_members_found": selected_members,
        "selected_members_missing": [],
        "duplicate_members": [],
        "rejected_unsafe_archive_members": [],
        "extracted_bytes": marker.get("extracted_bytes", 0),
        "elapsed_seconds": 0.0,
    }


def _safe_destination(root: Path, relative: Path) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterializationError(
            f"Destination escapes quarantine: {relative}"
        ) from exc
    return target


def _copy_exact(source: IO[bytes], target: Path, expected_size: int) -> int:
    written = 0
    with target.open("xb") as handle:
        while True:
            chunk = source.read(BUFFER_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written > expected_size:
                raise MaterializationError(
                    f"Archive member exceeds declared size: {target.name}"
                )
    if written != expected_size:
        raise MaterializationError(
            f"Archive member size mismatch for {target.name}: expected {expected_size}, got {written}"
        )
    return written


def _strip_geographic_suffix(value: str) -> str:
    parts = value.rsplit("_", 3)
    if (
        len(parts) != 4
        or len(parts[1]) != 5
        or not parts[2].isdigit()
        or not parts[3].isdigit()
    ):
        raise MaterializationError(
            f"Cannot derive canonical S1 archive parent from {value}"
        )
    return parts[0]


def _required_text(sample: Mapping[str, Any], field: str) -> str:
    value = sample.get(field)
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"Manifest sample has invalid {field}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--confirm-full-stream-transfer", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--modality", choices=tuple(ARCHIVES), action="append", dest="modalities"
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--package-output-dir", type=Path)
    parser.add_argument("--delete-loose-after-package", action="store_true")
    args = parser.parse_args()
    manifest = load_frozen_manifest(args.manifest)
    if args.plan:
        payload = build_plan(manifest, args.output_root)
        report = _base_report()
        report["plan"] = payload
        for modality in ARCHIVES:
            report["archives"][modality]["selected_members_expected"] = payload[
                "archives"
            ][modality]["selected_members_expected"]
        _write_json(args.report, report)
    else:
        modalities = tuple(dict.fromkeys(args.modalities or ARCHIVES.keys()))
        payload = materialize_from_http(
            manifest,
            output_root=args.output_root,
            report_path=args.report,
            modalities=modalities,
            timeout_seconds=args.timeout_seconds,
            package_output_dir=args.package_output_dir,
            delete_loose_after_package=args.delete_loose_after_package,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
