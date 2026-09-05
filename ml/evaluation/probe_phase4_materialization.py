"""Run bounded HTTP probes against the canonical BigEarthNet v2 archives.

The probe reads only archive headers and tails. It does not download or extract
imagery, and it deliberately does not treat HTTP byte ranges as tar-member
random access.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/phase4_bigearthnet_multisensor/results/access_probe.json"
)

ARCHIVES = {
    "sentinel_1": (
        "https://zenodo.org/records/10891137/files/" "BigEarthNet-S1.tar.zst?download=1"
    ),
    "sentinel_2": (
        "https://zenodo.org/records/10891137/files/" "BigEarthNet-S2.tar.zst?download=1"
    ),
}

ZSTD_FRAME_MAGIC = bytes.fromhex("28b52ffd")
ZSTD_SEEKABLE_FOOTER_MAGIC = bytes.fromhex("b1ea928f")
CONTENT_RANGE_RE = re.compile(r"^bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)$")


class AccessProbeError(RuntimeError):
    """Raised when a bounded archive probe cannot establish its contract."""


@dataclass(frozen=True, slots=True)
class ResponseSample:
    """A bounded portion of one HTTP response."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str
    runtime_seconds: float


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    """Parse a complete HTTP byte Content-Range header."""
    if value is None:
        return None
    match = CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None:
        return None
    return (
        int(match.group("start")),
        int(match.group("end")),
        int(match.group("total")),
    )


def inspect_zstd_markers(start: bytes, end: bytes) -> dict[str, object]:
    """Classify the observable zstd framing without claiming a full parse."""
    return {
        "start_magic_hex": start[:4].hex(),
        "standard_zstd_frame_at_start": start.startswith(ZSTD_FRAME_MAGIC),
        "seekable_zstd_footer_at_end": end.endswith(ZSTD_SEEKABLE_FOOTER_MAGIC),
        "end_sample_hex": end.hex(),
    }


def summarize_probe(
    head: ResponseSample,
    start: ResponseSample,
    end: ResponseSample,
) -> dict[str, object]:
    """Build the stable, serializable result for one archive."""
    content_range = parse_content_range(start.headers.get("content-range"))
    end_content_range = parse_content_range(end.headers.get("content-range"))
    content_length = int(head.headers["content-length"])
    if content_range is not None and content_range[2] != content_length:
        raise AccessProbeError("HEAD and range response lengths disagree")
    if end_content_range is not None and end_content_range[2] != content_length:
        raise AccessProbeError("HEAD and suffix-range response lengths disagree")

    range_supported = (
        start.status == 206
        and content_range is not None
        and content_range[0] == 0
        and len(start.body) == content_range[1] + 1
    )
    markers = inspect_zstd_markers(start.body, end.body)
    return {
        "canonical_url": head.final_url,
        "content_length_bytes": content_length,
        "http_range_supported": range_supported,
        "content_range": start.headers.get("content-range"),
        "suffix_content_range": end.headers.get("content-range"),
        "content_type": head.headers.get("content-type"),
        "container_markers": markers,
        "member_random_access": False,
        "member_random_access_reason": (
            "HTTP ranges address compressed bytes, while tar member offsets address "
            "the decompressed stream. No seekable-zstd footer or published member "
            "index was observed, so arbitrary member extraction still requires "
            "sequential decompression from the beginning."
        ),
        "bytes_read": len(start.body) + len(end.body),
        "request_runtime_seconds": round(
            head.runtime_seconds + start.runtime_seconds + end.runtime_seconds, 3
        ),
    }


def _request(
    url: str,
    *,
    method: str,
    range_header: str | None,
    read_limit: int,
    timeout_seconds: float,
) -> ResponseSample:
    headers = {"User-Agent": "SatQuery-Phase4C-Access-Probe/1.0"}
    if range_header is not None:
        headers["Range"] = range_header
    request = urllib.request.Request(url, headers=headers, method=method)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = b"" if method == "HEAD" else response.read(read_limit)
        response_headers = {
            key.lower(): value for key, value in response.headers.items()
        }
        return ResponseSample(
            status=response.status,
            headers=response_headers,
            body=body,
            final_url=response.url,
            runtime_seconds=time.monotonic() - started,
        )


def probe_archive(
    url: str,
    *,
    sample_bytes: int = 64,
    timeout_seconds: float = 45.0,
) -> dict[str, object]:
    """Probe one URL while reading at most twice ``sample_bytes`` body bytes."""
    if sample_bytes < 4:
        raise ValueError("sample_bytes must be at least 4")
    head = _request(
        url,
        method="HEAD",
        range_header=None,
        read_limit=0,
        timeout_seconds=timeout_seconds,
    )
    start = _request(
        url,
        method="GET",
        range_header=f"bytes=0-{sample_bytes - 1}",
        read_limit=sample_bytes,
        timeout_seconds=timeout_seconds,
    )
    end = _request(
        url,
        method="GET",
        range_header=f"bytes=-{sample_bytes}",
        read_limit=sample_bytes,
        timeout_seconds=timeout_seconds,
    )
    return summarize_probe(head, start, end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-bytes", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    args = parser.parse_args()

    results = {
        sensor: probe_archive(
            url,
            sample_bytes=args.sample_bytes,
            timeout_seconds=args.timeout_seconds,
        )
        for sensor, url in ARCHIVES.items()
    }
    payload = {
        "schema_version": 1,
        "probe": "bounded_head_prefix_suffix",
        "body_byte_limit_per_archive": args.sample_bytes * 2,
        "archives": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
