from __future__ import annotations

from ml.evaluation.probe_phase4_materialization import (
    ResponseSample,
    inspect_zstd_markers,
    parse_content_range,
    summarize_probe,
)


def _response(
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> ResponseSample:
    return ResponseSample(
        status=status,
        headers=headers or {},
        body=body,
        final_url="https://zenodo.org/example.tar.zst",
        runtime_seconds=0.1,
    )


def test_content_range_parser_is_strict() -> None:
    assert parse_content_range("bytes 0-63/1000") == (0, 63, 1000)
    assert parse_content_range("bytes */1000") is None
    assert parse_content_range(None) is None


def test_zstd_marker_check_does_not_confuse_standard_and_seekable_frames() -> None:
    markers = inspect_zstd_markers(
        bytes.fromhex("28b52ffd") + b"payload",
        b"tail-without-seek-table",
    )

    assert markers["standard_zstd_frame_at_start"] is True
    assert markers["seekable_zstd_footer_at_end"] is False


def test_http_ranges_are_not_reported_as_tar_member_random_access() -> None:
    head = _response(
        headers={"content-length": "1000", "content-type": "application/octet-stream"}
    )
    start = _response(
        status=206,
        headers={"content-range": "bytes 0-3/1000"},
        body=bytes.fromhex("28b52ffd"),
    )
    end = _response(
        status=206,
        headers={"content-range": "bytes 996-999/1000"},
        body=b"tail",
    )

    result = summarize_probe(head, start, end)

    assert result["http_range_supported"] is True
    assert result["member_random_access"] is False
    assert result["container_markers"]["seekable_zstd_footer_at_end"] is False
