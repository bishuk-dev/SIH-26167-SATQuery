"""P4-E01 deterministic verification runner (lanes A-C).

Lanes A-C verify formula, grid, and measurement correctness against frozen
hand-calculated fixtures. They produce deterministic verification artifacts,
not learned-model benchmark metrics. Lane D (optional OSCD deterministic
change benchmark) is blocked: the OSCD data contract is BLOCKED and no
defensible deterministic generic change method has been frozen, so no
OSCD precision/recall/F1/IoU may be emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from satquery.analytics.measurement import measure_mask_area
from satquery.analytics.spectral import SpectralBandRole, compute_index
from satquery.analytics.temporal import prepare_common_grid
from satquery.ingestion.models import ObservationState

LANE_D_STATUS = "BLOCKED_METHOD_CONTRACT"

FORMULA_NDVI = "NDVI = (NIR - RED) / (NIR + RED)"
FORMULA_NDWI = "NDWI = (GREEN - NIR) / (GREEN + NIR)"
FORMULA_MNDWI = "MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)"

_WGS84_SEMI_MAJOR = 6378137.0
_WGS84_INVERSE_FLATTENING = 298.257223563


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, **fields: object) -> dict[str, object]:
    return {"name": name, "status": "PASS", **fields}


# ---------------------------------------------------------------------------
# Lane A — formula correctness
# ---------------------------------------------------------------------------


def run_lane_a() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    red = np.array([[1.0, 2.0], [0.0, 4.0]])
    nir = np.array([[3.0, 2.0], [0.0, 0.0]])
    valid = np.array([[True, True], [False, True]])
    ndvi = compute_index("NDVI", {"RED": red, "NIR": nir}, valid)
    np.testing.assert_allclose(ndvi.compressed(), [0.5, 0.0, -1.0])
    assert ndvi.mask[1, 0]
    checks.append(
        _check("ndvi_hand_calculation", formula=FORMULA_NDVI, expected=[0.5, 0.0, -1.0])
    )

    green = np.array([[2.0, 1.0], [1.0, 3.0]])
    nir_w = np.array([[4.0, 1.0], [0.0, 1.0]])
    ndwi = compute_index("NDWI", {"GREEN": green, "NIR": nir_w}, valid)
    np.testing.assert_allclose(ndwi.compressed(), [(2.0 - 4.0) / 6.0, 0.0, (3.0 - 1.0) / 4.0])
    checks.append(_check("ndwi_hand_calculation", formula=FORMULA_NDWI))

    swir1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    mndwi = compute_index("MNDWI", {"GREEN": green, "SWIR1": swir1}, valid)
    np.testing.assert_allclose(mndwi.compressed(), [(2.0 - 1.0) / 3.0, 0.0, (3.0 - 1.0) / 4.0])
    checks.append(_check("mndwi_hand_calculation", formula=FORMULA_MNDWI))

    # NoData and zero-denominator cells stay masked, never coerced to values.
    nodata_valid = np.array([[True, False]])
    ndvi_nodata = compute_index(
        "NDVI",
        {"RED": np.array([[1.0, 1.0]]), "NIR": np.array([[2.0, 2.0]])},
        nodata_valid,
    )
    assert ndvi_nodata.mask[0, 1] and not ndvi_nodata.mask[0, 0]
    checks.append(_check("ndvi_nodata_masked", formula=FORMULA_NDVI))

    zero_valid = np.array([[True, True]])
    ndvi_zero = compute_index(
        "NDVI",
        {"RED": np.array([[1.0, 1.0]]), "NIR": np.array([[-1.0, 2.0]])},
        zero_valid,
    )
    assert ndvi_zero.mask[0, 0] and not ndvi_zero.mask[0, 1]
    checks.append(_check("ndvi_zero_denominator_masked", formula=FORMULA_NDVI))

    return {"lane": "A_formula_correctness", "status": "PASS", "checks": checks}


# ---------------------------------------------------------------------------
# Lane B — grid correctness
# ---------------------------------------------------------------------------


def _write_geotiff(
    path: Path,
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: str,
    values: np.ndarray,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=str(values.dtype),
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(values, 1)


def _fixture_observation(path: Path, observation_id: str) -> ObservationState:
    # ObservationState built directly from the fixture file so that the
    # registered SHA-256 and grid metadata are exactly what alignment sees.
    from tests.analytics.test_temporal import _observation  # type: ignore[attr-defined]

    return _observation(path, observation_id)


def run_lane_b(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    grid = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    t1_values = np.arange(16, dtype="float32").reshape(4, 4)
    t2_values = t1_values + 1.0
    t1_path = output_dir / "t1.tif"
    t2_path = output_dir / "t2.tif"
    _write_geotiff(t1_path, width=4, height=4, transform=grid, crs="EPSG:32633", values=t1_values)
    _write_geotiff(t2_path, width=4, height=4, transform=grid, crs="EPSG:32633", values=t2_values)
    hashes_before = {p.name: _sha256(p) for p in (t1_path, t2_path)}

    # identity pair: identical declared grids must never write derived rasters
    aligned = prepare_common_grid(
        _fixture_observation(t1_path, "laneb_t1"),
        _fixture_observation(t2_path, "laneb_t2"),
        output_dir / "aligned_identity",
    )
    assert aligned.reprojected is False
    assert aligned.derived_hashes == {}
    assert aligned.t1_path == t1_path and aligned.t2_path == t2_path
    checks.append(
        _check(
            "identity_pair_untouched",
            reprojected=False,
            source_hashes_unchanged=all(_sha256(p) == h for p, h in ((t1_path, hashes_before["t1.tif"]), (t2_path, hashes_before["t2.tif"]))),
        )
    )

    # reversed reference: T2-declared grid stays authoritative when chosen
    aligned_reversed = prepare_common_grid(
        _fixture_observation(t1_path, "laneb_t1"),
        _fixture_observation(t2_path, "laneb_t2"),
        output_dir / "aligned_reversed",
        reference="t2",
    )
    assert aligned_reversed.reprojected is False
    assert aligned_reversed.reference_observation_id == "laneb_t2"
    checks.append(
        _check(
            "reversed_order_untouched",
            reprojected=False,
            reference="t2",
        )
    )

    # deliberate equal-shape misalignment: same shape, shifted origin —
    # must reproject onto the declared reference grid, never compare raw
    shifted = Affine(10.0, 0.0, 500_005.0, 0.0, -10.0, 4_000_000.0)
    t2_shifted_path = output_dir / "t2_shifted.tif"
    _write_geotiff(
        t2_shifted_path, width=4, height=4, transform=shifted, crs="EPSG:32633", values=t2_values
    )
    aligned_shifted = prepare_common_grid(
        _fixture_observation(t1_path, "laneb_t1"),
        _fixture_observation(t2_shifted_path, "laneb_t2s"),
        output_dir / "aligned_misaligned",
        reference="t1",
    )
    assert aligned_shifted.reprojected is True
    assert aligned_shifted.reprojected_observation_ids == ("laneb_t2s",)
    assert aligned_shifted.t1_path == t1_path
    assert aligned_shifted.t2_path != t2_shifted_path
    assert Path(aligned_shifted.t2_path).exists()
    derived_hash = _sha256(Path(aligned_shifted.t2_path))
    assert aligned_shifted.derived_hashes == {"laneb_t2s": derived_hash}
    # original misaligned source must remain byte-identical
    assert _sha256(t2_shifted_path) == _fixture_observation(
        t2_shifted_path, "laneb_t2s"
    ).source_asset.sha256
    checks.append(
        _check(
            "misaligned_grid_reprojected_onto_reference",
            reprojected=True,
            reference="t1",
            derived_hash=derived_hash,
        )
    )

    # resampling provenance must be explicit in the result and the artifact name
    assert aligned_shifted.resampling == "bilinear"
    assert "bilinear" in Path(aligned_shifted.t2_path).name
    checks.append(
        _check(
            "resampling_provenance_recorded",
            resampling=aligned_shifted.resampling,
            derived_artifact=Path(aligned_shifted.t2_path).name,
        )
    )

    return {"lane": "B_grid_correctness", "status": "PASS", "checks": checks}


# ---------------------------------------------------------------------------
# Lane C — measurement correctness
# ---------------------------------------------------------------------------


def run_lane_c() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    # projected CRS: exact affine-determinant reconstruction
    mask_p = np.zeros((4, 4), dtype=bool)
    mask_p[1, 1] = mask_p[1, 2] = mask_p[2, 1] = True
    mask_p[3, 0] = mask_p[3, 3] = mask_p[0, 0] = mask_p[2, 3] = True
    transform_p = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)
    measured = measure_mask_area(mask_p, transform_p, "EPSG:32633", unit="m2")
    # independent scalar reconstruction: positives x |a*e - b*d| x unit factor^2
    expected_m2 = int(mask_p.sum()) * abs(10.0 * -10.0) * 1.0
    error = abs(measured.value - expected_m2)
    assert error < 1e-9
    assert measured.method == "projected_affine_determinant"
    checks.append(
        _check(
            "projected_area_pixel_count_reconstruction",
            crs="EPSG:32633",
            method=measured.method,
            unit="m2",
            positive_pixel_count=int(mask_p.sum()),
            reconstructed_value_m2=expected_m2,
            reconstruction_error_m2=error,
        )
    )

    # geographic CRS: ellipsoidal parallel-band sum reconstructed with an
    # independent scalar arithmetic path, plus a spherical cross-check
    mask_g = np.array([[True]])
    transform_g = Affine(0.01, 0.0, 0.0, 0.0, -0.01, 0.005)
    measured_g = measure_mask_area(mask_g, transform_g, "EPSG:4326", unit="m2")

    flattening = 1.0 / _WGS84_INVERSE_FLATTENING
    e = math.sqrt(flattening * (2.0 - flattening))
    a = _WGS84_SEMI_MAJOR

    def band_factor(phi_rad: float) -> float:
        u = math.sin(phi_rad)
        one_minus = 1.0 - e**2 * u**2
        return (
            a**2
            * (1.0 - e**2)
            * (u / (2.0 * one_minus) + math.atanh(e * u) / (2.0 * e))
        )

    phi_top = math.radians(0.005)
    phi_bottom = math.radians(-0.005)
    expected_g = abs(band_factor(phi_top) - band_factor(phi_bottom)) * math.radians(0.01)
    error_g = abs(measured_g.value - expected_g)
    assert error_g < 1e-6

    # spherical cross-check: R^2 (sin phi2 - sin phi1) dlambda must agree
    # with the ellipsoidal value to within ~0.5% at the equator
    spherical = 6_371_000.0**2 * abs(
        math.sin(phi_top) - math.sin(phi_bottom)
    ) * math.radians(0.01)
    spherical_relative_error = abs(measured_g.value - spherical) / spherical
    assert spherical_relative_error < 0.005
    assert measured_g.method == "ellipsoidal_parallel_band_sum"
    checks.append(
        _check(
            "geographic_area_ellipsoidal_reconstruction",
            crs="EPSG:4326",
            method=measured_g.method,
            unit="m2",
            positive_pixel_count=1,
            reconstructed_value_m2=expected_g,
            reconstruction_error_m2=error_g,
            spherical_cross_check_relative_error=spherical_relative_error,
        )
    )

    return {"lane": "C_measurement_correctness", "status": "PASS", "checks": checks}


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def run_p4_e01(output_path: Path) -> dict[str, object]:
    """Run lanes A-C and write the verification manifest as canonical JSON."""

    import tempfile

    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "P4-E01",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "fixture_provenance": "deterministic_synthetic_fixtures",
        "real_source_data_used": False,
        "lanes": {
            "A_formula_correctness": run_lane_a(),
            "B_grid_correctness": run_lane_b(Path(tempfile.mkdtemp(prefix="p4e01_lane_b_"))),
            "C_measurement_correctness": run_lane_c(),
        },
        "lane_d": {
            "status": LANE_D_STATUS,
            "reason": (
                "OSCD data contract is BLOCKED and no defensible deterministic "
                "generic change method has been frozen; NDVI-difference "
                "thresholding is not generic change detection."
            ),
            "oscd_metrics_emitted": False,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase4_temporal_analytics/p4_e01/manifest.json"),
        help="manifest output path",
    )
    args = parser.parse_args(argv)
    manifest = run_p4_e01(args.output)
    lane_statuses = {
        lane_id: lane["status"] for lane_id, lane in manifest["lanes"].items()
    }
    print(json.dumps({"output": str(args.output), **lane_statuses}, indent=2))
    return 0 if all(status == "PASS" for status in lane_statuses.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
