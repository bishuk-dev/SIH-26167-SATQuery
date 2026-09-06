"""Deterministic verifiers implementing R-VERIFY-001 through R-VERIFY-006."""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Mapping

from rasterio.crs import CRS
from rasterio.errors import CRSError

from satquery.evidence.models import EvidenceModelProvenance
from satquery.geo.spectral import BAND_ALIASES
from satquery.ingestion.models import Modality, ObservationState
from satquery.verification.models import CheckResult, VerificationReport, VerificationStatus

SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")


class GeometricVerifier:
    """Verifies CRS, bounds, overlap, grid compatibility, and calculation geometry."""

    NAME = "GeometricVerifier"

    @classmethod
    def verify_crs(
        cls,
        crs: str | None,
        *,
        rule_id: str = "GEO_CRS_VALIDITY",
        require_projected: bool = False,
    ) -> CheckResult:
        if crs is None:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="CRS is missing from spatial metadata",
            )

        try:
            parsed = CRS.from_user_input(crs)
        except (CRSError, ValueError) as exc:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"CRS {crs!r} cannot be parsed safely",
                details={"error": str(exc)},
            )

        if require_projected and parsed.is_geographic:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Projected metric CRS required, but geographic angular CRS was provided",
                details={"crs": crs},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"CRS {crs} is valid and verified",
            details={"is_projected": str(parsed.is_projected), "crs": crs},
        )

    @classmethod
    def verify_overlap(
        cls,
        overlap_fraction: float | None,
        *,
        min_overlap: float = 0.01,
        rule_id: str = "GEO_EXTENT_OVERLAP",
    ) -> CheckResult:
        if overlap_fraction is None:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message="Footprint spatial overlap is unknown",
            )

        if overlap_fraction <= 0.0:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Zero spatial overlap between observation footprints",
                details={"overlap_fraction": f"{overlap_fraction:.4f}"},
            )

        if overlap_fraction < min_overlap:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message=f"Low spatial overlap ({overlap_fraction:.2%}) below threshold ({min_overlap:.2%})",
                details={"overlap_fraction": f"{overlap_fraction:.4f}"},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Spatial overlap verified: {overlap_fraction:.2%}",
            details={"overlap_fraction": f"{overlap_fraction:.4f}"},
        )

    @classmethod
    def verify_grid_alignment(
        cls,
        aligned: bool | None,
        *,
        rule_id: str = "GEO_GRID_ALIGNMENT",
    ) -> CheckResult:
        if aligned is True:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.PASS,
                message="Pixel grids are co-registered and aligned",
            )
        if aligned is False:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message="Pixel grids are misaligned or require resampling/reprojection",
            )
        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.WARN,
            message="Grid alignment could not be determined",
        )

    @classmethod
    def verify_mask_dimensions(
        cls,
        mask_shape: tuple[int, ...],
        raster_shape: tuple[int, ...],
        *,
        rule_id: str = "GEO_MASK_DIMENSIONS",
    ) -> CheckResult:
        if len(mask_shape) < 2 or len(raster_shape) < 2:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Mask and raster must be at least two-dimensional",
            )

        if mask_shape[-2:] != raster_shape[-2:]:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Mask shape {mask_shape} does not match raster shape {raster_shape}",
                details={"mask_shape": str(mask_shape), "raster_shape": str(raster_shape)},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message="Mask dimensions match raster spatial dimensions",
        )

    @classmethod
    def verify_calculation_path(
        cls,
        path: str,
        *,
        rule_id: str = "GEO_CALCULATION_PATH",
    ) -> CheckResult:
        valid_paths = {"projected_planar", "geodesic_wgs84", "pixel_counting"}
        if path not in valid_paths:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Unsupported or unsafe area calculation path: {path!r}",
                details={"calculation_path": path},
            )
        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Area calculation path {path!r} is valid and safe",
        )


class TemporalVerifier:
    """Verifies temporal ordering, duplicate prevention, and time delta plausibility."""

    NAME = "TemporalVerifier"

    @classmethod
    def verify_temporal_order(
        cls,
        t1: datetime | None,
        t2: datetime | None,
        *,
        rule_id: str = "TEMP_ORDER_VALIDITY",
    ) -> CheckResult:
        if t1 is None or t2 is None:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Missing acquisition timestamp for temporal pair observation",
            )

        if t1 == t2:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Observations have identical timestamps ({t1.isoformat()}); change requires temporal difference",
                details={"t1": t1.isoformat(), "t2": t2.isoformat()},
            )

        if t1 > t2:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Temporal order inverted: T1 ({t1.isoformat()}) is after T2 ({t2.isoformat()})",
                details={"t1": t1.isoformat(), "t2": t2.isoformat()},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Temporal ordering verified: T1 ({t1.isoformat()}) < T2 ({t2.isoformat()})",
            details={"t1": t1.isoformat(), "t2": t2.isoformat()},
        )

    @classmethod
    def verify_no_duplicate(
        cls,
        obs_id_1: str,
        obs_id_2: str,
        *,
        sha256_1: str | None = None,
        sha256_2: str | None = None,
        rule_id: str = "TEMP_DUPLICATE_CHECK",
    ) -> CheckResult:
        if obs_id_1 == obs_id_2:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Duplicate observation ID {obs_id_1!r} provided for both temporal inputs",
            )

        if sha256_1 is not None and sha256_2 is not None and sha256_1 == sha256_2:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Observations share identical source asset SHA-256 hashes (duplicate imagery)",
                details={"sha256": sha256_1},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message="Temporal pair observations are distinct assets",
        )

    @classmethod
    def verify_time_delta(
        cls,
        delta_seconds: float,
        *,
        min_seconds: float = 1.0,
        max_years: float = 30.0,
        rule_id: str = "TEMP_DELTA_PLAUSIBILITY",
    ) -> CheckResult:
        if not math.isfinite(delta_seconds) or delta_seconds < min_seconds:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Time delta {delta_seconds}s is below minimum threshold ({min_seconds}s)",
            )

        max_seconds = max_years * 365.25 * 86400.0
        if delta_seconds > max_seconds:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message=f"Time delta ({delta_seconds / 86400:.1f} days) exceeds {max_years} years",
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Time delta verified: {delta_seconds / 86400.0:.2f} days",
            details={"delta_seconds": str(delta_seconds)},
        )

    @classmethod
    def verify_seasonality(
        cls,
        t1: datetime | None,
        t2: datetime | None,
        *,
        max_doy_diff: int = 60,
        rule_id: str = "TEMP_SEASONALITY_WARNING",
    ) -> CheckResult:
        if t1 is None or t2 is None:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message="Seasonality check skipped due to missing timestamps",
            )

        doy1 = t1.timetuple().tm_yday
        doy2 = t2.timetuple().tm_yday
        doy_diff = abs(doy1 - doy2)
        # Circular day of year difference (wrap around 365)
        doy_diff = min(doy_diff, 365 - doy_diff)

        if doy_diff > max_doy_diff:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message=f"Observations acquired in different seasons ({doy_diff} days apart in annual cycle); seasonal variation may affect change detection",
                details={"doy_t1": str(doy1), "doy_t2": str(doy2), "doy_diff": str(doy_diff)},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Observations are seasonally aligned ({doy_diff} days apart in annual cycle)",
        )


class PhysicalVerifier:
    """Verifies spectral band existence, physical output ranges, and measurement validity."""

    NAME = "PhysicalVerifier"

    @classmethod
    def verify_required_bands(
        cls,
        available_bands: Mapping[str, Any] | list[str] | tuple[str, ...],
        required_roles: list[str] | tuple[str, ...],
        *,
        rule_id: str = "PHYS_REQUIRED_BANDS",
    ) -> CheckResult:
        if isinstance(available_bands, Mapping):
            band_keys = list(available_bands.keys())
        else:
            band_keys = list(available_bands)

        missing_roles: list[str] = []
        for role in required_roles:
            role_norm = role.lower()
            aliases = BAND_ALIASES.get(role_norm, (role_norm,))
            found = False
            for key in band_keys:
                k_norm = key.strip().lower()
                if k_norm in aliases or any(alias in k_norm for alias in aliases):
                    found = True
                    break
            if not found:
                missing_roles.append(role)

        if missing_roles:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Missing required spectral bands for physical operation: {missing_roles}",
                details={"missing": ",".join(missing_roles), "available": ",".join(band_keys)},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"All required spectral bands present: {list(required_roles)}",
        )

    @classmethod
    def verify_index_range(
        cls,
        min_val: float,
        max_val: float,
        *,
        expected_min: float = -1.0,
        expected_max: float = 1.0,
        tolerance: float = 1e-4,
        rule_id: str = "PHYS_INDEX_RANGE",
    ) -> CheckResult:
        if not math.isfinite(min_val) or not math.isfinite(max_val):
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Spectral index statistics contain non-finite values",
            )

        if min_val < expected_min - tolerance or max_val > expected_max + tolerance:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Spectral index range [{min_val:.4f}, {max_val:.4f}] violates physical bounds [{expected_min}, {expected_max}]",
                details={"min": str(min_val), "max": str(max_val)},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Spectral index values [{min_val:.4f}, {max_val:.4f}] within physical bounds",
        )

    @classmethod
    def verify_measurement_positivity(
        cls,
        value: float,
        *,
        name: str = "Area",
        rule_id: str = "PHYS_MEASUREMENT_POSITIVITY",
    ) -> CheckResult:
        if not math.isfinite(value):
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"{name} measurement is non-finite: {value}",
            )
        if value < 0.0:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"{name} measurement cannot be negative: {value}",
            )
        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"{name} measurement is physically valid ({value})",
        )

    @classmethod
    def verify_modality_compatibility(
        cls,
        modality_a: Modality | str,
        modality_b: Modality | str,
        *,
        rule_id: str = "PHYS_MODALITY_COMPATIBILITY",
    ) -> CheckResult:
        mod_a = Modality(modality_a) if isinstance(modality_a, str) else modality_a
        mod_b = Modality(modality_b) if isinstance(modality_b, str) else modality_b

        if mod_a == Modality.UNKNOWN or mod_b == Modality.UNKNOWN:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message="Unknown sensor modality in pair",
            )

        if mod_a != mod_b:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message=f"Cross-modal pair ({mod_a.value} vs {mod_b.value}); direct pixel differencing may be invalid without cross-modal normalization",
                details={"modality_a": mod_a.value, "modality_b": mod_b.value},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Modalities compatible: {mod_a.value}",
        )


class ProvenanceVerifier:
    """Verifies asset identifiers, SHA256 integrity, and model/tool lineage."""

    NAME = "ProvenanceVerifier"

    @classmethod
    def verify_observation_provenance(
        cls,
        observation_id: str,
        sha256: str | None = None,
        *,
        rule_id: str = "PROV_OBSERVATION_INTEGRITY",
    ) -> CheckResult:
        if not observation_id or not observation_id.strip():
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Observation ID is missing or empty",
            )

        if sha256 is not None and not SHA256_REGEX.match(sha256):
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Invalid SHA-256 hash format: {sha256!r}",
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Observation provenance verified: {observation_id}",
            details={"observation_id": observation_id},
        )

    @classmethod
    def verify_tool_provenance(
        cls,
        tool_name: str,
        tool_version: str | None = None,
        *,
        rule_id: str = "PROV_TOOL_LINEAGE",
    ) -> CheckResult:
        if not tool_name or not tool_name.strip():
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Tool name is missing from execution provenance",
            )
        details = {"tool_name": tool_name}
        if tool_version:
            details["tool_version"] = tool_version
        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Tool provenance verified: {tool_name}",
            details=details,
        )

    @classmethod
    def verify_model_provenance(
        cls,
        model: EvidenceModelProvenance | None,
        *,
        rule_id: str = "PROV_MODEL_LINEAGE",
    ) -> CheckResult:
        if model is None:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.PASS,
                message="Deterministic operation (no ML model involved)",
            )

        if not SHA256_REGEX.match(model.checkpoint_sha256):
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message=f"Invalid model checkpoint SHA-256: {model.checkpoint_sha256}",
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Model provenance verified: {model.model_id} ({model.registry_id})",
            details={"model_id": model.model_id, "registry_id": model.registry_id},
        )


class StatisticalVerifier:
    """Verifies confidence calibration, valid pixel coverage, and statistical reliability."""

    NAME = "StatisticalVerifier"

    @classmethod
    def verify_pixel_coverage(
        cls,
        valid_pixels: int,
        total_pixels: int,
        *,
        min_coverage: float = 0.05,
        rule_id: str = "STAT_VALID_COVERAGE",
    ) -> CheckResult:
        if total_pixels <= 0:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.FAIL,
                message="Total pixel count must be positive",
            )

        ratio = valid_pixels / total_pixels
        if ratio < min_coverage:
            return CheckResult(
                rule_id=rule_id,
                verifier=cls.NAME,
                status=VerificationStatus.WARN,
                message=f"High NoData/invalid fraction: only {ratio:.1%} valid pixels (below {min_coverage:.1%})",
                details={"valid_pixels": str(valid_pixels), "total_pixels": str(total_pixels), "ratio": f"{ratio:.4f}"},
            )

        return CheckResult(
            rule_id=rule_id,
            verifier=cls.NAME,
            status=VerificationStatus.PASS,
            message=f"Valid pixel coverage verified: {ratio:.1%}",
            details={"ratio": f"{ratio:.4f}"},
        )


class VerificationEngine:
    """Unified engine to run multi-dimensional verification suites."""

    @classmethod
    def verify_measurement(
        cls,
        *,
        crs: str | None,
        calculation_path: str,
        area_value: float,
        pixel_count: int,
        tool_name: str = "deterministic_area_measurement",
        source_observation_id: str = "unknown",
    ) -> VerificationReport:
        checks: list[CheckResult] = [
            GeometricVerifier.verify_crs(crs),
            GeometricVerifier.verify_calculation_path(calculation_path),
            PhysicalVerifier.verify_measurement_positivity(area_value, name="Area"),
            PhysicalVerifier.verify_measurement_positivity(float(pixel_count), name="Pixel count"),
            ProvenanceVerifier.verify_observation_provenance(source_observation_id),
            ProvenanceVerifier.verify_tool_provenance(tool_name),
        ]
        return VerificationReport.from_checks(checks)

    @classmethod
    def verify_spectral_index(
        cls,
        *,
        available_bands: Mapping[str, Any] | list[str] | tuple[str, ...],
        required_roles: list[str] | tuple[str, ...],
        min_value: float,
        max_value: float,
        valid_pixels: int,
        total_pixels: int,
        tool_name: str = "spectral_index_computation",
        source_observation_id: str = "unknown",
    ) -> VerificationReport:
        checks: list[CheckResult] = [
            PhysicalVerifier.verify_required_bands(available_bands, required_roles),
            PhysicalVerifier.verify_index_range(min_value, max_value),
            StatisticalVerifier.verify_pixel_coverage(valid_pixels, total_pixels),
            ProvenanceVerifier.verify_observation_provenance(source_observation_id),
            ProvenanceVerifier.verify_tool_provenance(tool_name),
        ]
        return VerificationReport.from_checks(checks)

    @classmethod
    def verify_temporal_pair(
        cls,
        *,
        obs_1: ObservationState,
        obs_2: ObservationState,
        overlap_fraction: float | None = None,
        grid_aligned: bool | None = None,
        tool_name: str = "bitemporal_change_detection",
    ) -> VerificationReport:
        t1 = obs_1.temporal.acquisition_time
        t2 = obs_2.temporal.acquisition_time
        sha1 = obs_1.source_asset.sha256
        sha2 = obs_2.source_asset.sha256

        checks: list[CheckResult] = [
            ProvenanceVerifier.verify_observation_provenance(obs_1.observation_id, sha1),
            ProvenanceVerifier.verify_observation_provenance(obs_2.observation_id, sha2),
            TemporalVerifier.verify_no_duplicate(obs_1.observation_id, obs_2.observation_id, sha256_1=sha1, sha256_2=sha2),
            TemporalVerifier.verify_temporal_order(t1, t2),
            GeometricVerifier.verify_crs(obs_1.geo.crs),
            GeometricVerifier.verify_crs(obs_2.geo.crs),
            GeometricVerifier.verify_overlap(overlap_fraction),
            GeometricVerifier.verify_grid_alignment(grid_aligned),
            PhysicalVerifier.verify_modality_compatibility(obs_1.sensor.modality, obs_2.sensor.modality),
            TemporalVerifier.verify_seasonality(t1, t2),
            ProvenanceVerifier.verify_tool_provenance(tool_name),
        ]

        if t1 is not None and t2 is not None and t1 < t2:
            delta = (t2 - t1).total_seconds()
            checks.append(TemporalVerifier.verify_time_delta(delta))

        return VerificationReport.from_checks(checks)
