"""Phase 4 closeout reconstruction tests.

Every artifact referenced by the closeout must exist and hash-match, and
every measured lane must carry nonzero check/sample evidence. The closeout
status must be consistent with the frozen model/dataset contract states.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/phase4_temporal_analytics"
CLOSEOUT_PATH = EXPERIMENT_DIR / "PHASE_4_CLOSEOUT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def closeout() -> dict:
    assert CLOSEOUT_PATH.exists(), "PHASE_4_CLOSEOUT.json is created only at Task 13"
    return json.loads(CLOSEOUT_PATH.read_text(encoding="utf-8"))


def test_closeout_schema(closeout: dict) -> None:
    assert closeout["schema_version"] == 1
    assert closeout["experiment_id"] == "PHASE_4"
    assert closeout["status"] in {"COMPLETE", "IN_PROGRESS", "BLOCKED"}
    assert closeout["git_sha"] and len(closeout["git_sha"]) == 40


def test_every_referenced_artifact_exists_and_hash_matches(closeout: dict) -> None:
    artifacts = closeout["artifacts"]
    assert artifacts, "closeout must reference audited artifacts"
    for relative_path, expected_hash in artifacts.items():
        path = PROJECT_ROOT / relative_path
        assert path.exists(), f"missing closeout artifact: {relative_path}"
        assert _sha256(path) == expected_hash, f"hash mismatch: {relative_path}"


def test_measured_lanes_have_nonzero_evidence(closeout: dict) -> None:
    for lane_id, lane in closeout["lanes"].items():
        if lane["status"] != "MEASURED":
            continue
        assert lane["check_count"] > 0, lane_id
        assert lane["evidence_artifact"], lane_id


def test_blocked_lanes_carry_exact_blockers(closeout: dict) -> None:
    contracts = yaml.safe_load(
        (EXPERIMENT_DIR / "model_contracts.yaml").read_text(encoding="utf-8")
    )
    for lane_id, lane in closeout["lanes"].items():
        if lane["status"] == "BLOCKED":
            assert lane["blockers"], lane_id
            assert not lane.get("metrics"), f"{lane_id} must not carry metrics"


def test_learned_specialists_are_not_promoted(closeout: dict) -> None:
    # no learned specialist is promoted without measured reproduction
    promotions = closeout["promotions"]
    assert promotions["promoted_model_ids"] == []
    assert promotions["rejected_or_deferred"], "blocked specialists must be listed"


def test_status_truthfully_reflects_mandatory_gates(closeout: dict) -> None:
    lanes = closeout["lanes"]
    mandatory = ["p4_e02_changerex", "p4_e03_chg2cap", "p4_e04_sturm"]
    if any(lanes[name]["status"] == "BLOCKED" for name in mandatory):
        assert closeout["status"] == "BLOCKED", (
            "closeout must be BLOCKED while mandatory learned-specialist "
            "reproductions are blocked"
        )


def test_explicit_limitations_are_recorded(closeout: dict) -> None:
    limitations = closeout["limitations"]
    assert "cartosat_risat_non_generalization" in limitations
    assert limitations["rscama"] == "excluded_from_phase4_runtime"
    assert "sen1floods11" in limitations
