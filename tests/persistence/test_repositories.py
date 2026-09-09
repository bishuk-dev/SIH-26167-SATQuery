"""Task 2 — MetadataRepository behavior: records, transitions, pagination,
events, restart recovery, and schema security."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pydantic import ValidationError

from satquery.persistence import (
    AnalysisRecord,
    AnalysisStatus,
    Database,
    ExecutionEvent,
    JobRecord,
    JobStatus,
    MetadataRepository,
    ObservationRecord,
    PageCursor,
    PairRecord,
)
from satquery.persistence.repositories import (
    IllegalTransitionError,
    PersistenceIntegrityError,
    RecordAlreadyExistsError,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _ts(seconds: int = 0) -> datetime:
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def repo(tmp_path: Path) -> MetadataRepository:
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    return MetadataRepository(db)


def _observation(observation_id: str, created_at: datetime, payload: dict | None = None) -> ObservationRecord:
    return ObservationRecord(
        observation_id=observation_id,
        created_at=created_at,
        payload=payload or {"sensor": "TestSat", "crs": "EPSG:32633"},
    )


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------


def test_observation_round_trip(repo: MetadataRepository) -> None:
    record = _observation("obs_" + "a" * 32, _ts())
    repo.create_observation(record)
    loaded = repo.get_observation(record.observation_id)
    assert loaded == record
    assert loaded is not None
    assert loaded.created_at.tzinfo is UTC


def test_duplicate_observation_rejected(repo: MetadataRepository) -> None:
    record = _observation("obs_" + "a" * 32, _ts())
    repo.create_observation(record)
    with pytest.raises(RecordAlreadyExistsError):
        repo.create_observation(record)


def test_canonical_payload_json_stored_identically(
    repo: MetadataRepository, tmp_path: Path
) -> None:
    record = _observation("obs_" + "a" * 32, _ts(), {"z": 1, "a": {"d": 4, "c": 3}})
    repo.create_observation(record)
    with repo._db.transaction() as connection:
        stored = connection.execute(
            "SELECT payload_json FROM observations WHERE observation_id = ?",
            (record.observation_id,),
        ).fetchone()["payload_json"]
    assert stored == '{"a":{"c":3,"d":4},"z":1}'


def test_nan_payload_rejected(repo: MetadataRepository) -> None:
    # NaN survives record construction (plain dict) but canonical encoding
    # rejects it at persistence time
    record = _observation("obs_" + "a" * 32, _ts(), {"score": float("nan")})
    with pytest.raises(ValueError):
        repo.create_observation(record)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        ObservationRecord(
            observation_id="obs_" + "a" * 32,
            created_at=datetime(2026, 1, 1),  # naive
            payload={},
        )


def test_non_utc_datetime_normalized_to_utc(repo: MetadataRepository) -> None:
    plus_two = timezone(timedelta(hours=2))
    record = _observation("obs_" + "a" * 32, datetime(2026, 1, 1, 14, 0, 0, tzinfo=plus_two))
    repo.create_observation(record)
    loaded = repo.get_observation(record.observation_id)
    assert loaded.created_at == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_observation_list_ordering_deterministic(repo: MetadataRepository) -> None:
    for i, hex_char in enumerate("abc"):
        repo.create_observation(_observation("obs_" + hex_char * 32, _ts(i)))
    records = repo.list_observations()
    # created_at DESC, id DESC
    assert [r.observation_id for r in records] == [
        "obs_" + "c" * 32,
        "obs_" + "b" * 32,
        "obs_" + "a" * 32,
    ]


def test_keyset_pagination_across_equal_timestamps(repo: MetadataRepository) -> None:
    same_time = _ts(0)
    for hex_char in "abcd":
        repo.create_observation(_observation("obs_" + hex_char * 32, same_time))
    page1 = repo.list_observations(limit=2)
    assert [r.observation_id[-1] for r in page1] == ["d", "c"]
    cursor = PageCursor(created_at=page1[-1].created_at, record_id=page1[-1].observation_id)
    page2 = repo.list_observations(cursor=cursor, limit=2)
    assert [r.observation_id[-1] for r in page2] == ["b", "a"]
    assert repo.list_observations(cursor=PageCursor(created_at=page2[-1].created_at, record_id=page2[-1].observation_id), limit=2) == ()


def test_page_size_is_bounded(repo: MetadataRepository) -> None:
    with pytest.raises(ValueError):
        repo.list_observations(limit=0)
    with pytest.raises(ValueError):
        repo.list_observations(limit=101)


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------


def _pair(pair_id: str, a: str, b: str, created_at: datetime) -> PairRecord:
    return PairRecord(
        pair_id=pair_id,
        observation_a_id=a,
        observation_b_id=b,
        created_at=created_at,
        payload={"pair_type": "temporal"},
    )


def test_valid_pair_persists(repo: MetadataRepository) -> None:
    a, b = "obs_" + "a" * 32, "obs_" + "b" * 32
    repo.create_observation(_observation(a, _ts()))
    repo.create_observation(_observation(b, _ts()))
    record = _pair("pair_" + "a" * 32, a, b, _ts())
    repo.create_pair(record)
    assert repo.get_pair(record.pair_id) == record


def test_pair_with_same_observation_twice_rejected(repo: MetadataRepository) -> None:
    a = "obs_" + "a" * 32
    repo.create_observation(_observation(a, _ts()))
    with pytest.raises(ValueError):
        _pair("pair_" + "a" * 32, a, a, _ts())


def test_pair_missing_observation_fk_rejected(repo: MetadataRepository) -> None:
    with pytest.raises(PersistenceIntegrityError):
        repo.create_pair(
            _pair("pair_" + "a" * 32, "obs_" + "0" * 32, "obs_" + "1" * 32, _ts())
        )


def test_duplicate_pair_id_rejected(repo: MetadataRepository) -> None:
    a, b = "obs_" + "a" * 32, "obs_" + "b" * 32
    repo.create_observation(_observation(a, _ts()))
    repo.create_observation(_observation(b, _ts()))
    record = _pair("pair_" + "a" * 32, a, b, _ts())
    repo.create_pair(record)
    with pytest.raises(RecordAlreadyExistsError):
        repo.create_pair(record)


def test_pair_payload_is_immutable(repo: MetadataRepository) -> None:
    a, b = "obs_" + "a" * 32, "obs_" + "b" * 32
    repo.create_observation(_observation(a, _ts()))
    repo.create_observation(_observation(b, _ts()))
    record = _pair("pair_" + "a" * 32, a, b, _ts())
    # the record is a frozen model: field reassignment is rejected, and no
    # repository method can update a stored pair payload
    with pytest.raises(ValidationError):
        record.payload = {"pair_type": "mutated"}  # type: ignore[misc]
    assert record.payload["pair_type"] == "temporal"


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------


def _analysis(analysis_id: str, created_at: datetime, status: AnalysisStatus = AnalysisStatus.PENDING) -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id=analysis_id,
        status=status,
        created_at=created_at,
        updated_at=created_at,
        payload={"query_digest": "x"},
    )


def test_analysis_round_trip(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    assert repo.get_analysis(record.analysis_id) == record


def test_analysis_pending_to_running(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    assert repo.transition_analysis(
        record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=_ts(1)
    )
    assert repo.get_analysis(record.analysis_id).status == AnalysisStatus.RUNNING


def test_analysis_running_to_succeeded(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    repo.transition_analysis(record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=_ts(1))
    assert repo.transition_analysis(
        record.analysis_id, AnalysisStatus.RUNNING, AnalysisStatus.SUCCEEDED, updated_at=_ts(2)
    )


def test_analysis_stale_cas_returns_false(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    repo.transition_analysis(record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=_ts(1))
    # expected=PENDING is legal to move to RUNNING, but the row is already
    # RUNNING: the compare-and-set must return False, not raise
    assert not repo.transition_analysis(
        record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=_ts(2)
    )


def test_analysis_illegal_pending_to_succeeded_rejected(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    with pytest.raises(IllegalTransitionError):
        repo.transition_analysis(
            record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.SUCCEEDED, updated_at=_ts(1)
        )


def test_analysis_terminal_state_frozen(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    repo.transition_analysis(record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.ABSTAINED, updated_at=_ts(1))
    with pytest.raises(IllegalTransitionError):
        repo.transition_analysis(
            record.analysis_id, AnalysisStatus.ABSTAINED, AnalysisStatus.RUNNING, updated_at=_ts(2)
        )


def test_analysis_pagination_deterministic(repo: MetadataRepository) -> None:
    for hex_char in "abc":
        repo.create_analysis(_analysis("ana_" + hex_char * 32, _ts()))
    records = repo.list_analyses()
    assert [r.analysis_id[-1] for r in records] == ["c", "b", "a"]


def test_analysis_status_closed_set_enforced_by_database(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as connection:
            connection.execute(
                "INSERT INTO analyses(analysis_id, status, created_at, updated_at, payload_json)"
                " VALUES (?, 'INVENTED', '2026-01-01T00:00:00.000000Z',"
                " '2026-01-01T00:00:00.000000Z', '{}')",
                ("ana_" + "a" * 32,),
            )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _job(job_id: str, analysis_id: str, created_at: datetime) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        analysis_id=analysis_id,
        status=JobStatus.QUEUED,
        created_at=created_at,
        updated_at=created_at,
        payload={"plan_hash": "x"},
    )


def test_job_requires_existing_analysis(repo: MetadataRepository) -> None:
    with pytest.raises(PersistenceIntegrityError):
        repo.create_job(_job("job_" + "a" * 32, "ana_" + "0" * 32, _ts()))


def test_new_job_must_be_queued(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    premature = JobRecord(
        job_id="job_" + "a" * 32,
        analysis_id=analysis.analysis_id,
        status=JobStatus.RUNNING,
        created_at=_ts(),
        updated_at=_ts(),
        payload={},
    )
    with pytest.raises(ValueError):
        repo.create_job(premature)


def test_job_lifecycle_transitions(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    assert repo.transition_job(job.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(1))
    assert repo.transition_job(job.job_id, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, updated_at=_ts(2))
    assert repo.transition_job(job.job_id, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED, updated_at=_ts(3))
    assert repo.get_job(job.job_id).status == JobStatus.CANCELLED


def test_job_illegal_transition_rejected(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    with pytest.raises(IllegalTransitionError):
        repo.transition_job(job.job_id, JobStatus.QUEUED, JobStatus.SUCCEEDED, updated_at=_ts(1))


def test_job_stale_cas_returns_false(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    assert not repo.transition_job(job.job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, updated_at=_ts(1))


def test_job_terminal_transition_rejected(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    repo.transition_job(job.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(1))
    repo.transition_job(job.job_id, JobStatus.RUNNING, JobStatus.FAILED, updated_at=_ts(2))
    with pytest.raises(IllegalTransitionError):
        repo.transition_job(job.job_id, JobStatus.FAILED, JobStatus.RUNNING, updated_at=_ts(3))


def test_direct_invalid_job_status_rejected_by_database(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    with pytest.raises((sqlite3.IntegrityError, PersistenceIntegrityError)):
        with repo._db.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(job_id, analysis_id, status, created_at, updated_at, payload_json)"
                " VALUES (?, ?, 'TELEPORTED', 'x', 'x', '{}')",
                ("job_" + "a" * 32, analysis.analysis_id),
            )


# ---------------------------------------------------------------------------
# restart recovery
# ---------------------------------------------------------------------------


def test_restart_recovery_only_touches_running_jobs(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    running = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    queued = _job("job_" + "b" * 32, analysis.analysis_id, _ts())
    done = _job("job_" + "c" * 32, analysis.analysis_id, _ts())
    for job in (running, queued, done):
        repo.create_job(job)
    repo.transition_job(running.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(1))
    repo.transition_job(done.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(1))
    repo.transition_job(done.job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, updated_at=_ts(2))

    recovered = repo.mark_running_jobs_interrupted(updated_at=_ts(10))

    assert recovered == 1
    assert repo.get_job(running.job_id).status == JobStatus.INTERRUPTED
    assert repo.get_job(queued.job_id).status == JobStatus.QUEUED
    assert repo.get_job(done.job_id).status == JobStatus.SUCCEEDED


def test_restart_recovery_is_idempotent(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    repo.transition_job(job.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(1))
    assert repo.mark_running_jobs_interrupted(updated_at=_ts(10)) == 1
    assert repo.mark_running_jobs_interrupted(updated_at=_ts(11)) == 0


# ---------------------------------------------------------------------------
# execution events
# ---------------------------------------------------------------------------


def _event(event_id: str, job_id: str, sequence: int, created_at: datetime) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        job_id=job_id,
        sequence=sequence,
        event_type="JOB_STARTED",
        created_at=created_at,
        payload={"step": sequence},
    )


@pytest.fixture
def queued_job(repo: MetadataRepository) -> JobRecord:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    return job


def test_events_append_and_list_in_sequence_order(
    repo: MetadataRepository, queued_job: JobRecord
) -> None:
    repo.append_event(_event("event_" + "a" * 32, queued_job.job_id, 0, _ts()))
    repo.append_event(_event("event_" + "b" * 32, queued_job.job_id, 1, _ts(1)))
    events = repo.list_events(queued_job.job_id)
    assert [e.sequence for e in events] == [0, 1]


def test_duplicate_event_sequence_rejected(
    repo: MetadataRepository, queued_job: JobRecord
) -> None:
    repo.append_event(_event("event_" + "a" * 32, queued_job.job_id, 0, _ts()))
    with pytest.raises(PersistenceIntegrityError):
        repo.append_event(_event("event_" + "b" * 32, queued_job.job_id, 0, _ts()))


def test_event_requires_existing_job(repo: MetadataRepository) -> None:
    with pytest.raises(PersistenceIntegrityError):
        repo.append_event(_event("event_" + "a" * 32, "job_" + "0" * 32, 0, _ts()))


def test_event_invalid_sequence_rejected(
    repo: MetadataRepository, queued_job: JobRecord
) -> None:
    with pytest.raises(ValueError):
        ExecutionEvent(
            event_id="event_" + "a" * 32,
            job_id=queued_job.job_id,
            sequence=-1,
            event_type="X",
            created_at=_ts(),
            payload={},
        )


def test_event_sequence_is_contiguous(
    repo: MetadataRepository, queued_job: JobRecord
) -> None:
    repo.append_event(_event("event_" + "a" * 32, queued_job.job_id, 0, _ts()))
    repo.append_event(_event("event_" + "b" * 32, queued_job.job_id, 1, _ts(1)))
    with pytest.raises(PersistenceIntegrityError):
        repo.append_event(_event("event_" + "c" * 32, queued_job.job_id, 5, _ts(2)))


# ---------------------------------------------------------------------------
# schema security
# ---------------------------------------------------------------------------


def test_idempotency_table_stores_hash_not_plaintext(repo: MetadataRepository) -> None:
    with repo._db.transaction() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(idempotency_keys)")
        }
    assert "key_hash" in columns
    for banned in ("raw_key", "idempotency_key", "idempotency_key_plaintext"):
        assert banned not in columns


def test_artifact_table_stores_no_blob_contents(tmp_path: Path) -> None:
    db = Database(tmp_path / "satquery.db")
    db.migrate()
    with db.transaction() as connection:
        columns = list(connection.execute("PRAGMA table_info(artifacts)"))
    assert all(row["type"].upper() != "BLOB" for row in columns)
    assert "storage_key" in {row["name"] for row in columns}


def test_no_delete_api_on_repository(repo: MetadataRepository) -> None:
    for banned in ("delete", "delete_observation", "delete_pair", "delete_analysis",
                   "delete_job", "delete_event", "update_record", "execute_sql",
                   "patch_payload"):
        assert not hasattr(repo, banned)


def test_foreign_keys_enforced_after_reopen(repo: MetadataRepository, tmp_path: Path) -> None:
    a = "obs_" + "a" * 32
    repo.create_observation(_observation(a, _ts()))
    reopened = MetadataRepository(Database(tmp_path / "satquery.db"))
    with pytest.raises(PersistenceIntegrityError):
        reopened.create_pair(
            _pair("pair_" + "a" * 32, a, "obs_" + "1" * 32, _ts())
        )


# ---------------------------------------------------------------------------
# canonical exception contract (Task 2 follow-up FIX 1)
# ---------------------------------------------------------------------------


def test_persistence_integrity_error_is_canonical() -> None:
    import satquery.persistence
    from satquery.persistence.database import (
        PersistenceIntegrityError as A,
    )
    from satquery.persistence.repositories import (
        PersistenceIntegrityError as B,
    )

    assert satquery.persistence.PersistenceIntegrityError is A
    assert A is B


def test_pair_missing_fk_catchable_via_public_export(repo: MetadataRepository) -> None:
    import satquery.persistence

    with pytest.raises(satquery.persistence.PersistenceIntegrityError):
        repo.create_pair(
            _pair("pair_" + "e" * 32, "obs_" + "0" * 32, "obs_" + "1" * 32, _ts())
        )


def test_job_missing_analysis_catchable_via_public_export(
    repo: MetadataRepository,
) -> None:
    import satquery.persistence

    with pytest.raises(satquery.persistence.PersistenceIntegrityError):
        repo.create_job(_job("job_" + "e" * 32, "ana_" + "0" * 32, _ts()))


def test_invalid_event_sequence_catchable_via_public_export(
    repo: MetadataRepository,
) -> None:
    import satquery.persistence

    with pytest.raises(satquery.persistence.PersistenceIntegrityError):
        repo.append_event(_event("event_" + "e" * 32, "job_" + "0" * 32, 0, _ts()))


# ---------------------------------------------------------------------------
# read/write transaction boundary (Task 2 follow-up FIX 2)
# ---------------------------------------------------------------------------


def test_reader_does_not_reserve_writer_slot(tmp_path: Path) -> None:
    """WAL concurrency: an open read transaction must not block a writer."""


    db_path = tmp_path / "satquery.db"
    reader_db = Database(db_path)
    reader_db.migrate()
    reader = MetadataRepository(reader_db)
    reader.create_observation(_observation("obs_" + "a" * 32, _ts()))

    writer_db = Database(db_path)
    writer = MetadataRepository(writer_db)

    with reader_db.read_transaction() as reader_connection:
        reader_connection.execute("SELECT COUNT(*) FROM observations").fetchone()
        # while the read transaction is open, a second connection must be
        # able to commit a write (old BEGIN IMMEDIATE reads blocked this)
        writer.create_observation(
            _observation("obs_" + "b" * 32, _ts(1))
        )

    assert writer.get_observation("obs_" + "b" * 32) is not None


# ---------------------------------------------------------------------------
# timestamp monotonicity (Task 2 follow-up hardening)
# ---------------------------------------------------------------------------


def test_job_updated_at_cannot_move_backwards(repo: MetadataRepository) -> None:
    analysis = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(analysis)
    job = _job("job_" + "a" * 32, analysis.analysis_id, _ts())
    repo.create_job(job)
    repo.transition_job(job.job_id, JobStatus.QUEUED, JobStatus.RUNNING, updated_at=_ts(5))
    # a later transition stamped before the persisted updated_at is stale
    assert not repo.transition_job(
        job.job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, updated_at=_ts(3)
    )
    # a properly ordered transition still succeeds afterwards
    assert repo.transition_job(
        job.job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, updated_at=_ts(6)
    )


def test_analysis_updated_at_cannot_move_backwards(repo: MetadataRepository) -> None:
    record = _analysis("ana_" + "a" * 32, _ts())
    repo.create_analysis(record)
    repo.transition_analysis(
        record.analysis_id, AnalysisStatus.PENDING, AnalysisStatus.RUNNING, updated_at=_ts(5)
    )
    assert not repo.transition_analysis(
        record.analysis_id, AnalysisStatus.RUNNING, AnalysisStatus.SUCCEEDED, updated_at=_ts(1)
    )


def test_record_updated_at_before_created_at_rejected() -> None:
    with pytest.raises(ValueError):
        AnalysisRecord(
            analysis_id="ana_" + "a" * 32,
            status=AnalysisStatus.PENDING,
            created_at=_ts(10),
            updated_at=_ts(5),
            payload={},
        )
    with pytest.raises(ValueError):
        JobRecord(
            job_id="job_" + "a" * 32,
            analysis_id="ana_" + "a" * 32,
            status=JobStatus.QUEUED,
            created_at=_ts(10),
            updated_at=_ts(5),
            payload={},
        )
