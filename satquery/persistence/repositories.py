"""Strict persistence records and the metadata repository.

Records are frozen Pydantic contracts (same conventions as ``ContractModel``).
The persistence layer validates and stores server-generated IDs — it never
generates random IDs itself. Contract payloads are encoded as canonical JSON;
timestamps are normalized to UTC text so lexical order preserves chronology.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from satquery.ingestion.models import ContractModel
from satquery.persistence.database import (
    Database,
    PersistenceError,
    PersistenceIntegrityError,
)

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class RecordAlreadyExistsError(PersistenceError):
    """A record with the same primary key already exists."""


class IllegalTransitionError(PersistenceError):
    """A status transition outside the frozen state machine was requested."""


# ---------------------------------------------------------------------------
# state enums (frozen at Task 0; no extra states)
# ---------------------------------------------------------------------------


class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABSTAINED = "ABSTAINED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


_ANALYSIS_TRANSITIONS: dict[AnalysisStatus, frozenset[AnalysisStatus]] = {
    AnalysisStatus.PENDING: frozenset(
        {
            AnalysisStatus.RUNNING,
            AnalysisStatus.FAILED,
            AnalysisStatus.ABSTAINED,
            AnalysisStatus.REJECTED,
            AnalysisStatus.CANCELLED,
        }
    ),
    AnalysisStatus.RUNNING: frozenset(
        {
            AnalysisStatus.SUCCEEDED,
            AnalysisStatus.FAILED,
            AnalysisStatus.ABSTAINED,
            AnalysisStatus.CANCELLED,
            AnalysisStatus.INTERRUPTED,
        }
    ),
    # terminal states transition nowhere
}
for _terminal in (
    AnalysisStatus.SUCCEEDED,
    AnalysisStatus.FAILED,
    AnalysisStatus.ABSTAINED,
    AnalysisStatus.REJECTED,
    AnalysisStatus.CANCELLED,
    AnalysisStatus.INTERRUPTED,
):
    _ANALYSIS_TRANSITIONS.setdefault(_terminal, frozenset())


_JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.INTERRUPTED,
        }
    ),
    JobStatus.CANCEL_REQUESTED: frozenset(
        {JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.INTERRUPTED}
    ),
}
for _terminal in (
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.INTERRUPTED,
):
    _JOB_TRANSITIONS.setdefault(_terminal, frozenset())


# ---------------------------------------------------------------------------
# canonical encoding
# ---------------------------------------------------------------------------

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def encode_timestamp(value: datetime) -> str:
    """Normalize a timezone-aware datetime to canonical UTC text.

    Naive datetimes are rejected; aware non-UTC values are converted. The
    fixed-width ``...ffffffZ`` form keeps lexical ordering chronological.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT) + "Z"


def decode_timestamp(text: str) -> datetime:
    value = datetime.strptime(text, _TIMESTAMP_FORMAT + "Z")
    return value.replace(tzinfo=timezone.utc)


def canonical_json(payload: Any) -> str:
    """Frozen canonical encoding: sorted keys, no whitespace, no NaN/Inf."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# ---------------------------------------------------------------------------
# records (frozen, strict)
# ---------------------------------------------------------------------------


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class _PersistenceModel(ContractModel):
    # only models that declare these fields normalize them; check_fields=False
    # keeps the shared validator applicable across the record family
    @model_validator(mode="after")
    def _require_monotonic_timestamps(self) -> "_PersistenceModel":
        updated = getattr(self, "updated_at", None)
        created = getattr(self, "created_at", None)
        if updated is not None and created is not None and updated < created:
            raise ValueError("updated_at must not precede created_at")
        return self

    @field_validator("created_at", "updated_at", check_fields=False)
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)


class ObservationRecord(_PersistenceModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    created_at: datetime
    payload: dict[str, Any]


class PairRecord(_PersistenceModel):
    pair_id: str = Field(pattern=r"^pair_[0-9a-f]{32}$")
    observation_a_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    observation_b_id: str = Field(pattern=r"^obs_[0-9a-f]{32}$")
    created_at: datetime
    payload: dict[str, Any]

    @field_validator("observation_b_id")
    @classmethod
    def _require_distinct_observations(cls, value: str, info: Any) -> str:
        a = info.data.get("observation_a_id")
        if a is not None and value == a:
            raise ValueError("pair observations must be distinct")
        return value


class AnalysisRecord(_PersistenceModel):
    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any]


class JobRecord(_PersistenceModel):
    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    analysis_id: str = Field(pattern=r"^ana_[0-9a-f]{32}$")
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any]


class ExecutionEvent(_PersistenceModel):
    event_id: str = Field(pattern=r"^event_[0-9a-f]{32}$")
    job_id: str = Field(pattern=r"^job_[0-9a-f]{32}$")
    sequence: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    created_at: datetime
    payload: dict[str, Any]


class PageCursor(_PersistenceModel):
    """Internal keyset cursor: ``created_at DESC, id DESC`` boundary."""

    created_at: datetime
    record_id: str = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        return _require_utc(value)


_PAGE_LIMIT_MIN = 1
_PAGE_LIMIT_MAX = 100


# ---------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------


class MetadataRepository:
    """Append-only metadata persistence with compare-and-set transitions.

    Observation and pair scientific metadata are immutable: no repository
    method updates their payloads or identifiers. Only analyses and jobs
    change status, through explicit compare-and-set methods.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- shared helpers ----------------------------------------------------

    @staticmethod
    def _wrap_integrity_error(exc: sqlite3.IntegrityError, table: str) -> PersistenceError:
        message = str(exc)
        if message.startswith("UNIQUE constraint failed") and table in message:
            return RecordAlreadyExistsError(
                f"a {table} record with this ID already exists"
            )
        if "FOREIGN KEY constraint failed" in message:
            return PersistenceIntegrityError(f"referenced {table} record is missing")
        return PersistenceIntegrityError(f"{table} persistence failed: {message}")

    def _page_clause(
        self, cursor: PageCursor | None, table: str, id_column: str
    ) -> tuple[str, list[Any]]:
        if cursor is None:
            return "", []
        return (
            f"WHERE (created_at < ?) OR (created_at = ? AND {id_column} < ?)",
            [
                encode_timestamp(cursor.created_at),
                encode_timestamp(cursor.created_at),
                cursor.record_id,
            ],
        )

    @staticmethod
    def _validate_limit(limit: int) -> int:
        if not _PAGE_LIMIT_MIN <= limit <= _PAGE_LIMIT_MAX:
            raise ValueError(
                f"page limit must be between {_PAGE_LIMIT_MIN} and {_PAGE_LIMIT_MAX}"
            )
        return limit

    # -- observations (immutable) ------------------------------------------

    def create_observation(self, record: ObservationRecord) -> None:
        with self._db.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO observations(observation_id, created_at, payload_json)"
                    " VALUES (?, ?, ?)",
                    (
                        record.observation_id,
                        encode_timestamp(record.created_at),
                        canonical_json(record.payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._wrap_integrity_error(exc, "observations") from exc

    def get_observation(self, observation_id: str) -> ObservationRecord | None:
        with self._db.read_transaction() as connection:
            row = connection.execute(
                "SELECT observation_id, created_at, payload_json FROM observations"
                " WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._observation_from_row(row) if row is not None else None

    def list_observations(
        self, cursor: PageCursor | None = None, limit: int = 50
    ) -> tuple[ObservationRecord, ...]:
        self._validate_limit(limit)
        clause, parameters = self._page_clause(cursor, "observations", "observation_id")
        with self._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT observation_id, created_at, payload_json FROM observations"
                f" {clause} ORDER BY created_at DESC, observation_id DESC LIMIT ?",
                [*parameters, limit],
            ).fetchall()
        return tuple(self._observation_from_row(row) for row in rows)

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> ObservationRecord:
        return ObservationRecord(
            observation_id=row["observation_id"],
            created_at=decode_timestamp(row["created_at"]),
            payload=json.loads(row["payload_json"]),
        )

    # -- pairs (immutable) ---------------------------------------------------

    def create_pair(self, record: PairRecord) -> None:
        with self._db.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO pairs(pair_id, observation_a_id, observation_b_id,"
                    " created_at, payload_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.pair_id,
                        record.observation_a_id,
                        record.observation_b_id,
                        encode_timestamp(record.created_at),
                        canonical_json(record.payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._wrap_integrity_error(exc, "pairs") from exc

    def get_pair(self, pair_id: str) -> PairRecord | None:
        with self._db.read_transaction() as connection:
            row = connection.execute(
                "SELECT pair_id, observation_a_id, observation_b_id, created_at,"
                " payload_json FROM pairs WHERE pair_id = ?",
                (pair_id,),
            ).fetchone()
        return self._pair_from_row(row) if row is not None else None

    def list_pairs(
        self, cursor: PageCursor | None = None, limit: int = 50
    ) -> tuple[PairRecord, ...]:
        self._validate_limit(limit)
        clause, parameters = self._page_clause(cursor, "pairs", "pair_id")
        with self._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT pair_id, observation_a_id, observation_b_id, created_at,"
                f" payload_json FROM pairs {clause}"
                " ORDER BY created_at DESC, pair_id DESC LIMIT ?",
                [*parameters, limit],
            ).fetchall()
        return tuple(self._pair_from_row(row) for row in rows)

    @staticmethod
    def _pair_from_row(row: sqlite3.Row) -> PairRecord:
        return PairRecord(
            pair_id=row["pair_id"],
            observation_a_id=row["observation_a_id"],
            observation_b_id=row["observation_b_id"],
            created_at=decode_timestamp(row["created_at"]),
            payload=json.loads(row["payload_json"]),
        )

    # -- analyses ------------------------------------------------------------

    def create_analysis(self, record: AnalysisRecord) -> None:
        with self._db.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO analyses(analysis_id, status, created_at, updated_at,"
                    " payload_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.analysis_id,
                        record.status.value,
                        encode_timestamp(record.created_at),
                        encode_timestamp(record.updated_at),
                        canonical_json(record.payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._wrap_integrity_error(exc, "analyses") from exc

    def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        with self._db.read_transaction() as connection:
            row = connection.execute(
                "SELECT analysis_id, status, created_at, updated_at, payload_json"
                " FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return self._analysis_from_row(row) if row is not None else None

    def list_analyses(
        self, cursor: PageCursor | None = None, limit: int = 50
    ) -> tuple[AnalysisRecord, ...]:
        self._validate_limit(limit)
        clause, parameters = self._page_clause(cursor, "analyses", "analysis_id")
        with self._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT analysis_id, status, created_at, updated_at, payload_json"
                f" FROM analyses {clause}"
                " ORDER BY created_at DESC, analysis_id DESC LIMIT ?",
                [*parameters, limit],
            ).fetchall()
        return tuple(self._analysis_from_row(row) for row in rows)

    @staticmethod
    def _analysis_from_row(row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord(
            analysis_id=row["analysis_id"],
            status=AnalysisStatus(row["status"]),
            created_at=decode_timestamp(row["created_at"]),
            updated_at=decode_timestamp(row["updated_at"]),
            payload=json.loads(row["payload_json"]),
        )

    def transition_analysis(
        self,
        analysis_id: str,
        expected: AnalysisStatus,
        target: AnalysisStatus,
        *,
        updated_at: datetime,
    ) -> bool:
        allowed = _ANALYSIS_TRANSITIONS[expected]
        if target not in allowed:
            raise IllegalTransitionError(
                f"analysis transition {expected.value} -> {target.value} is illegal"
            )
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE analyses SET status = ?, updated_at = ?"
                " WHERE analysis_id = ? AND status = ? AND updated_at <= ?",
                (
                    target.value,
                    encode_timestamp(updated_at),
                    analysis_id,
                    expected.value,
                    encode_timestamp(updated_at),
                ),
            )
            return cursor.rowcount == 1

    # -- jobs ------------------------------------------------------------

    def create_job(self, record: JobRecord) -> None:
        if record.status is not JobStatus.QUEUED:
            raise ValueError("a new job must be created in QUEUED state")
        with self._db.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO jobs(job_id, analysis_id, status, created_at,"
                    " updated_at, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.job_id,
                        record.analysis_id,
                        record.status.value,
                        encode_timestamp(record.created_at),
                        encode_timestamp(record.updated_at),
                        canonical_json(record.payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._wrap_integrity_error(exc, "jobs") from exc

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._db.read_transaction() as connection:
            row = connection.execute(
                "SELECT job_id, analysis_id, status, created_at, updated_at,"
                " payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def list_jobs(
        self, cursor: PageCursor | None = None, limit: int = 50
    ) -> tuple[JobRecord, ...]:
        self._validate_limit(limit)
        clause, parameters = self._page_clause(cursor, "jobs", "job_id")
        with self._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT job_id, analysis_id, status, created_at, updated_at,"
                f" payload_json FROM jobs {clause}"
                " ORDER BY created_at DESC, job_id DESC LIMIT ?",
                [*parameters, limit],
            ).fetchall()
        return tuple(self._job_from_row(row) for row in rows)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            analysis_id=row["analysis_id"],
            status=JobStatus(row["status"]),
            created_at=decode_timestamp(row["created_at"]),
            updated_at=decode_timestamp(row["updated_at"]),
            payload=json.loads(row["payload_json"]),
        )

    def transition_job(
        self,
        job_id: str,
        expected: JobStatus,
        target: JobStatus,
        *,
        updated_at: datetime,
    ) -> bool:
        allowed = _JOB_TRANSITIONS[expected]
        if target not in allowed:
            raise IllegalTransitionError(
                f"job transition {expected.value} -> {target.value} is illegal"
            )
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?"
                " WHERE job_id = ? AND status = ? AND updated_at <= ?",
                (
                    target.value,
                    encode_timestamp(updated_at),
                    job_id,
                    expected.value,
                    encode_timestamp(updated_at),
                ),
            )
            return cursor.rowcount == 1

    def mark_running_jobs_interrupted(self, *, updated_at: datetime) -> int:
        """Restart recovery: RUNNING -> INTERRUPTED for orphaned jobs.

        Deliberately bypasses the per-pair transition graph: this is the one
        sanctioned recovery primitive, executed atomically at startup. Other
        states are untouched.
        """

        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                (
                    JobStatus.INTERRUPTED.value,
                    encode_timestamp(updated_at),
                    JobStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount

    # -- execution events ----------------------------------------------

    def append_event(self, event: ExecutionEvent) -> None:
        with self._db.transaction() as connection:
            last = connection.execute(
                "SELECT MAX(sequence) AS last FROM execution_events WHERE job_id = ?",
                (event.job_id,),
            ).fetchone()
            previous = last["last"]
            required = 0 if previous is None else previous + 1
            if event.sequence != required:
                raise PersistenceIntegrityError(
                    f"event sequence must be contiguous: expected {required}, "
                    f"got {event.sequence}"
                )
            try:
                connection.execute(
                    "INSERT INTO execution_events(event_id, job_id, sequence,"
                    " event_type, created_at, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.job_id,
                        event.sequence,
                        event.event_type,
                        encode_timestamp(event.created_at),
                        canonical_json(event.payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._wrap_integrity_error(exc, "execution_events") from exc

    def list_events(self, job_id: str) -> tuple[ExecutionEvent, ...]:
        with self._db.read_transaction() as connection:
            rows = connection.execute(
                "SELECT event_id, job_id, sequence, event_type, created_at,"
                " payload_json FROM execution_events WHERE job_id = ?"
                " ORDER BY sequence ASC",
                (job_id,),
            ).fetchall()
        return tuple(
            ExecutionEvent(
                event_id=row["event_id"],
                job_id=row["job_id"],
                sequence=row["sequence"],
                event_type=row["event_type"],
                created_at=decode_timestamp(row["created_at"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )
