"""Durable SQLite metadata database for the Phase 5 backend.

One connection per transaction, no pool, stdlib ``sqlite3`` only. Contract
payloads are stored as canonical JSON text and all timestamps as normalized
UTC strings, so lexical ordering preserves chronology.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection

SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 5_000

# exact Phase-5 foundation table set; nothing speculative
_REQUIRED_TABLES = (
    "schema_meta",
    "observations",
    "pairs",
    "analyses",
    "analysis_inputs",
    "jobs",
    "plan_steps",
    "evidence",
    "evidence_edges",
    "artifacts",
    "execution_events",
    "cache_entries",
    "idempotency_keys",
)

_SCHEMA = (
    # singleton authoritative schema-version record
    """
    CREATE TABLE schema_meta (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        version INTEGER NOT NULL,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE observations (
        observation_id TEXT PRIMARY KEY
            CHECK (length(observation_id) = 36 AND substr(observation_id, 1, 4) = 'obs_'),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE pairs (
        pair_id TEXT PRIMARY KEY
            CHECK (length(pair_id) = 37 AND substr(pair_id, 1, 5) = 'pair_'),
        observation_a_id TEXT NOT NULL REFERENCES observations(observation_id),
        observation_b_id TEXT NOT NULL REFERENCES observations(observation_id),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        CHECK (observation_a_id <> observation_b_id)
    )
    """,
    """
    CREATE TABLE analyses (
        analysis_id TEXT PRIMARY KEY
            CHECK (length(analysis_id) = 36 AND substr(analysis_id, 1, 4) = 'ana_'),
        status TEXT NOT NULL CHECK (status IN (
            'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED',
            'ABSTAINED', 'REJECTED', 'CANCELLED', 'INTERRUPTED')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE analysis_inputs (
        analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
        position INTEGER NOT NULL CHECK (position >= 0),
        observation_id TEXT REFERENCES observations(observation_id),
        pair_id TEXT REFERENCES pairs(pair_id),
        PRIMARY KEY (analysis_id, position),
        CHECK ((observation_id IS NULL) <> (pair_id IS NULL))
    )
    """,
    """
    CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY
            CHECK (length(job_id) = 36 AND substr(job_id, 1, 4) = 'job_'),
        analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
        status TEXT NOT NULL CHECK (status IN (
            'QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED',
            'CANCEL_REQUESTED', 'CANCELLED', 'INTERRUPTED')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE plan_steps (
        analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
        step_index INTEGER NOT NULL CHECK (step_index >= 0),
        tool_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (analysis_id, step_index)
    )
    """,
    """
    CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY
            CHECK (length(evidence_id) = 41 AND substr(evidence_id, 1, 9) = 'evidence_'),
        analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE evidence_edges (
        analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
        source_evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
        target_evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
        edge_type TEXT NOT NULL,
        PRIMARY KEY (analysis_id, source_evidence_id, target_evidence_id, edge_type),
        CHECK (source_evidence_id <> target_evidence_id)
    )
    """,
    """
    CREATE TABLE artifacts (
        artifact_id TEXT PRIMARY KEY
            CHECK (length(artifact_id) = 41 AND substr(artifact_id, 1, 9) = 'artifact_'),
        analysis_id TEXT REFERENCES analyses(analysis_id),
        evidence_id TEXT REFERENCES evidence(evidence_id),
        created_at TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
        media_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        storage_key TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE execution_events (
        event_id TEXT PRIMARY KEY
            CHECK (length(event_id) = 38 AND substr(event_id, 1, 6) = 'event_'),
        job_id TEXT NOT NULL REFERENCES jobs(job_id),
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        event_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        UNIQUE (job_id, sequence)
    )
    """,
    """
    CREATE TABLE cache_entries (
        cache_key TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    # client idempotency keys are stored hashed only, never as plaintext
    """
    CREATE TABLE idempotency_keys (
        key_hash TEXT PRIMARY KEY,
        request_hash TEXT NOT NULL,
        analysis_id TEXT REFERENCES analyses(analysis_id),
        created_at TEXT NOT NULL
    )
    """,
)


class PersistenceError(RuntimeError):
    """Base class for persistence-layer failures."""


class UnsupportedSchemaVersionError(PersistenceError):
    """The database was written by a newer application; fail closed."""


class PersistenceIntegrityError(PersistenceError):
    """A database constraint or structural expectation was violated."""


class Database:
    """Filesystem SQLite metadata database; one connection per transaction."""

    def __init__(self, path: Path | str) -> None:
        path_string = str(path)
        if path_string == ":memory:" or path_string.startswith("file::memory:"):
            # one connection per transaction makes :memory: a fresh empty
            # database every time; rejecting it prevents silent misuse
            raise ValueError(
                ":memory: is not supported; the metadata database must be a "
                "durable filesystem file"
            )
        self._path = Path(path_string)

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> Connection:
        connection = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Yield a connection inside an explicit BEGIN IMMEDIATE transaction.

        Commits on success, rolls back on any exception, and always closes
        the connection.
        """

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        finally:
            connection.close()

    def _read_version(self, connection: Connection) -> int:
        row = connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("schema_meta is missing its singleton row")
        version = row["version"]
        if not isinstance(version, int):
            raise PersistenceIntegrityError(
                f"schema_meta version is corrupted: {version!r}"
            )
        return version

    def _validate_tables(self, connection: Connection) -> None:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        present = {row["name"] for row in rows}
        missing = set(_REQUIRED_TABLES) - present
        if missing:
            raise PersistenceIntegrityError(
                f"database claims schema version {SCHEMA_VERSION} but is "
                f"missing tables: {', '.join(sorted(missing))}"
            )

    def migrate(self) -> None:
        """Create or validate the schema; transactional, idempotent, safe on
        restart. Never downgrades a newer database."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")  # persistent file setting
            existing = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                " AND name = 'schema_meta'"
            ).fetchone()
            if existing is not None:
                version = self._read_version(connection)
                if version > SCHEMA_VERSION:
                    raise UnsupportedSchemaVersionError(
                        f"database schema version {version} is newer than the "
                        f"supported version {SCHEMA_VERSION}; refusing to open"
                    )
                if version < SCHEMA_VERSION:
                    raise PersistenceIntegrityError(
                        f"database schema version {version} is older than the "
                        f"supported version {SCHEMA_VERSION}; upgrade path not "
                        "implemented"
                    )
                self._validate_tables(connection)
                return
            connection.execute("BEGIN IMMEDIATE")
            try:
                for statement in _SCHEMA:
                    connection.execute(statement)
                from datetime import datetime, timezone

                connection.execute(
                    "INSERT INTO schema_meta(singleton, version, applied_at) VALUES (1, ?, ?)",
                    (
                        SCHEMA_VERSION,
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        finally:
            connection.close()

    def schema_version(self) -> int:
        connection = self._connect()
        try:
            return self._read_version(connection)
        finally:
            connection.close()
