"""Task 2 — Database: migration, transactions, and connection policy tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from satquery.persistence import Database
from satquery.persistence.database import SCHEMA_VERSION, UnsupportedSchemaVersionError

REQUIRED_TABLES = {
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
}


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "satquery.db")


def test_migrate_creates_database(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    assert (tmp_path / "satquery.db").exists()


def test_schema_version_is_frozen_at_one(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    assert SCHEMA_VERSION == 1
    assert db.schema_version() == 1


def test_expected_table_set_exists(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    assert {row["name"] for row in rows} == REQUIRED_TABLES


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    db.migrate()
    db.migrate()
    assert db.schema_version() == 1


def test_newer_schema_version_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "satquery.db"
    Database(db_path).migrate()
    # simulate a database written by a newer application version
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE schema_meta SET version = 99")
    with pytest.raises(UnsupportedSchemaVersionError):
        Database(db_path).migrate()
    # the database must not be downgraded
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 99


def test_foreign_keys_enforced_on_every_connection(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_wal_journal_mode(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_row_factory_is_sqlite_row(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        row = connection.execute("SELECT 1 AS value, 'x' AS label").fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["value"] == 1 and row["label"] == "x"


def test_rollback_restores_prior_state(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with pytest.raises(RuntimeError):
        with db.transaction() as connection:
            connection.execute(
                "INSERT INTO observations(observation_id, created_at, payload_json)"
                " VALUES ('obs_' || hex(randomblob(16)), '2026-01-01T00:00:00.000000Z', '{}')"
            )
            raise RuntimeError("boom")
    reopened = Database(tmp_path / "satquery.db")
    with reopened.transaction() as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
    assert count == 0


def test_committed_transaction_survives_reopen(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO observations(observation_id, created_at, payload_json)"
            " VALUES (?, ?, ?)",
            ("obs_" + "a" * 32, "2026-01-01T00:00:00.000000Z", "{}"),
        )
    reopened = Database(tmp_path / "satquery.db")
    with reopened.transaction() as connection:
        row = connection.execute(
            "SELECT observation_id FROM observations"
        ).fetchone()
    assert row["observation_id"] == "obs_" + "a" * 32


def test_parent_directory_created_safely(tmp_path: Path) -> None:
    db = Database(tmp_path / "nested" / "dirs" / "satquery.db")
    db.migrate()
    assert db.schema_version() == 1


def test_in_memory_database_rejected() -> None:
    with pytest.raises(ValueError, match=":memory:"):
        Database(":memory:")  # type: ignore[arg-type]


def test_connection_closes_after_transaction(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction():
        pass
    # after the transaction the WAL has no lingering connection: an exclusive
    # lock must be acquirable immediately by a fresh connection
    probe = sqlite3.connect(tmp_path / "satquery.db", timeout=0.1)
    probe.execute("BEGIN EXCLUSIVE")
    probe.rollback()
    probe.close()


def test_corrupted_schema_version_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "satquery.db"
    Database(db_path).migrate()
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE schema_meta SET version = 'not-a-number'")
    with pytest.raises(Exception):
        Database(db_path).migrate()


def test_corrupted_missing_tables_fail_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "satquery.db"
    Database(db_path).migrate()
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE jobs")
    with pytest.raises(Exception):
        Database(db_path).migrate()


def test_schema_meta_is_singleton(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.migrate()
    with db.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO schema_meta(singleton, version, applied_at) VALUES (2, 1, 'x')"
            )
