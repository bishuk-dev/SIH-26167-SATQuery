"""Durable SQLite metadata persistence for the Phase 5 backend.

Stable Task-2 API: ``Database`` owns migrations/transactions; the strict
records and ``MetadataRepository`` own typed metadata storage. ID generation
and application wiring belong to later tasks.
"""

from satquery.persistence.database import (
    SCHEMA_VERSION,
    Database,
    PersistenceError,
    PersistenceIntegrityError,
    UnsupportedSchemaVersionError,
)
from satquery.persistence.repositories import (
    AnalysisRecord,
    AnalysisStatus,
    ExecutionEvent,
    IllegalTransitionError,
    JobRecord,
    JobStatus,
    MetadataRepository,
    ObservationRecord,
    PageCursor,
    PairRecord,
    RecordAlreadyExistsError,
)

__all__ = [
    "AnalysisRecord",
    "AnalysisStatus",
    "Database",
    "ExecutionEvent",
    "IllegalTransitionError",
    "JobRecord",
    "JobStatus",
    "MetadataRepository",
    "ObservationRecord",
    "PageCursor",
    "PairRecord",
    "PersistenceError",
    "PersistenceIntegrityError",
    "RecordAlreadyExistsError",
    "SCHEMA_VERSION",
    "UnsupportedSchemaVersionError",
]
