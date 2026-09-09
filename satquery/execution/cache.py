"""Exact scientific cache keys and durable reuse of successful results.

A cache key must cover every scientific input that can change a result: the
tool identity/version, model checkpoint hash, preprocessing hash, input
hashes, canonical parameters, ROI, planner version, and the tool registry
hash. Keys are SHA-256 over a canonical JSON serialization, so dictionary
insertion order never changes the key.

Only immutable successful results are cached. Entries reference published
artifacts by ID plus recorded SHA-256; a lookup verifies those hashes against
the artifact store and fails closed (returns a miss) when verification is
impossible or fails. Failures and mutable staging paths are never cached.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from satquery.artifacts import ArtifactStore
    from satquery.persistence import MetadataRepository


_CACHE_KEY = re.compile(r"^[0-9a-f]{64}$")


class CacheKeyError(ValueError):
    """A cache key or cache payload could not be canonically serialized."""


def _canonical(value: Any) -> str:
    """Frozen canonical encoding; NaN/Infinity and non-JSON values fail."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CacheKeyError("value is not canonically serializable") from exc


def build_cache_key(
    tool: Mapping[str, Any],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    roi: Mapping[str, Any] | None,
    *,
    planner_version: str,
    registry_hash: str,
) -> str:
    """Build the exact scientific cache key for one tool execution.

    ``tool`` is the tool descriptor (identity, version, and any checkpoint or
    preprocessing hashes), ``inputs`` maps input names to content hashes, and
    ``parameters``/``roi`` are the canonical request parameters and region.
    """

    if not isinstance(tool, Mapping) or not tool:
        raise CacheKeyError("cache key requires a non-empty tool descriptor")
    if not isinstance(inputs, Mapping) or not isinstance(parameters, Mapping):
        raise CacheKeyError("cache key inputs and parameters must be mappings")
    if roi is not None and not isinstance(roi, Mapping):
        raise CacheKeyError("cache key ROI must be a mapping or None")
    if not isinstance(planner_version, str) or not planner_version:
        raise CacheKeyError("cache key requires a planner version")
    if not isinstance(registry_hash, str) or not registry_hash:
        raise CacheKeyError("cache key requires a registry hash")
    payload = {
        "cache_schema_version": 1,
        "tool": tool,
        "inputs": inputs,
        "parameters": parameters,
        "roi": roi,
        "planner_version": planner_version,
        "registry_hash": registry_hash,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class AnalysisCache:
    """Durable store of successful tool results keyed by exact cache keys."""

    def __init__(self, repository: MetadataRepository) -> None:
        self._repository = repository

    def put(self, cache_key: str, payload: Mapping[str, Any]) -> bool:
        """Persist one successful result payload; rejects malformed input."""

        if not isinstance(cache_key, str) or not _CACHE_KEY.fullmatch(cache_key):
            raise CacheKeyError("cache key must be a 64-character hex digest")
        encoded = _canonical(dict(payload))
        with self._repository._db.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO cache_entries(cache_key, created_at, payload_json)"
                " VALUES (?, ?, ?)",
                (
                    cache_key,
                    datetime.now(timezone.utc).isoformat(),
                    encoded,
                ),
            )
        return True

    def get(
        self,
        cache_key: str,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> dict[str, Any] | None:
        """Return the cached payload, or None on miss/corruption/hash failure.

        Artifact references are verified against the artifact store; any
        verification failure is a miss, never a partial or unverified reuse.
        """

        if not isinstance(cache_key, str) or not _CACHE_KEY.fullmatch(cache_key):
            return None
        with self._repository._db.read_transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        artifacts = payload.get("artifacts")
        if artifacts:
            if artifact_store is None:
                return None
            from satquery.artifacts import ArtifactNotFoundError

            for item in artifacts:
                if not isinstance(item, dict):
                    return None
                artifact_id = item.get("artifact_id")
                recorded = item.get("sha256")
                if not isinstance(artifact_id, str) or not isinstance(recorded, str):
                    return None
                try:
                    record, _path = artifact_store.resolve(artifact_id)
                except Exception:
                    return None
                if record.sha256 != recorded:
                    return None
        return payload


__all__ = ["AnalysisCache", "CacheKeyError", "build_cache_key"]
