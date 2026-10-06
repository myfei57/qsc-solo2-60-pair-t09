"""Startup recovery: migrate the legacy document, replay segments, audit.

Recovery policy (fixed): recovery runs automatically inside ``Store.open``
before the store is handed to the console. No operator confirmation is needed
and no operation is accepted until recovery has finished. Replaying a
committed operation is idempotent, so the whole procedure can be retried after
another crash and always converges on the same state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import recovery_audit
from .recovery_audit import RecoveryEvent
from .wal import (
    DEFAULT_SEGMENT_BYTES,
    WriteAheadLog,
    fsync_directory,
    replay,
    truncate_tail,
)

WAL_SUFFIX = ".wal"
LEGACY_SUFFIX = ".legacy.json"
MIGRATION_MARKER = "legacy.migrated"
TMP_SUFFIX = ".tmp"

# Recovery outcome labels written to the audit trail.
OUTCOME_CLEAN = "clean"
OUTCOME_DISCARDED_TAIL = "tail_discarded"
OUTCOME_MIGRATED = "legacy_migrated"


@dataclass(frozen=True)
class RecoveryReport:
    """Everything the console needs to describe the last startup recovery."""

    wal_directory: str
    migrated_from_legacy: bool
    migrated_keys: int
    replay: dict[str, object]
    event: RecoveryEvent
    history: list[RecoveryEvent]

    def discarded(self) -> list[dict[str, object]]:
        discarded = self.replay.get("discarded", [])
        return [item for item in discarded if isinstance(item, dict)]

    def as_dict(self) -> dict[str, object]:
        return {
            "wal_directory": self.wal_directory,
            "migrated_from_legacy": self.migrated_from_legacy,
            "migrated_keys": self.migrated_keys,
            "recovered_to_sequence": self.event.recovered_to_sequence,
            "segments": self.event.segments,
            "applied_operations": self.event.applied_operations,
            "discarded": self.event.discarded,
            "outcome": self.event.outcome,
            "detail": self.event.detail,
            "at": self.event.at,
            "history": [item.as_dict() for item in self.history],
        }


def resolve_paths(target: str) -> tuple[str, str]:
    """Return ``(wal_directory, legacy_path)`` for the configured store path."""

    absolute = os.path.abspath(target)
    return absolute + WAL_SUFFIX, absolute


def _load_legacy_document(path: str) -> dict[str, str] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return None
    if not raw.strip():
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("store document must be a JSON object")
    return {str(key): str(value) for key, value in loaded.items()}


def _migrate_legacy(wal: WriteAheadLog, wal_directory: str, legacy_path: str, data: dict[str, str]) -> int:
    """Import the legacy document, then mark it archived.

    Crash ordering::

        1. segment rows written + fsynced
        2. migration marker atomically renamed into the WAL directory + fsync
        3. legacy document renamed aside

    The marker is the durable commit point of the migration. A crash before it
    leaves the document in place with no (or a regenerable) segment, so the
    migration is retried from scratch. A crash after it means migration is
    finished and the document is never read again.
    """

    from .wal import segment_names

    # Without the marker, any existing segments can only be rows written by a
    # migration that crashed before committing; remove them before re-importing.
    for stale in segment_names(wal_directory):
        os.remove(os.path.join(wal_directory, stale))
    records = [
        (
            index,
            "put",
            {"key": key, "value": data[key]},
        )
        for index, key in enumerate(sorted(data), start=1)
    ]
    if records:
        wal.open_writer()
        wal.append_batch(records)
        wal.close()
    marker = os.path.join(wal_directory, MIGRATION_MARKER)
    temporary = marker + TMP_SUFFIX
    with open(temporary, "wb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker)
    fsync_directory(wal_directory)
    if os.path.exists(legacy_path):
        archived = legacy_path + LEGACY_SUFFIX
        os.replace(legacy_path, archived)
        fsync_directory(os.path.dirname(os.path.abspath(legacy_path)))
    return len(records)


def recover(target: str, segment_bytes: int = DEFAULT_SEGMENT_BYTES) -> tuple[dict[str, str], int, RecoveryReport]:
    """Run startup recovery and return the reconstructed state and report."""

    wal_directory, legacy_path = resolve_paths(target)
    os.makedirs(wal_directory, exist_ok=True)
    wal = WriteAheadLog(wal_directory, segment_bytes=segment_bytes)

    migrated = False
    migrated_keys = 0
    # Migration is committed by its marker file. Without the marker nothing is
    # considered migrated, so a crash at any earlier point simply reruns the
    # import; the marker makes that decision crash safe and idempotent.
    marker_present = os.path.exists(os.path.join(wal_directory, MIGRATION_MARKER))
    if not marker_present and os.path.exists(legacy_path):
        legacy = _load_legacy_document(legacy_path)
        if legacy is not None:
            migrated_keys = _migrate_legacy(wal, wal_directory, legacy_path, legacy)
            migrated = True

    result = replay(wal_directory)

    if migrated:
        outcome = OUTCOME_MIGRATED
        detail = (
            f"legacy document migrated: {migrated_keys} keys imported, "
            f"recovered to sequence {result.last_sequence}"
        )
    elif result.discarded:
        outcome = OUTCOME_DISCARDED_TAIL
        detail = _describe_discarded(result)
    else:
        outcome = OUTCOME_CLEAN
        detail = f"clean recovery to sequence {result.last_sequence}"

    # Audit what will be thrown away before the bytes are cut. If truncation
    # itself is interrupted, the next boot rediscovers the same tail, records
    # it again and retries: the repair is idempotent and converges.
    event = recovery_audit.record_recovery(
        wal_directory, outcome=outcome, result=result, detail=detail
    )
    truncate_tail(wal_directory, result)
    history = recovery_audit.list_events(wal_directory)
    report = RecoveryReport(
        wal_directory=wal_directory,
        migrated_from_legacy=migrated,
        migrated_keys=migrated_keys,
        replay=result.as_dict(),
        event=event,
        history=history,
    )
    return result.data, result.last_sequence, report


def record_failure(target: str, reason: str) -> RecoveryReport | None:
    """Audit a recovery that refused to proceed, keeping the store stopped."""

    wal_directory, _ = resolve_paths(target)
    os.makedirs(wal_directory, exist_ok=True)
    event = recovery_audit.record_recovery(
        wal_directory, outcome="corrupt", result=None, detail=reason
    )
    history = recovery_audit.list_events(wal_directory)
    return RecoveryReport(
        wal_directory=wal_directory,
        migrated_from_legacy=False,
        migrated_keys=0,
        replay={"last_sequence": 0, "segments": 0, "applied_operations": 0, "discarded": []},
        event=event,
        history=history,
    )


def _describe_discarded(result) -> str:
    parts = []
    for item in result.discarded:
        if item.operation == "?" and not item.key:
            parts.append(f"{item.segment}@{item.byte_offset} unparseable tail bytes")
        else:
            parts.append(
                f"{item.segment}#{item.sequence} {item.operation}({item.key or '-'})"
            )
    dropped = ", ".join(parts)
    return (
        f"partial write detected, recovered to sequence {result.last_sequence}; "
        f"discarded incomplete write(s): {dropped}"
    )
