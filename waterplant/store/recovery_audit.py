"""Durable audit trail of startup recovery.

Every boot appends one structured JSON line to ``recovery-audit.log`` inside
the WAL directory. The line is fsynced before the service accepts traffic, so
the history of crash points and discarded partial writes survives the very
next crash. The console reads it back verbatim for operator display.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from .wal import ReplayResult

AUDIT_FILE = "recovery-audit.log"


@dataclass(frozen=True)
class RecoveryEvent:
    """One persisted recovery outcome."""

    at: str
    epoch: str
    outcome: str
    recovered_to_sequence: int
    segments: int
    applied_operations: int
    discarded: list[dict[str, object]]
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "at": self.at,
            "epoch": self.epoch,
            "outcome": self.outcome,
            "recovered_to_sequence": self.recovered_to_sequence,
            "segments": self.segments,
            "applied_operations": self.applied_operations,
            "discarded": self.discarded,
            "detail": self.detail,
        }


def _now() -> tuple[str, str]:
    return (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        str(time.time()),
    )


def append_event(directory: str, event: RecoveryEvent) -> None:
    """Append and fsync one recovery event."""

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, AUDIT_FILE)
    line = json.dumps(event.as_dict(), ensure_ascii=False) + "\n"
    with open(path, "ab", buffering=0) as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def record_recovery(
    directory: str,
    *,
    outcome: str,
    result: ReplayResult | None,
    detail: str,
) -> RecoveryEvent:
    """Build, persist and return the audit row for this boot."""

    timestamp, epoch = _now()
    event = RecoveryEvent(
        at=timestamp,
        epoch=epoch,
        outcome=outcome,
        recovered_to_sequence=result.last_sequence if result else 0,
        segments=result.segments if result else 0,
        applied_operations=result.applied_operations if result else 0,
        discarded=[item.as_dict() for item in result.discarded] if result else [],
        detail=detail,
    )
    append_event(directory, event)
    return event


def list_events(directory: str) -> list[RecoveryEvent]:
    """Return every well formed recovery event, newest last."""

    path = os.path.join(directory, AUDIT_FILE)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return []
    events: list[RecoveryEvent] = []
    for line in raw.splitlines():
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            # The final line can be torn by a crash mid-append; skip it.
            continue
        if not isinstance(payload, dict):
            continue
        discarded = payload.get("discarded", [])
        if not isinstance(discarded, list):
            discarded = []
        events.append(
            RecoveryEvent(
                at=str(payload.get("at", "")),
                epoch=str(payload.get("epoch", "0")),
                outcome=str(payload.get("outcome", "")),
                recovered_to_sequence=int(payload.get("recovered_to_sequence", 0)),
                segments=int(payload.get("segments", 0)),
                applied_operations=int(payload.get("applied_operations", 0)),
                discarded=[item for item in discarded if isinstance(item, dict)],
                detail=str(payload.get("detail", "")),
            )
        )
    return events
