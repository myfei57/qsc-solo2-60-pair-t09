"""Thread safe persistent mapping behind the control service.

Persistence is a segmented write ahead log (see :mod:`waterplant.store.wal`).
Each mutating operation is appended as a data plus commit frame pair and
fsynced before it returns, so an acknowledged write is a durable write and a
crash can only ever lose the single write that was still landing.

Lifecycle::

    bootstrapping -> recovering -> ready
                                   \\-> failed (write error / corrupt log)

``Store.open`` does not finish until recovery has replayed every complete
segment, so a store handed to the console is always recovered. Mutating
methods raise :class:`StoreNotReadyError` in any phase except ``ready``; reads
stay available so the console can still show what was recovered.
"""

from __future__ import annotations

import threading

from .recovery import RecoveryReport, recover, record_failure
from .wal import (
    DEFAULT_SEGMENT_BYTES,
    SegmentCorruptError,
    WriteAheadLog,
)

PHASE_BOOTSTRAPPING = "bootstrapping"
PHASE_RECOVERING = "recovering"
PHASE_READY = "ready"
PHASE_FAILED = "failed"
PHASE_CLOSED = "closed"


class StoreNotReadyError(RuntimeError):
    """Raised when a mutation is attempted outside the ready phase."""


class Store:
    """A log backed mapping from string keys to string values."""

    def __init__(
        self,
        path: str,
        data: dict[str, str] | None = None,
        *,
        wal: WriteAheadLog | None = None,
        sequence: int = 0,
        report: RecoveryReport | None = None,
        phase: str = PHASE_READY,
        failure_reason: str = "",
        segment_bytes: int = DEFAULT_SEGMENT_BYTES,
    ) -> None:
        self._path = path
        self._data: dict[str, str] = dict(data) if data else {}
        self._lock = threading.RLock()
        self._wal = wal
        self._sequence = sequence
        self._report = report
        self._phase = phase
        self._failure_reason = failure_reason
        self._segment_bytes = segment_bytes

    @property
    def path(self) -> str:
        return self._path

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def failure_reason(self) -> str:
        with self._lock:
            return self._failure_reason

    @property
    def last_sequence(self) -> int:
        """Sequence number of the last durable, acknowledged operation."""

        with self._lock:
            return self._sequence

    def ready(self) -> bool:
        return self.phase == PHASE_READY

    def recovery_report(self) -> RecoveryReport | None:
        return self._report

    @classmethod
    def open(cls, path: str, *, segment_bytes: int = DEFAULT_SEGMENT_BYTES) -> "Store":
        """Recover or create the store at ``path``.

        Recovery is synchronous: legacy migration, segment replay and the
        recovery audit write all finish before this method returns. The store
        is failed (never ready) when the log is structurally corrupt, so the
        console can refuse mutations instead of serving an unknown state.
        """

        try:
            data, sequence, report = recover(path, segment_bytes=segment_bytes)
        except SegmentCorruptError as exc:
            report = record_failure(path, str(exc))
            store = cls(
                path,
                dict(exc.partial_data),
                sequence=exc.partial_sequence,
                phase=PHASE_FAILED,
                failure_reason=str(exc),
                report=report,
                segment_bytes=segment_bytes,
            )
            return store
        wal_directory = report.wal_directory
        wal = WriteAheadLog(wal_directory, segment_bytes=segment_bytes)
        wal.open_writer()
        store = cls(
            path,
            data,
            wal=wal,
            sequence=sequence,
            report=report,
            phase=PHASE_READY,
            segment_bytes=segment_bytes,
        )
        return store

    def mark_failed(self, reason: str) -> None:
        """Take the store out of service after an unrecoverable write error."""

        with self._lock:
            if self._phase != PHASE_CLOSED:
                self._phase = PHASE_FAILED
                self._failure_reason = reason

    def get(self, key: str) -> tuple[str, bool]:
        with self._lock:
            if key in self._data:
                return self._data[key], True
            return "", False

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._require_ready()
            sequence = self._sequence + 1
            self._commit(sequence, "put", key=key, value=value)
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._require_ready()
            if key not in self._data:
                return
            sequence = self._sequence + 1
            self._commit(sequence, "delete", key=key)
            del self._data[key]

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._data)

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._require_ready()
            sequence = self._sequence + 1
            self._commit(sequence, "clear")
            self._data = {}

    def close(self) -> None:
        with self._lock:
            if self._wal is not None:
                self._wal.close()
                self._wal = None
            self._phase = PHASE_CLOSED

    def _require_ready(self) -> None:
        if self._phase == PHASE_READY:
            return
        if self._phase == PHASE_FAILED:
            raise StoreNotReadyError(f"store failed: {self._failure_reason}")
        raise StoreNotReadyError(f"store is {self._phase}, recovery not finished")

    def _commit(self, sequence: int, operation: str, **fields: object) -> None:
        """Durably append one operation; the in-memory update follows on success."""

        try:
            assert self._wal is not None
            self._wal.append(sequence, operation, **fields)
        except OSError as exc:
            self.mark_failed(f"write ahead log append failed: {exc}")
            raise StoreNotReadyError(f"store failed: {exc}") from exc
        self._sequence = sequence
