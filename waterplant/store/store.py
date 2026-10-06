"""Crash safe, journaled key/value document behind the control service.

The store no longer rewrites one whole JSON file per mutation. Every mutation
is appended as a checksummed frame to the current journal segment and fsynced
before it is acknowledged, so an acknowledged operation survives a power loss.
On open, recovery replays the last checkpoint plus every complete frame after
it; the incomplete tail of an interrupted flush is quarantined and reported.

Recovery and new writes are strictly ordered: the store refuses every
operation until recovery has finished, and recovery is idempotent — replaying
it after a crash during recovery converges on the same consistent state.
"""

from __future__ import annotations

import json
import os
import threading

from .journal import DEFAULT_SEGMENT_MAX_BYTES, Journal, RecoveryReport


class RecoveryPendingError(RuntimeError):
    """Raised when an operation arrives before startup recovery finished."""


class Store:
    """A journaled mapping from string keys to string values.

    Durability policy: every mutation is flushed and fsynced immediately —
    these are low-rate, high-value control operations, so no batching window
    is allowed to lose an acknowledged write. Recovery policy: automatic on
    :meth:`open`, with the discarded tail quarantined and audited rather than
    silently dropped.
    """

    def __init__(
        self, path: str, *, segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES
    ) -> None:
        self._path = os.path.abspath(path)
        self._journal = Journal(f"{self._path}.d", segment_max_bytes)
        self._data: dict[str, str] = {}
        self._seq = 0
        self._lock = threading.RLock()
        self._ready = False
        self._report: RecoveryReport | None = None
        self._notes: list[str] = []

    @classmethod
    def open(
        cls,
        path: str,
        *,
        auto_recover: bool = True,
        segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES,
    ) -> "Store":
        """Open the store, migrating any legacy document and recovering.

        Recovery runs automatically so the service comes back unattended;
        pass ``auto_recover=False`` only to drive recovery explicitly (every
        operation is rejected with :class:`RecoveryPendingError` until
        :meth:`recover` completes).
        """

        store = cls(path, segment_max_bytes=segment_max_bytes)
        store._migrate_legacy_document()
        if auto_recover:
            store.recover()
        return store

    @property
    def path(self) -> str:
        return self._path

    @property
    def ready(self) -> bool:
        """True once startup recovery has finished and writes are accepted."""

        return self._ready

    def recover(self) -> RecoveryReport:
        """Replay the journal back to the last consistent state.

        Idempotent: a second call returns the cached report, and a crash
        midway simply causes the next call to redo the same deterministic
        steps against the same on-disk prefix.
        """

        with self._lock:
            if self._ready and self._report is not None:
                return self._report
            checkpoint_seq, data, frames, report = self._journal.recover(self._notes)
            self._data = data
            self._seq = checkpoint_seq
            for frame in frames:
                if frame.seq <= self._seq:
                    continue  # already folded into the checkpoint
                self._apply(frame.operation())
                self._seq = frame.seq
            self._report = report
            self._ready = True
            return report

    def recovery_report(self) -> dict[str, object]:
        """The most recent recovery report, for the console and audit trail."""

        with self._lock:
            if self._report is None:
                persisted = self._journal.read_report()
                return persisted if persisted is not None else {"status": "pending"}
            return self._report.as_dict()

    def get(self, key: str) -> tuple[str, bool]:
        with self._lock:
            self._require_ready()
            if key in self._data:
                return self._data[key], True
            return "", False

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._require_ready()
            self._append({"op": "put", "key": key, "value": value})
            self._data[key] = value
            self._maybe_compact()

    def delete(self, key: str) -> None:
        with self._lock:
            self._require_ready()
            if key not in self._data:
                return
            self._append({"op": "delete", "key": key})
            del self._data[key]
            self._maybe_compact()

    def keys(self) -> list[str]:
        with self._lock:
            self._require_ready()
            return sorted(self._data)

    def count(self) -> int:
        with self._lock:
            self._require_ready()
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._require_ready()
            self._append({"op": "clear"})
            self._data = {}
            self._maybe_compact()

    def close(self) -> None:
        with self._lock:
            self._journal.close()

    # ----------------------------------------------------------------- interns

    def _require_ready(self) -> None:
        if not self._ready:
            raise RecoveryPendingError(
                "startup recovery has not finished; new operations are rejected"
            )

    def _append(self, operation: dict[str, object]) -> None:
        self._seq += 1
        payload = json.dumps(operation, ensure_ascii=False).encode("utf-8")
        self._journal.append(self._seq, payload)

    def _maybe_compact(self) -> None:
        """Rotate the segment, then checkpoint the state it now covers.

        Runs only after the in-memory apply, so the checkpoint always
        includes the frame that triggered the rotation — pruning the rotated
        segment can never drop the newest operation.
        """

        if self._journal.active_size >= self._journal.segment_max_bytes:
            self._journal.rotate()
            self._journal.write_checkpoint(self._seq, dict(self._data))
            self._journal.prune_segments()

    def _apply(self, operation: dict[str, object]) -> None:
        kind = operation.get("op")
        if kind == "put":
            self._data[str(operation["key"])] = str(operation["value"])
        elif kind == "delete":
            self._data.pop(str(operation["key"]), None)
        elif kind == "clear":
            self._data = {}

    def _migrate_legacy_document(self) -> None:
        """Import a pre-journal single-file document once, then set it aside."""

        if not os.path.isfile(self._path):
            return
        if self._journal.is_empty():
            try:
                with open(self._path, "r", encoding="utf-8") as handle:
                    raw = handle.read()
            except OSError:
                raw = ""
            if raw.strip():
                try:
                    loaded = json.loads(raw)
                    if not isinstance(loaded, dict):
                        raise ValueError("legacy store document must be a JSON object")
                except ValueError:
                    os.replace(self._path, f"{self._path}.corrupt")
                    self._notes.append(
                        "legacy store document was corrupt; quarantined to "
                        f"{self._path}.corrupt"
                    )
                    return
                data = {str(key): str(value) for key, value in loaded.items()}
                self._journal.write_checkpoint(0, data)
                self._notes.append(
                    f"migrated {len(data)} keys from the legacy single-file store"
                )
        os.replace(self._path, f"{self._path}.migrated")
