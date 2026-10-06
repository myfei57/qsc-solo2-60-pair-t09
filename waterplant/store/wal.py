"""Segmented write ahead log backing the key/value store.

The log is a directory of ordered segment files. One segment is open for
appending at a time; the rest are sealed. Every mutation is written as a data
frame immediately followed by a commit frame, and the file is flushed and
fsynced before the call returns. A segment that never receives a commit frame
for its last data frame is therefore unmistakably the site of a crash.

Durability policy (fixed): each operation lands and fsyncs immediately. The
service never batches writes, so the acknowledged state is always the on-disk
state. The only configurable limit is the size at which the active segment is
rotated; rotation is a bookkeeping operation that never loses data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .framing import COMMIT_KIND, DATA_KIND, encode_frame, parse_frame

SEGMENT_PREFIX = "seg-"
SEGMENT_SUFFIX = ".log"
DEFAULT_SEGMENT_BYTES = 4 * 1024 * 1024


class SegmentCorruptError(Exception):
    """Raised when a sealed part of the log is structurally inconsistent.

    A torn tail of the last segment is a recoverable partial failure and is
    reported through :class:`ReplayResult` instead. Any other mismatch means
    the log cannot be reconstructed safely and the operator must intervene.
    The state recovered up to the failure point rides along so the console can
    still show the last readable snapshot, read only.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_data: dict[str, str] | None = None,
        partial_sequence: int = 0,
    ) -> None:
        super().__init__(message)
        self.partial_data = partial_data or {}
        self.partial_sequence = partial_sequence


@dataclass(frozen=True)
class DiscardedWrite:
    """One incomplete write identified and dropped during recovery."""

    segment: str
    sequence: int
    operation: str
    key: str
    reason: str
    byte_offset: int

    def as_dict(self) -> dict[str, object]:
        return {
            "segment": self.segment,
            "sequence": self.sequence,
            "operation": self.operation,
            "key": self.key,
            "reason": self.reason,
            "byte_offset": self.byte_offset,
        }


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying every segment during startup recovery."""

    last_sequence: int
    segments: int
    applied_operations: int
    discarded: list[DiscardedWrite]
    data: dict[str, str]
    truncate_segment: str
    valid_bytes: int

    def recovery_point(self) -> int:
        """Sequence number of the last operation that survived recovery."""

        return self.last_sequence

    def needs_truncation(self) -> bool:
        """Whether the final segment carries a torn tail that must be cut."""

        return bool(self.discarded) and self.truncate_segment != ""

    def as_dict(self) -> dict[str, object]:
        return {
            "last_sequence": self.last_sequence,
            "segments": self.segments,
            "applied_operations": self.applied_operations,
            "discarded": [item.as_dict() for item in self.discarded],
        }


def segment_names(directory: str) -> list[str]:
    """Return every segment file name in ascending numeric order."""

    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return []
    segments = [name for name in names if _segment_number(name) is not None]
    return sorted(segments, key=lambda name: _segment_number(name) or 0)


def _segment_number(name: str) -> int | None:
    if not (name.startswith(SEGMENT_PREFIX) and name.endswith(SEGMENT_SUFFIX)):
        return None
    middle = name[len(SEGMENT_PREFIX) : -len(SEGMENT_SUFFIX)]
    if len(middle) != 8 or not middle.isdigit():
        return None
    return int(middle)


def fsync_directory(path: str) -> None:
    """Fsync a directory so a newly created directory entry is durable."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _payload_operation(payload: dict[str, object]) -> tuple[str, str]:
    return str(payload.get("op", "")), str(payload.get("key", ""))


def apply_operation(data: dict[str, str], payload: dict[str, object]) -> None:
    """Replay one committed operation onto ``data``.

    Replay is idempotent: applying the same committed sequence twice leaves the
    mapping in the same state, so recovery can be retried without side effects.
    """

    operation = payload.get("op")
    if operation == "put":
        data[str(payload["key"])] = str(payload["value"])
    elif operation == "delete":
        data.pop(str(payload.get("key", "")), None)
    elif operation == "clear":
        data.clear()
    else:
        raise SegmentCorruptError(f"unknown operation {operation!r}")


class WriteAheadLog:
    """Append only segmented log.

    Each append opens the active segment with ``O_APPEND``, writes the frame
    pair, fsyncs and closes it again. Writes are per operation anyway, so there
    is no batching benefit to a long lived descriptor, while open-per-write
    guarantees no descriptor survives a crash or a forgotten ``close``.
    """

    def __init__(self, directory: str, segment_bytes: int = DEFAULT_SEGMENT_BYTES) -> None:
        self._directory = os.path.abspath(directory)
        self._segment_bytes = segment_bytes
        self._active_number = 0
        self._active_name = ""

    @property
    def directory(self) -> str:
        return self._directory

    def open_writer(self) -> None:
        """Pick the highest existing segment, or prepare to create the first."""

        os.makedirs(self._directory, exist_ok=True)
        names = segment_names(self._directory)
        if names:
            self._active_name = names[-1]
            self._active_number = _segment_number(self._active_name) or 0
        else:
            self._active_number = 1
            self._active_name = self._format_name(self._active_number)

    def close(self) -> None:
        # Descriptors are per append; nothing is held open.
        self._active_number = 0
        self._active_name = ""

    def _append_blob(self, blob: bytes) -> None:
        if self._active_number == 0:
            self.open_writer()
        path = os.path.join(self._directory, self._active_name)
        created = not os.path.exists(path)
        descriptor = os.open(path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            view = memoryview(blob)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            fsync_directory(self._directory)
        if os.path.getsize(path) >= self._segment_bytes:
            self._active_number += 1
            self._active_name = self._format_name(self._active_number)

    def append(self, sequence: int, operation: str, **fields: object) -> None:
        """Durably append one data plus commit frame pair for an operation."""

        data_payload: dict[str, object] = {"seq": sequence, "op": operation}
        data_payload.update(fields)
        record = encode_frame(DATA_KIND, data_payload) + encode_frame(
            COMMIT_KIND, {"seq": sequence}
        )
        self._append_blob(record)

    def append_batch(self, records: list[tuple[int, str, dict[str, object]]]) -> None:
        """Append many already sequenced operations with one fsync.

        Used only by one-time legacy migration; runtime writes always go
        through :meth:`append` one operation at a time.
        """

        blob = bytearray()
        for sequence, operation, fields in records:
            data_payload: dict[str, object] = {"seq": sequence, "op": operation}
            data_payload.update(fields)
            blob += encode_frame(DATA_KIND, data_payload)
            blob += encode_frame(COMMIT_KIND, {"seq": sequence})
        self._append_blob(bytes(blob))

    @staticmethod
    def _format_name(number: int) -> str:
        return f"{SEGMENT_PREFIX}{number:08d}{SEGMENT_SUFFIX}"


def replay(directory: str) -> ReplayResult:
    """Replay every segment and recover the last consistent state.

    Segments are processed in order and each data frame must be followed by a
    commit frame carrying the same sequence number. A missing or garbled frame
    at the end of the final segment is the torn tail of a crashed write: the
    uncommitted operation is reported as discarded and replay stops there. The
    same shape anywhere earlier is fatal because committed history would have
    a hole.
    """

    data: dict[str, str] = {}
    applied = 0
    discarded: list[DiscardedWrite] = []
    last_sequence = 0
    names = segment_names(directory)
    truncate_segment = ""
    valid_bytes = 0
    for position, name in enumerate(names):
        is_final = position == len(names) - 1
        try:
            segment_applied, segment_last, segment_valid = _replay_segment(
                os.path.join(directory, name),
                name=name,
                is_final=is_final,
                last_sequence=last_sequence,
                data=data,
                discarded=discarded,
            )
        except SegmentCorruptError as exc:
            raise SegmentCorruptError(
                str(exc), partial_data=dict(data), partial_sequence=last_sequence
            ) from None
        applied += segment_applied
        last_sequence = segment_last
        if is_final:
            truncate_segment = name if discarded else ""
            valid_bytes = segment_valid
    return ReplayResult(
        last_sequence=last_sequence,
        segments=len(names),
        applied_operations=applied,
        discarded=discarded,
        data=data,
        truncate_segment=truncate_segment,
        valid_bytes=valid_bytes,
    )


def truncate_tail(directory: str, result: ReplayResult) -> None:
    """Cut a torn tail off the final segment and fsync the shortened file.

    Runs only after the discarded bytes have already been recorded in the
    recovery audit, so the description of what was dropped outlives the bytes
    themselves.
    """

    if not result.needs_truncation():
        return
    path = os.path.join(directory, result.truncate_segment)
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(descriptor, result.valid_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replay_segment(
    path: str,
    *,
    name: str,
    is_final: bool,
    last_sequence: int,
    data: dict[str, str],
    discarded: list[DiscardedWrite],
) -> tuple[int, int, int]:
    with open(path, "rb") as handle:
        raw = handle.read()

    applied = 0
    expected = last_sequence + 1
    # Byte offset immediately after the last committed frame pair.
    valid_bytes = 0
    pos = 0
    total = len(raw)
    while pos < total:
        newline = raw.find(b"\n", pos)
        line_end = total if newline == -1 else newline
        line = raw[pos:line_end]
        data_offset = pos
        data_frame = parse_frame(line)
        if data_frame is None:
            if is_final:
                discarded.append(
                    DiscardedWrite(
                        segment=name,
                        sequence=0,
                        operation="?",
                        key="",
                        reason="torn frame line could not be parsed",
                        byte_offset=data_offset,
                    )
                )
                return applied, last_sequence, valid_bytes
            raise SegmentCorruptError(
                f"{name}: unreadable frame at byte {data_offset}"
            )
        if data_frame.kind != DATA_KIND:
            raise SegmentCorruptError(
                f"{name}: expected data frame at byte {data_offset}"
            )
        operation, key = _payload_operation(data_frame.payload)
        sequence = _sequence_of(data_frame.payload, name, data_offset)

        # Locate the following line, which must be the matching commit frame.
        next_start = line_end + 1 if newline != -1 else total
        if next_start >= total:
            if not is_final:
                raise SegmentCorruptError(
                    f"{name}: sealed segment ends without commit for seq {sequence}"
                )
            discarded.append(
                DiscardedWrite(
                    segment=name,
                    sequence=sequence,
                    operation=operation,
                    key=key,
                    reason="data frame landed but commit frame never did",
                    byte_offset=data_offset,
                )
            )
            return applied, last_sequence, valid_bytes

        commit_newline = raw.find(b"\n", next_start)
        commit_end = total if commit_newline == -1 else commit_newline
        commit_line = raw[next_start:commit_end]
        commit_frame = parse_frame(commit_line)
        if commit_frame is None or commit_frame.kind != COMMIT_KIND:
            if not is_final:
                raise SegmentCorruptError(
                    f"{name}: missing valid commit for seq {sequence}"
                )
            discarded.append(
                DiscardedWrite(
                    segment=name,
                    sequence=sequence,
                    operation=operation,
                    key=key,
                    reason="commit frame is torn or garbled",
                    byte_offset=data_offset,
                )
            )
            return applied, last_sequence, valid_bytes
        commit_sequence = _sequence_of(commit_frame.payload, name, next_start)
        if commit_sequence != sequence:
            raise SegmentCorruptError(
                f"{name}: commit seq {commit_sequence} does not match data seq {sequence}"
            )
        if sequence != expected:
            raise SegmentCorruptError(
                f"{name}: expected seq {expected}, found {sequence}"
            )
        if commit_newline == -1:
            # A commit frame is only valid when fully landed including its
            # newline; treat a missing terminator as a torn commit.
            if not is_final:
                raise SegmentCorruptError(
                    f"{name}: commit for seq {sequence} has no line terminator"
                )
            discarded.append(
                DiscardedWrite(
                    segment=name,
                    sequence=sequence,
                    operation=operation,
                    key=key,
                    reason="commit frame is missing its line terminator",
                    byte_offset=data_offset,
                )
            )
            return applied, last_sequence, valid_bytes
        apply_operation(data, data_frame.payload)
        applied += 1
        expected += 1
        last_sequence = sequence
        valid_bytes = commit_newline + 1
        pos = commit_newline + 1
    return applied, last_sequence, valid_bytes


def _sequence_of(payload: dict[str, object], name: str, offset: int) -> int:
    try:
        return int(payload["seq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SegmentCorruptError(
            f"{name}: frame at byte {offset} has no valid sequence"
        ) from exc
