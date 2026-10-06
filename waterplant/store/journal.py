"""Segmented append-only journal with checkpoints, quarantine and recovery.

Layout under the journal root::

    segments/seg-<index>.wal     append-only frame logs, rotated by size
    checkpoints/chk-<seq>.chk    consistent snapshots, written tmp + rename
    quarantine/<segment>.<off>.torn   discarded torn tails kept for forensics
    recovery.json                the most recent recovery report

Crash safety rules:

* every frame is flushed and fsynced before the caller is acknowledged;
* a checkpoint is written to a temporary file, fsynced, then atomically
  renamed over the previous one, so a crash can never leave half a
  checkpoint;
* a torn tail left by an interrupted flush is quarantined and the segment is
  truncated back to the last complete frame — rerunning recovery after a
  crash during recovery itself simply redoes the same steps.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import dataclass, field

from .frames import Frame, decode_frames, describe_torn, encode_frame

SEGMENT_DIR = "segments"
CHECKPOINT_DIR = "checkpoints"
QUARANTINE_DIR = "quarantine"
REPORT_FILE = "recovery.json"
DEFAULT_SEGMENT_MAX_BYTES = 1024 * 1024


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _checkpoint_crc(data: dict[str, str]) -> int:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return zlib.crc32(canonical) & 0xFFFFFFFF


@dataclass(frozen=True)
class TornTail:
    """The incomplete tail that recovery identified, quarantined and dropped."""

    segment: str
    offset: int
    discarded_bytes: int
    detail: str
    quarantine: str

    def as_dict(self) -> dict[str, object]:
        return {
            "segment": self.segment,
            "offset": self.offset,
            "discarded_bytes": self.discarded_bytes,
            "detail": self.detail,
            "quarantine": self.quarantine,
        }


@dataclass(frozen=True)
class RecoveryReport:
    """What startup recovery did; persisted for the audit trail and console."""

    recovered_at: str
    status: str  # "clean" | "recovered" | "recovered_with_loss"
    checkpoint_seq: int
    recovered_seq: int
    frames_replayed: int
    frames_skipped: int
    segments_scanned: list[str] = field(default_factory=list)
    torn: TornTail | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "recovered_at": self.recovered_at,
            "status": self.status,
            "checkpoint_seq": self.checkpoint_seq,
            "recovered_seq": self.recovered_seq,
            "frames_replayed": self.frames_replayed,
            "frames_skipped": self.frames_skipped,
            "segments_scanned": list(self.segments_scanned),
            "torn": None if self.torn is None else self.torn.as_dict(),
            "notes": list(self.notes),
        }


class Journal:
    """Owns the on-disk layout the store appends to and recovers from."""

    def __init__(self, root: str, segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES) -> None:
        self._root = root
        self._segment_max = segment_max_bytes
        self._segments = os.path.join(root, SEGMENT_DIR)
        self._checkpoints = os.path.join(root, CHECKPOINT_DIR)
        self._quarantine = os.path.join(root, QUARANTINE_DIR)
        for directory in (self._segments, self._checkpoints, self._quarantine):
            os.makedirs(directory, exist_ok=True)
        self._active_index = 0
        self._active_path = ""
        self._active_handle: object | None = None
        self._active_size = 0

    @property
    def root(self) -> str:
        return self._root

    @property
    def segment_max_bytes(self) -> int:
        return self._segment_max

    @property
    def active_size(self) -> int:
        return self._active_size

    def is_empty(self) -> bool:
        """True when no checkpoint and no segment file exists yet."""

        return not self._segment_names() and not self._checkpoint_names()

    # ------------------------------------------------------------------ writes

    def append(self, seq: int, payload: bytes) -> None:
        """Append one frame and fsync it before returning."""

        handle = self._ensure_handle()
        handle.write(encode_frame(seq, payload))
        handle.flush()
        os.fsync(handle.fileno())
        self._active_size = os.path.getsize(self._active_path)

    def rotate(self) -> None:
        """Close the current segment and start the next one."""

        self._close_handle()
        self._active_index += 1
        self._active_path = self._segment_path(self._active_index)
        self._active_size = 0
        self._ensure_handle()
        _fsync_dir(self._segments)

    def write_checkpoint(self, seq: int, data: dict[str, str]) -> None:
        """Atomically publish a snapshot covering every record up to ``seq``."""

        document = {"seq": seq, "crc": _checkpoint_crc(data), "data": dict(data)}
        raw = json.dumps(document, ensure_ascii=False, sort_keys=True)
        final_name = f"chk-{seq:016d}.chk"
        temporary = os.path.join(self._checkpoints, final_name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, os.path.join(self._checkpoints, final_name))
        for name in self._checkpoint_names():
            if name != final_name:
                os.unlink(os.path.join(self._checkpoints, name))
        _fsync_dir(self._checkpoints)

    def prune_segments(self) -> None:
        """Drop segments fully covered by the latest checkpoint."""

        removed = False
        for name in self._segment_names():
            if self._segment_index(name) < self._active_index:
                os.unlink(os.path.join(self._segments, name))
                removed = True
        if removed:
            _fsync_dir(self._segments)

    def close(self) -> None:
        self._close_handle()

    # ---------------------------------------------------------------- recovery

    def recover(
        self, notes: list[str] | None = None
    ) -> tuple[int, dict[str, str], list[Frame], RecoveryReport]:
        """Scan checkpoints and segments back into a consistent prefix.

        Returns the checkpoint sequence, the checkpoint data, the frames that
        still need replaying (sequence above the checkpoint, in order) and the
        recovery report. Safe to rerun: quarantine and truncation are
        idempotent, so a crash during recovery is itself recoverable.
        """

        report_notes = list(notes or [])
        self._discard_stray_temp_files()
        checkpoint_seq, checkpoint_data = self._load_checkpoint()
        replay: list[Frame] = []
        skipped = 0
        scanned: list[str] = []
        torn: TornTail | None = None
        recovered_seq = checkpoint_seq

        names = self._segment_names()
        for position, name in enumerate(names):
            scanned.append(name)
            path = os.path.join(self._segments, name)
            with open(path, "rb") as handle:
                region = handle.read()
            frames, stop, reason = decode_frames(region)
            for frame in frames:
                if frame.seq <= checkpoint_seq:
                    skipped += 1  # already folded into the checkpoint
                    continue
                replay.append(frame)
                recovered_seq = max(recovered_seq, frame.seq)
            if reason is not None:
                torn = self._handle_torn_tail(name, path, stop, region[stop:])
                leftover = names[position + 1 :]
                if leftover:
                    report_notes.append(
                        "segments after the torn tail were left unread: " + ", ".join(leftover)
                    )
                break

        self._open_active(names[-1] if names else None)
        status = "clean"
        if replay or checkpoint_seq or skipped:
            status = "recovered"
        if torn is not None:
            status = "recovered_with_loss"
        report = RecoveryReport(
            recovered_at=_utcnow(),
            status=status,
            checkpoint_seq=checkpoint_seq,
            recovered_seq=recovered_seq,
            frames_replayed=len(replay),
            frames_skipped=skipped,
            segments_scanned=scanned,
            torn=torn,
            notes=report_notes,
        )
        self._persist_report(report)
        return checkpoint_seq, checkpoint_data, replay, report

    def read_report(self) -> dict[str, object] | None:
        """Return the last persisted recovery report, when present."""

        try:
            with open(os.path.join(self._root, REPORT_FILE), "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None

    # ----------------------------------------------------------------- interns

    def _handle_torn_tail(self, name: str, path: str, offset: int, discarded: bytes) -> TornTail:
        quarantine_name = f"{name}.{offset}.torn"
        quarantine_path = os.path.join(self._quarantine, quarantine_name)
        temporary = quarantine_path + ".tmp"
        with open(temporary, "wb") as handle:
            handle.write(discarded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, quarantine_path)
        _fsync_dir(self._quarantine)
        with open(path, "r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
        return TornTail(
            segment=name,
            offset=offset,
            discarded_bytes=len(discarded),
            detail=describe_torn(discarded),
            quarantine=os.path.join(QUARANTINE_DIR, quarantine_name),
        )

    def _open_active(self, last_segment: str | None) -> None:
        if last_segment is None:
            self._active_index = 1
            self._active_path = self._segment_path(self._active_index)
        else:
            self._active_index = self._segment_index(last_segment)
            self._active_path = os.path.join(self._segments, last_segment)
        # a torn segment was already truncated, so the current size is the
        # offset of the last complete frame in every case; a fresh segment
        # does not exist until the first append creates it
        self._active_size = (
            os.path.getsize(self._active_path) if os.path.exists(self._active_path) else 0
        )
        self._ensure_handle()
        _fsync_dir(self._segments)

    def _ensure_handle(self):
        if self._active_handle is None:
            if not self._active_path:
                self._active_index = 1
                self._active_path = self._segment_path(self._active_index)
            self._active_handle = open(self._active_path, "ab")
        return self._active_handle

    def _close_handle(self) -> None:
        if self._active_handle is not None:
            self._active_handle.flush()
            os.fsync(self._active_handle.fileno())
            self._active_handle.close()
            self._active_handle = None

    def _segment_path(self, index: int) -> str:
        return os.path.join(self._segments, f"seg-{index:016d}.wal")

    @staticmethod
    def _segment_index(name: str) -> int:
        return int(name[len("seg-") : -len(".wal")])

    def _segment_names(self) -> list[str]:
        try:
            names = os.listdir(self._segments)
        except FileNotFoundError:
            return []
        return sorted(
            name for name in names if name.startswith("seg-") and name.endswith(".wal")
        )

    def _checkpoint_names(self) -> list[str]:
        try:
            names = os.listdir(self._checkpoints)
        except FileNotFoundError:
            return []
        return sorted(name for name in names if name.startswith("chk-") and name.endswith(".chk"))

    def _load_checkpoint(self) -> tuple[int, dict[str, str]]:
        for name in reversed(self._checkpoint_names()):
            try:
                with open(os.path.join(self._checkpoints, name), "r", encoding="utf-8") as handle:
                    document = json.load(handle)
                seq = int(document["seq"])
                data = document["data"]
                if not isinstance(data, dict):
                    continue
                if int(document["crc"]) != _checkpoint_crc({str(k): str(v) for k, v in data.items()}):
                    continue  # corrupt snapshot: fall back to an older one
                return seq, {str(key): str(value) for key, value in data.items()}
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return 0, {}

    def _discard_stray_temp_files(self) -> None:
        for directory in (self._checkpoints, self._quarantine):
            for name in os.listdir(directory):
                if name.endswith(".tmp"):
                    os.unlink(os.path.join(directory, name))

    def _persist_report(self, report: RecoveryReport) -> None:
        raw = json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        temporary = os.path.join(self._root, REPORT_FILE + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, os.path.join(self._root, REPORT_FILE))
        _fsync_dir(self._root)
