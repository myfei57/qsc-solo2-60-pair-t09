"""Line based framing for the segmented write ahead log.

Every persisted mutation is two newline delimited frames:

* a data frame ``D <crc> <json>`` describing the operation;
* a commit frame ``C <crc> <json>`` carrying the same sequence number.

A data frame without its matching commit frame is an operation that never
finished landing on disk. Recovery treats such a tail as the torn remainder of
a partial write and discards it. The checksum is a CRC32 over the JSON body so
a frame that was garbled mid write can never be mistaken for a good one.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass

DATA_KIND = "D"
COMMIT_KIND = "C"
_KINDS = (DATA_KIND, COMMIT_KIND)


@dataclass(frozen=True)
class Frame:
    """One parsed log frame."""

    kind: str
    payload: dict[str, object]


def _checksum(body: bytes) -> str:
    return f"{zlib.crc32(body) & 0xFFFFFFFF:08x}"


def encode_frame(kind: str, payload: dict[str, object]) -> bytes:
    """Serialise one frame including its trailing newline."""

    if kind not in _KINDS:
        raise ValueError(f"unknown frame kind {kind!r}")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    body_bytes = body.encode("utf-8")
    line = f"{kind} {_checksum(body_bytes)} {body}\n"
    return line.encode("utf-8")


def parse_frame(raw: bytes) -> Frame | None:
    """Parse a frame, returning ``None`` for any torn or malformed line."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    parts = text.split(" ", 2)
    if len(parts) != 3:
        return None
    kind, checksum, body = parts
    if kind not in _KINDS or len(checksum) != 8:
        return None
    body_bytes = body.encode("utf-8")
    if _checksum(body_bytes) != checksum.lower():
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return Frame(kind=kind, payload=payload)
