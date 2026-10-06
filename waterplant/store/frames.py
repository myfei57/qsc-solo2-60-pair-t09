"""Length prefixed, checksummed log frames stored inside journal segments.

Every record is framed as::

    magic(4) | seq(8) | payload length(4) | crc32(4) | payload(bytes)

The crc covers the sequence number, the length and the payload, so a write
that was interrupted mid-flush — a torn header, a short payload or a corrupt
checksum — is detected when the segment is scanned during recovery.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass

MAGIC = b"WPL1"
HEADER = struct.Struct(">4sQII")
HEADER_SIZE = HEADER.size
MAX_PAYLOAD = 16 * 1024 * 1024  # sanity bound so a corrupt length cannot ask for gigabytes


@dataclass(frozen=True)
class Frame:
    """One decoded record: a sequenced operation payload."""

    seq: int
    payload: bytes

    def operation(self) -> dict[str, object]:
        decoded = json.loads(self.payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("frame payload must be a JSON object")
        return decoded


def encode_frame(seq: int, payload: bytes) -> bytes:
    """Serialise one record with its integrity header."""

    body = struct.pack(">QI", seq, len(payload)) + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return HEADER.pack(MAGIC, seq, len(payload), crc) + payload


def decode_frames(region: bytes) -> tuple[list[Frame], int, str | None]:
    """Decode every complete frame in ``region``.

    Returns the decoded frames, the offset decoding stopped at and a torn
    reason. The reason is ``None`` when the region ends exactly on a frame
    boundary; otherwise the bytes from the stop offset onward are the
    incomplete tail left by an interrupted flush.
    """

    frames: list[Frame] = []
    pos = 0
    while pos < len(region):
        remaining = len(region) - pos
        if remaining < HEADER_SIZE:
            return frames, pos, f"partial frame header ({remaining} of {HEADER_SIZE} bytes)"
        magic, seq, length, crc = HEADER.unpack_from(region, pos)
        if magic != MAGIC:
            return frames, pos, "unrecognised frame magic"
        if length > MAX_PAYLOAD:
            return frames, pos, f"frame seq={seq} length {length} exceeds limit"
        if remaining < HEADER_SIZE + length:
            present = remaining - HEADER_SIZE
            return frames, pos, f"frame seq={seq} truncated ({present} of {length} payload bytes)"
        payload = region[pos + HEADER_SIZE : pos + HEADER_SIZE + length]
        expected = zlib.crc32(struct.pack(">QI", seq, length) + payload) & 0xFFFFFFFF
        if expected != crc:
            return frames, pos, f"frame seq={seq} failed checksum"
        frames.append(Frame(seq=seq, payload=payload))
        pos += HEADER_SIZE + length
    return frames, pos, None


def describe_torn(region: bytes) -> str:
    """Best effort description of a discarded tail for the recovery audit."""

    if not region:
        return "no bytes discarded"
    parts = [f"{len(region)} bytes"]
    if len(region) < HEADER_SIZE:
        parts.append(f"incomplete frame header ({len(region)} of {HEADER_SIZE} bytes)")
        return ", ".join(parts)
    magic, seq, length, _crc = HEADER.unpack_from(region, 0)
    if magic != MAGIC:
        parts.append("unrecognised frame header")
        return ", ".join(parts)
    payload = region[HEADER_SIZE:]
    if len(payload) < length:
        parts.append(f"frame seq={seq} truncated ({len(payload)} of {length} payload bytes)")
    else:
        parts.append(f"frame seq={seq} failed checksum")
    try:
        op = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ", ".join(parts)
    if isinstance(op, dict):
        parts.append(f"dropped op={op.get('op')} key={op.get('key')!r}")
    return ", ".join(parts)
