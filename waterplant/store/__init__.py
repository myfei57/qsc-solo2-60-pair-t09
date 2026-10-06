"""File backed persistence used by every control component."""

from .commands import append_command, clear_commands, load_commands
from .epoch import advance_epoch, load_float, save_float
from .framing import COMMIT_KIND, DATA_KIND, Frame, encode_frame, parse_frame
from .history import Event, append_event, event_count, list_events
from .recovery import RecoveryReport, recover, resolve_paths
from .recovery_audit import RecoveryEvent, list_events as list_recovery_events
from .report import StoreState, describe_store, export_state, has_key, store_state
from .store import (
    PHASE_BOOTSTRAPPING,
    PHASE_CLOSED,
    PHASE_FAILED,
    PHASE_READY,
    PHASE_RECOVERING,
    Store,
    StoreNotReadyError,
)
from .wal import (
    DEFAULT_SEGMENT_BYTES,
    DiscardedWrite,
    ReplayResult,
    SegmentCorruptError,
    WriteAheadLog,
)

__all__ = [
    "COMMIT_KIND",
    "DATA_KIND",
    "DEFAULT_SEGMENT_BYTES",
    "DiscardedWrite",
    "Event",
    "Frame",
    "PHASE_BOOTSTRAPPING",
    "PHASE_CLOSED",
    "PHASE_FAILED",
    "PHASE_READY",
    "PHASE_RECOVERING",
    "RecoveryEvent",
    "RecoveryReport",
    "ReplayResult",
    "SegmentCorruptError",
    "Store",
    "StoreNotReadyError",
    "StoreState",
    "WriteAheadLog",
    "advance_epoch",
    "append_command",
    "append_event",
    "clear_commands",
    "describe_store",
    "encode_frame",
    "event_count",
    "export_state",
    "has_key",
    "list_events",
    "list_recovery_events",
    "load_commands",
    "load_float",
    "parse_frame",
    "recover",
    "resolve_paths",
    "save_float",
    "store_state",
]
