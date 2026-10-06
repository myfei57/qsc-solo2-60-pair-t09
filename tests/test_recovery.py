"""Crash safety tests for segmented persistence and startup recovery."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from waterplant.console.server import Server
from waterplant.store import (
    PHASE_BOOTSTRAPPING,
    PHASE_FAILED,
    PHASE_READY,
    Store,
    StoreNotReadyError,
    encode_frame,
    recover,
    resolve_paths,
)
from waterplant.store.framing import DATA_KIND
from waterplant.store.recovery import MIGRATION_MARKER
from waterplant.store.recovery_audit import list_events
from waterplant.store.wal import segment_names

SEGMENT = "seg-00000001.log"


def call(server: Server, method: str, path: str, payload: dict | None = None):
    response = server.dispatch(method, path, payload)
    body = json.loads(response.body.decode("utf-8"))
    return response.status, body


def wal_directory(path: str) -> str:
    return resolve_paths(path)[0]


def active_segment(path: str) -> str:
    directory = wal_directory(path)
    names = segment_names(directory)
    return os.path.join(directory, names[-1])


def raw_data_frame(sequence: int, key: str, value: str) -> bytes:
    return encode_frame(
        DATA_KIND, {"seq": sequence, "op": "put", "key": key, "value": value}
    )


def append_bytes(path: str, blob: bytes) -> None:
    with open(active_segment(path), "ab") as handle:
        handle.write(blob)


def read_audit(path: str) -> list[dict[str, object]]:
    return [event.as_dict() for event in list_events(wal_directory(path))]


class SegmentedStoreCase(unittest.TestCase):
    def test_writes_segment_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.put("b", "2")
            self.assertEqual(store.last_sequence, 2)
            self.assertTrue(os.path.isdir(wal_directory(path)))
            self.assertEqual(segment_names(wal_directory(path)), [SEGMENT])
            store.close()

            reopened = Store.open(path)
            self.assertEqual(reopened.get("a"), ("1", True))
            self.assertEqual(reopened.get("b"), ("2", True))
            self.assertEqual(reopened.last_sequence, 2)
            reopened.put("c", "3")
            reopened.close()

            again = Store.open(path)
            self.assertEqual(again.get("c"), ("3", True))
            self.assertEqual(again.last_sequence, 3)

    def test_segment_rotation_replays_across_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path, segment_bytes=128)
            for index in range(20):
                store.put(f"key-{index:02d}", f"value-{index}")
            self.assertGreater(len(segment_names(wal_directory(path))), 1)
            store.close()

            reopened = Store.open(path, segment_bytes=128)
            for index in range(20):
                self.assertEqual(reopened.get(f"key-{index:02d}"), (f"value-{index}", True))
            self.assertEqual(reopened.last_sequence, 20)

    def test_clear_is_a_committed_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.clear()
            store.close()
            reopened = Store.open(path)
            self.assertEqual(reopened.count(), 0)
            self.assertEqual(reopened.get("a"), ("", False))


class TornTailCase(unittest.TestCase):
    def test_data_frame_without_commit_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.put("b", "2")
            store.close()

            # Simulate a crash after the data frame but before the commit.
            append_bytes(path, raw_data_frame(3, "c", "3"))

            recovered = Store.open(path)
            self.assertEqual(recovered.phase, PHASE_READY)
            self.assertEqual(recovered.get("a"), ("1", True))
            self.assertEqual(recovered.get("b"), ("2", True))
            self.assertEqual(recovered.get("c"), ("", False))
            self.assertEqual(recovered.last_sequence, 2)

            report = recovered.recovery_report()
            self.assertIsNotNone(report)
            discarded = report.discarded()
            self.assertEqual(len(discarded), 1)
            self.assertEqual(discarded[0]["operation"], "put")
            self.assertEqual(discarded[0]["key"], "c")
            self.assertEqual(discarded[0]["sequence"], 3)
            self.assertIn("commit", str(discarded[0]["reason"]))

            # The next operation continues the sequence and lands cleanly.
            recovered.put("d", "4")
            self.assertEqual(recovered.last_sequence, 3)
            recovered.close()

            clean = Store.open(path)
            self.assertEqual(clean.get("d"), ("4", True))
            self.assertEqual(clean.last_sequence, 3)
            self.assertEqual(clean.recovery_report().discarded(), [])

    def test_garbled_commit_frame_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.close()
            append_bytes(path, raw_data_frame(2, "b", "2"))
            append_bytes(path, b"X 00000000 not-a-commit\n")

            recovered = Store.open(path)
            self.assertEqual(recovered.get("a"), ("1", True))
            self.assertEqual(recovered.get("b"), ("", False))
            self.assertEqual(recovered.last_sequence, 1)
            discarded = recovered.recovery_report().discarded()
            self.assertEqual(len(discarded), 1)
            self.assertEqual(discarded[0]["sequence"], 2)

            # The torn bytes were physically cut, so re-recovery is clean.
            recovered.close()
            again = Store.open(path)
            self.assertEqual(again.recovery_report().discarded(), [])

    def test_torn_bytes_are_physically_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.close()
            append_bytes(path, raw_data_frame(2, "b", "2"))
            before = os.path.getsize(active_segment(path))

            Store.open(path).close()
            after = os.path.getsize(active_segment(path))
            self.assertLess(after, before)


class CorruptLogCase(unittest.TestCase):
    def test_corruption_inside_sealed_segment_fails_the_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path, segment_bytes=128)
            for index in range(20):
                store.put(f"key-{index:02d}", f"value-{index}")
            store.close()
            self.assertGreater(len(segment_names(wal_directory(path))), 1)

            # Flip a checksum digit of a frame inside the first, now sealed,
            # segment, but after intact frames so a readable prefix remains.
            first = os.path.join(wal_directory(path), "seg-00000001.log")
            with open(first, "rb") as handle:
                blob = bytearray(handle.read())
            first_newline = blob.index(b"\n")
            second_newline = blob.index(b"\n", first_newline + 1)
            target = second_newline + 3
            blob[target] = ord("0") if blob[target] != ord("0") else ord("1")
            with open(first, "wb") as handle:
                handle.write(blob)

            failed = Store.open(path)
            self.assertEqual(failed.phase, PHASE_FAILED)
            self.assertTrue(failed.failure_reason)
            with self.assertRaises(StoreNotReadyError):
                failed.put("x", "y")
            # Frames before the corrupt one stay readable, read only.
            self.assertEqual(failed.get("key-00"), ("value-0", True))
            self.assertFalse(failed.ready())

            audit = read_audit(path)
            self.assertEqual(audit[-1]["outcome"], "corrupt")

    def test_replaying_committed_operations_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.put("b", "2")
            store.delete("a")
            store.close()

            first_data, first_seq, first = recover(path)
            second_data, second_seq, second = recover(path)
            self.assertEqual(first_data, second_data)
            self.assertEqual(first_seq, second_seq)
            self.assertEqual(second_data, {"b": "2"})
            self.assertEqual(second_seq, 3)


class LifecycleGateCase(unittest.TestCase):
    def test_bootstrapping_store_rejects_mutations_but_serves_reads(self) -> None:
        store = Store("unused/state.json", {"a": "1"}, phase=PHASE_BOOTSTRAPPING)
        with self.assertRaises(StoreNotReadyError):
            store.put("b", "2")
        server = Server(store)
        status, body = call(server, "POST", "/history", {"kind": "x"})
        self.assertEqual(status, 503)
        self.assertIn("recover", body["error"])
        status, body = call(server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["phase"], PHASE_BOOTSTRAPPING)
        self.assertEqual(body["status"], "degraded")

    def test_failed_store_returns_503_for_posts(self) -> None:
        store = Store("unused/state.json", phase=PHASE_FAILED, failure_reason="corrupt log")
        server = Server(store)
        status, body = call(server, "POST", "/ops/reset-store", {})
        self.assertEqual(status, 503)
        status, _ = call(server, "GET", "/recovery")
        self.assertEqual(status, 200)


class LegacyMigrationCase(unittest.TestCase):
    def test_legacy_json_document_is_imported_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"a": "1", "b": "2"}, handle)

            store = Store.open(path)
            self.assertEqual(store.get("a"), ("1", True))
            self.assertEqual(store.get("b"), ("2", True))
            self.assertEqual(store.last_sequence, 2)
            self.assertTrue(os.path.exists(os.path.join(wal_directory(path), MIGRATION_MARKER)))
            self.assertTrue(os.path.exists(f"{path}.legacy.json"))
            self.assertFalse(os.path.exists(path))
            self.assertEqual(read_audit(path)[-1]["outcome"], "legacy_migrated")
            store.close()

            # A second boot must not import the archived document again.
            reopened = Store.open(path)
            self.assertEqual(reopened.last_sequence, 2)
            reopened.put("c", "3")
            self.assertEqual(reopened.last_sequence, 3)

    def test_marker_commits_migration_even_if_document_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"a": "1"}, handle)
            migrated = Store.open(path)
            migrated.close()

            # Crash window: marker durable but the rename of the document did
            # not happen. Recovery must treat the migration as already done.
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"a": "999", "ghost": "1"}, handle)
            reopened = Store.open(path)
            self.assertEqual(reopened.get("a"), ("1", True))
            self.assertEqual(reopened.get("ghost"), ("", False))
            self.assertEqual(read_audit(path)[-1]["outcome"], "clean")


class RecoveryConsoleCase(unittest.TestCase):
    def test_recovery_endpoint_exposes_point_discard_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("a", "1")
            store.close()
            append_bytes(path, raw_data_frame(2, "lost", "x"))

            server = Server(Store.open(path))
            status, body = call(server, "GET", "/recovery")
            self.assertEqual(status, 200)
            self.assertEqual(body["phase"], PHASE_READY)
            self.assertTrue(body["ready"])
            self.assertEqual(body["last_sequence"], 1)
            self.assertEqual(body["outcome"], "tail_discarded")
            self.assertEqual(body["discarded"][0]["key"], "lost")
            self.assertIn("fsync", body["policy"]["flush"])
            self.assertIn("automatic", body["policy"]["recovery"])
            outcomes = [event["outcome"] for event in body["history"]]
            self.assertIn("tail_discarded", outcomes)
            self.assertTrue(any(event["recovered_to_sequence"] == 1 for event in body["history"]))


if __name__ == "__main__":
    unittest.main()
