"""Crash safety tests for the segmented journal store."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from unittest import mock

from waterplant.console.seed import seed_defaults
from waterplant.console.server import Server
from waterplant.store import RecoveryPendingError, Store
from waterplant.store.frames import encode_frame
from waterplant.store.journal import (
    CHECKPOINT_DIR,
    QUARANTINE_DIR,
    REPORT_FILE,
    SEGMENT_DIR,
    Journal,
)


def journal_root(path: str) -> str:
    return f"{path}.d"


def segment_dir(path: str) -> str:
    return os.path.join(journal_root(path), SEGMENT_DIR)


def only_segment(path: str) -> str:
    names = [name for name in os.listdir(segment_dir(path)) if name.endswith(".wal")]
    assert len(names) == 1, f"expected one segment, found {names}"
    return os.path.join(segment_dir(path), names[0])


class RecoveryCase(unittest.TestCase):
    def test_operations_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.put("beta", "2")
            store.delete("alpha")
            store.close()

            reopened = Store.open(path)
            self.assertEqual(reopened.get("alpha"), ("", False))
            self.assertEqual(reopened.get("beta"), ("2", True))
            report = reopened.recovery_report()
            self.assertEqual(report["status"], "recovered")
            self.assertEqual(report["frames_replayed"], 3)
            reopened.close()

    def test_torn_tail_is_identified_quarantined_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.put("beta", "2")
            store.close()

            segment = only_segment(path)
            clean_size = os.path.getsize(segment)
            # a kill mid-flush: the header and half the payload of op gamma
            partial = encode_frame(3, b'{"op":"put","key":"gamma","value":"3"}')[:25]
            with open(segment, "ab") as handle:
                handle.write(partial)

            reopened = Store.open(path)
            self.assertEqual(reopened.get("beta"), ("2", True))
            self.assertEqual(reopened.get("gamma"), ("", False))

            report = reopened.recovery_report()
            self.assertEqual(report["status"], "recovered_with_loss")
            torn = report["torn"]
            self.assertIsNotNone(torn)
            self.assertEqual(torn["offset"], clean_size)
            self.assertEqual(torn["discarded_bytes"], len(partial))
            self.assertIn("seq=3", torn["detail"])

            quarantine = os.path.join(journal_root(path), torn["quarantine"])
            self.assertTrue(os.path.exists(quarantine))
            with open(quarantine, "rb") as handle:
                self.assertEqual(handle.read(), partial)

            # the segment was truncated back to the last complete frame
            self.assertEqual(os.path.getsize(segment), clean_size)
            reopened.close()

    def test_checksum_failure_names_the_dropped_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.close()

            segment = only_segment(path)
            frame = encode_frame(2, b'{"op":"put","key":"beta","value":"2"}')
            # tamper with the value but keep the JSON valid, so the checksum
            # fails while the dropped operation stays identifiable
            frame = frame.replace(b'"value":"2"', b'"value":"3"')
            with open(segment, "ab") as handle:
                handle.write(frame)

            reopened = Store.open(path)
            self.assertEqual(reopened.get("beta"), ("", False))
            torn = reopened.recovery_report()["torn"]
            self.assertIn("failed checksum", torn["detail"])
            self.assertIn("key='beta'", torn["detail"])
            reopened.close()

    def test_recovers_to_last_consistent_state_when_killed_mid_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            for index in range(5):
                store.put(f"key{index}", str(index))
            # process is killed here: no close, no clean shutdown — every put
            # was already fsynced, so the on-disk state is exactly seq 5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                del store

            segment = only_segment(path)
            with open(segment, "ab") as handle:
                handle.write(encode_frame(6, b'{"op":"put","key":"key5","value":"5"}')[:22])

            reopened = Store.open(path)
            for index in range(5):
                self.assertEqual(reopened.get(f"key{index}"), (str(index), True))
            self.assertEqual(reopened.get("key5"), ("", False))
            self.assertEqual(reopened.recovery_report()["recovered_seq"], 5)

            # new writes reuse the discarded sequence number without collision
            reopened.put("key5", "5")
            reopened.close()
            again = Store.open(path)
            self.assertEqual(again.get("key5"), ("5", True))
            self.assertEqual(again.count(), 6)
            again.close()

    def test_recovery_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.close()
            segment = only_segment(path)
            with open(segment, "ab") as handle:
                handle.write(b"\xde\xad\xbe")

            first = Store.open(path)
            self.assertEqual(first.recovery_report()["status"], "recovered_with_loss")
            first.close()

            second = Store.open(path)
            self.assertEqual(second.get("alpha"), ("1", True))
            report = second.recovery_report()
            self.assertEqual(report["status"], "recovered")
            self.assertIsNone(report["torn"])
            quarantine = os.path.join(journal_root(path), QUARANTINE_DIR)
            self.assertEqual(len(os.listdir(quarantine)), 1)  # not duplicated
            second.close()

    def test_operations_are_rejected_until_recovery_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path, auto_recover=False)
            self.assertFalse(store.ready)
            with self.assertRaises(RecoveryPendingError):
                store.put("alpha", "1")
            with self.assertRaises(RecoveryPendingError):
                store.get("alpha")
            with self.assertRaises(RecoveryPendingError):
                store.keys()
            with self.assertRaises(RecoveryPendingError):
                store.clear()

            report = store.recover()
            self.assertTrue(store.ready)
            self.assertEqual(report.status, "clean")
            store.put("alpha", "1")
            self.assertEqual(store.get("alpha"), ("1", True))
            # recovering again is a no-op returning the same report
            self.assertIs(store.recover(), report)
            store.close()

    def test_rotation_writes_checkpoint_and_prunes_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path, segment_max_bytes=256)
            for index in range(50):
                store.put(f"key{index:02d}", f"value-{index}")
            store.close()

            checkpoints = os.listdir(os.path.join(journal_root(path), CHECKPOINT_DIR))
            self.assertEqual(len([name for name in checkpoints if name.endswith(".chk")]), 1)
            segments = [n for n in os.listdir(segment_dir(path)) if n.endswith(".wal")]
            self.assertEqual(len(segments), 1)  # covered segments were pruned

            reopened = Store.open(path, segment_max_bytes=256)
            self.assertEqual(reopened.count(), 50)
            self.assertEqual(reopened.get("key00"), ("value-0", True))
            self.assertEqual(reopened.get("key49"), ("value-49", True))
            report = reopened.recovery_report()
            self.assertGreater(report["checkpoint_seq"], 0)
            reopened.close()

    def test_crash_during_checkpoint_write_leaves_no_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.close()

            # a checkpoint that was never renamed because the process died
            checkpoints = os.path.join(journal_root(path), CHECKPOINT_DIR)
            with open(os.path.join(checkpoints, "chk-0000000000000009.chk.tmp"), "w") as handle:
                handle.write('{"seq": 9, "crc": 0, "da')

            reopened = Store.open(path)
            self.assertEqual(reopened.get("alpha"), ("1", True))
            self.assertEqual(os.listdir(checkpoints), [])  # stray temp cleaned up
            reopened.close()

    def test_corrupt_checkpoint_falls_back_to_older_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            journal = Journal(journal_root(path))
            journal.append(1, b'{"op":"put","key":"a","value":"1"}')
            journal.write_checkpoint(1, {"a": "1"})
            journal.append(2, b'{"op":"put","key":"b","value":"2"}')
            journal.write_checkpoint(2, {"a": "1", "b": "2"})
            journal.append(3, b'{"op":"put","key":"c","value":"3"}')
            journal.close()

            # bit rot in the newest checkpoint: keep the seq, break the crc
            checkpoints = os.path.join(journal_root(path), CHECKPOINT_DIR)
            name = os.listdir(checkpoints)[0]
            with open(os.path.join(checkpoints, name), "r", encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["seq"], 2)
            document["data"]["b"] = "tampered"
            with open(os.path.join(checkpoints, name), "w", encoding="utf-8") as handle:
                json.dump(document, handle)

            # no older checkpoint survives rotation, so recovery replays the log
            store = Store.open(path)
            self.assertEqual(store.get("a"), ("1", True))
            self.assertEqual(store.get("b"), ("2", True))
            self.assertEqual(store.get("c"), ("3", True))
            store.close()

    def test_every_mutation_is_fsynced_before_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            with mock.patch("os.fsync", wraps=os.fsync) as fsync:
                store.put("alpha", "1")
                self.assertGreaterEqual(fsync.call_count, 1)
                fsync.reset_mock()
                store.delete("alpha")
                self.assertGreaterEqual(fsync.call_count, 1)
            store.close()

    def test_clear_is_durable_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.clear()
            store.close()
            reopened = Store.open(path)
            self.assertEqual(reopened.count(), 0)
            self.assertEqual(reopened.get("alpha"), ("", False))
            reopened.close()

    def test_recovery_report_is_persisted_for_the_audit_trail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.close()

            report_path = os.path.join(journal_root(path), REPORT_FILE)
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["status"], "clean")
            self.assertIn("recovered_at", persisted)

            reopened = Store.open(path)
            self.assertEqual(reopened.recovery_report()["frames_replayed"], 1)
            reopened.close()

    def test_legacy_single_file_store_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"alpha": "1", "beta": "2"}, handle)

            store = Store.open(path)
            self.assertEqual(store.get("alpha"), ("1", True))
            self.assertEqual(store.get("beta"), ("2", True))
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.exists(f"{path}.migrated"))
            self.assertIn(
                "migrated 2 keys from the legacy single-file store",
                store.recovery_report()["notes"],
            )
            store.put("gamma", "3")
            store.close()

            reopened = Store.open(path)
            self.assertEqual(reopened.count(), 3)
            reopened.close()

    def test_corrupt_legacy_document_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"alpha": "1", "be')

            store = Store.open(path)
            self.assertEqual(store.count(), 0)
            self.assertTrue(os.path.exists(f"{path}.corrupt"))
            self.assertIn(
                "legacy store document was corrupt; quarantined to "
                f"{os.path.abspath(path)}.corrupt",
                store.recovery_report()["notes"],
            )
            store.close()


class ConsoleRecoveryCase(unittest.TestCase):
    def test_console_displays_recovery_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            store = Store.open(path)
            store.put("alpha", "1")
            store.close()
            segment = only_segment(path)
            with open(segment, "ab") as handle:
                handle.write(encode_frame(2, b'{"op":"put","key":"beta","value":"2"}')[:24])

            reopened = Store.open(path)
            seed_defaults(reopened)
            server = Server(reopened)
            response = server.dispatch("GET", "/ops/recovery")
            self.assertEqual(response.status, 200)
            body = json.loads(response.body.decode("utf-8"))
            self.assertEqual(body["status"], "recovered_with_loss")
            self.assertEqual(body["torn"]["discarded_bytes"], 24)
            self.assertIn("seg-", body["torn"]["segment"])
            self.assertEqual(body["recovered_seq"], 1)

            catalog = json.loads(server.dispatch("GET", "/catalog").body.decode("utf-8"))
            self.assertIn(
                "/ops/recovery", [route["path"] for route in catalog["routes"]]
            )
            reopened.close()


if __name__ == "__main__":
    unittest.main()
