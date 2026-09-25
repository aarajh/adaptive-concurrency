import tempfile
import unittest
from pathlib import Path

from adaptive_concurrency import AdaptiveConfig, Controller, audit_log, clear_override, current_overrides, set_override
from adaptive_concurrency.control import CLEARED


class ControlLog(unittest.TestCase):
    def setUp(self):
        self.db = Path(tempfile.mkdtemp()) / "control.sqlite"

    def test_missing_db_reads_empty_without_creating_it(self):
        self.assertEqual(current_overrides(self.db), {})
        self.assertEqual(audit_log(self.db), [])
        self.assertFalse(self.db.exists())

    def test_latest_row_wins_and_history_is_kept(self):
        set_override(self.db, "max_workers", 4, note="a")
        set_override(self.db, "max_workers", 8, note="b")
        set_override(self.db, "force_batch_size", 20)
        self.assertEqual(current_overrides(self.db), {"max_workers": "8", "force_batch_size": "20"})
        self.assertEqual([r[2:] for r in audit_log(self.db)],
                         [("max_workers", "4", "a"), ("max_workers", "8", "b"), ("force_batch_size", "20", None)])

    def test_clear_is_a_tombstone_not_a_delete(self):
        set_override(self.db, "max_workers", 8)
        clear_override(self.db, "max_workers", note="back to config")
        self.assertEqual(current_overrides(self.db), {})
        self.assertEqual([r[3] for r in audit_log(self.db)], ["8", CLEARED])


class ApplyOverrides(unittest.TestCase):
    def test_force_is_clamped_to_ceiling(self):
        ctl = Controller(AdaptiveConfig(workers=2, max_workers=4, batch_size=10, max_batch_size=40))
        changes = ctl.apply_overrides({"force_workers": "99", "force_batch_size": "1"})
        self.assertEqual((ctl.setting.workers, ctl.setting.batch_size), (4, 1))
        self.assertEqual(changes, ["force workers 2->4", "force batch_size 10->1"])

    def test_unknown_keys_are_ignored(self):
        ctl = Controller(AdaptiveConfig(workers=2, max_workers=4))
        self.assertEqual(ctl.apply_overrides({"max_wrokers": "8"}), [])

    def test_not_thread_safe_run_ignores_worker_keys_but_applies_batch_keys(self):
        ctl = Controller(AdaptiveConfig(workers=1, max_workers=1, batch_size=10, max_batch_size=40))
        changes = ctl.apply_overrides({"max_workers": "8", "force_workers": "8", "force_batch_size": "20"})
        self.assertEqual((ctl.setting.workers, ctl.setting.batch_size), (1, 20))
        self.assertTrue(changes[0].startswith("ignored max_workers, force_workers"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
