"""UsageLog is durable and queryable, and both runners log every call -- bisection retries included.
Real sqlite file on disk; no mock of UsageLog itself."""

import tempfile
import unittest
from pathlib import Path

from adaptive_concurrency import AdaptiveConfig, PoolConfig, UsageLog, run_adaptive, run_batches


def _attempt(log, **kw):
    log.log_batch_attempt(label="t", workers=1, batch_size=10, n_items=10, elapsed_seconds=1.0,
                          outcome=kw.pop("outcome", "ok"), **kw)


class UsageLogBasics(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "usage.db"
        self.log = UsageLog(self.path)

    def tearDown(self):
        self.log.close()

    def test_persists_across_reopen(self):
        _attempt(self.log)
        self.log.close()
        self.log = UsageLog(self.path)
        self.assertEqual(self.log.requests_today(), 1)

    def test_requests_in_window_excludes_old_rows(self):
        _attempt(self.log, timestamp=1000.0)
        _attempt(self.log, timestamp=5000.0)
        self.assertEqual(self.log.requests_in_window(100, now=5050.0), 1)
        self.assertEqual(self.log.requests_in_window(10000, now=5050.0), 2)

    def test_bytes_in_window_sums_only_non_null(self):
        _attempt(self.log, n_bytes=100, timestamp=1000.0)
        _attempt(self.log, timestamp=1001.0)
        self.assertEqual(self.log.bytes_in_window(10, now=1005.0), 100)

    def test_rejects_unknown_outcome(self):
        with self.assertRaises(ValueError):
            _attempt(self.log, outcome="bogus")


class WiredIntoRunBatches(unittest.TestCase):
    def setUp(self):
        self.usage = UsageLog(Path(tempfile.mkdtemp()) / "usage.db")

    def tearDown(self):
        self.usage.close()

    def test_logs_every_top_level_batch(self):
        cfg = PoolConfig(label="wired", batch_size=10, workers=1, usage_log=self.usage)
        written = run_batches(list(range(37)), lambda c: (True, len(c), ""), cfg)
        self.assertEqual(written, 37)
        self.assertEqual(self.usage.requests_today(), 4)
        self.assertEqual(self.usage.requests_in_window(3600), 4)

    def test_logs_bisection_retries_as_separate_attempts(self):
        cfg = PoolConfig(label="bisect", batch_size=4, workers=1, bisect_on_failure=True, usage_log=self.usage)
        run_batches(list(range(4)), lambda c: (False, 0, "boom") if len(c) > 1 else (True, 1, ""), cfg)
        # one size-4 failure, two size-2 failures, four size-1 successes
        self.assertEqual(self.usage.requests_today(), 7)


class WiredIntoRunAdaptive(unittest.TestCase):
    def test_logs_every_dispatched_attempt(self):
        usage = UsageLog(Path(tempfile.mkdtemp()) / "usage.db")
        calls = {"n": 0}

        def process_batch(chunk):
            calls["n"] += 1
            return True, len(chunk), ""

        cfg = AdaptiveConfig(label="adaptive", batch_size=5, workers=1, max_workers=1, usage_log=usage)
        self.assertEqual(run_adaptive(list(range(23)), process_batch, cfg), 23)
        self.assertEqual(usage.requests_today(), calls["n"])
        self.assertEqual(usage.requests_in_window(3600), calls["n"])
        usage.close()

    def test_omitting_usage_log_changes_nothing(self):
        cfg = AdaptiveConfig(label="no-log", batch_size=5, workers=1, max_workers=1)
        self.assertEqual(run_adaptive(list(range(23)), lambda c: (True, len(c), ""), cfg), 23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
