import time
import unittest

from adaptive_concurrency import AdaptiveConfig, PoolConfig, run_adaptive, run_batches
from tests.support import captured_log, in_empty_cwd


def always_fail(chunk):
    return False, 0, "HTTPError: 500"


class CircuitBreaker(unittest.TestCase):
    def test_run_batches_stops_after_max_consecutive_failures(self):
        calls = []
        cfg = PoolConfig(batch_size=1, max_consecutive_failures=3)
        with captured_log() as lines:
            written = run_batches(list(range(10)), lambda c: calls.append(c) or always_fail(c), cfg)
        self.assertEqual((written, len(calls)), (0, 3))
        self.assertTrue(any("SYSTEMATIC FAILURE" in line for line in lines))

    def test_run_adaptive_stops_after_max_consecutive_failures(self):
        calls = []
        cfg = AdaptiveConfig(batch_size=1, workers=1, max_workers=1, max_consecutive_failures=3)
        written = run_adaptive(list(range(10)), lambda c: calls.append(c) or always_fail(c), cfg)
        self.assertEqual((written, len(calls)), (0, 3))

    def test_a_success_resets_the_count(self):
        cfg = PoolConfig(batch_size=1, max_consecutive_failures=2)
        written = run_batches(list(range(6)), lambda c: (True, 1, "") if c[0] % 2 else always_fail(c), cfg)
        self.assertEqual(written, 3)


class FailuresCsv(unittest.TestCase):
    def test_path_gets_a_header_and_one_row_per_failed_batch(self):
        with in_empty_cwd() as tmp:
            cfg = PoolConfig(batch_size=2, failures_csv=tmp / "fail.csv")
            run_batches([1, 2, 3, 4], lambda c: always_fail(c) if 3 in c else (True, 2, ""), cfg)
            lines = (tmp / "fail.csv").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], "batch,error,timestamp")
            self.assertTrue(lines[1].startswith("3;4,HTTPError: 500,"))

    def test_none_writes_nothing(self):
        with in_empty_cwd() as tmp:
            run_batches([1, 2, 3], always_fail, PoolConfig(batch_size=1))
            run_adaptive([1, 2, 3], always_fail, AdaptiveConfig(batch_size=1, workers=2, max_workers=2))
            self.assertEqual(list(tmp.iterdir()), [])


class ThreadedRunBatches(unittest.TestCase):
    def test_writes_everything_on_worker_threads(self):
        written = run_batches(list(range(100)), lambda c: (True, len(c), ""), PoolConfig(batch_size=7, workers=4))
        self.assertEqual(written, 100)

    def test_an_exception_propagates_without_running_the_queued_batches(self):
        calls = []

        def process(chunk):
            calls.append(chunk)
            time.sleep(0.005)
            if chunk[0] == 2:
                raise RuntimeError("boom")
            return True, 1, ""

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_batches(list(range(100)), process, PoolConfig(batch_size=1, workers=4))
        self.assertLess(len(calls), 20, "queued batches must be cancelled, not run after the error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
