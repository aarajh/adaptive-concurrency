"""RateLimiter against a fake clock whose sleep advances it, so nothing here waits in real time."""

import tempfile
import time
import unittest
from pathlib import Path

from adaptive_concurrency import RateLimiter
from tests.support import in_empty_cwd


class FakeClock:
    def __init__(self, start):
        self.now = start
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def local(y, mo, d, h, mi, s=0):
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


def limiter(clock, **kw):
    kw.setdefault("max_per_sec", 1000)
    return RateLimiter(clock=clock, sleep=clock.sleep, **kw)


class PerSecondCap(unittest.TestCase):
    def test_blocks_the_request_over_the_cap_until_the_window_slides(self):
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock, max_per_sec=3)
        for _ in range(3):
            rl.acquire()
        self.assertEqual(clock.sleeps, [])
        rl.acquire()
        self.assertGreater(sum(clock.sleeps), 1.0)
        self.assertEqual(rl.stats()["total_requests_this_process"], 4)


class BytesCap(unittest.TestCase):
    def test_recorded_bytes_block_until_a_minute_passes(self):
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock, max_bytes_per_min=1000)
        rl.acquire()
        rl.record_bytes(1500)
        start = clock.now
        rl.acquire()
        self.assertGreater(clock.now - start, 59.0)


class BreachCooldown(unittest.TestCase):
    def test_escalates_and_caps_at_max_doublings(self):
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock)
        cooldowns = []
        for _ in range(6):
            cooldowns.append(rl.report_breach())
            clock.now += 1
        self.assertEqual(cooldowns, [60, 120, 240, 480, 960, 960])

    def test_resets_after_quiet_window(self):
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock)
        rl.report_breach()
        clock.now += 10
        self.assertEqual(rl.report_breach(), 120)
        clock.now += 301
        self.assertEqual(rl.report_breach(), 60)

    def test_acquire_waits_out_the_cooldown_and_is_backing_off_tracks_it(self):
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock)
        self.assertFalse(rl.is_backing_off())
        start = clock.now
        rl.report_breach()
        self.assertTrue(rl.is_backing_off())
        rl.acquire()
        self.assertGreaterEqual(clock.now - start, 60)
        self.assertFalse(rl.is_backing_off())


class DailyCap(unittest.TestCase):
    def test_blocks_at_cap_and_resets_at_local_midnight(self):
        clock = FakeClock(local(2026, 1, 5, 23, 58))
        rl = limiter(clock, max_per_day=2)
        rl.acquire()
        rl.acquire()
        rl.acquire()
        self.assertGreaterEqual(clock.now, local(2026, 1, 6, 0, 0))
        self.assertEqual(rl.stats()["daily_count"], 1)

    def test_state_file_carries_count_across_restarts_same_day_only(self):
        path = Path(tempfile.mkdtemp()) / "state.json"
        clock = FakeClock(local(2026, 1, 5, 12, 0))
        rl = limiter(clock, max_per_day=10, state_file=path)
        for _ in range(3):
            rl.acquire()
        self.assertEqual(limiter(clock, state_file=path).stats()["daily_count"], 3)
        clock.now = local(2026, 1, 6, 9, 0)
        self.assertEqual(limiter(clock, state_file=path).stats()["daily_count"], 0)

    def test_no_state_file_writes_nothing(self):
        with in_empty_cwd() as tmp:
            clock = FakeClock(local(2026, 1, 5, 12, 0))
            rl = limiter(clock, max_per_day=10)
            rl.acquire()
            self.assertEqual(list(tmp.iterdir()), [])
        self.assertEqual(rl.stats()["daily_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
