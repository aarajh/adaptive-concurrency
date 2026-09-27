"""Regression tests for three defects found on this package's first contact with a real
rate-limited API (~12k requests), after only ever being validated against synthetic mocks. No
MockServer here: every latency value and outcome below is copied from that run's own durable
request log (a local SQLite `requests` table, 12180 rows) or its verbatim tune-decision lines, not
modelled.

`AdaptiveConfig(min_window=1)` below removes noise-driven window sizing so a fixed number of
identical observations deterministically fills one evaluation window, the same technique
`test_controller.py`'s `SizeFailureCeiling` already uses for a noise-free reproduction.
"""

import unittest

from adaptive_concurrency import AdaptiveConfig, Controller, Observation
from adaptive_concurrency.adaptive import observation
from adaptive_concurrency.pool import Call


class CongestionMisclassifiedAsSizeFault(unittest.TestCase):
    """Real "fail" rows at workers=6, elapsed 7.08s/4.68s/180.09s/180.12s/180.08s, all a slow
    server-side "internal error" with no HTTP 429. Before the fix these never set `breach` (only a
    429 does) and were treated as size faults, so batch_size was cut on every one while workers --
    the variable actually causing them -- was never touched. Measured live: batch_size collapsed
    12->6->4->2->1 while workers held at 5-6, every batch still failing."""

    REAL_SLOW_FAILURES = [7.0803303071297705, 4.681998076848686, 180.0863904231228,
                          180.12247628718615, 180.07563255773857]
    REAL_ERR = "internal error: asynchronous query failed for the requested identifiers"

    @staticmethod
    def slow_is_congestion(elapsed: float, err: str) -> bool:
        return elapsed > 1.0

    def test_slow_failures_cut_workers_not_batch_size(self):
        cfg = AdaptiveConfig(label="t", workers=6, batch_size=15, min_workers=1,
                             congestion_classifier=self.slow_is_congestion)
        ctl = Controller(cfg)
        for elapsed in self.REAL_SLOW_FAILURES:
            setting = ctl.setting
            calls = [Call(setting.batch_size, False, elapsed, False)]
            obs = observation(cfg, setting, calls, self.REAL_ERR)
            self.assertTrue(obs.congestion)
            self.assertFalse(obs.size_fault)
            ctl.observe(obs)
        self.assertEqual(ctl.setting.batch_size, 15, "a congestion failure must never cut batch_size")
        self.assertLess(ctl.setting.workers, 6, "workers must be the variable that moves")

    def test_fast_failures_from_the_same_run_still_classify_as_size_faults(self):
        """Non-vacuity: real "fail" rows at workers=6, elapsed 0.038s/0.033s -- a fast
        content-not-available error, not a slow one -- must still cut batch_size: the classifier
        has to discriminate, not default everything to congestion. Two failures, not more: a third
        would hit the default floor and trigger the pre-existing (unrelated) "cut workers once
        batch_size bottoms out" fallback, which is not what this test is isolating."""
        cfg = AdaptiveConfig(label="t", workers=6, batch_size=15, min_workers=1,
                             congestion_classifier=self.slow_is_congestion)
        ctl = Controller(cfg)
        for elapsed in (0.03782017668709159, 0.033493873197585344):
            setting = ctl.setting
            calls = [Call(setting.batch_size, False, elapsed, False)]
            obs = observation(cfg, setting, calls, "content not available for the requested identifier")
            self.assertFalse(obs.congestion)
            self.assertTrue(obs.size_fault)
            ctl.observe(obs)
        self.assertLess(ctl.setting.batch_size, 15)
        self.assertEqual(ctl.setting.workers, 6, "a real size fault must not touch workers")

    def test_without_a_classifier_the_pre_fix_collapse_still_happens_by_design(self):
        """The fix is opt-in: no `congestion_classifier` means every non-breach failure is still a
        size fault, exactly as before. Three of the five real failures (batch_decrease=0.5 on a
        floor of 1) are enough to walk batch_size 15->7->3->1 with workers still untouched -- this
        documents that choice rather than asserting a defect."""
        cfg = AdaptiveConfig(label="t", workers=6, batch_size=15, min_workers=1, min_batch_size=1)
        ctl = Controller(cfg)
        for elapsed in self.REAL_SLOW_FAILURES[:3]:
            setting = ctl.setting
            calls = [Call(setting.batch_size, False, elapsed, False)]
            ctl.observe(observation(cfg, setting, calls, self.REAL_ERR))
        self.assertEqual(ctl.setting.batch_size, 1)
        self.assertEqual(ctl.setting.workers, 6)


class RebaselineRestoresPreCutWorkers(unittest.TestCase):
    """Real incident: workers cut 3->2 on "latency 1.72x this setting's history", then the very
    next window logged "rebaseline: latency did not fall after cutting workers, not load-driven"
    -- but workers stayed at 2 for the rest of the run (~8h projected vs ~2h) instead of being
    restored to 3. 1.72 and the sqrt(1.5)-derived 1.30 are the real ratios the controller computed
    that night; 0.264s/item is the real median for a (workers=3, batch_size=14) window (62 rows)
    from the same run's request log, used here as the per-item baseline scale."""

    OWN = 0.264

    def test_not_load_driven_restores_workers_to_pre_cut_level(self):
        cfg = AdaptiveConfig(label="t", workers=3, batch_size=15, min_workers=1, max_workers=3, min_window=1)
        ctl = Controller(cfg)
        for _ in range(3):
            ctl.observe(Observation(ctl.setting, 15, self.OWN * 15, True, False))
        self.assertEqual(ctl.setting.workers, 3)

        change = None
        for _ in range(3):
            change = ctl.observe(Observation(ctl.setting, 15, 1.72 * self.OWN * 15, True, False))
        self.assertIn("decrease workers 3->2", change)
        self.assertEqual(ctl.setting.workers, 2)

        change = None
        for _ in range(2):
            change = ctl.observe(Observation(ctl.setting, 15, 1.30 * self.OWN * 15, True, False))
        self.assertEqual(change, "rebaseline: latency did not fall after cutting workers, not load-driven")
        self.assertEqual(ctl.setting.workers, 3, "workers must be restored, not stuck at the reduced level")


class RebaselineDoesNotRestoreWhenLoadDriven(unittest.TestCase):
    """Contrast case: the same run's own (workers=3, batch_size=30) -> (workers=2, batch_size=30)
    transition shows a real, load-driven fall -- median per-item latency dropped from 0.776s (184
    rows) to 0.182s (26 rows), a 0.182/0.776 = 0.235x fall. A cut that genuinely relieves load like
    that must be kept, not undone."""

    OWN = 0.264
    REAL_FALL_RATIO = 0.182 / 0.776

    def test_load_driven_cut_is_kept(self):
        cfg = AdaptiveConfig(label="t", workers=3, batch_size=15, min_workers=1, max_workers=3, min_window=1)
        ctl = Controller(cfg)
        for _ in range(3):
            ctl.observe(Observation(ctl.setting, 15, self.OWN * 15, True, False))

        change = None
        for _ in range(3):
            change = ctl.observe(Observation(ctl.setting, 15, 1.72 * self.OWN * 15, True, False))
        self.assertIn("decrease workers 3->2", change)

        fallen_elapsed = self.REAL_FALL_RATIO * 1.72 * self.OWN * 15
        change = None
        for _ in range(2):
            change = ctl.observe(Observation(ctl.setting, 15, fallen_elapsed, True, False))
        self.assertNotEqual(change, "rebaseline: latency did not fall after cutting workers, not load-driven")
        self.assertEqual(ctl.setting.workers, 2, "a genuine load-driven cut must not be restored")


class BatchFloorIsRespected(unittest.TestCase):
    """Real incident: batch_size collapsed 12->6->4->2->1 while workers held at 5-6, multiplying
    request count against an identical-sized set of items. 38.2s is a real elapsed value from that
    run's own "internal error" failures at (workers=4, batch_size=12)."""

    REAL_SIZE_FAULT_ELAPSED = 38.2
    REAL_ERR = "internal error: asynchronous query failed for the requested identifiers"

    def test_default_floor_stops_the_collapse_to_one(self):
        cfg = AdaptiveConfig(label="t", workers=6, batch_size=12, min_workers=1)
        ctl = Controller(cfg)
        for _ in range(10):
            setting = ctl.setting
            calls = [Call(setting.batch_size, False, self.REAL_SIZE_FAULT_ELAPSED, False)]
            obs = observation(cfg, setting, calls, self.REAL_ERR)
            ctl.observe(obs)
        self.assertGreaterEqual(ctl.setting.batch_size, 3, "batch_size must not collapse below the default floor")

    def test_explicit_min_batch_size_of_one_is_still_honoured(self):
        """Non-vacuity: the floor is a default change, not a hardcoded clamp -- a caller who
        explicitly asks for min_batch_size=1 still gets it."""
        cfg = AdaptiveConfig(label="t", workers=6, batch_size=12, min_workers=1, min_batch_size=1)
        ctl = Controller(cfg)
        for _ in range(10):
            setting = ctl.setting
            calls = [Call(setting.batch_size, False, self.REAL_SIZE_FAULT_ELAPSED, False)]
            obs = observation(cfg, setting, calls, self.REAL_ERR)
            ctl.observe(obs)
        self.assertEqual(ctl.setting.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
