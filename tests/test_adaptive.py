"""End-to-end tests through run_adaptive with real threads against a synthetic MockServer."""

import tempfile
import unittest
from pathlib import Path
from statistics import median

from adaptive_concurrency import audit_log, current_overrides, set_override
from tests.support import MockServer, regression_penalty, run, summarise


class LinearCostBatchGrowth(unittest.TestCase):
    def test_grows_batch_then_backs_off_on_explicit_failure(self):
        """The server rejects any batch larger than 30 items."""
        server = MockServer(per_item=0.001, fail=lambda c: "HTTPError: 502 Bad Gateway" if len(c) > 30 else None)
        r = run(list(range(10000)), server, batch_size=5, max_batch_size=100, batch_step=5,
                workers=2, max_workers=2, bisect_on_failure=True)
        summarise("linear+fail>30", r, server)
        self.assertEqual(r["written"], 10000, "bisection must recover every item of a size-failed batch")
        self.assertTrue(any("decrease batch_size" in t for t in r["tunes"]), "a >30 failure must shrink batch")
        self.assertGreaterEqual(median(c.n for c in server.calls[-60:]), 15, "should converge toward a large batch")
        # Not final_batch: once the ceiling expires the controller re-probes past 30, and a run can
        # end mid-probe. Clean runs measure <=2.0% (150 runs); an ignored ceiling exceeds 3% in ~9 runs
        # of 10, so this is a statistical guard. test_controller.SizeFailureCeiling is the exact one.
        over = sum(c.n > 30 for c in server.calls) / len(server.calls)
        self.assertLessEqual(over, 0.03, f"{over:.1%} of calls exceeded the 30-item limit")

    def test_batch_growth_stops_short_of_timeout(self):
        server = MockServer(per_item=0.002)
        r = run(list(range(4000)), server, batch_size=5, max_batch_size=200, batch_step=5,
                workers=2, max_workers=2, max_batch_seconds=0.1)
        summarise("timeout headroom", r, server)
        self.assertLess(max(c.n for c in server.calls), 50, "50 items is the timeout-equivalent size")
        self.assertGreaterEqual(r["final_batch"], 20)


class LatencyOnlyRegression(unittest.TestCase):
    def test_backs_off_on_latency_with_no_error(self):
        """Past the knee latency climbs steeply and nothing ever fails."""
        items = list(range(8000))
        server = MockServer(per_item=0.002, penalty=regression_penalty)
        r = run(items, server, batch_size=10, workers=3, max_workers=16)
        summarise("adaptive", r, server)

        control = MockServer(per_item=0.002, penalty=regression_penalty)
        rc = run(items, control, batch_size=10, workers=3, max_workers=16,
                 congestion_ratio=float("inf"), worker_probe_min_ratio=0.0, batch_probe_min_ratio=0.0)
        summarise("error-only control", rc, control)

        self.assertTrue(all(c.ok for c in server.calls), "scenario must contain no explicit errors")
        late = [c.active for c in server.calls[len(server.calls) // 2:]]
        self.assertLessEqual(sum(a >= 8 for a in late) / len(late), 0.10, "excursions past the knee stay transient")
        self.assertLessEqual(max(c.active for c in server.calls), 10)
        self.assertEqual(round(median(late)), 6, "measured optimum of this mock at fixed W is 6")
        self.assertGreaterEqual(max(c.active for c in control.calls), 12, "control proves errors alone don't stop it")
        self.assertLess(r["wall"], rc["wall"])


class ThreadSafetyCeiling(unittest.TestCase):
    def test_max_workers_1_is_absolute_and_runs_on_calling_thread(self):
        server = MockServer(per_item=0.0005, not_thread_safe=True)
        r = run(list(range(3000)), server, batch_size=10, max_batch_size=60, batch_step=5,
                workers=1, max_workers=1)
        summarise("max_workers=1", r, server)
        self.assertEqual(server.violations, 0)
        self.assertEqual(max(c.active for c in server.calls), 1)
        self.assertTrue(all(c.main_thread for c in server.calls), "runs on the calling thread")
        self.assertGreater(r["final_batch"], 10, "batch dimension still tunes")

    def test_starting_workers_above_ceiling_is_clamped(self):
        server = MockServer(per_item=0.0005, not_thread_safe=True)
        run(list(range(500)), server, batch_size=10, workers=8, max_workers=1)
        self.assertEqual(server.violations, 0)
        self.assertEqual(max(c.active for c in server.calls), 1)

    def test_undeclared_ceiling_defaults_to_starting_workers(self):
        server = MockServer(per_item=0.001)
        r = run(list(range(4000)), server, batch_size=10, workers=4)
        summarise("max_workers=None", r, server)
        self.assertLessEqual(max(c.active for c in server.calls), 4)

    def test_live_override_cannot_make_a_non_thread_safe_run_concurrent(self):
        db = Path(tempfile.mkdtemp()) / "control.sqlite"
        set_override(db, "max_workers", "8")
        set_override(db, "force_workers", "8")
        server = MockServer(per_item=0.0005, not_thread_safe=True)
        r = run(list(range(500)), server, batch_size=10, workers=1, max_workers=1,
                control_db=db, control_poll_seconds=0.0)
        self.assertEqual(server.violations, 0)
        self.assertEqual(max(c.active for c in server.calls), 1)
        self.assertTrue(any("not thread-safe" in line for line in r["log"]))


class LiveOverrideReachesTheServer(unittest.TestCase):
    def test_raised_max_workers_is_real_server_concurrency(self):
        """The override is written mid-run from inside a batch. What counts is concurrency the
        server observed, not what the controller believes it is running."""
        db = Path(tempfile.mkdtemp()) / "control.sqlite"
        server = MockServer(per_item=0.001, sigma=0.1, seed=3)
        state = {"n": 0}

        def process(chunk):
            state["n"] += 1
            if state["n"] == 20:
                set_override(db, "max_workers", "8", note="raise ceiling mid-run")
            return server(chunk)

        r = run(list(range(3000)), process, batch_size=5, workers=2, max_workers=2,
                control_db=db, control_poll_seconds=0.0)
        summarise("live override", r, server)
        self.assertTrue(any("control override: max_workers 2->8" in line for line in r["log"]))
        self.assertGreater(r["final_workers"], 2, "controller must have probed above the old ceiling")
        after = [c.active for c in server.calls[40:]]
        self.assertGreater(max(after), 2, "server never saw more than the starting ceiling")
        self.assertEqual(r["written"], 3000)
        self.assertEqual(current_overrides(db), {"max_workers": "8"})
        self.assertEqual([row[2:5] for row in audit_log(db)], [("max_workers", "8", "raise ceiling mid-run")])


class BisectionThroughAdaptiveDispatch(unittest.TestCase):
    def test_isolates_bad_items_and_does_not_shrink_batch(self):
        bad = {17, 93, 150, 151}
        server = MockServer(per_item=0.001,
                            fail=lambda c: next((f"KeyError: bad item {x}" for x in c if x in bad), None))
        r = run(list(range(400)), server, batch_size=20, max_batch_size=40, batch_step=5,
                workers=2, max_workers=2, bisect_on_failure=True)
        summarise("bad items", r, server)
        leaf_failed = {c.items[0] for c in server.calls if not c.ok and c.n == 1}
        self.assertEqual(r["written"], 400 - len(bad))
        self.assertEqual(leaf_failed, bad)
        for x in bad:
            self.assertIn(f"bad item {x}", r["csv"])
        self.assertFalse(any("decrease batch_size" in t for t in r["tunes"]),
                         "an item fault is bisection's job, not a reason to shrink")


class ExogenousSlowdown(unittest.TestCase):
    def test_costlier_items_do_not_collapse_workers(self):
        """Items tripling in cost mid-run looks like congestion but cutting workers can't fix it."""
        server = MockServer(per_item=lambda c: 0.002 if c[0] < 3000 else 0.006)
        r = run(list(range(6000)), server, batch_size=10, workers=6, max_workers=6)
        summarise("exogenous 3x", r, server)
        self.assertTrue(any("rebaseline" in t for t in r["tunes"]))
        cuts = sum("decrease workers" in t for t in r["tunes"])
        self.assertLessEqual(cuts, 1, "one cut, then re-learn, not a spiral")
        self.assertGreaterEqual(median(c.active for c in server.calls[-60:]), 5)


class FakeLimiter:
    def __init__(self):
        self.backing_off = False

    def acquire(self):
        pass

    def is_backing_off(self):
        return self.backing_off


class BreachIsNotASizeSignal(unittest.TestCase):
    def _run(self, report_breach):
        limiter = FakeLimiter()
        state = {"first": True}

        def process(chunk):
            limiter.backing_off = False
            if state["first"]:
                state["first"] = False
                limiter.backing_off = report_breach
                return False, 0, "HTTPError: 429 Too Many Requests"
            return True, len(chunk), ""

        return run(list(range(200)), process, batch_size=20, workers=1, max_workers=1,
                   bisect_on_failure=True, rate_limiter=limiter)

    def test_failure_under_breach_leaves_batch_alone(self):
        r = self._run(report_breach=True)
        self.assertEqual(r["written"], 200)
        self.assertFalse(any("decrease batch_size" in t for t in r["tunes"]))

    def test_non_vacuity_same_failure_without_breach_shrinks(self):
        r = self._run(report_breach=False)
        self.assertTrue(any("decrease batch_size" in t for t in r["tunes"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
