"""The async runners end to end, on one event loop, against AsyncMockServer."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from statistics import mean

from adaptive_concurrency import (
    AdaptiveConfig,
    PoolConfig,
    RateLimiter,
    arun_adaptive,
    arun_batches,
    audit_log,
    current_overrides,
    set_override,
)
from tests.support import AsyncMockServer, arun, captured_log, summarise

KNEE = 4


async def always_fail(chunk):
    return False, 0, "HTTPError: 500"


class NoThreads(unittest.IsolatedAsyncioTestCase):
    async def test_arun_batches_runs_workers_batches_concurrently_on_the_loop_thread(self):
        server = AsyncMockServer(per_item=0.002, sigma=0.0)
        threads_before = threading.active_count()
        written = await arun_batches(list(range(400)), server, PoolConfig(batch_size=10, workers=4))
        self.assertEqual(written, 400)
        self.assertEqual(max(c.active for c in server.calls), 4)
        self.assertTrue(all(c.main_thread for c in server.calls))
        self.assertEqual(threading.active_count(), threads_before)

    async def test_arun_adaptive_writes_everything_on_the_loop_thread(self):
        server = AsyncMockServer(per_item=0.001)
        r = await arun(list(range(2000)), server, batch_size=10, workers=2, max_workers=8)
        summarise("async adaptive", r, server)
        self.assertEqual(r["written"], 2000)
        self.assertTrue(all(c.main_thread for c in server.calls))


class AsyncPlateauDoesNotDrift(unittest.IsolatedAsyncioTestCase):
    async def test_server_seen_concurrency_stays_near_the_knee(self):
        """Past a flat-topped knee an extra batch in flight buys nothing. Measured as the concurrency
        the server saw, so it covers the async dispatch, not just the controller."""
        per_seed = []
        for seed in range(4):
            server = AsyncMockServer(per_item=0.001, penalty=lambda a: max(1.0, a / KNEE), sigma=0.1, seed=seed)
            r = await arun(list(range(15000)), server, batch_size=10, workers=2, max_workers=16)
            self.assertEqual(r["written"], 15000)
            per_seed.append(mean(c.active for c in server.calls[len(server.calls) // 2:]))
        print(f"\n  async plateau knee={KNEE} cap=16 sigma=0.1: back-half server concurrency "
              f"mean {mean(per_seed):.2f}, per seed {[round(w, 2) for w in per_seed]}")
        self.assertLessEqual(mean(per_seed), KNEE * 1.4)
        self.assertGreaterEqual(mean(per_seed), KNEE * 0.8)
        for w in per_seed:
            self.assertLessEqual(w, KNEE * 2, "no seed may settle at 2x the knee")


class AsyncLiveOverride(unittest.IsolatedAsyncioTestCase):
    async def test_raised_max_workers_is_real_server_concurrency(self):
        db = Path(tempfile.mkdtemp()) / "control.sqlite"
        server = AsyncMockServer(per_item=0.001, sigma=0.1, seed=3)
        state = {"n": 0}

        async def process(chunk):
            state["n"] += 1
            if state["n"] == 20:
                set_override(db, "max_workers", "8", note="raise ceiling mid-run")
            return await server(chunk)

        r = await arun(list(range(3000)), process, batch_size=5, workers=2, max_workers=2,
                       control_db=db, control_poll_seconds=0.0)
        summarise("async live override", r, server)
        self.assertTrue(any("control override: max_workers 2->8" in line for line in r["log"]))
        self.assertGreater(max(c.active for c in server.calls[40:]), 2, "server never saw more than 2 in flight")
        self.assertEqual(r["written"], 3000)
        self.assertEqual(current_overrides(db), {"max_workers": "8"})
        self.assertEqual([row[2:5] for row in audit_log(db)], [("max_workers", "8", "raise ceiling mid-run")])


class AsyncBisection(unittest.IsolatedAsyncioTestCase):
    async def test_isolates_bad_items_and_does_not_shrink_batch(self):
        bad = {17, 93, 150, 151}
        server = AsyncMockServer(per_item=0.001,
                                 fail=lambda c: next((f"KeyError: bad item {x}" for x in c if x in bad), None))
        r = await arun(list(range(400)), server, batch_size=20, max_batch_size=40, batch_step=5,
                       workers=2, max_workers=2, bisect_on_failure=True)
        self.assertEqual(r["written"], 400 - len(bad))
        self.assertEqual({c.items[0] for c in server.calls if not c.ok and c.n == 1}, bad)
        for x in bad:
            self.assertIn(f"bad item {x}", r["csv"])
        self.assertFalse(any("decrease batch_size" in t for t in r["tunes"]))

    async def test_arun_batches_bisects(self):
        server = AsyncMockServer(per_item=0.0, fail=lambda c: "KeyError: 7" if 7 in c else None)
        cfg = PoolConfig(batch_size=16, workers=2, bisect_on_failure=True)
        self.assertEqual(await arun_batches(list(range(64)), server, cfg), 63)


class AsyncCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    async def test_arun_batches_stops_after_max_consecutive_failures(self):
        calls = []

        async def process(chunk):
            calls.append(chunk)
            return await always_fail(chunk)

        with captured_log() as lines:
            written = await arun_batches(list(range(10)), process, PoolConfig(batch_size=1, max_consecutive_failures=3))
        self.assertEqual((written, len(calls)), (0, 3))
        self.assertTrue(any("SYSTEMATIC FAILURE" in line for line in lines))

    async def test_arun_adaptive_stops_and_drains_what_is_in_flight(self):
        """With 4 in flight, the batches already started when the breaker trips still finish and are
        recorded; nothing new starts."""
        calls = []

        async def process(chunk):
            calls.append(chunk)
            await asyncio.sleep(0.01)
            return False, 0, "HTTPError: 500"

        cfg = AdaptiveConfig(batch_size=1, workers=4, max_workers=4, max_consecutive_failures=3)
        with captured_log() as lines:
            written = await arun_adaptive(list(range(100)), process, cfg)
        self.assertEqual(written, 0)
        self.assertLessEqual(len(calls), 3 + 3, "at most workers-1 batches may still be in flight at the trip")
        self.assertEqual(sum("batch FAILED" in line for line in lines), len(calls), "every started batch is recorded")

    async def test_an_exception_cancels_the_batches_in_flight(self):
        cancelled = []

        async def process(chunk):
            if chunk[0] == 0:
                await asyncio.sleep(0.01)
                raise RuntimeError("boom")
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(chunk[0])
                raise
            return True, 1, ""

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await arun_batches(list(range(4)), process, PoolConfig(batch_size=1, workers=4))
        self.assertEqual(sorted(cancelled), [1, 2, 3])


class AsyncRateLimiter(unittest.IsolatedAsyncioTestCase):
    async def test_a_limiter_without_aacquire_is_rejected_before_any_call(self):
        class SyncOnly:
            def acquire(self):
                pass

            def is_backing_off(self):
                return False

        calls = []

        async def process(chunk):
            calls.append(chunk)
            return True, len(chunk), ""

        with self.assertRaisesRegex(TypeError, "aacquire"):
            await arun_adaptive(list(range(10)), process, AdaptiveConfig(rate_limiter=SyncOnly()))
        self.assertEqual(calls, [])

    async def test_rate_limiter_caps_request_rate_without_blocking_the_loop(self):
        limiter = RateLimiter(max_per_sec=20)
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        async def process(chunk):
            return True, len(chunk), ""

        tick_task = asyncio.ensure_future(ticker())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        cfg = PoolConfig(batch_size=1, workers=4, rate_limiter=limiter)
        written = await arun_batches(list(range(50)), process, cfg)
        elapsed = loop.time() - t0
        tick_task.cancel()
        self.assertEqual(written, 50)
        self.assertGreaterEqual(elapsed, 1.5, "50 requests at 20/s span three one-second windows")
        self.assertGreater(ticks, elapsed / 0.01 * 0.5, "the loop must keep running while the limiter waits")

    async def test_breach_is_not_a_size_signal(self):
        for report_breach in (True, False):
            limiter = RateLimiter(max_per_sec=1000, breach_base_cooldown=0.01)
            state = {"first": True}

            async def process(chunk, limiter=limiter, state=state, report_breach=report_breach):
                if state["first"]:
                    state["first"] = False
                    if report_breach:
                        limiter.report_breach()
                    return False, 0, "HTTPError: 429 Too Many Requests"
                return True, len(chunk), ""

            r = await arun(list(range(200)), process, batch_size=20, workers=1, max_workers=1,
                           bisect_on_failure=True, rate_limiter=limiter)
            self.assertEqual(r["written"], 200)
            shrank = any("decrease batch_size" in t for t in r["tunes"])
            self.assertEqual(shrank, not report_breach, f"report_breach={report_breach}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
