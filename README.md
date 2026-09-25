# adaptive-concurrency

[![CI](https://github.com/aarajh/adaptive-concurrency/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aarajh/adaptive-concurrency/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adaptive-concurrency)](https://pypi.org/project/adaptive-concurrency/)
[![Python versions](https://img.shields.io/pypi/pyversions/adaptive-concurrency)](https://pypi.org/project/adaptive-concurrency/)
[![License](https://img.shields.io/pypi/l/adaptive-concurrency)](https://github.com/aarajh/adaptive-concurrency/blob/main/LICENSE)

Latency-aware adaptive concurrency and batch sizing for bulk calls to slow, rate-limited APIs.

It runs your batch callable over a list of items and, while it runs, tunes how many batches are in
flight and how many items go in each, from measured per-item latency and explicit failures. It
keeps a latency history for every (workers, batch size) pair it has tried, keeps an extra worker
only if it beats the best throughput measured at any setting by a margin, and takes live ceiling
changes through an audited SQLite file. Threads or asyncio. Zero runtime dependencies, Python
3.10+, typed.

It converges slowly under noise: from 3 workers to a knee at 6, the benchmark takes a median of 19
batches at a latency spread of σ=0.1 and 126 at σ=0.5 (`benchmarks/sweep.py`, sweep 4). For a short
job, or when you already know the server's limits, use `run_batches` with fixed settings instead.
Known limitations are in
[docs/limitations.md](https://github.com/aarajh/adaptive-concurrency/blob/main/docs/limitations.md).

## Install

```
pip install adaptive-concurrency
```

## Usage

You provide a callable that processes one batch and returns `(ok, items_written, error_message)`.
The runner owns dispatch, timing, tuning, logging, the failures CSV (`failures_csv`) and a circuit
breaker that stops after `max_consecutive_failures` (default 8) failed batches in a row. The caller
owns resumability (pass only items not yet done) and writing results inside the callable, counting
only what was written. Worker threads are reused for the whole run, so a `threading.local()` client
is created once per thread.

Every runner returns the total `items_written` summed over the calls that returned `ok=True`. It
is less than `len(items)` when batches failed or the circuit breaker stopped the run early.

Every example here runs against a local stand-in for a remote API, so you can paste it and run it.
This one gets slower once it has more than 8 calls in flight, and never returns an error:

```python
import logging
import threading
import time

from adaptive_concurrency import AdaptiveConfig, run_adaptive

logging.basicConfig(level=logging.INFO, format="%(message)s")

in_flight = 0
lock = threading.Lock()


def call_api(batch):
    """Stand-in for a remote API: flat latency up to 8 concurrent calls, slower past that."""
    global in_flight
    with lock:
        in_flight += 1
        load = in_flight
    try:
        time.sleep(0.004 * len(batch) * (1.0 if load <= 8 else 1.0 + 0.4 * (load - 8)))
        return True, len(batch), ""
    finally:
        with lock:
            in_flight -= 1


config = AdaptiveConfig(workers=2, max_workers=32, batch_size=1)
written = run_adaptive(list(range(4000)), call_api, config)
print(f"wrote {written}")
```

The log records every probe, revert, and cut. Each line starts with the config's `label` in
brackets (default `batches`), so several jobs can share one log. This is an excerpt from one run:

```
[batches] tune: probe workers 2->3 -> workers=3, batch_size=1
...
[batches] tune: probe workers 10->11 -> workers=11, batch_size=1
[batches] tune: revert workers 11->10 (throughput 0.51x best) -> workers=10, batch_size=1
[batches] tune: decrease workers 10->7 (latency 1.77x this setting's history) -> workers=7, batch_size=1
[batches] tune: probe workers 7->8 -> workers=8, batch_size=1
[batches] tune: probe workers 8->9 -> workers=9, batch_size=1
[batches] tune: revert workers 9->8 (throughput 0.81x best) -> workers=8, batch_size=1
```

It overshoots, cuts, and then holds at 8. It keeps re-testing 9 now and then, because a real
server's capacity changes. The exact path varies from run to run, because the threads race.
`batch_size` stays fixed because `max_batch_size` defaults to the starting `batch_size`.

## Batch size

Many bulk APIs accept a list of items per request, with a fixed overhead per request. Bigger
batches amortise that overhead until they run into a size limit or a timeout. Give the tuner a
ceiling to probe up to and, if requests time out, the timeout:

```python
import time

from adaptive_concurrency import AdaptiveConfig, run_adaptive


def call_api(batch):
    """Stand-in: 20 ms per request plus 1 ms per item, and batches over 60 items are rejected."""
    if len(batch) > 60:
        return False, 0, "HTTP 413: batch too large"
    time.sleep(0.02 + 0.001 * len(batch))
    return True, len(batch), ""


config = AdaptiveConfig(
    workers=4,
    max_workers=4,
    batch_size=5,
    max_batch_size=200,
    max_batch_seconds=30.0,
    bisect_on_failure=True,
)
written = run_adaptive(list(range(6000)), call_api, config)
print(f"wrote {written}")
```

`bisect_on_failure=True` splits a failed batch in half and retries each half, recursively. One bad
item in a batch of 50 then costs about a dozen extra calls to isolate, rather than losing the
other 49. A batch-size cut happens only when the failure followed from the size. A failure traced
to one bad item is left to bisection, and one during a rate-limit cooldown is ignored.

## Failures

- **A failed batch** is one whose call returned `ok=False`. It is logged, written to the failures
  CSV, counted toward the circuit breaker, and its items are dropped unless `bisect_on_failure`
  recovers them. It is not retried as a whole.
- **A raised exception** is not caught. It propagates out of the runner, and nothing new is
  dispatched. The sync runners let batches already running finish and discard their results; the
  async runners cancel them. The written count is lost with the return value, so keep your own
  record of what was written.
- **The circuit breaker** logs `SYSTEMATIC FAILURE` at `ERROR` and returns normally, with the
  written count so far. It raises nothing, so compare the return value with `len(items)`.
- **Bisection re-sends items.** If a call wrote part of its batch before returning `ok=False`, the
  halves send those items again, so writes must be idempotent (an upsert) or all-or-nothing per
  call. `items_written` from a call that returned `ok=False` is never counted, with or without
  bisection. A batch counts as a success if bisection wrote any of it.
- **The failures CSV** has the columns `batch,error,timestamp`: `describe_batch(batch)` (default:
  the items joined by `;`), the error, which joins the errors of every failed part with ` | `, and
  a Unix time. There is one row per batch that failed or partly failed, and the row lists the whole
  batch, including items bisection wrote. It is a log, not a resume file: resume by passing only the
  items your own record says are not yet written.

## Rate limits

`RateLimiter` enforces requests per second, and optionally bytes per minute and requests per
day. Every call goes through it, bisection retries included. When the server rejects you, call
`report_breach()` and every worker waits out one shared cooldown. By default the cooldown starts
at 60 s, doubles on each repeat up to 960 s, and resets after 300 s with no breach.

The runner does not know what a 429 is: a rejected call is a failed batch like any other. With
`bisect_on_failure=True` its halves wait out the cooldown and are retried, and without it the
batch's items are dropped and logged to the failures CSV. Either way the rejection is not taken as
evidence about batch size. This stand-in server allows 30 requests per second, fewer than the
limiter's 50, so it rejects some calls, and the cooldown is shortened so the example is quick:

```python
import threading
import time
from collections import deque

from adaptive_concurrency import AdaptiveConfig, RateLimiter, run_adaptive

limiter = RateLimiter(max_per_sec=50, breach_base_cooldown=0.5, breach_max_doublings=1)
recent = deque()
lock = threading.Lock()
rejected = 0


def call_api(batch):
    """Stand-in: rejects any request past 30 in the last second."""
    global rejected
    with lock:
        now = time.monotonic()
        while recent and now - recent[0] > 1.0:
            recent.popleft()
        recent.append(now)
        over = len(recent) > 30
    time.sleep(0.01)
    if over:
        rejected += 1
        limiter.report_breach()
        return False, 0, "HTTP 429"
    return True, len(batch), ""


config = AdaptiveConfig(workers=4, max_workers=4, batch_size=10, rate_limiter=limiter, bisect_on_failure=True)
written = run_adaptive(list(range(1000)), call_api, config)
print(f"wrote {written} of 1000, {rejected} requests rejected with 429")
```

Pass `state_file=Path(...)` to keep the daily count across restarts (local calendar day). If your
HTTP layer knows response sizes, call `limiter.record_bytes(n)` from it so the bytes-per-minute
cap can engage.

## Live control

Point `control_db` at a SQLite file. The running job checks it on batch completion, at most every
`control_poll_seconds` (default 5 s), and applies any change without restarting:

```python
from pathlib import Path

from adaptive_concurrency import audit_log, clear_override, current_overrides, set_override

db = Path("control.sqlite")  # the same path you passed as AdaptiveConfig(control_db=...)

set_override(db, "max_workers", 16, note="off-peak: allow more")
set_override(db, "force_batch_size", 25)
print(current_overrides(db))  # {'max_workers': '16', 'force_batch_size': '25'}

clear_override(db, "force_batch_size", note="let it tune again")
for row in audit_log(db):
    print(row)  # (id, timestamp, key, value, note), oldest first
```

| Key | Effect |
|---|---|
| `max_workers`, `max_batch_size` | Move the ceiling the tuner probes within. |
| `min_workers`, `min_batch_size` | Move the floor. |
| `force_workers`, `force_batch_size` | Jump to a level now, clamped to the current floor and ceiling. |

Every set and every clear is an `INSERT`. Nothing is updated or deleted. The current value of a key
is its latest row, and a clear writes a tombstone, so the table is both the live config and a
complete history of who changed what, and why. Unknown keys are stored and ignored, so a typo is a
no-op rather than a crash. On a run started with `max_workers=1`, the worker keys are ignored and
the ignore is logged.

## Usage log

`UsageLog(path)` records every call, bisection retries included: the setting it ran under, item
count, elapsed time, and outcome (`ok`, `fail`, or `breach`). Use it to answer "how many requests
did we make today?" against a quota, across restarts:

```python
from pathlib import Path

from adaptive_concurrency import AdaptiveConfig, UsageLog, run_adaptive

usage = UsageLog(Path("usage.sqlite"))
config = AdaptiveConfig(batch_size=10, usage_log=usage)
run_adaptive(list(range(100)), lambda batch: (True, len(batch), ""), config)
print(usage.requests_today(), usage.requests_in_window(3600))
usage.close()
```

## Fixed settings

`run_batches` has the same callable contract with no tuning: fixed `batch_size`, fixed `workers`.
It still gives you bisection, the failures CSV, the rate limiter, the usage log, and the circuit
breaker.

```python
from adaptive_concurrency import PoolConfig, run_batches

written = run_batches(list(range(100)), lambda batch: (True, len(batch), ""), PoolConfig(batch_size=20, workers=4))
print(written)
```

## Sync or async

`run_adaptive` and `run_batches` are thread-based: `workers` is the number of threads calling your
function; `run_batches` with `workers=1` and `run_adaptive` with `max_workers=1` run on the
calling thread. `arun_adaptive` and `arun_batches`
are `asyncio`-native: they take a coroutine with the same contract and the same config, run on the
event loop with no threads, and `workers` is the number of batches in flight. Both pairs share the
same `Controller`, so they tune identically. Use the async runners when your client is async (an
`httpx.AsyncClient`, `aiohttp`), and the sync ones when it blocks. With the async runners a
`rate_limiter` must provide `aacquire()`, and `RateLimiter` does.

```python
import asyncio

from adaptive_concurrency import AdaptiveConfig, PoolConfig, arun_adaptive, arun_batches


async def call_api(batch):
    await asyncio.sleep(0.001 * len(batch))
    return True, len(batch), ""


async def main():
    print(await arun_adaptive(list(range(2000)), call_api, AdaptiveConfig(workers=2, max_workers=16)))
    print(await arun_batches(list(range(100)), call_api, PoolConfig(batch_size=20, workers=4)))


asyncio.run(main())
```

## Configuration

`PoolConfig` configures `run_batches` and `arun_batches`. `AdaptiveConfig` extends it for the
adaptive runners, where `batch_size` and `workers` are the starting point.

| Field | Default | Meaning |
|---|---|---|
| `label` | `"batches"` | Log-line prefix. |
| `batch_size` | `50` | Items per call. |
| `workers` | `1` | Concurrent calls. |
| `max_consecutive_failures` | `8` | Failed batches in a row that stop the run. |
| `rate_limiter` | `None` | A `RateLimiter`, or anything with `acquire()` (`aacquire()` for async) and `is_backing_off()`. |
| `failures_csv` | `None` | Path for the failures CSV; `None` writes nothing. |
| `bisect_on_failure` | `False` | Split and retry a failed batch. |
| `bisect_min_size` | `1` | Smallest part bisection retries. |
| `usage_log` | `None` | A `UsageLog` recording every call. |
| `max_workers` | `None` | Worker ceiling; `None` means the starting `workers`. `1` declares the callable not thread-safe. |
| `min_workers` | `1` | Worker floor. |
| `max_batch_size` | `None` | Batch-size ceiling; `None` means the starting `batch_size`, so batch size is not tuned. |
| `min_batch_size` | `1` | Batch-size floor. |
| `batch_step` | `None` | Additive batch-size increase; `None` means `max(1, batch_size // 4)`. |
| `max_batch_seconds` | `None` | Request timeout; batch growth stops short of it. |
| `control_db` | `None` | SQLite file for live control. |
| `control_poll_seconds` | `5.0` | How often `control_db` is read. |

The remaining `AdaptiveConfig` fields tune the algorithm, and the defaults suit nearly every
workload: `batch_seconds_headroom=0.7`, `congestion_ratio=1.5`, `good_ratio=1.2`,
`worker_probe_min_ratio=1.05`, `batch_probe_min_ratio=0.85`, `worker_decrease=0.7`,
`batch_decrease=0.5`, `min_window=5`, `max_window=60`, `target_log_se=0.08`, `baseline_epochs=5`,
`ceiling_ttl_epochs=10`, `history_ttl_windows=20`. Each is documented where it is defined, in
[`adaptive.py`](https://github.com/aarajh/adaptive-concurrency/blob/main/src/adaptive_concurrency/adaptive.py).

## Logging

Everything goes to the standard `logging` logger `"adaptive_concurrency"`. The library prints
nothing and writes no files unless you pass it a path.

## Development

```
python -m unittest discover -s tests
PYTHONPATH=src python -m benchmarks.sweep
```

The sweep drives `Controller` directly across noise levels and thresholds and is the evidence
behind the defaults. The decision rules are documented on `Controller` in
[`adaptive.py`](https://github.com/aarajh/adaptive-concurrency/blob/main/src/adaptive_concurrency/adaptive.py).

## Related work

- [Netflix concurrency-limits](https://github.com/Netflix/concurrency-limits) is a Java library
  that applies TCP congestion control (Vegas, Gradient2, AIMD limits) to the number of concurrent
  requests. It limits concurrency only, and is aimed at services and RPC clients rather than bulk
  batch jobs.
- [Envoy's adaptive concurrency filter](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/adaptive_concurrency_filter)
  adjusts outstanding requests to a cluster by comparing sampled latency against a minRTT. It
  measures the minRTT periodically by pinning concurrency low. It is a proxy filter, and it does
  not size batches.

## License

Apache-2.0. See [LICENSE](https://github.com/aarajh/adaptive-concurrency/blob/main/LICENSE).
