"""Fixed-setting batch runner, plus the pieces the adaptive runner shares with it.

Callers own what "already done" means (resumability), the batch contents, and how a successful
batch's result is written. This module owns batching, dispatch (sequential or threaded), progress
logging, the failures CSV, bisection recovery and the systematic-failure circuit breaker.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

from .rate_limiter import Limiter
from .usage_log import UsageLog

T = TypeVar("T")

BatchResult = tuple[bool, int, str]
"""(ok, written_count, error_message). Timing is the runner's job, not the callable's."""

BatchFn = Callable[[list[T]], BatchResult]

logger = logging.getLogger("adaptive_concurrency")


@dataclass
class PoolConfig:
    label: str = "batches"
    batch_size: int = 50
    workers: int = 1
    max_consecutive_failures: int = 8
    rate_limiter: Limiter | None = None
    failures_csv: Path | None = None
    """One row per failed or partially failed batch. None writes nothing."""
    bisect_on_failure: bool = False
    """On failure, split the batch in half and retry each half recursively down to
    bisect_min_size, instead of discarding it. One bad item in a batch of 50 then costs about
    2*log2(50) extra calls to isolate, not the other 49 items."""
    bisect_min_size: int = 1
    usage_log: UsageLog | None = None
    """Optional durable record of every call, bisection retries included."""


def default_describe(chunk: Sequence[object]) -> str:
    return ";".join(map(str, chunk))


class Call(NamedTuple):
    n_items: int
    ok: bool
    elapsed: float
    breach: bool


def metered(
    process_batch: BatchFn[T],
    config: PoolConfig,
    workers: int,
    batch_size: int,
    on_call: Callable[[Call], None] | None = None,
) -> BatchFn[T]:
    """Wraps every real request, bisection retries included: acquire() runs before each, so a
    bisection fired by a 429 cannot retry straight through the limiter's cooldown."""
    limiter = config.rate_limiter

    def wrapped(chunk: list[T]) -> BatchResult:
        if limiter is not None:
            limiter.acquire()
        t0 = time.perf_counter()
        ok, written, err = process_batch(chunk)
        account_call(config, workers, batch_size, on_call, len(chunk), ok, time.perf_counter() - t0)
        return ok, written, err

    return wrapped


def account_call(
    config: PoolConfig,
    workers: int,
    batch_size: int,
    on_call: Callable[[Call], None] | None,
    n_items: int,
    ok: bool,
    elapsed: float,
) -> None:
    limiter = config.rate_limiter
    call = Call(n_items, ok, elapsed, not ok and limiter is not None and limiter.is_backing_off())
    if on_call is not None:
        on_call(call)
    if config.usage_log is not None:
        config.usage_log.log_batch_attempt(
            label=config.label, workers=workers, batch_size=batch_size,
            n_items=call.n_items, elapsed_seconds=call.elapsed,
            outcome="breach" if call.breach else ("ok" if ok else "fail"),
        )


def bisecting(process_batch: BatchFn[T], min_size: int) -> BatchFn[T]:
    # A rate-limit breach is bisected like any failure, so one 429 still costs a full split-and-retry.
    def wrapped(chunk: list[T]) -> BatchResult:
        ok, written, err = process_batch(chunk)
        if ok:
            return True, written, ""
        if len(chunk) <= min_size:
            return False, 0, err
        mid = len(chunk) // 2
        return merge_halves(wrapped(chunk[:mid]), wrapped(chunk[mid:]))

    return wrapped


def merge_halves(first: BatchResult, second: BatchResult) -> BatchResult:
    written = first[1] + second[1]
    return written > 0, written, " | ".join(e for e in (first[2], second[2]) if e)


class FailureWriter(Generic[T]):
    def __init__(self, path: Path | None, describe: Callable[[list[T]], str]) -> None:
        self._path = path
        self._describe = describe
        self._lock = threading.Lock()
        if path is not None and not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(["batch", "error", "timestamp"])

    def write(self, chunk: list[T], err: str) -> None:
        if self._path is None:
            return
        with self._lock, self._path.open("a", newline="") as f:
            csv.writer(f).writerow([self._describe(chunk), err, time.time()])


class RunTally(Generic[T]):
    """Per-batch accounting both runners share: written total, progress log, failures CSV and the
    consecutive-failure circuit breaker."""

    def __init__(self, config: PoolConfig, describe: Callable[[list[T]], str]) -> None:
        self._label = config.label
        self._max_consecutive = config.max_consecutive_failures
        self._failures = FailureWriter(config.failures_csv, describe)
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._consecutive = 0
        self.total_written = 0

    def record(self, chunk: list[T], ok: bool, written: int, err: str, elapsed: float, tag: str = "") -> bool:
        """Returns False once the run should stop on systematic failure."""
        with self._lock:
            if ok:
                self._consecutive = 0
                self.total_written += written
            else:
                self._consecutive += 1
            consecutive, total = self._consecutive, self.total_written
        wall = time.monotonic() - self._start
        where = f" ({tag})" if tag else ""
        if ok:
            logger.info(
                "[%s] %d/%d written, %.1fs%s | run_avg=%.2f/s, total=%d, wall=%.1fmin%s",
                self._label, written, len(chunk), elapsed, where, total / wall if wall else 0.0, total, wall / 60,
                f" | partial failure after bisection: {err}" if err else "",
            )
        else:
            logger.warning("[%s] batch FAILED after %.1fs%s (%s) -- continuing", self._label, elapsed, where, err)
        if err:
            self._failures.write(chunk, err)
        if not ok and consecutive >= self._max_consecutive:
            logger.error("[%s] SYSTEMATIC FAILURE: %d consecutive. Stopping.", self._label, consecutive)
            return False
        return True

    def finish(self) -> int:
        logger.info("[%s] done. %d written this run", self._label, self.total_written)
        return self.total_written


def run_batches(
    items: Sequence[T],
    process_batch: BatchFn[T],
    config: PoolConfig | None = None,
    *,
    describe_batch: Callable[[list[T]], str] = default_describe,
) -> int:
    """Runs `process_batch` over `items` in fixed chunks of `config.batch_size`, sequentially or on
    `config.workers` threads. Returns items written by this call."""
    config = config or PoolConfig()
    tally: RunTally[T] = RunTally(config, describe_batch)
    call = metered(process_batch, config, config.workers, config.batch_size)
    if config.bisect_on_failure:
        call = bisecting(call, config.bisect_min_size)
    batches = plan_batches(items, config)

    def timed(chunk: list[T]) -> tuple[list[T], bool, int, str, float]:
        t0 = time.perf_counter()
        ok, written, err = call(chunk)
        return chunk, ok, written, err, time.perf_counter() - t0

    if config.workers <= 1:
        for chunk in batches:
            if not tally.record(*timed(chunk)):
                break
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            try:
                for fut in as_completed([pool.submit(timed, b) for b in batches]):
                    if not tally.record(*fut.result()):
                        break
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

    return tally.finish()


def plan_batches(items: Sequence[T], config: PoolConfig) -> list[list[T]]:
    batches = [list(items[i : i + config.batch_size]) for i in range(0, len(items), config.batch_size)]
    logger.info(
        "[%s] %d items, %d batches, batch_size=%d, workers=%d",
        config.label, len(items), len(batches), config.batch_size, config.workers,
    )
    return batches
