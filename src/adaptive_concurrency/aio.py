"""Async counterparts of `run_batches` and `run_adaptive`, for callables that are coroutines.

Same contract, same config, same `Controller`, on one event loop with no threads. Concurrency is
the number of batches in flight at once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, TypeVar

from .adaptive import AdaptiveConfig, AdaptiveRun, Setting, _Attempt, observation
from .pool import BatchResult, Call, PoolConfig, RunTally, account_call, default_describe, merge_halves, plan_batches
from .rate_limiter import AsyncLimiter

T = TypeVar("T")
R = TypeVar("R")

AsyncBatchFn = Callable[[list[T]], Awaitable[BatchResult]]


def _limiter(config: PoolConfig) -> AsyncLimiter | None:
    limiter = config.rate_limiter
    if limiter is None:
        return None
    if not isinstance(limiter, AsyncLimiter):
        raise TypeError(f"the async runners need a rate_limiter with aacquire(); got {type(limiter).__name__}")
    return limiter


def _metered(
    process_batch: AsyncBatchFn[T],
    config: PoolConfig,
    limiter: AsyncLimiter | None,
    workers: int,
    batch_size: int,
    on_call: Callable[[Call], None] | None,
) -> AsyncBatchFn[T]:
    async def wrapped(chunk: list[T]) -> BatchResult:
        if limiter is not None:
            await limiter.aacquire()
        t0 = time.perf_counter()
        ok, written, err = await process_batch(chunk)
        account_call(config, workers, batch_size, on_call, len(chunk), ok, time.perf_counter() - t0)
        return ok, written, err

    return wrapped


def _bisecting(process_batch: AsyncBatchFn[T], min_size: int) -> AsyncBatchFn[T]:
    async def wrapped(chunk: list[T]) -> BatchResult:
        ok, written, err = await process_batch(chunk)
        if ok:
            return True, written, ""
        if len(chunk) <= min_size:
            return False, 0, err
        mid = len(chunk) // 2
        return merge_halves(await wrapped(chunk[:mid]), await wrapped(chunk[mid:]))

    return wrapped


def _prepared(
    process_batch: AsyncBatchFn[T],
    config: PoolConfig,
    limiter: AsyncLimiter | None,
    workers: int,
    batch_size: int,
    on_call: Callable[[Call], None] | None = None,
) -> AsyncBatchFn[T]:
    call = _metered(process_batch, config, limiter, workers, batch_size, on_call)
    return _bisecting(call, config.bisect_min_size) if config.bisect_on_failure else call


async def _dispatch(
    start: Callable[[], tuple[list[T], Coroutine[Any, Any, R]] | None],
    limit: Callable[[], int],
    handle: Callable[[list[T], R], bool],
) -> None:
    """Keeps up to limit() batches in flight, re-read on every completion so a raised limit takes
    effect at once. Results are handled here, one at a time. After handle() returns False nothing
    new starts, and batches already in flight finish and are recorded."""
    in_flight: dict[asyncio.Task[R], list[T]] = {}
    stop = False
    try:
        while True:
            while not stop and len(in_flight) < limit() and (job := start()) is not None:
                chunk, coro = job
                in_flight[asyncio.ensure_future(coro)] = chunk
            if not in_flight:
                return
            done, _ = await asyncio.wait(set(in_flight), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not handle(in_flight.pop(task), task.result()):
                    stop = True
    finally:
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*in_flight, return_exceptions=True)


async def arun_batches(
    items: Sequence[T],
    process_batch: AsyncBatchFn[T],
    config: PoolConfig | None = None,
    *,
    describe_batch: Callable[[list[T]], str] = default_describe,
) -> int:
    """`run_batches` for a coroutine `process_batch`: up to `config.workers` batches in flight."""
    config = config or PoolConfig()
    tally: RunTally[T] = RunTally(config, describe_batch)
    call = _prepared(process_batch, config, _limiter(config), config.workers, config.batch_size)
    batches = iter(plan_batches(items, config))

    async def timed(chunk: list[T]) -> tuple[bool, int, str, float]:
        t0 = time.perf_counter()
        ok, written, err = await call(chunk)
        return ok, written, err, time.perf_counter() - t0

    def start() -> tuple[list[T], Coroutine[Any, Any, tuple[bool, int, str, float]]] | None:
        chunk = next(batches, None)
        return None if chunk is None else (chunk, timed(chunk))

    def handle(chunk: list[T], result: tuple[bool, int, str, float]) -> bool:
        return tally.record(chunk, *result)

    await _dispatch(start, lambda: max(1, config.workers), handle)
    return tally.finish()


async def arun_adaptive(
    items: Sequence[T],
    process_batch: AsyncBatchFn[T],
    config: AdaptiveConfig | None = None,
    *,
    describe_batch: Callable[[list[T]], str] = default_describe,
) -> int:
    """`run_adaptive` for a coroutine `process_batch`: workers is the number of batches in flight."""
    config = config or AdaptiveConfig()
    limiter = _limiter(config)
    run: AdaptiveRun[T] = AdaptiveRun(items, config, describe_batch)

    async def attempt(chunk: list[T], setting: Setting) -> _Attempt:
        calls: list[Call] = []
        call = _prepared(process_batch, config, limiter, setting.workers, setting.batch_size, calls.append)
        t0 = time.perf_counter()
        ok, written, err = await call(chunk)
        return observation(config, setting, calls), ok, written, err, time.perf_counter() - t0

    def start() -> tuple[list[T], Coroutine[Any, Any, _Attempt]] | None:
        job = run.take()
        return None if job is None else (job[0], attempt(*job))

    await _dispatch(start, lambda: run.ctl.setting.workers, run.handle)
    return run.finish()
