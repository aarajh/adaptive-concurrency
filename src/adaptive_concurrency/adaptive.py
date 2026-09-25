"""Adaptive worker-count / batch-size runner. Same `process_batch` contract as `run_batches`; the
difference is that (workers, batch_size) is re-decided continuously from what the server is doing.

Why latency and not only errors: adding workers past a server's saturation point can multiply batch
latency without producing a single error, and an error-only controller cannot see that. So the
primary signal is per-item latency against per-setting history, and explicit batch failures are a
second, independent signal feeding the same decisions.

Why a work queue: batch size is only tunable if each batch is cut at dispatch time.

What this does NOT do: replace a 429 circuit breaker. `RateLimiter.report_breach()` owns hard
rejections; a failure observed while the limiter is backing off is not a tuning signal.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Generic, TypeVar

from .control import current_overrides
from .pool import BatchFn, Call, PoolConfig, RunTally, bisecting, default_describe, metered

T = TypeVar("T")
R = TypeVar("R")

WORKERS = "workers"
BATCH = "batch_size"
_DIMS = (WORKERS, BATCH)

logger = logging.getLogger("adaptive_concurrency")


@dataclass
class AdaptiveConfig(PoolConfig):
    """Inherited `batch_size` / `workers` are the starting point. Growth past them is opt-in via
    max_workers / max_batch_size."""

    max_workers: int | None = None
    """Hard ceiling, never probed past. None means the starting `workers`: thread-safety must be
    declared, not assumed. 1 declares the callable not thread-safe: it then runs on the calling
    thread, and no live override can raise it."""
    min_workers: int = 1
    max_batch_size: int | None = None
    """None means the starting `batch_size`."""
    min_batch_size: int = 1
    batch_step: int | None = None
    """Additive batch increase; None means max(1, batch_size // 4)."""
    max_batch_seconds: float | None = None
    """Set to the request timeout. Batch growth stops when the predicted slowest batch would
    exceed batch_seconds_headroom of this."""
    control_db: Path | None = None
    """SQLite file (see `control`) polled on batch completion for live tuning without a restart."""
    control_poll_seconds: float = 5.0

    # --- Advanced: algorithm tuning. The defaults are fine for nearly all workloads. ---

    batch_seconds_headroom: float = 0.7
    congestion_ratio: float = 1.5
    good_ratio: float = 1.2
    worker_probe_min_ratio: float = 1.05
    """The marginal throughput gain an extra worker must buy to be kept, and to not be stepped
    back later. More concurrency is pure cost to the server unless it buys throughput. At 1.0,
    past a flat-topped knee every probe is a coin flip that noise wins half the time, so workers
    drift to the ceiling. A policy, not a noise bound: raising it towards 1 + z*target_log_se
    (~1.13) also refuses workers that genuinely help on a concave curve."""
    batch_probe_min_ratio: float = 0.85
    """A batch probe is kept unless throughput is clearly worse: at equal throughput a bigger
    batch still means fewer requests against req/sec and daily quotas."""
    worker_decrease: float = 0.7
    batch_decrease: float = 0.5
    min_window: int = 5
    max_window: int = 60
    target_log_se: float = 0.08
    """Window length is sized so the window's median per-item latency has about this standard
    error in log terms (se(median) ~= 1.25 sigma / sqrt(n)), from the measured batch-to-batch
    spread. At 0.08 a two-window comparison resolves a ~17% throughput step at z~1.6. At a spread
    of sigma~0.5, windows run at max_window."""
    baseline_epochs: int = 5
    ceiling_ttl_epochs: int = 10
    history_ttl_windows: int = 20
    """History for a setting not re-measured in this many windows is dropped. A stale best from
    before a server-wide slowdown otherwise rejects every probe for the rest of the run."""


@dataclass(frozen=True)
class Setting:
    epoch: int
    workers: int
    batch_size: int


@dataclass(frozen=True)
class Observation:
    setting: Setting
    """The setting the batch was *dispatched* under; results from superseded epochs are not
    latency evidence for the current one."""
    n_items: int
    elapsed: float
    """First attempt only -- bisection retries are recovery, not a latency sample."""
    ok: bool
    size_fault: bool
    """First attempt failed and it is attributable to batch size: not a rate-limit breach, and
    bisection did not isolate a failing single item (a bad item is fixed by bisecting, not by
    shrinking)."""


@dataclass(frozen=True)
class _Probe:
    dim: str
    prev_level: int


class Controller:
    """Pure decision logic: no threads, no clock. Called only from the dispatching thread.

    History is kept per (workers, batch_size) setting -- the last `baseline_epochs` window medians
    of per-item latency -- rather than as one rolling baseline. A single rolling baseline absorbs
    every probe that noise lets through, so it ratchets upward and walks past the knee.

    Per evaluation window (successful batches from the current epoch; length from window_size,
    which sizes it from measured noise so a ~17% throughput step is distinguishable):
      - per_item = median(elapsed / n_items); est. throughput = workers / per_item.
      - A probe is kept only if the probed setting's throughput is at least *_probe_min_ratio of
        the best throughput measured at any setting, else it is reverted and its level becomes a
        temporary ceiling. Comparing against the best, not the previous level, stops a slow walk
        down a declining plateau one noisy accept at a time.
      - Unprobed: if latency exceeds congestion_ratio x this setting's own history, cut workers
        multiplicatively. If latency does not fall after the cut, the slowdown is not load-driven
        (costlier items, a slow server): history -- including the now-stale best -- is discarded
        and re-learned instead of cutting on down to the floor.
      - Otherwise the window joins history. A probe accepted on one noisy window is re-judged once
        its setting has pooled history: if one fewer worker measures better, step back. Then, if
        latency is within good_ratio of history, one dimension is probed upward, alternating, so a
        change is attributable to a single knob.
      - History for a setting not re-measured in history_ttl_windows windows is dropped.

    Provenance. Two constants match independent prior art: congestion_ratio=1.5 is Netflix
    concurrency-limits' Gradient2Limit rttTolerance (the limit shrinks once short-term RTT exceeds
    1.5x long-term), and worker_decrease=0.7 is the multiplicative decrease of CUBIC (RFC 9438) and
    BBR (BBR.Beta). batch_decrease=0.5 is Reno's. The rest -- good_ratio, the separate
    *_probe_min_ratio values and comparing probes against the best-ever setting -- have no prior
    art we know of: they are original choices for one client finding a knee in two dimensions,
    which is a different problem from sharing a link fairly across many flows.

    The cuts are asymmetric because the signals are. A batch cut fires on an explicit failure,
    which is loss-like, so it halves as Reno does on loss; a worker cut fires on latency drift,
    which is delay-like and noisier, so it backs off more gently, as delay-based schemes do.
    """

    def __init__(self, cfg: AdaptiveConfig) -> None:
        self.cfg = cfg
        self._max = {
            WORKERS: cfg.max_workers if cfg.max_workers is not None else cfg.workers,
            BATCH: cfg.max_batch_size if cfg.max_batch_size is not None else cfg.batch_size,
        }
        self._min = {WORKERS: cfg.min_workers, BATCH: cfg.min_batch_size}
        self.thread_safe = self._max[WORKERS] > 1
        self._levels = {
            d: max(self._min[d], min(start, self._max[d]))
            for d, start in ((WORKERS, cfg.workers), (BATCH, cfg.batch_size))
        }
        self._batch_step = cfg.batch_step or max(1, cfg.batch_size // 4)
        self._epoch = 0
        self._samples: list[Observation] = []
        self._history: dict[tuple[int, int], deque[float]] = {}
        self._probe: _Probe | None = None
        self._ceilings: dict[str, list[int]] = {}
        self._cut_ref: float | None = None
        self._windows = 0
        self._touched: dict[tuple[int, int], int] = {}
        self._size_cut_epoch = 0
        self._resid: deque[float] = deque(maxlen=200)
        self._next_dim = 0
        self._last_good = dict(self._levels)

    @property
    def setting(self) -> Setting:
        return Setting(self._epoch, self._levels[WORKERS], self._levels[BATCH])

    @property
    def hard_max_workers(self) -> int:
        return self._max[WORKERS]

    @property
    def hard_max_batch_size(self) -> int:
        return self._max[BATCH]

    def observe(self, obs: Observation) -> str | None:
        """Returns a description of any change made, for logging."""
        if not obs.ok:
            # Batches dispatched before the last size cut are still in flight and will fail at the
            # old size; they are not evidence against the new one.
            if obs.size_fault and obs.setting.epoch >= self._size_cut_epoch:
                if self._levels[BATCH] > self._min[BATCH]:
                    change = self._decrease(BATCH, "batch failed", ceiling=self._last_good[BATCH])
                else:
                    change = self._decrease(WORKERS, "batch failed at min batch_size")
                self._size_cut_epoch = self._epoch
                return change
            return None
        if obs.setting.epoch != self._epoch:
            return None
        self._samples.append(obs)
        if len(self._samples) < self.window_size():
            return None
        return self._evaluate()

    def window_size(self) -> int:
        floor = max(self.cfg.min_window, self._levels[WORKERS])
        if len(self._resid) < self.cfg.min_window:
            return floor
        sigma = 1.4826 * median(abs(r) for r in self._resid)
        n = math.ceil((1.2533 * sigma / self.cfg.target_log_se) ** 2)
        return max(floor, min(self.cfg.max_window, n))

    def _throughput(self, key: tuple[int, int]) -> float:
        return key[0] / median(self._history[key])

    def _record(self, key: tuple[int, int], per_item: float) -> None:
        self._history.setdefault(key, deque(maxlen=self.cfg.baseline_epochs)).append(per_item)
        self._touched[key] = self._windows
        for k, at in list(self._touched.items()):
            if self._windows - at > self.cfg.history_ttl_windows:
                del self._touched[k], self._history[k]

    def _rebaseline(self, key: tuple[int, int], per_item: float) -> None:
        self._history.clear()
        self._touched.clear()
        self._record(key, per_item)

    def _evaluate(self) -> str | None:
        per_item = median(s.elapsed / s.n_items for s in self._samples)
        worst = max(s.elapsed for s in self._samples)
        self._resid.extend(math.log(s.elapsed / s.n_items / per_item) for s in self._samples)
        self._samples = []
        self._windows += 1
        key = (self._levels[WORKERS], self._levels[BATCH])
        own = median(self._history[key]) if key in self._history else None
        probe, self._probe = self._probe, None
        cut_ref, self._cut_ref = self._cut_ref, None

        # Load-driven congestion returns to normal after a cut; exogenous stays near the trigger.
        # The geometric midpoint of the two is the split least sensitive to noise either side.
        if cut_ref is not None and per_item >= math.sqrt(self.cfg.congestion_ratio) * cut_ref:
            self._rebaseline(key, per_item)
            return "rebaseline: latency did not fall after cutting workers, not load-driven"
        if probe is not None:
            best = max((self._throughput(k) for k in self._history if k != key), default=None)
            self._record(key, per_item)
            est = self._throughput(key)
            min_ratio = self.cfg.worker_probe_min_ratio if probe.dim == WORKERS else self.cfg.batch_probe_min_ratio
            if best is not None and est < min_ratio * best:
                probed = self._levels[probe.dim]
                self._set(probe.dim, probe.prev_level, ceiling=probed - 1)
                return f"revert {probe.dim} {probed}->{probe.prev_level} (throughput {est / best:.2f}x best)"
        elif own is not None and per_item > self.cfg.congestion_ratio * own:
            ratio = per_item / own
            change = self._decrease(WORKERS, f"latency {ratio:.2f}x this setting's history")
            if change is None:
                self._rebaseline(key, per_item)
                return f"rebaseline: latency {ratio:.2f}x history at min workers"
            self._cut_ref = own
            return change
        else:
            self._record(key, per_item)
            below = (key[0] - 1, key[1])
            if (
                len(self._history[key]) >= 2
                and below in self._history
                and key[0] - 1 >= self._min[WORKERS]
                and self._throughput(key) < self.cfg.worker_probe_min_ratio * self._throughput(below)
            ):
                ratio = self._throughput(key) / self._throughput(below)
                self._set(WORKERS, key[0] - 1, ceiling=key[0] - 1)
                return f"step back workers {key[0]}->{key[0] - 1} (pooled throughput {ratio:.2f}x one fewer)"

        self._last_good = dict(self._levels)
        self._tick_ceilings()
        if own is None or per_item <= self.cfg.good_ratio * own:
            return self._increase(worst)
        return None

    def _limit(self, dim: str) -> int:
        ceiling = self._ceilings.get(dim)
        return min(self._max[dim], ceiling[0]) if ceiling else self._max[dim]

    def _increase(self, worst: float) -> str | None:
        for i in range(len(_DIMS)):
            dim = _DIMS[(self._next_dim + i) % len(_DIMS)]
            prev = self._levels[dim]
            new = min(self._limit(dim), prev + (1 if dim == WORKERS else self._batch_step))
            if new <= prev:
                continue
            if dim == BATCH and self.cfg.max_batch_seconds is not None:
                if worst * new / prev > self.cfg.batch_seconds_headroom * self.cfg.max_batch_seconds:
                    continue
            self._next_dim = (self._next_dim + i + 1) % len(_DIMS)
            self._set(dim, new)
            self._probe = _Probe(dim, prev)
            return f"probe {dim} {prev}->{new}"
        return None

    def _decrease(self, dim: str, reason: str, ceiling: int | None = None) -> str | None:
        prev = self._levels[dim]
        factor = self.cfg.worker_decrease if dim == WORKERS else self.cfg.batch_decrease
        new = max(self._min[dim], min(prev - 1, int(prev * factor)))
        if new >= prev:
            return None
        # A failure's ceiling is the last size measured good, not prev-1: with batch_step > 1,
        # prev-1 is usually still above the real limit, so the next probe would fail again.
        self._set(dim, new, ceiling=ceiling if ceiling is not None and ceiling < prev else prev - 1)
        return f"decrease {dim} {prev}->{new} ({reason})"

    def _set(self, dim: str, level: int, ceiling: int | None = None) -> None:
        self._levels[dim] = level
        self._epoch += 1
        self._samples = []
        self._probe = None
        if ceiling is not None:
            self._ceilings[dim] = [ceiling, self.cfg.ceiling_ttl_epochs]

    def _tick_ceilings(self) -> None:
        for dim in list(self._ceilings):
            self._ceilings[dim][1] -= 1
            if self._ceilings[dim][1] <= 0:
                del self._ceilings[dim]

    def apply_overrides(self, overrides: dict[str, str]) -> list[str]:
        """Applies whichever control keys are present. A force_* jump goes through _set(), like a
        probe or a cut -- fresh epoch, fresh evidence window -- so a manual override is not treated
        as already measured good. On a run declared not thread-safe, worker keys are ignored."""
        changes: list[str] = []
        specs = (
            (WORKERS, "max_workers", "min_workers", "force_workers"),
            (BATCH, "max_batch_size", "min_batch_size", "force_batch_size"),
        )
        for dim, max_key, min_key, force_key in specs:
            if dim == WORKERS and not self.thread_safe:
                ignored = [k for k in (max_key, min_key, force_key) if k in overrides]
                if ignored:
                    changes.append(f"ignored {', '.join(ignored)}: run declared not thread-safe (max_workers=1)")
                continue
            if max_key in overrides:
                new_max = int(overrides[max_key])
                if new_max != self._max[dim]:
                    changes.append(f"{max_key} {self._max[dim]}->{new_max}")
                    self._max[dim] = new_max
            if min_key in overrides:
                new_min = int(overrides[min_key])
                if new_min != self._min[dim]:
                    changes.append(f"{min_key} {self._min[dim]}->{new_min}")
                    self._min[dim] = new_min
            if force_key in overrides:
                forced = max(self._min[dim], min(int(overrides[force_key]), self._max[dim]))
                if forced != self._levels[dim]:
                    changes.append(f"force {dim} {self._levels[dim]}->{forced}")
                    self._set(dim, forced)
        return changes


class _GrowingPool(Generic[R]):
    """Worker threads started on demand, up to the most jobs ever in flight at once, then reused.

    A fixed-size executor would cap real concurrency at its starting size even after a live
    override raised the target, while the controller credited itself with the higher count.
    Threads are reused rather than spawned per batch so thread-local clients survive.
    """

    def __init__(self, fn: Callable[..., R]) -> None:
        self._fn = fn
        self._jobs: queue.SimpleQueue[tuple[int, tuple[object, ...]] | None] = queue.SimpleQueue()
        self._results: queue.SimpleQueue[tuple[int, R | BaseException]] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._next_id = 0
        self.in_flight = 0

    def submit(self, *args: object) -> int:
        job_id = self._next_id
        self._next_id += 1
        self.in_flight += 1
        if self.in_flight > len(self._threads):
            t = threading.Thread(target=self._loop, name=f"adaptive-concurrency-{len(self._threads)}", daemon=True)
            t.start()
            self._threads.append(t)
        self._jobs.put((job_id, args))
        return job_id

    def _loop(self) -> None:
        while (job := self._jobs.get()) is not None:
            job_id, args = job
            try:
                result: R | BaseException = self._fn(*args)
            except BaseException as e:  # re-raised on the dispatching thread by next_result()
                result = e
            self._results.put((job_id, result))

    def next_result(self) -> tuple[int, R]:
        job_id, result = self._results.get()
        self.in_flight -= 1
        if isinstance(result, BaseException):
            raise result
        return job_id, result

    def shutdown(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)
        for t in self._threads:
            t.join()

    @property
    def thread_count(self) -> int:
        return len(self._threads)


_Attempt = tuple[Observation, bool, int, str, float]


def _attempt(process_batch: BatchFn[T], config: AdaptiveConfig, chunk: list[T], setting: Setting) -> _Attempt:
    calls: list[Call] = []
    call = metered(process_batch, config, setting.workers, setting.batch_size, on_call=calls.append)
    if config.bisect_on_failure:
        call = bisecting(call, config.bisect_min_size)
    t_start = time.perf_counter()
    ok, written, err = call(chunk)
    return observation(config, setting, calls), ok, written, err, time.perf_counter() - t_start


def observation(config: AdaptiveConfig, setting: Setting, calls: list[Call]) -> Observation:
    """`calls` are every request one batch made, first attempt first, bisection retries after."""
    first = calls[0]
    item_fault = config.bisect_on_failure and any(not c.ok and c.n_items <= config.bisect_min_size for c in calls)
    return Observation(
        setting, first.n_items, first.elapsed, first.ok,
        size_fault=not first.ok and not first.breach and not item_fault,
    )


class _ControlPoller:
    """One query per control_poll_seconds, and it only acts when something changed."""

    def __init__(self, ctl: Controller, config: AdaptiveConfig) -> None:
        self._ctl = ctl
        self._config = config
        self._last_check = float("-inf")
        self._last: dict[str, str] = {}

    def poll(self) -> None:
        db = self._config.control_db
        now = time.monotonic()
        if db is None or now - self._last_check < self._config.control_poll_seconds:
            return
        self._last_check = now
        overrides = current_overrides(db)
        if overrides == self._last:
            return
        self._last = overrides
        for change in self._ctl.apply_overrides(overrides):
            logger.info("[%s] control override: %s", self._config.label, change)


def run_adaptive(
    items: Sequence[T],
    process_batch: BatchFn[T],
    config: AdaptiveConfig | None = None,
    *,
    describe_batch: Callable[[list[T]], str] = default_describe,
) -> int:
    """`run_batches`' contract, with (workers, batch_size) chosen per dispatch by `Controller`.
    Returns items written by this call."""
    config = config or AdaptiveConfig()
    run: AdaptiveRun[T] = AdaptiveRun(items, config, describe_batch)

    def attempt(chunk: list[T], setting: Setting) -> _Attempt:
        return _attempt(process_batch, config, chunk, setting)

    if not run.ctl.thread_safe:
        while (job := run.take()) is not None:
            if not run.handle(job[0], attempt(*job)):
                break
    else:
        stop = False
        chunks: dict[int, list[T]] = {}
        pool: _GrowingPool[_Attempt] = _GrowingPool(attempt)
        try:
            while True:
                while not stop and pool.in_flight < run.ctl.setting.workers and (job := run.take()) is not None:
                    chunks[pool.submit(*job)] = job[0]
                if pool.in_flight == 0:
                    break
                job_id, result = pool.next_result()
                if not run.handle(chunks.pop(job_id), result):
                    stop = True
        finally:
            pool.shutdown()
    return run.finish()


class AdaptiveRun(Generic[T]):
    """Dispatch-side state of one adaptive run, shared by the sync and async runners. Every method
    runs on the dispatching thread or coroutine, which is what keeps `Controller` single-threaded."""

    def __init__(self, items: Sequence[T], config: AdaptiveConfig, describe: Callable[[list[T]], str]) -> None:
        self.config = config
        self.ctl = Controller(config)
        self._tally: RunTally[T] = RunTally(config, describe)
        self._control = _ControlPoller(self.ctl, config)
        self._work: deque[T] = deque(items)
        s = self.ctl.setting
        logger.info(
            "[%s] %d items, adaptive: start workers=%d batch_size=%d, ceilings workers<=%d batch_size<=%d",
            config.label, len(items), s.workers, s.batch_size, self.ctl.hard_max_workers, self.ctl.hard_max_batch_size,
        )
        self._control.poll()

    def take(self) -> tuple[list[T], Setting] | None:
        """The next batch, cut at the current batch_size, and the setting it is dispatched under."""
        if not self._work:
            return None
        setting = self.ctl.setting
        return [self._work.popleft() for _ in range(min(setting.batch_size, len(self._work)))], setting

    def handle(self, chunk: list[T], result: _Attempt) -> bool:
        """Returns False on systematic failure."""
        obs, ok, written, err, elapsed = result
        self._control.poll()
        tag = f"w={obs.setting.workers} b={obs.setting.batch_size}"
        keep_going = self._tally.record(chunk, ok, written, err, elapsed, tag)
        change = self.ctl.observe(obs)
        if change:
            s = self.ctl.setting
            logger.info(
                "[%s] tune: %s -> workers=%d, batch_size=%d", self.config.label, change, s.workers, s.batch_size,
            )
        return keep_going

    def finish(self) -> int:
        s = self.ctl.setting
        logger.info(
            "[%s] done. %d written this run, %d not dispatched, final workers=%d batch_size=%d",
            self.config.label, self._tally.total_written, len(self._work), s.workers, s.batch_size,
        )
        return self._tally.total_written
