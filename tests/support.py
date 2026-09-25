"""Shared synthetic fixtures. No network: every process_batch is a mock that sleeps for a modelled
latency and measures its own concurrency, so assertions are about what the server saw."""

import asyncio
import contextlib
import logging
import math
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from adaptive_concurrency import AdaptiveConfig, Controller, Observation, arun_adaptive, run_adaptive

LOGGER = "adaptive_concurrency"


@dataclass
class Call:
    n: int
    active: int
    main_thread: bool
    ok: bool
    items: tuple


class MockServer:
    """latency = n * per_item(items) * concurrency_penalty(active) * lognormal noise."""

    def __init__(self, per_item=0.002, penalty=lambda active: 1.0, sigma=0.2, fail=None,
                 not_thread_safe=False, seed=0):
        self.per_item = per_item if callable(per_item) else (lambda items, c=per_item: c)
        self.penalty = penalty
        self.sigma = sigma
        self.fail = fail
        self.not_thread_safe = not_thread_safe
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.active = 0
        self.calls: list[Call] = []
        self.violations = 0

    def __call__(self, chunk):
        with self.lock:
            self.active += 1
            active = self.active
            noise = math.exp(self.rng.gauss(0, self.sigma))
        try:
            time.sleep(len(chunk) * self.per_item(chunk) * self.penalty(active) * noise)
            err = ""
            if self.not_thread_safe and active > 1:
                err = "RuntimeError: client is not thread-safe"
                with self.lock:
                    self.violations += 1
            elif self.fail is not None:
                err = self.fail(chunk) or ""
            ok = not err
            with self.lock:
                self.calls.append(Call(len(chunk), active, threading.current_thread() is threading.main_thread(),
                                       ok, tuple(chunk)))
            return (True, len(chunk), "") if ok else (False, 0, err)
        finally:
            with self.lock:
                self.active -= 1


class AsyncMockServer(MockServer):
    """MockServer's latency model on asyncio.sleep: `active` counts coroutines in flight."""

    async def __call__(self, chunk):
        self.active += 1
        active = self.active
        noise = math.exp(self.rng.gauss(0, self.sigma))
        try:
            await asyncio.sleep(len(chunk) * self.per_item(chunk) * self.penalty(active) * noise)
            err = (self.fail(chunk) or "") if self.fail is not None else ""
            ok = not err
            self.calls.append(Call(len(chunk), active, threading.current_thread() is threading.main_thread(),
                                   ok, tuple(chunk)))
            return (True, len(chunk), "") if ok else (False, 0, err)
        finally:
            self.active -= 1


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@contextlib.contextmanager
def captured_log() -> Iterator[list[str]]:
    logger = logging.getLogger(LOGGER)
    handler = _ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@contextlib.contextmanager
def in_empty_cwd() -> Iterator[Path]:
    """For asserting an optional path left unset writes nothing, not even a relative default."""
    tmp = Path(tempfile.mkdtemp())
    old = Path.cwd()
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(old)


def run(items, server, **cfg_kwargs):
    cfg, tmp = _config(cfg_kwargs)
    t0 = time.time()
    with captured_log() as lines:
        written = run_adaptive(items, server, cfg)
    return _result(written, time.time() - t0, lines, tmp)


async def arun(items, server, **cfg_kwargs):
    cfg, tmp = _config(cfg_kwargs)
    t0 = time.time()
    with captured_log() as lines:
        written = await arun_adaptive(items, server, cfg)
    return _result(written, time.time() - t0, lines, tmp)


def _config(cfg_kwargs):
    tmp = Path(tempfile.mkdtemp())
    return AdaptiveConfig(label="t", failures_csv=tmp / "fail.csv", **cfg_kwargs), tmp


def _result(written, wall, lines, tmp):
    tunes = [line.split("tune: ", 1)[1] for line in lines if "tune: " in line]
    final = next(m for line in lines if (m := re.search(r"final workers=(\d+) batch_size=(\d+)", line)))
    return {
        "written": written, "wall": wall, "tunes": tunes, "log": lines, "csv": (tmp / "fail.csv").read_text(),
        "final_workers": int(final.group(1)), "final_batch": int(final.group(2)),
    }


def summarise(name, r, server):
    actives = [c.active for c in server.calls]
    print(f"    [{name}] written={r['written']} wall={r['wall']:.2f}s tunes={len(r['tunes'])} "
          f"final workers={r['final_workers']} batch={r['final_batch']} max_concurrency_seen={max(actives)}")


def regression_penalty(active):
    """Synthetic knee: flat up to 6 concurrent calls, then +40% latency per extra call."""
    return 1.0 if active <= 6 else 1.0 + 0.4 * (active - 6)


def simulate(cfg, mean_latency, sigma, rounds, seed):
    """Drive Controller directly: each round, `workers` batches complete under the current setting."""
    rng = random.Random(seed)
    ctl = Controller(cfg)
    seen = []
    for _ in range(rounds):
        s = ctl.setting
        seen.append(s.workers)
        for _ in range(s.workers):
            lat = mean_latency(s.workers, s.batch_size) * math.exp(rng.gauss(0, sigma))
            ctl.observe(Observation(s, s.batch_size, lat, True, False))
    return seen
