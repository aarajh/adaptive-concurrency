"""Thread-safe rate limiter shared by every worker, sync or async: requests/sec, bytes/min and
requests/day caps, plus a shared circuit breaker for rejections the server actually returned.

Byte accounting needs a hook into your HTTP layer, since only it learns response sizes: call
`record_bytes()` from there. Without it the bytes/min cap never engages.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable


class Limiter(Protocol):
    """What the runners need from a limiter."""

    def acquire(self) -> None: ...

    def is_backing_off(self) -> bool: ...


@runtime_checkable
class AsyncLimiter(Protocol):
    """What the async runners need from a limiter. `RateLimiter` is both."""

    async def aacquire(self) -> None: ...

    def is_backing_off(self) -> bool: ...


class RateLimiterStats(TypedDict):
    req_per_sec_now: int
    bytes_per_min_now: int
    daily_count: int
    total_requests_this_process: int
    backing_off: bool
    backoff_remaining_s: float


class RateLimiter:
    def __init__(
        self,
        max_per_sec: float,
        max_bytes_per_min: float | None = None,
        max_per_day: int | None = None,
        state_file: Path | None = None,
        *,
        breach_base_cooldown: float = 60.0,
        breach_max_doublings: int = 4,
        breach_quiet_window: float = 300.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`state_file` persists the daily count across restarts (local calendar day). None keeps
        it in memory only."""
        self.max_per_sec = max_per_sec
        self.max_bytes_per_min = max_bytes_per_min
        self.max_per_day = max_per_day
        self.state_file = state_file
        self._breach_base_cooldown = breach_base_cooldown
        self._breach_max_doublings = breach_max_doublings
        self._breach_quiet_window = breach_quiet_window
        self._clock = clock
        self._sleep = sleep

        self._lock = threading.Lock()
        self._request_starts: deque[float] = deque()
        self._byte_events: deque[tuple[float, int]] = deque()
        self._total_bytes = 0
        self._total_requests = 0
        self._day, self._daily_count = self._load_state()

        self._backoff_until = 0.0
        self._breach_count = 0
        self._last_breach_time = float("-inf")

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self._clock()))

    def _load_state(self) -> tuple[str, int]:
        today = self._today()
        if self.state_file is not None and self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            if data.get("day") == today:
                return today, int(data.get("count", 0))
        return today, 0

    def _save_state(self) -> None:
        if self.state_file is not None:
            self.state_file.write_text(json.dumps({"day": self._day, "count": self._daily_count}))

    def report_breach(self) -> float:
        """Call when the server actually rejected a request (e.g. HTTP 429). Forces every worker
        into one shared cooldown, doubling on repeat breaches and resetting after a quiet window.
        Returns the cooldown applied, for logging."""
        with self._lock:
            now = self._clock()
            if now - self._last_breach_time > self._breach_quiet_window:
                self._breach_count = 0
            self._breach_count += 1
            self._last_breach_time = now
            cooldown = self._breach_base_cooldown * 2.0 ** min(self._breach_count - 1, self._breach_max_doublings)
            self._backoff_until = max(self._backoff_until, now + cooldown)
            return cooldown

    def is_backing_off(self) -> bool:
        with self._lock:
            return self._clock() < self._backoff_until

    def acquire(self) -> None:
        """Blocks until a new request is allowed under every cap. A breach cooldown overrides
        local accounting unconditionally: it is the server's word, not our estimate."""
        while wait := self._try_acquire():
            self._sleep(wait)

    async def aacquire(self) -> None:
        """`acquire` for the async runners: waits without blocking the event loop."""
        while wait := self._try_acquire():
            await asyncio.sleep(wait)

    def _try_acquire(self) -> float:
        """Takes a request slot and returns 0.0, or returns the seconds to wait before retrying."""
        with self._lock:
            now = self._clock()
            if now < self._backoff_until:
                return self._backoff_until - now
            while self._request_starts and now - self._request_starts[0] > 1.0:
                self._request_starts.popleft()
            while self._byte_events and now - self._byte_events[0][0] > 60.0:
                self._total_bytes -= self._byte_events.popleft()[1]
            today = self._today()
            if today != self._day:
                self._day, self._daily_count = today, 0

            rate_ok = len(self._request_starts) < self.max_per_sec
            bytes_ok = self.max_bytes_per_min is None or self._total_bytes < self.max_bytes_per_min
            day_ok = self.max_per_day is None or self._daily_count < self.max_per_day
            if not (rate_ok and bytes_ok and day_ok):
                return 60.0 if not day_ok else 0.25
            self._request_starts.append(now)
            self._daily_count += 1
            self._total_requests += 1
            self._save_state()
            return 0.0

    def record_bytes(self, n: int) -> None:
        with self._lock:
            self._byte_events.append((self._clock(), n))
            self._total_bytes += n

    def stats(self) -> RateLimiterStats:
        with self._lock:
            now = self._clock()
            return {
                "req_per_sec_now": sum(1 for t in self._request_starts if now - t <= 1.0),
                "bytes_per_min_now": sum(n for t, n in self._byte_events if now - t <= 60.0),
                "daily_count": self._daily_count,
                "total_requests_this_process": self._total_requests,
                "backing_off": now < self._backoff_until,
                "backoff_remaining_s": max(0.0, self._backoff_until - now),
            }
