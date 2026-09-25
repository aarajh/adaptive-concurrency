"""SQLite durable log of every call a runner makes, bisection retries included.

The rate limiter's windows are in memory and reset on restart. This answers "how many requests did
we make yesterday" for any past window, and survives restarts. Rollups are plain SQL over the one
`requests` table, not a separately maintained running total.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    label TEXT NOT NULL,
    workers INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    n_items INTEGER NOT NULL,
    elapsed_seconds REAL NOT NULL,
    outcome TEXT NOT NULL,
    bytes INTEGER
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
"""

OUTCOMES = ("ok", "fail", "breach")


def _day_start(ts: float) -> float:
    """Local midnight of the day containing ts: the same day boundary RateLimiter's daily cap uses."""
    t = time.localtime(ts)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


class UsageLog:
    """Safe for concurrent writers in one process (a lock serialises the shared connection) and
    across processes (WAL plus a busy timeout)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log_batch_attempt(
        self,
        *,
        label: str,
        workers: int,
        batch_size: int,
        n_items: int,
        elapsed_seconds: float,
        outcome: str,
        n_bytes: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            self._conn.execute(
                "INSERT INTO requests (timestamp, label, workers, batch_size, n_items, "
                "elapsed_seconds, outcome, bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, label, workers, batch_size, n_items, elapsed_seconds, outcome, n_bytes),
            )
            self._conn.commit()

    def _scalar(self, sql: str, params: tuple[float, ...]) -> int:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row[0])

    def requests_today(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return self._scalar("SELECT COUNT(*) FROM requests WHERE timestamp >= ? AND timestamp <= ?",
                            (_day_start(now), now))

    def requests_in_window(self, seconds: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return self._scalar("SELECT COUNT(*) FROM requests WHERE timestamp >= ? AND timestamp <= ?",
                            (now - seconds, now))

    def bytes_in_window(self, seconds: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return self._scalar("SELECT COALESCE(SUM(bytes), 0) FROM requests "
                            "WHERE timestamp >= ? AND timestamp <= ? AND bytes IS NOT NULL",
                            (now - seconds, now))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
