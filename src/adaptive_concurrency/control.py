"""SQLite live-control log for a running adaptive job. The log is both the config and its audit
trail: the effective value of a key is its latest row, and every set or clear is an INSERT, never
an UPDATE or DELETE, so no history is lost.

Keys `Controller.apply_overrides()` understands: max_workers, min_workers, max_batch_size,
min_batch_size (move the ceiling/floor it probes within) and force_workers, force_batch_size (jump
to a level now). Other keys are stored and returned but ignored, so a typo is a no-op, not a crash.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS control_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS ix_control_events_key ON control_events(key, id);
"""

CLEARED = "__cleared__"
"""Tombstone value: a key whose latest row is this is not currently overridden."""

AuditRow = tuple[int, float, str, str, "str | None"]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    return conn


def set_override(db_path: Path, key: str, value: object, note: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO control_events (timestamp, key, value, note) VALUES (?, ?, ?, ?)",
                (time.time(), key, str(value), note),
            )
    finally:
        conn.close()


def clear_override(db_path: Path, key: str, note: str | None = None) -> None:
    set_override(db_path, key, CLEARED, note)


def current_overrides(db_path: Path) -> dict[str, str]:
    """Latest value per key. The (key, id) index keeps this cheap on a long job's history."""
    if not db_path.exists():
        return {}
    conn = _connect(db_path)
    try:
        rows: list[tuple[str, str]] = conn.execute(
            "SELECT key, value FROM control_events c WHERE id = "
            "(SELECT MAX(id) FROM control_events WHERE key = c.key)"
        ).fetchall()
    finally:
        conn.close()
    return {k: v for k, v in rows if v != CLEARED}


def audit_log(db_path: Path) -> list[AuditRow]:
    """Full history, oldest first: every set and every clear."""
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        rows: list[AuditRow] = conn.execute(
            "SELECT id, timestamp, key, value, note FROM control_events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return rows
