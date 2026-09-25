"""Latency-aware adaptive concurrency and batch sizing for bulk calls to slow, rate-limited APIs."""

import logging

from .adaptive import AdaptiveConfig, Controller, Observation, Setting, run_adaptive
from .aio import AsyncBatchFn, arun_adaptive, arun_batches
from .control import audit_log, clear_override, current_overrides, set_override
from .pool import BatchResult, PoolConfig, run_batches
from .rate_limiter import AsyncLimiter, Limiter, RateLimiter
from .usage_log import UsageLog

logging.getLogger("adaptive_concurrency").addHandler(logging.NullHandler())

__all__ = [
    "AdaptiveConfig",
    "AsyncBatchFn",
    "AsyncLimiter",
    "BatchResult",
    "Controller",
    "Limiter",
    "Observation",
    "PoolConfig",
    "RateLimiter",
    "Setting",
    "UsageLog",
    "arun_adaptive",
    "arun_batches",
    "audit_log",
    "clear_override",
    "current_overrides",
    "run_adaptive",
    "run_batches",
    "set_override",
]
