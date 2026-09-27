# Changelog

While the version is 0.x, a minor version may break the API.

## Unreleased

First contact with a real rate-limited API (~12k requests) surfaced three defects, all backwards
compatible:

- `AdaptiveConfig.congestion_classifier` lets a caller mark a failure as congestion by latency
  (`(elapsed_seconds, error_message) -> bool`) instead of a size fault, so it cuts `workers`
  instead of `batch_size`. `None` (the default) keeps the old behaviour.
- A "not load-driven" rebaseline now restores `workers` to its pre-cut level instead of leaving it
  at the reduced setting for the rest of the run.
- `AdaptiveConfig.min_batch_size` defaults to `max(1, batch_size // 4)` instead of `1`, so a
  batch-failure spiral on a batched API no longer collapses to single-item requests. An explicit
  `min_batch_size` is unaffected.

## 0.1.1

Tests only; the library is unchanged. The async plateau test averages 8 seeds instead of 4 and its
bound is re-sized from measured runs, so it no longer fails intermittently on Python 3.10.

## 0.1.0

First release: `run_adaptive` and `run_batches`, their async counterparts `arun_adaptive` and
`arun_batches`, `Controller`, `RateLimiter`, live control through a SQLite file, and `UsageLog`.
