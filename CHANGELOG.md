# Changelog

While the version is 0.x, a minor version may break the API.

## 0.1.1

Tests only; the library is unchanged. The async plateau test averages 8 seeds instead of 4 and its
bound is re-sized from measured runs, so it no longer fails intermittently on Python 3.10.

## 0.1.0

First release: `run_adaptive` and `run_batches`, their async counterparts `arun_adaptive` and
`arun_batches`, `Controller`, `RateLimiter`, live control through a SQLite file, and `UsageLog`.
