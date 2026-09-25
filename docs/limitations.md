# Limitations

Validated only against synthetic servers. The test suite and `benchmarks/sweep.py` drive the tuner
against mock servers with modelled latency curves and noise; it has not been benchmarked against a
real public API.

It converges slowly under realistic noise. Each decision waits for a window large enough to tell a
real change from noise, and at a batch-to-batch latency spread of σ≈0.5 in log terms that is 60
batches per decision. That suits long bulk jobs (roughly 30 minutes or more); on a short job it may
never move far from its starting setting, so start from a sensible guess. Window sizing is
documented on `AdaptiveConfig.target_log_se`.

It is not a 429 circuit breaker. Hard rejections belong to `RateLimiter.report_breach()`, which puts
every worker into one shared cooldown. A failure seen during that cooldown is not treated as
evidence about batch size or concurrency, but with `bisect_on_failure=True` it is still split and
retried, so one 429 costs a full bisection's worth of (throttled) requests.

Thread safety is declared, not assumed. Growth past the starting `workers` needs an explicit
`max_workers`. `max_workers=1` declares the callable not thread-safe: it runs on the calling thread,
and no live override can make it concurrent. The async runners follow the same rule for batches in
flight.

The async runners still do their bookkeeping synchronously on the event loop: the control-file
poll, the usage-log insert and the failures-CSV append are short local SQLite or file writes per
batch, not awaited.
