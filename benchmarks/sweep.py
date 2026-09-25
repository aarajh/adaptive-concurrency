"""Parameter sweep over the Controller alone (no threads, no sleeps). Run from the repo root:
PYTHONPATH=src python3 -m benchmarks.sweep"""

from dataclasses import replace
from statistics import median

from adaptive_concurrency import AdaptiveConfig
from tests.support import regression_penalty, simulate


def sweep() -> None:
    base = AdaptiveConfig(label="sim", batch_size=10, workers=3, max_workers=16)

    def knee(w, b):
        return b * regression_penalty(w)

    print("Sweep 1 -- knee at 6 workers, 2000 rounds x 5 seeds, last 1000 rounds:")
    print("    sigma  worker_min_ratio  at_6   >=8   rel_throughput")
    for sigma in (0.1, 0.3, 0.5, 0.8):
        for wmr in (0.95, 1.0, 1.05, 1.1):
            cfg = replace(base, worker_probe_min_ratio=wmr)
            tail = [w for seed in range(5) for w in simulate(cfg, knee, sigma, 2000, seed)[1000:]]
            tput = sum(w / regression_penalty(w) for w in tail) / len(tail) / 6
            print(f"    {sigma:4.1f}  {wmr:16.2f}  {tail.count(6) / len(tail):4.0%}  "
                  f"{sum(w >= 8 for w in tail) / len(tail):4.0%}  {tput:6.1%}")

    print("\nSweep 2 -- stationary server, workers fixed at 6: false cuts per 100 rounds")
    print("    sigma  " + "  ".join(f"ratio={r}" for r in (1.3, 1.5, 2.0)))

    def flat(w, b):
        return float(b)

    for sigma in (0.1, 0.3, 0.5, 0.8):
        row = []
        for ratio in (1.3, 1.5, 2.0):
            cfg = replace(base, workers=6, max_workers=6, congestion_ratio=ratio)
            cuts = sum(sum(1 for a, b in zip(s, s[1:], strict=False) if b < a)
                       for s in (simulate(cfg, flat, sigma, 1200, seed) for seed in range(5)))
            row.append(f"{cuts / (5 * 1200) * 100:9.1f}")
        print(f"    {sigma:4.1f}  " + "  ".join(row))

    print("\nSweep 3 -- worker_probe_min_ratio on a flat top (knee at 4) and a concave curve (no knee),")
    print("2000 rounds x 5 seeds, last 1000 rounds. Plateau: mean workers. Concave: throughput vs cap.")
    print("    sigma  worker_min_ratio  plateau_workers  concave_throughput")

    def plateau(w, b):
        return b * max(1.0, w / 4)

    def concave(w, b):
        return b * (1 + w / 8)

    for sigma in (0.1, 0.3, 0.5):
        for wmr in (1.0, 1.05, 1.1):
            cfg = replace(base, workers=2, worker_probe_min_ratio=wmr)
            flat_top = [w for seed in range(5) for w in simulate(cfg, plateau, sigma, 2000, seed)[1000:]]
            bowed = [w for seed in range(5) for w in simulate(cfg, concave, sigma, 2000, seed)[1000:]]
            tput = sum(w / (1 + w / 8) for w in bowed) / len(bowed) / (16 / 3)
            print(f"    {sigma:4.1f}  {wmr:16.2f}  {sum(flat_top) / len(flat_top):15.2f}  {tput:18.1%}")

    print("\nSweep 4 -- convergence: batches completed before workers first reach the knee (6) from 3,")
    print("defaults, 20 seeds.")
    print("    sigma  median  max")
    for sigma in (0.1, 0.3, 0.5, 0.8):
        taken = []
        for seed in range(20):
            seen = simulate(base, knee, sigma, 3000, seed)
            taken.append(sum(seen[: seen.index(6)]))
        print(f"    {sigma:4.1f}  {median(taken):6.0f}  {max(taken):4d}")


if __name__ == "__main__":
    sweep()
