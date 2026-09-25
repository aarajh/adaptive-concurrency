"""Controller driven directly, no threads: long runs against a known throughput curve."""

import unittest
from dataclasses import replace
from statistics import mean

from adaptive_concurrency import AdaptiveConfig, Controller, Observation
from tests.support import simulate

KNEE = 4
CAP = 16
SEEDS = range(4)


def plateau(workers, batch_size):
    """Batch latency is flat up to the knee and grows linearly past it: throughput tops out flat."""
    return batch_size * max(1.0, workers / KNEE)


def back_half_workers(cfg, sigma, seed, rounds=2000):
    return mean(simulate(cfg, plateau, sigma, rounds, seed)[rounds // 2:])


class PlateauDoesNotDrift(unittest.TestCase):
    """Past a flat-topped knee an extra worker buys nothing; noise alone must not walk workers up."""

    cfg = AdaptiveConfig(label="sim", batch_size=10, workers=2, max_workers=CAP)

    def test_workers_stay_near_the_knee(self):
        for sigma in (0.1, 0.3):
            per_seed = [back_half_workers(self.cfg, sigma, s) for s in SEEDS]
            print(f"\n  plateau knee={KNEE} cap={CAP} sigma={sigma}: "
                  f"mean workers {mean(per_seed):.2f}, per seed {[round(w, 2) for w in per_seed]}")
            self.assertLessEqual(mean(per_seed), KNEE * 1.4)
            self.assertGreaterEqual(mean(per_seed), KNEE * 0.95)
            for w in per_seed:
                self.assertLessEqual(w, KNEE * 2, "no seed may settle at 2x the knee")

    def test_non_vacuity_ratio_1_drifts(self):
        """At the old threshold of 1.0 the same curve drifts well past the knee."""
        cfg = replace(self.cfg, worker_probe_min_ratio=1.0)
        per_seed = [back_half_workers(cfg, 0.1, s) for s in SEEDS]
        print(f"\n  plateau at worker_probe_min_ratio=1.0, sigma=0.1: mean workers {mean(per_seed):.2f}")
        self.assertGreaterEqual(mean(per_seed), KNEE * 2)


class SizeFailureCeiling(unittest.TestCase):
    """Noise-free: a server rejecting batches over LIMIT. After each rejection the controller must
    climb back to the last size measured good and hold there until the ceiling expires, not probe
    just past it again."""

    LIMIT = 30

    def test_regrows_to_the_last_good_size_between_failures(self):
        ctl = Controller(AdaptiveConfig(batch_size=5, max_batch_size=100, batch_step=5, workers=2, max_workers=2))
        dispatched = []
        for _ in range(600):
            s = ctl.setting
            for _ in range(s.workers):
                fail = s.batch_size > self.LIMIT
                dispatched.append((s.batch_size, fail))
                ctl.observe(Observation(s, s.batch_size, 0.001 * s.batch_size, not fail, fail))
        fails = [i for i, (_, failed) in enumerate(dispatched) if failed]
        gaps = [(a, z) for a, z in zip(fails, fails[1:], strict=False) if z - a > 1]
        held = [max(b for b, failed in dispatched[a:z] if not failed) for a, z in gaps]
        self.assertGreaterEqual(len(held), 3, "the ceiling must expire and be re-probed, repeatedly")
        self.assertEqual(set(held), {self.LIMIT})


if __name__ == "__main__":
    unittest.main()
