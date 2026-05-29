"""Smoke tests: solve produces a valid permutation that beats the greedy
baseline, and results are deterministic for a fixed seed."""
import numpy as np
from aco import ACOConfig, TSPProblem, DistributedMMAS, tour_length


def _problem(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return TSPProblem(rng.uniform(0, 1000, (n, 2)), nn_size=16)


def test_valid_and_beats_baseline():
    p = _problem()
    cfg = ACOConfig(n_ants=20, n_workers=4, max_iterations=40, seed=0)
    with DistributedMMAS(p, cfg) as e:
        r = e.solve()
    assert sorted(r.best_tour.tolist()) == list(range(p.n))
    nn = tour_length(p.nearest_neighbour_tour(), p.dist)
    assert r.best_length <= nn + 1e-6


def test_deterministic():
    def run():
        p = _problem()
        cfg = ACOConfig(n_ants=16, n_workers=4, max_iterations=25, seed=7)
        with DistributedMMAS(p, cfg) as e:
            return e.solve().best_length
    assert abs(run() - run()) < 1e-9
