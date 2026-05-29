"""Runnable demo.

    python -m aco.demo --cities 150 --workers 4 --ants 48 --iters 150
"""
from __future__ import annotations
import argparse, logging, time
import numpy as np
from .core import ACOConfig, TSPProblem, tour_length
from .distributed import DistributedMMAS

def make_random_problem(n, seed, nn_size):
    rng = np.random.default_rng(seed)
    return TSPProblem(rng.uniform(0.0, 1000.0, size=(n, 2)), nn_size=nn_size)

def main():
    p = argparse.ArgumentParser(description="Distributed MMAS ant colony TSP solver")
    p.add_argument("--cities",  type=int,   default=150)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--ants",    type=int,   default=48)
    p.add_argument("--iters",   type=int,   default=150)
    p.add_argument("--alpha",   type=float, default=1.0)
    p.add_argument("--beta",    type=float, default=3.0)
    p.add_argument("--rho",     type=float, default=0.10)
    p.add_argument("--no-2opt", action="store_true")
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = ACOConfig(
        n_ants=args.ants, alpha=args.alpha, beta=args.beta, rho=args.rho,
        max_iterations=args.iters, use_2opt=not args.no_2opt,
        n_workers=args.workers, nn_size=16, seed=args.seed,
    )
    problem = make_random_problem(args.cities, args.seed, cfg.nn_size)
    nn_len  = tour_length(problem.nearest_neighbour_tour(), problem.dist)

    print(f"\nTSP: {args.cities} cities | workers={cfg.n_workers} ants={cfg.n_ants} "
          f"iters={cfg.max_iterations} 2opt={cfg.use_2opt}")
    print(f"nearest-neighbour baseline : {nn_len:10.2f}")

    t0 = time.perf_counter()
    with DistributedMMAS(problem, cfg) as engine:
        result = engine.solve()
    elapsed = time.perf_counter() - t0

    gap = 100.0 * (nn_len - result.best_length) / nn_len
    print(f"distributed MMAS best tour : {result.best_length:10.2f}  ({gap:+.2f}% vs NN)")
    print(f"wall time                  : {elapsed:.2f}s")

    tour = result.best_tour
    assert sorted(tour.tolist()) == list(range(args.cities)), "tour is not a permutation"
    print("validation                 : tour is a valid permutation \u2713\n")

if __name__ == "__main__":
    main()
