"""Distributed master-worker ACO engine.

Master owns MMASPheromone and a POSIX shared-memory segment for the pheromone
matrix. Workers attach zero-copy numpy views; only a seed tuple crosses the
process boundary per iteration. Each worker returns only its best ant.

To scale to distributed memory: replace mp.Pool with Ray/Dask/MPI and replace
the shared-memory broadcast with the framework's broadcast primitive.
"""
from __future__ import annotations
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import List, Optional, Tuple
import numpy as np
from .core import (ACOConfig, MMASPheromone, TSPProblem,
                   construct_tour, tour_length, two_opt)

logger = logging.getLogger("aco")
NDArray = np.ndarray
_AntResult = Tuple[np.ndarray, float]

# Per-process scratch space populated once by the pool initializer.
_W: dict = {}


def _init_worker(shm_name, shape, dtype_str, dist, heuristic, nn_list,
                 alpha, beta, use_2opt):
    shm = shared_memory.SharedMemory(name=shm_name)
    tau_view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    _W.update(shm=shm, tau=tau_view, dist=dist, heuristic=heuristic,
              nn_list=nn_list, alpha=alpha, beta=beta, use_2opt=use_2opt, n=shape[0])


def _worker_batch(args: Tuple[int, int]) -> _AntResult:
    """Construct n_ants tours and return only the best (tour, length)."""
    n_ants, seed = args
    rng = np.random.default_rng(seed)
    tau = _W["tau"]
    choice = np.power(tau, _W["alpha"]) * np.power(_W["heuristic"], _W["beta"])
    nn_list = _W["nn_list"]
    dist = _W["dist"]
    use_2opt = _W["use_2opt"]
    best_tour: Optional[np.ndarray] = None
    best_len = float("inf")
    for _ in range(n_ants):
        t = construct_tour(choice, nn_list, rng)
        if use_2opt:
            t = two_opt(t, dist, nn_list)
        length = tour_length(t, dist)
        if length < best_len:
            best_len, best_tour = length, t
    return best_tour, best_len


@dataclass
class SolveResult:
    best_tour: np.ndarray
    best_length: float
    iterations: int
    history: List[float] = field(default_factory=list)


class DistributedMMAS:
    """Synchronous master-worker MMAS engine with zero-copy pheromone broadcast."""

    def __init__(self, problem: TSPProblem, config: ACOConfig):
        config.validate()
        self.problem = problem
        self.cfg = config
        self.pher = MMASPheromone(problem, config)
        self._shm = shared_memory.SharedMemory(create=True, size=self.pher.tau.nbytes)
        self._shared_tau = np.ndarray(
            self.pher.tau.shape, dtype=self.pher.tau.dtype, buffer=self._shm.buf)
        self._shared_tau[:] = self.pher.tau

    def _publish_pheromone(self) -> None:
        self._shared_tau[:] = self.pher.tau

    def _split_ants(self) -> List[int]:
        nw, na = self.cfg.n_workers, self.cfg.n_ants
        base, extra = divmod(na, nw)
        return [base + (1 if w < extra else 0) for w in range(nw)]

    def solve(self) -> SolveResult:
        cfg = self.cfg
        ants_per_worker = self._split_ants()
        init_args = (self._shm.name, self.pher.tau.shape, self.pher.tau.dtype.str,
                     self.problem.dist, self.problem.heuristic, self.problem.nn_list,
                     cfg.alpha, cfg.beta, cfg.use_2opt)
        gbest_tour: Optional[np.ndarray] = None
        gbest_len = float("inf")
        history: List[float] = []
        stagnation = 0
        ctx = mp.get_context()
        with ctx.Pool(processes=cfg.n_workers, initializer=_init_worker,
                      initargs=init_args) as pool:
            for it in range(cfg.max_iterations):
                self._publish_pheromone()
                tasks = [(n_ants, cfg.seed + it * cfg.n_workers + w)
                         for w, n_ants in enumerate(ants_per_worker) if n_ants > 0]
                results: List[_AntResult] = pool.map(_worker_batch, tasks)
                iter_tour, iter_len = min(results, key=lambda r: r[1])
                if iter_len < gbest_len - 1e-9:
                    gbest_tour, gbest_len = iter_tour.copy(), iter_len
                    stagnation = 0
                else:
                    stagnation += 1
                history.append(gbest_len)
                use_global = (it + 1) % cfg.gb_frequency == 0 and gbest_tour is not None
                if use_global:
                    self.pher.update(gbest_tour, gbest_len)
                else:
                    self.pher.update(iter_tour, iter_len)
                if stagnation >= cfg.stagnation_limit:
                    logger.info("iter %d: stagnation -> restart", it)
                    self.pher.restart()
                    stagnation = 0
                if it % 20 == 0 or it == cfg.max_iterations - 1:
                    logger.info("iter %4d | global best = %.4f", it, gbest_len)
        return SolveResult(best_tour=gbest_tour, best_length=gbest_len,
                           iterations=cfg.max_iterations, history=history)

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "DistributedMMAS":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
