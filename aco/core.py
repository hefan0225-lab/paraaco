"""Core algorithm: problem model, MMAS pheromone state, and stateless
solution-construction / local-search kernels that workers execute.

Kernel/state separation: everything a worker calls (construct_tour, two_opt,
tour_length) is a PURE FUNCTION of arrays + an RNG. No global state, no
pheromone mutation. That is what makes the algorithm distributable.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger("aco")
NDArray = np.ndarray


@dataclass(frozen=True)
class ACOConfig:
    # Colony behaviour
    n_ants: int = 32
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.10
    p_best: float = 0.05
    # Search control
    max_iterations: int = 200
    stagnation_limit: int = 50
    gb_frequency: int = 25
    use_2opt: bool = True
    nn_size: int = 16
    # Distribution
    n_workers: int = 4
    seed: int = 42

    def validate(self) -> None:
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must be in (0, 1]")
        if self.n_ants < self.n_workers:
            raise ValueError("n_ants must be >= n_workers")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha/beta must be non-negative")


class TSPProblem:
    """Symmetric Euclidean TSP. Precomputes distance matrix, heuristic matrix
    and per-city nearest-neighbour candidate lists."""

    def __init__(self, coords: NDArray, nn_size: int = 16):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        self.coords = coords
        self.n = coords.shape[0]
        self.dist = self._distance_matrix(coords)
        with np.errstate(divide="ignore"):
            inv = np.where(self.dist > 0.0, 1.0 / self.dist, 0.0)
        np.fill_diagonal(inv, 0.0)
        self.heuristic = inv
        self.nn_size = min(nn_size, self.n - 1)
        self.nn_list = self._nearest_neighbour_lists(self.dist, self.nn_size)

    @staticmethod
    def _distance_matrix(coords: NDArray) -> NDArray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    @staticmethod
    def _nearest_neighbour_lists(dist: NDArray, k: int) -> NDArray:
        return np.argsort(dist, axis=1)[:, 1: k + 1].astype(np.int32)

    def nearest_neighbour_tour(self, start: int = 0) -> NDArray:
        n = self.n
        visited = np.zeros(n, dtype=bool)
        tour = np.empty(n, dtype=np.int32)
        tour[0] = start
        visited[start] = True
        cur = start
        for i in range(1, n):
            d = self.dist[cur].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour[i] = nxt
            visited[nxt] = True
            cur = nxt
        return tour


def tour_length(tour: NDArray, dist: NDArray) -> float:
    return float(dist[tour, np.roll(tour, -1)].sum())


def construct_tour(choice: NDArray, nn_list: NDArray, rng: np.random.Generator) -> NDArray:
    """Build one tour using candidate-list stochastic construction."""
    n = choice.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int32)
    start = int(rng.integers(n))
    tour[0] = start
    visited[start] = True
    cur = start
    for step in range(1, n):
        cand = nn_list[cur]
        cand = cand[~visited[cand]]
        if cand.size == 0:
            cand = np.where(~visited)[0]
        w = choice[cur, cand]
        total = w.sum()
        if total <= 0.0:
            nxt = int(cand[rng.integers(cand.size)])
        else:
            r = rng.random() * total
            idx = int(np.searchsorted(np.cumsum(w), r))
            idx = min(idx, cand.size - 1)
            nxt = int(cand[idx])
        tour[step] = nxt
        visited[nxt] = True
        cur = nxt
    return tour


def _reverse_segment(tour: NDArray, pos: NDArray, a: int, b: int) -> None:
    while a < b:
        tour[a], tour[b] = tour[b], tour[a]
        pos[tour[a]] = a
        pos[tour[b]] = b
        a += 1
        b -= 1


def two_opt(tour: NDArray, dist: NDArray, nn_list: NDArray) -> NDArray:
    """Neighbour-list 2-opt local search to a local optimum."""
    tour = tour.copy()
    n = tour.shape[0]
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            c1 = int(tour[i])
            succ1 = int(tour[(i + 1) % n])
            d_c1_succ1 = dist[c1, succ1]
            for c2 in nn_list[c1]:
                c2 = int(c2)
                d_c1_c2 = dist[c1, c2]
                if d_c1_c2 >= d_c1_succ1:
                    break
                j = int(pos[c2])
                if j <= i:
                    continue
                succ2 = int(tour[(j + 1) % n])
                delta = (d_c1_c2 + dist[succ1, succ2]) - (d_c1_succ1 + dist[c2, succ2])
                if delta < -1e-10:
                    _reverse_segment(tour, pos, i + 1, j)
                    improved = True
                    succ1 = int(tour[(i + 1) % n])
                    d_c1_succ1 = dist[c1, succ1]
    return tour


class MMASPheromone:
    """MAX-MIN Ant System pheromone matrix (master-only mutable state)."""

    def __init__(self, problem: TSPProblem, config: ACOConfig):
        self.problem = problem
        self.cfg = config
        self.n = problem.n
        nn_len = tour_length(problem.nearest_neighbour_tour(), problem.dist)
        self.tau_max = 1.0 / (config.rho * nn_len)
        self.tau_min = self._tau_min(self.tau_max, self.n, config.p_best)
        self.tau = np.full((self.n, self.n), self.tau_max, dtype=np.float64)

    @staticmethod
    def _tau_min(tau_max: float, n: int, p_best: float) -> float:
        avg = n / 2.0
        p = p_best ** (1.0 / n)
        tau_min = tau_max * (1.0 - p) / max((avg - 1.0) * p, 1e-12)
        return min(tau_min, tau_max)

    def update(self, best_tour: NDArray, best_len: float) -> None:
        cfg = self.cfg
        self.tau *= (1.0 - cfg.rho)
        self.tau_max = 1.0 / (cfg.rho * best_len)
        self.tau_min = self._tau_min(self.tau_max, self.n, cfg.p_best)
        deposit = 1.0 / best_len
        a = best_tour
        b = np.roll(best_tour, -1)
        self.tau[a, b] += deposit
        self.tau[b, a] += deposit
        np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

    def restart(self) -> None:
        self.tau.fill(self.tau_max)
