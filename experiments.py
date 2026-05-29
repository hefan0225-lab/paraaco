"""Generates all experimental data for the paper."""
import json, time, pickle, logging
import numpy as np
from aco import ACOConfig, TSPProblem, DistributedMMAS, tour_length

logging.disable(logging.CRITICAL)
AREA = 1000.0 * 1000.0
BHH = 0.7124
def bhh(n): return BHH * np.sqrt(n * AREA)

def make(n, seed):
    rng = np.random.default_rng(seed)
    return TSPProblem(rng.uniform(0, 1000, (n, 2)), nn_size=16)

def run(problem, **kw):
    cfg = ACOConfig(**kw)
    t = time.perf_counter()
    with DistributedMMAS(problem, cfg) as e:
        r = e.solve()
    return r, time.perf_counter() - t

SEEDS = [0, 1, 2]
out = {}

print("E1 quality scaling")
e1 = []
for n in [50, 100, 200, 300]:
    nn, best, gaps, times = [], [], [], []
    for s in SEEDS:
        p = make(n, s)
        nn_len = tour_length(p.nearest_neighbour_tour(), p.dist)
        r, dt = run(p, n_ants=40, n_workers=4, max_iterations=100, use_2opt=True, seed=s)
        nn.append(nn_len); best.append(r.best_length)
        gaps.append(100*(r.best_length - bhh(n))/bhh(n)); times.append(dt)
    e1.append(dict(n=n, bhh=bhh(n), nn_mean=float(np.mean(nn)),
        best_mean=float(np.mean(best)), best_std=float(np.std(best)),
        gap_bhh_mean=float(np.mean(gaps)),
        improve_vs_nn=float(100*(np.mean(nn)-np.mean(best))/np.mean(nn)),
        time_mean=float(np.mean(times))))
    print(f"  n={n}: best={np.mean(best):.1f}+/-{np.std(best):.1f}")
out["quality"] = e1

print("E2 2-opt ablation (n=200)")
e2 = {}
for use in (False, True):
    best, times = [], []
    for s in SEEDS:
        p = make(200, s)
        r, dt = run(p, n_ants=40, n_workers=4, max_iterations=100, use_2opt=use, seed=s)
        best.append(r.best_length); times.append(dt)
    e2["with_2opt" if use else "no_2opt"] = dict(
        best_mean=float(np.mean(best)), best_std=float(np.std(best)),
        gap_bhh=float(100*(np.mean(best)-bhh(200))/bhh(200)),
        time_mean=float(np.mean(times)))
out["ablation_2opt"] = e2

print("E3 ant-count sensitivity (n=200)")
e3 = []
for na in [16, 32, 64]:
    best = []
    for s in SEEDS:
        p = make(200, s)
        r, _ = run(p, n_ants=na, n_workers=4, max_iterations=100, use_2opt=True, seed=s)
        best.append(r.best_length)
    e3.append(dict(n_ants=na, best_mean=float(np.mean(best)), best_std=float(np.std(best))))
out["ant_sensitivity"] = e3

with open("results.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved results.json")
