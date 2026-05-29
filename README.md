# ParaACO

A portable, shared-memory **parallel MAX-MIN Ant System (MMAS)** with neighbour-list **2-opt** local search for the symmetric Euclidean **Travelling Salesman Problem (TSP)**.

ParaACO separates a stateless construction kernel from a master-only pheromone manager. The kernel is a pure function of read-only arrays and a random seed, so the same code runs unchanged under the bundled shared-memory multiprocessing engine or, by substituting the executor, on a distributed-memory cluster.

## Installation

```bash
git clone https://github.com/hefan0225-lab/paraaco.git
cd paraaco
pip install -e .
pip install -e ".[plot]"
```

Requires Python >= 3.10 on a POSIX system (Linux/macOS).

## Quick start

```python
import numpy as np
from aco import ACOConfig, TSPProblem, DistributedMMAS, tour_length

rng = np.random.default_rng(0)
problem = TSPProblem(rng.uniform(0, 1000, (200, 2)), nn_size=16)
cfg = ACOConfig(n_ants=40, n_workers=4, max_iterations=100, use_2opt=True, seed=0)

with DistributedMMAS(problem, cfg) as engine:
    result = engine.solve()
print(result.best_length)
```

## License

MIT
