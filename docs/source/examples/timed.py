from pprint import pprint
import numpy as np
from gramag import MagGraph, format_table, DirectSum
import time


def experiment(N=100, l_max=8):
    # Create graph (tuple of edges, no support for isolated vertices atm)
    mg = MagGraph(
        [(i, (i + 1) % N) for i in range(N)] + [((i + 1) % N, i) for i in range(N)]
    )
    tic = time.time()
    mg.populate_paths(l_max=l_max)
    toc = time.time()
    del mg
    return toc - tic


times = np.array([experiment() for _ in range(50)])
median = np.median(times)
std = np.std(times)
print(f"{median:.2f}±{std:.2f}")
pprint(times)

# print(format_table(mg.rank_generators()))
