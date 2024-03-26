import joblib
from gramag import MagGraph, format_rank_table
import os

def experiment(N=100, k_max=3):
    from gramag import MagGraph, format_rank_table

    os.environ['RAYON_NUM_THREADS']='2'
    mg = MagGraph([(i, (i+1)%N) for i in range(N)])
    mg.populate_paths(k_max=k_max)
    return mg.rank_homology()

results = joblib.Parallel(backend="threading", n_jobs=4)(
    joblib.delayed(experiment)() for _ in range(100)
)
