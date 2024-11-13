from gramag import MagGraph, format_rank_table
import logging

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

# Create your graph
# A few rules:
# 1. Nodes are labelled by integers
# 2. Edges are provided as a list of tuples of vertices
# 3. Isolated vertices are not supported at the moment
mg = MagGraph([(0, 1), (0, 2), (0, 3), (1, 6), (2, 6), (3, 4), (4, 5), (5, 6)])

# Compute generators of all MC^{(s, t)}_{k, l} for l<=6
mg.populate_paths(l_max=6)

# Reports the ranks of MC^{(s, t)}_{k, l}, summed over (s, t)
rk_gens = mg.rank_generators()

# For each l, in parallel across each (s, t), computes MH^{(s, t)}_{k, l}
# Adds up the rank for each k, l
rk_hom = mg.rank_homology()

# Pretty print
print("Rank of MC:")
print(format_rank_table(rk_gens))

print("Rank of MH:")
print(format_rank_table(rk_hom))

# Compute homology summed over a given list of (s, t)
print("Rank of MH^{(0, 6)}:")
print(format_rank_table(mg.rank_homology(node_pairs=[(0, i) for i in range(7)])))

# Compute homology with representatives, at a given l
homology = mg.l_homology(4, node_pairs=[(0, 6)])
print("Representatives for MH_{2, 4}:")
print(homology.representatives(2))
