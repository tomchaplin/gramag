from gramag import MagGraph, format_rank_table

# Create your graph
# A few rules:
# 1. Nodes are labelled by integers
# 2. Edges are provided as a list of tuples of vertices
# 3. Isolated vertices are not supported at the moment
N = 7
mg = MagGraph(
    [(i, (i + 1) % N) for i in range(N)] + [((i + 1) % N, i) for i in range(N)]
)

# Compute generators of all MC^{(s, t)}_{k, l} for l<=6
mg.populate_paths(l_max=11)

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
