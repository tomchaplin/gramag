from gramag import MagGraph, format_rank_table

# Create your graph based on a square distance matrix
# use -1 to denote an infinite distance

# Undirected cycle graph on N nodes
N = 7
distance_matrix = [[min((j - i) % N, (i - j) % N) for j in range(N)] for i in range(N)]

mg = MagGraph.from_distance_matrix(distance_matrix)
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
