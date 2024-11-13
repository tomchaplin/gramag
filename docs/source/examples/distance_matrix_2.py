from gramag import MagGraph, format_rank_table

# Create your graph based on a square distance matrix
# use -1 to denote an infinite distance

# We do a digraph made up of two paths 0 -> 1 -> 2 -> 4 and 0 -> 3 -> 4

distance_matrix = [
    [0, 1, 2, 1, 2],
    [-1, 0, 1, -1, 2],
    [-1, -1, 0, -1, 1],
    [-1, -1, -1, 0, 1],
    [-1, -1, -1, -1, 0],
]

mg = MagGraph.from_distance_matrix(distance_matrix)
# Compute generators of all MC^{(s, t)}_{k, l} for l<=6
mg.populate_paths(l_max=3)

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
