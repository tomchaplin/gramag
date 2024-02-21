from gramag import MagGraph, format_table

# Creation of mh may be slow, it does the following:
# 1. Compute all pairwise distances
# 2. Compute all the generators of each MC^{(s, t)}_{k, l}
l_max = 10
mh = MagGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
mh.populate_paths(10)

# This is just counting, should be quick
rk_gens = mh.rank_generators()

# For each l, in parallel across each (s, t), computes MH^{(s, t)}_{k, l}
# Adds up the rank for each k, l
rk_hom = mh.rank_homology()

# Pretty print
print("Rank of MC:")
print(format_table(rk_gens))

print("Rank of MH:")
print(format_table(rk_hom))


# Count generators for a given list (s, t)
print("Rank of MC^{(0, 1)}:")
print(format_table(mh.rank_generators([(0, 1)])))
