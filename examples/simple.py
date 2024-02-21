from gramag import MagGraph, format_table

# Creation of mh may be slow, it does the following:
# 1. Compute all pairwise distances
# 2. Compute all the generators of each MC^{(s, t)}_{k, l}
l_max = 10
# mh = MagGraph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
mh = MagGraph([(0, 1), (0, 2), (0, 3), (1, 5), (2, 5), (3, 4), (4, 5)])
mh.populate_paths(5)

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
print(format_table(mh.rank_generators([(0, 5)])))

for s in range(6):
    for t in range(6):
        for l in range(2, 6):
            hom = mh.stl_homology((s, t), l, True)
            if len(hom.ranks) == 0:
                continue
            print(((s, t), l))
            print(hom.ranks)
            print(hom.representatives)
            print("")
