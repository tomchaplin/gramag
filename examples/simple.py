from gramag import MagGraph, format_table

# Create graph (tuple of edges, no support for isolated vertices atm)
mh = MagGraph([(0, 1), (0, 2), (0, 3), (1, 5), (2, 5), (3, 4), (4, 5)])
# Compute generators of all MC^{(s, t)}_{k, l} for l<=l_max
l_max = 10
mh.populate_paths(l_max)

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
        # Ignoring l<=1 because they're usually boring
        for l in range(2, 6):
            # Gets Magnitude homology for fixed (s, t), l (and all k<=l)
            # Will return None if l > l_max
            # Final boolean parameter determines whether representatives should be computed
            hom = mh.stl_homology((s, t), l, True)
            if len(hom.ranks) == 0:
                continue
            print(((s, t), l))
            print(hom.ranks)
            print(hom.representatives)
            print("")
