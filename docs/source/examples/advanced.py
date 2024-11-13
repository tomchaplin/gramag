from gramag import MagGraph, format_rank_table, DirectSum

# Create graph (tuple of edges, no support for isolated vertices atm)
mg = MagGraph([(0, 1), (0, 2), (0, 3), (1, 6), (2, 6), (3, 4), (4, 5), (5, 6)])
# Compute generators of all MC^{(s, t)}_{k, l} for l<=l_max
l_max = 6
mg.populate_paths(l_max=l_max)

# This is just counting, should be quick
rk_gens = mg.rank_generators()

# For each l, in parallel across each (s, t), computes MH^{(s, t)}_{k, l}
# Adds up the rank for each k, l
rk_hom = mg.rank_homology()

# Pretty print
print("Rank of MC:")
print(format_rank_table(rk_gens))

print("Rank of MH:")
print(format_rank_table(rk_hom))


# Count generators for a given list (s, t)
print("Rank of MC^{(0, 6)}:")
print(format_rank_table(mg.rank_generators([(0, 6)])))

print("l=2 homology ranks:")
l2_hom = mg.l_homology(2)
print(l2_hom.ranks)
print("(k,l)=(2,2) reps:")
print(l2_hom.representatives(2))
print("")

print("l=4 homology ranks:")
l4_hom = mg.l_homology(4)
print(l4_hom.ranks)
print("(k,l)=(2,4) reps:")
print(l4_hom.representatives(2))
print("")


print("Custom direct sum:")
print("We add up all of the l rows and hence we see rank 2 homology in k=2:")
ds = DirectSum(
    [
        mg.stl_homology((s, t), l)
        for s in range(7)
        for t in range(7)
        for l in range(2, 7)
    ]
)
print(ds.ranks)
print(ds.representatives(2))
print("")

print("Fixed l=1, custom (s, t) list - computed in parallel!")
ds2 = mg.l_homology(1, node_pairs=[(s, 6) for s in range(7)])
print(ds2.ranks)
print("This shows that there are 3 edges ending at 6")
# print(ds2.representatives)
print("")

print("Errors:")
try:
    mg.l_homology(12)
except TypeError as e:
    print(e)

try:
    mg.populate_paths(k_max=5, l_max=5)
except TypeError as e:
    print(e)

try:
    mg.l_homology(4).representatives(5)
except TypeError as e:
    print(e)

print("")


print("Stopping condition based on k_max:")
# No limit on l, but k_max up to 3 so can compute homology up to k=2
mg.populate_paths(k_max=3)
print("MC:")
print(format_rank_table(mg.rank_generators()))
print("MH: (change table format)")
print(format_rank_table(mg.rank_homology(), zero="0", unknown="-", above_diagonal="/"))
