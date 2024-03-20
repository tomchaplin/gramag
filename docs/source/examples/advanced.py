from gramag import MagGraph, format_table, DirectSum

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
print(format_table(rk_gens))

print("Rank of MH:")
print(format_table(rk_hom))


# Count generators for a given list (s, t)
print("Rank of MC^{(0, 6)}:")
print(format_table(mg.rank_generators([(0, 6)])))

for s in range(6):
    for t in range(6):
        # Ignoring l<=1 because they're usually boring
        for l in range(2, 6):
            # Gets Magnitude homology for fixed (s, t), l (and all k<=l)
            # Will return None if l > l_max
            # Final boolean parameter determines whether representatives should be computed
            hom = mg.stl_homology((s, t), l, representatives=True)
            total_rank = sum([rk for rk in hom.ranks.values()])
            if total_rank == 0:
                continue
            print(f"({(s, t)}, l={l})")
            print(hom.ranks)
            print(hom.representatives)
            print("")


print("l=2")
l2_hom = mg.l_homology(2, representatives=True)
print(l2_hom.ranks)
print(l2_hom.representatives)
print("")

print("l=4")
l2_hom = mg.l_homology(4, representatives=True)
print(l2_hom.ranks)
print(l2_hom.representatives)
print("")


print("Custom direct sum:")
ds = DirectSum(
    [
        mg.stl_homology((s, t), l, representatives=True)
        for s in range(7)
        for t in range(7)
        for l in range(2, 7)
    ]
)
print(ds.ranks)
print(ds.representatives)
print("")

print("Fixed l=1, custom (s, t) list - computed in parallel!")
ds2 = mg.l_homology(1, representatives=True, node_pairs=[(s, 6) for s in range(7)])
print(ds2.ranks)
print(ds2.representatives)
print("")

print("Errors:")
try:
    mg.l_homology(12)
except TypeError as e:
    print(e)

try:
    mg.l_homology(2, representatives=False).representatives
except TypeError as e:
    print(e)

try:
    mg.populate_paths(k_max=5, l_max=5)
except TypeError as e:
    print(e)

# No error because no homology
print(mg.l_homology(5, representatives=False).representatives)
print("")


print("Stopping condition based on k_max:")
# No limit on l, but k_max up to 3 so can compute homology up to k=2
mg.populate_paths(k_max=3)
print("MC:")
print(format_table(mg.rank_generators()))
print("MH:")
print(format_table(mg.rank_homology()))
print("l=4 representatives")
print(mg.l_homology(4, representatives=True).representatives)
