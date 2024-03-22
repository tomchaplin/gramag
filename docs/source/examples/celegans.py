#!/usr/bin/env python
# coding: utf-8

# This script illustrates how you can use a custom GrPPHATI filtration
# to compute betti_1 of regular path homology (no persistence)

# Pull in deps

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

from gramag import MagGraph, format_rank_table
import time

print("=== CELEGANS ===")

# Read in neurons

# Available here https://www.wormatlas.org/neuronalwiring.html (Section 2.1 in Excel format)
df = pd.read_csv(Path(__file__).parent / "NeuronConnect.csv")
print("Loaded data")

# Just look at synapses from the sending side

sending_types = ["S", "Sp", "EJ"]
sending_df = df[df["Type"].isin(sending_types)]

# Aggregate #synapses between each neuron pair

n_connections_df = (
    sending_df.groupby(["Neuron 1", "Neuron 2"])["Nbr"].sum().reset_index()
)


# Weighted digraph with weight = 1/(number of synapses)

G = nx.DiGraph()
G.add_edges_from(
    (row["Neuron 1"], row["Neuron 2"])  # , {"weight": 1.0 / row["Nbr"]})
    for _, row in n_connections_df.iterrows()
)

# grpphati-rs required integer-labelled nodes
G = nx.convert_node_labels_to_integers(G, label_attribute="Neuron")
print("Built digraph")

mg = MagGraph(G.edges)
l_max = 4
mg.populate_paths(l_max=l_max)
print(f"Populated paths up to l={l_max}")
rk_hom = mg.rank_homology()
print(format_rank_table(rk_hom))
rk_gen = mg.rank_generators()
print(format_rank_table(rk_gen))

tic = time.time()
for dim in range(l_max+1):
    omega = mg.l_homology(l_max, representatives=True)
    print(f"Got Ω_{dim}")
toc = time.time()
print(toc - tic)

tic = time.time()
ph_groups = mg.path_homology_groups()
for dim in range(l_max+1):
    grp = ph_groups[dim]
    print((dim, len(grp) ))
toc = time.time()
print(toc - tic)
