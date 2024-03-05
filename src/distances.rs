use core::hash::Hash;
use dashmap::DashMap;
use petgraph::{
    algo::dijkstra,
    visit::{IntoEdges, IntoNodeIdentifiers, Visitable},
};
use rayon::prelude::*;
use std::{collections::HashMap, ops::Add};

#[derive(Debug)]
pub struct DistanceMatrix<NodeId: Eq + Hash>(DashMap<NodeId, HashMap<NodeId, usize>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Finite(usize),
    Infinite,
}

impl Add for Distance {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Distance::Finite(a), Distance::Finite(b)) => Distance::Finite(a + b),
            _ => Distance::Infinite,
        }
    }
}

impl<NodeId: Eq + Hash> DistanceMatrix<NodeId> {
    pub fn distance(&self, i: &NodeId, j: &NodeId) -> Distance {
        self.0
            .get(i)
            .and_then(|d_from_i| d_from_i.get(j).copied())
            .map(Distance::Finite)
            .unwrap_or(Distance::Infinite)
    }
}

pub fn parallel_all_pairs_distance<G>(g: G) -> DistanceMatrix<G::NodeId>
where
    G: IntoEdges + Visitable + IntoNodeIdentifiers + Sync,
    G::NodeId: Eq + Hash + Sync + Send,
    <G as IntoNodeIdentifiers>::NodeIdentifiers: Send,
{
    DistanceMatrix(
        g.node_identifiers()
            .par_bridge()
            .map(|node_id| (node_id, dijkstra(g, node_id, None, |_edge| 1)))
            .collect(),
    )
}
