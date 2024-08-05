use par_dfs::sync::FastNode;
use petgraph::{
    graph::NodeIndex,
    visit::{GraphRef, IntoNodeIdentifiers, NodeCount},
};
use std::hash::Hash;
use std::{fmt::Debug, iter};

use crate::{
    distances::Distance,
    path_search::{PathKey, PathQuery, StoppingCondition},
    phlite_types::PathIndex,
};

#[derive(Clone)]
pub struct GraphPathSearchNode<'a, G: GraphRef>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount,
{
    pub path: PathIndex,
    k: usize,
    l: usize,
    final_vertex: usize,
    path_query: &'a PathQuery<G>,
    n_nodes: usize,
}

impl<'a, G> GraphPathSearchNode<'a, G>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount,
{
    pub fn path_key(&self) -> PathKey<NodeIndex> {
        PathKey {
            s: NodeIndex::from(self.path.initial_vertex(self.n_nodes) as u32),
            t: NodeIndex::from(self.final_vertex as u32),
            k: self.k,
            l: self.l,
        }
    }

    pub fn init(node: NodeIndex, path_query: &'a PathQuery<G>) -> Self {
        let n_nodes = path_query.g.node_count();

        let node_usize = node.index();

        let path_index = PathIndex::from_indices(iter::once(node_usize), n_nodes).unwrap();

        Self {
            path: path_index,
            final_vertex: node_usize,
            l: 0,
            k: 0,
            path_query,
            n_nodes,
        }
    }
}

// Derive Debug, Hash, PartialEq, Eq for these nodes
// These should only depend on self.path
// The other information is just kept around for score keeping

impl<'a, G> Debug for GraphPathSearchNode<'a, G>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

impl<'a, G> Hash for GraphPathSearchNode<'a, G>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state)
    }
}

impl<'a, G> PartialEq for GraphPathSearchNode<'a, G>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.path.eq(&other.path)
    }
}

impl<'a, G> Eq for GraphPathSearchNode<'a, G> where G: GraphRef<NodeId = NodeIndex> + NodeCount {}

impl<'a, G> FastNode for GraphPathSearchNode<'a, G>
where
    G: GraphRef<NodeId = NodeIndex> + NodeCount + IntoNodeIdentifiers,
{
    type Error = anyhow::Error;

    fn add_children<E>(&self, _depth: usize, queue: &mut E) -> Result<(), Self::Error>
    where
        E: par_dfs::sync::ExtendQueue<Self, Self::Error>,
    {
        if let StoppingCondition::KMax(k_max) = self.path_query.stopping_condition {
            if self.k >= k_max {
                // Stop the iteration because we already have a (k,?)-path
                return Ok(());
            }
        }

        let final_node = NodeIndex::from(self.final_vertex as u32);

        let longer_paths = self
            .path_query
            .g
            .node_identifiers()
            .filter_map(|next_node| {
                // Next node must be different
                if next_node == final_node {
                    return None;
                }

                // Next node must be reachable in finite time
                let next_hop_distance = self.path_query.d.distance(&final_node, &next_node);
                let new_l = match next_hop_distance {
                    Distance::Finite(next_hop) => Some(self.l + next_hop),
                    Distance::Infinite => None,
                }?;

                // Total length of new paths must be at most l_max (assuming that is the stopping condition)
                match self.path_query.stopping_condition {
                    StoppingCondition::LMax(l_max) if new_l > l_max => None,
                    _ => Some((next_node, new_l)),
                }
            })
            .map(|(next_node, new_l)| {
                // Convert the paths into Path Search Nodes
                let next_path =
                    self.path
                        .push_unchecked(next_node.index(), self.n_nodes, self.k as u32);
                Ok(GraphPathSearchNode {
                    path: next_path,
                    l: new_l,
                    k: self.k + 1,
                    path_query: &self.path_query,
                    final_vertex: next_node.index(),
                    n_nodes: self.n_nodes,
                })
            });

        queue.add_all(longer_paths);
        Ok(())
    }
}
