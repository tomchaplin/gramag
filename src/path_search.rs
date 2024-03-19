use crate::distances::{Distance, DistanceMatrix};
use crate::homology::{build_stl_homology, StlHomology};
use crate::Path;

use core::hash::Hash;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use dashmap::DashMap;
use lophat::algorithms::{DecompositionAlgo, SerialAlgorithm, SerialDecomposition};
use lophat::columns::{Column, VecColumn};
use lophat::options::LoPhatOptions;
use par_dfs::sync::par::IntoParallelIterator;
use par_dfs::sync::{FastBfs, FastNode};
use petgraph::visit::{GraphBase, GraphRef, IntoEdges, IntoNodeIdentifiers, Visitable};
use rayon::prelude::{ParallelBridge, ParallelIterator};

pub trait SensibleNode: Debug + Eq + Hash + Clone + Copy {}
impl<T> SensibleNode for T where T: Debug + Eq + Hash + Clone + Copy {}

// (s, t, k, l)
// s = start vertex
// t = end vertex
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct PathKey<NodeId: SensibleNode> {
    pub s: NodeId, // Start vertex
    pub t: NodeId, // Terminal vertex
    pub k: usize,  // (k+1) = number of vertices
    pub l: usize,  // Length of the path (shortest-path metric)
}

impl<NodeId: SensibleNode> PathKey<NodeId> {
    fn build_from_path(path: &Path<NodeId>, l: usize) -> Self {
        Self {
            s: path.first().copied().expect("Path should be non-empty"),
            t: path.last().copied().expect("Path should be non-empty"),
            k: path.len() - 1,
            l,
        }
    }
}

// TODO: Allow both stopping conditions?

#[derive(Debug, Clone)]
pub enum StoppingCondition {
    KMax(usize),
    LMax(usize),
}

#[derive(Debug, Clone)]
pub struct PathQuery<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    pub g: G,
    pub d: Arc<DistanceMatrix<G::NodeId>>,
    pub stopping_condition: StoppingCondition,
}

impl<G> PathQuery<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    pub fn build(
        g: G,
        d: Arc<DistanceMatrix<G::NodeId>>,
        stopping_condition: StoppingCondition,
    ) -> Self {
        Self {
            g,
            d,
            stopping_condition,
        }
    }
}

impl<G> PathQuery<G>
where
    G: IntoEdges + Visitable + IntoNodeIdentifiers + Sync + Send,
    G::NodeId: SensibleNode + Send + Sync,
    <G as IntoNodeIdentifiers>::NodeIdentifiers: Send,
{
    pub fn run(&self) -> PathContainer<G::NodeId> {
        let k_max = match self.stopping_condition {
            StoppingCondition::KMax(k_max) => k_max,
            StoppingCondition::LMax(l_max) => l_max,
        };
        let l_max = match self.stopping_condition {
            StoppingCondition::KMax(_) => None,
            StoppingCondition::LMax(l_max) => Some(l_max),
        };

        // Setup container for paths and their indexes
        let container = PathContainer::new(self.d.clone(), k_max, l_max);

        // Setup counters for the number of (s, t, k, l) paths encountered
        // This allows us to index such paths as we find them
        let counters: DashMap<_, AtomicUsize> = DashMap::default();

        let store_node = |node: GraphPathSearchNode<G>| {
            let key = PathKey::build_from_path(&node.path, node.l);
            let idx = counters
                .entry(key)
                .or_default()
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            container.store(&key, node.path, idx);
        };

        // Start parallelised BFS
        self.g
            .node_identifiers()
            .map(|start_node| GraphPathSearchNode::init(start_node, self.clone()))
            .par_bridge()
            .for_each(|start_search_node| {
                // Include the start path
                store_node(start_search_node.clone());

                // Start the search
                // TODO: Experiment with Dfs vs Bfs
                FastBfs::<GraphPathSearchNode<G>>::new(start_search_node, None, true)
                    .into_par_iter()
                    .for_each(|path_node| {
                        store_node(path_node.expect("Search node never errors"));
                    })
            });

        container
    }
}

#[derive(Debug)]
pub struct PathContainer<NodeId>
where
    NodeId: SensibleNode,
{
    pub paths: DashMap<PathKey<NodeId>, DashMap<Path<NodeId>, usize>>,
    pub d: Arc<DistanceMatrix<NodeId>>,
    pub k_max: usize,
    pub l_max: Option<usize>,
}

impl<NodeId> PathContainer<NodeId>
where
    NodeId: SensibleNode,
{
    fn new(d: Arc<DistanceMatrix<NodeId>>, k_max: usize, l_max: Option<usize>) -> Self {
        Self {
            paths: DashMap::new(),
            d,
            k_max,
            l_max,
        }
    }

    fn store(&self, key: &PathKey<NodeId>, path: Path<NodeId>, idx: usize) {
        // We use or_default to insert an empty DashMap
        // if we haven't found a path with this key before
        // Note: To prevent race conditions, if key is not in self.paths
        //        then we have to keep it locked until we've inserted the empty DashMap
        self.paths.entry(*key).or_default().insert(path, idx);
    }

    pub fn num_paths(&self, key: &PathKey<NodeId>) -> usize {
        if let Some(sub_paths) = self.paths.get(key) {
            sub_paths.len()
        } else {
            0
        }
    }

    pub fn index_of(&self, key: &PathKey<NodeId>, path: &Path<NodeId>) -> Option<usize> {
        Some(*self.paths.get(key)?.get(path)?.value())
    }

    pub fn path_at_index(&self, key: &PathKey<NodeId>, idx: usize) -> Option<Path<NodeId>> {
        self.paths.get(key)?.iter().find_map(|entry| {
            if *entry.value() == idx {
                Some(entry.key().clone())
            } else {
                None
            }
        })
    }

    pub fn aggregated_rank<I: Iterator<Item = (NodeId, NodeId)>>(
        &self,
        node_identifiers: impl Fn() -> I,
        k: usize,
        l: usize,
    ) -> usize {
        node_identifiers()
            .map(|(s, t)| self.num_paths(&PathKey { s, t, k, l }))
            .sum()
    }

    pub fn rank_matrix<I: Iterator<Item = (NodeId, NodeId)>>(
        &self,
        node_identifiers: impl Fn() -> I + Copy,
    ) -> Vec<Vec<usize>> {
        let k_max = self.k_max;
        let l_max = self.l_max.unwrap_or_else(|| self.max_found_l());
        (0..=l_max)
            .map(|l| {
                (0..=k_max)
                    .map(|k| self.aggregated_rank(node_identifiers, k, l))
                    .collect()
            })
            .collect()
    }

    pub fn max_found_l(&self) -> usize {
        self.paths
            .iter()
            .map(|entry| entry.key().l)
            .max()
            .unwrap_or(0)
    }

    pub fn stl(&self, node_pair: (NodeId, NodeId), l: usize) -> StlPathContainer<&Self, NodeId> {
        StlPathContainer {
            parent_container: self,
            node_pair,
            l,
        }
    }
}

#[derive(Clone)]
struct GraphPathSearchNode<G: GraphRef>
where
    <G as GraphBase>::NodeId: Eq + Hash,
{
    path: Vec<G::NodeId>,
    l: usize,
    path_query: PathQuery<G>,
}

impl<G> GraphPathSearchNode<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    fn init(node: G::NodeId, path_query: PathQuery<G>) -> Self {
        Self {
            path: vec![node],
            l: 0,
            path_query,
        }
    }
}

// Derive Debug, Hash, PartialEq, Eq for these nodes
// These should only depend on self.path
// The other information is just kept around for score keeping

impl<G> Debug for GraphPathSearchNode<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

impl<G> Hash for GraphPathSearchNode<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.as_slice().hash(state)
    }
}

impl<G> PartialEq for GraphPathSearchNode<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.path.as_slice().eq(other.path.as_slice())
    }
}

impl<G> Eq for GraphPathSearchNode<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
}

impl<G> FastNode for GraphPathSearchNode<G>
where
    G: GraphRef + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Debug,
{
    type Error = anyhow::Error;

    fn add_children<E>(&self, _depth: usize, queue: &mut E) -> Result<(), Self::Error>
    where
        E: par_dfs::sync::ExtendQueue<Self, Self::Error>,
    {
        if let StoppingCondition::KMax(k_max) = self.path_query.stopping_condition {
            #[allow(clippy::int_plus_one)]
            if self.path.len() >= k_max + 1 {
                // Stop the iteration because we already have a (k,?)-path
                return Ok(());
            }
        }

        let final_vertex = self.path.last().expect("Path should be non-empty");

        let longer_paths = self
            .path_query
            .g
            .node_identifiers()
            .filter_map(|next_node| {
                // Next node must be different
                if next_node == *final_vertex {
                    return None;
                }

                // Next node must be reachable in finite time
                let next_hop_distance = self.path_query.d.distance(final_vertex, &next_node);
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
                let mut path_vec = self.path.clone();
                path_vec.push(next_node);
                Ok(GraphPathSearchNode {
                    path: path_vec,
                    l: new_l,
                    path_query: self.path_query.clone(),
                })
            });

        queue.add_all(longer_paths);
        Ok(())
    }
}

// TODO: Needs to know k_max
#[derive(Debug)]
pub struct StlPathContainer<Ref, NodeId>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
{
    pub parent_container: Ref,
    pub node_pair: (NodeId, NodeId),
    pub l: usize,
}

impl<Ref, NodeId> StlPathContainer<Ref, NodeId>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
{
    pub fn key_from_k(&self, k: usize) -> PathKey<NodeId> {
        PathKey {
            s: self.node_pair.0,
            t: self.node_pair.1,
            k,
            l: self.l,
        }
    }

    pub fn num_paths(&self, k: usize) -> usize {
        self.parent_container.num_paths(&self.key_from_k(k))
    }

    pub fn index_of(&self, path: &Path<NodeId>) -> Option<usize> {
        self.parent_container
            .index_of(&self.key_from_k(path.len() - 1), path)
    }

    pub fn path_at_index(&self, k: usize, idx: usize) -> Option<Path<NodeId>> {
        self.parent_container
            .path_at_index(&self.key_from_k(k), idx)
    }

    pub fn chain_group_sizes(&self, k_max: usize) -> impl Iterator<Item = usize> + '_ {
        (0..=k_max).map(|k| self.num_paths(k))
    }

    pub fn max_homology_dim(&self) -> usize {
        if self.parent_container.k_max == self.l {
            self.l
        } else {
            self.parent_container.k_max - 1
        }
    }

    pub fn serial_homology(
        self,
        representatives: bool,
    ) -> StlHomology<Ref, NodeId, VecColumn, SerialDecomposition<VecColumn>> {
        let options = LoPhatOptions {
            maintain_v: representatives,
            ..Default::default()
        };

        self.homology::<VecColumn, SerialAlgorithm<VecColumn>>(Some(options))
    }

    pub fn homology<C, Algo>(
        self,
        options: Option<Algo::Options>,
    ) -> StlHomology<Ref, NodeId, C, Algo::Decomposition>
    where
        C: Column,
        Algo: DecompositionAlgo<C>,
    {
        build_stl_homology::<Ref, NodeId, C, Algo>(self, options)
    }

    // TODO: Add a k-homology function that only compute homology for a given k?
}

impl<NodeId> StlPathContainer<Arc<PathContainer<NodeId>>, NodeId>
where
    NodeId: SensibleNode,
{
    pub fn new(
        parent_container: Arc<PathContainer<NodeId>>,
        node_pair: (NodeId, NodeId),
        l: usize,
    ) -> Self {
        Self {
            parent_container,
            node_pair,
            l,
        }
    }
}
