use crate::distances::{Distance, DistanceMatrix};
use crate::Path;

use core::hash::Hash;
use std::iter;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

use dashmap::DashMap;
use par_dfs::sync::par::IntoParallelIterator;
use par_dfs::sync::{FastBfs, FastNode};
use petgraph::visit::{GraphBase, GraphRef, IntoEdges, IntoNodeIdentifiers, Visitable};
use rayon::prelude::{ParallelBridge, ParallelIterator};

use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::style::{HorizontalLine, VerticalLine};
use tabled::settings::{Alignment, Style};

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
            s: path.first().copied().unwrap(),
            t: path.last().copied().unwrap(),
            k: path.len() - 1,
            l,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathQuery<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    g: G,
    d: Arc<DistanceMatrix<G::NodeId>>,
    l_max: usize,
}

impl<G> PathQuery<G>
where
    G: GraphRef,
    <G as GraphBase>::NodeId: Eq + Hash,
{
    pub fn build(g: G, d: Arc<DistanceMatrix<G::NodeId>>, l_max: usize) -> Self {
        Self { g, d, l_max }
    }
}

impl<G> PathQuery<G>
where
    G: GraphRef + IntoNodeIdentifiers,
    <G as GraphBase>::NodeId: SensibleNode,
{
    pub fn key_iterator<'a>(&'a self) -> impl Iterator<Item = PathKey<G::NodeId>> + 'a {
        self.g
            .node_identifiers()
            .flat_map(|s| self.g.node_identifiers().map(move |t| (s, t)))
            .flat_map(|(s, t)| (0..=self.l_max).map(move |k| (s, t, k)))
            .flat_map(|(s, t, k)| (0..=self.l_max).map(move |l| (s, t, k, l)))
            .map(|(s, t, k, l)| PathKey { s, t, k, l })
    }
}

#[derive(Debug)]
pub struct PathContainer<G>
where
    G: GraphRef,
    G::NodeId: SensibleNode,
{
    pub paths: DashMap<PathKey<G::NodeId>, DashMap<Path<G::NodeId>, usize>>,
    query: PathQuery<G>,
}

impl<G> PathContainer<G>
where
    G: GraphRef,
    G::NodeId: SensibleNode,
{
    fn new(query: &PathQuery<G>) -> Self {
        Self {
            paths: DashMap::new(),
            query: query.clone(),
        }
    }

    fn store(&self, key: &PathKey<G::NodeId>, path: Path<G::NodeId>, idx: usize) {
        // We use or_default to insert an empty DashMap
        // if we haven't found a path with this key before
        // Note: To prevent race conditions, if key is not in self.paths
        //        then we have to keep it locked until we've inserted the empty DashMap
        self.paths.entry(*key).or_default().insert(path, idx);
    }

    fn num_paths(&self, key: &PathKey<G::NodeId>) -> usize {
        if let Some(sub_paths) = self.paths.get(key) {
            sub_paths.len()
        } else {
            0
        }
    }
}

// TODO: Allow fixing s and/or t

impl<G> PathContainer<G>
where
    G: GraphRef + IntoNodeIdentifiers,
    G::NodeId: SensibleNode,
{
    pub fn aggregated_rank(&self, k: usize, l: usize) -> usize {
        self.query
            .g
            .node_identifiers()
            .flat_map(|s| self.query.g.node_identifiers().map(move |t| (s, t)))
            .map(|(s, t)| self.num_paths(&PathKey { s, t, k, l }))
            .sum()
    }

    pub fn rank_matrix(&self) -> Vec<Vec<usize>> {
        (0..(self.query.l_max + 1))
            .map(|l| {
                (0..(self.query.l_max + 1))
                    .map(|k| self.aggregated_rank(k, l))
                    .collect()
            })
            .collect()
    }

    pub fn rank_table(&self) -> String {
        let mut builder = Builder::new();

        // Header

        let header = iter::once(format!("k="))
            .chain((0..(self.query.l_max + 1)).map(move |k| format!("{}", k)));
        builder.push_record(header);

        for l in 0..(self.query.l_max + 1) {
            let ranks =
                (0..(self.query.l_max + 1)).map(move |k| format!("{}", self.aggregated_rank(k, l)));
            let row = iter::once(format!("l={}", l)).chain(ranks);

            builder.push_record(row)
        }

        let theme = Style::modern_rounded()
            .horizontals([(1, HorizontalLine::inherit(Style::modern_rounded()))])
            .verticals([(1, VerticalLine::inherit(Style::modern_rounded()))])
            .remove_horizontal()
            .remove_vertical();

        builder
            .build()
            .with(theme)
            .modify(Columns::new(1..), Alignment::right())
            .to_string()
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
        //if depth - 1 > self.path_query.l_max {
        //    return Ok(());
        //}

        let final_vertex = self.path.last().unwrap();

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
                let next_hop_distance = self.path_query.d.distance(&final_vertex, &next_node);
                let new_l = match next_hop_distance {
                    Distance::Finite(next_hop) => Some(self.l + next_hop),
                    Distance::Infinite => None,
                }?;

                // Total length of new paths must be at most l_max
                if new_l <= self.path_query.l_max {
                    Some((next_node, new_l))
                } else {
                    None
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

impl<G> PathQuery<G>
where
    G: IntoEdges + Visitable + IntoNodeIdentifiers + Sync + Send,
    G::NodeId: SensibleNode + Send + Sync,
    <G as IntoNodeIdentifiers>::NodeIdentifiers: Send,
{
    pub fn run(&self) -> PathContainer<G>
    where
        G: IntoEdges + Visitable + IntoNodeIdentifiers + Sync + Send,
        G::NodeId: SensibleNode + Send + Sync,
        <G as IntoNodeIdentifiers>::NodeIdentifiers: Send,
    {
        // Setup container for paths and their indexes
        let container = PathContainer::new(&self);

        // Setup counters for the number of (s, t, k, l) paths encountered
        // This allows us to index such paths as we find them
        let mut counters = HashMap::new();
        for key in self.key_iterator() {
            counters.insert(key, AtomicUsize::new(0));
        }

        let store_node = |node: GraphPathSearchNode<G>| {
            let key = PathKey::build_from_path(&node.path, node.l);
            let idx = counters
                .get(&key)
                .unwrap()
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
                        store_node(path_node.unwrap());
                    })
            });

        container
    }
}
