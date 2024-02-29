use std::{collections::HashMap, marker::PhantomData, ops::Deref};

use lophat::{algorithms::Decomposition, columns::Column};
use petgraph::visit::{GraphRef, IntoNodeIdentifiers};
use rustc_hash::FxHashMap;

use rayon::prelude::*;

use crate::{
    path_search::{PathContainer, PathQuery, SensibleNode, StlPathContainer},
    utils::rank_map_to_rank_vec,
    Path,
};

// Returns a map indexed by dimension k
// map[k] is a list of column-indices wherein non-dead homology is created

pub fn homology_idxs<C, Algo>(decomposition: &Algo) -> FxHashMap<usize, Vec<usize>>
where
    C: Column,
    Algo: Decomposition<C>,
{
    let mut map: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let diagram = decomposition.diagram();
    for idx in diagram.unpaired {
        let dim = decomposition.get_r_col(idx).dimension();
        map.entry(dim).or_default().push(idx);
    }
    map
}

pub fn all_homology_ranks_default<G>(
    path_container: &PathContainer<G::NodeId>,
    query: PathQuery<G>,
) -> Vec<Vec<usize>>
where
    G: GraphRef + IntoNodeIdentifiers + Sync,
    G::NodeId: SensibleNode + Send + Sync,
{
    let all_node_pairs: Vec<_> = query.node_pair_iterator().collect();

    (0..=query.l_max)
        .map(|l| {
            // Compute MH^{(s, t)} for each (s, t) in arbitrary order
            let node_pair_wise: Vec<_> = all_node_pairs
                .iter()
                .par_bridge()
                .map(|node_pair| {
                    let homology = path_container
                        .stl(node_pair.clone(), l)
                        .serial_homology(query.d.clone(), false);

                    ((node_pair.clone(), l), homology)
                })
                .collect();

            let ds = DirectSum::new(node_pair_wise.into_iter());
            rank_map_to_rank_vec(&ds.ranks(), query.l_max)
        })
        .collect()
}

pub struct StlHomology<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    pub stl_paths: StlPathContainer<Ref, NodeId>,
    pub decomposition: Decomp,
    pub homology_idxs: FxHashMap<usize, Vec<usize>>,
    c: PhantomData<C>,
}

impl<Ref, NodeId, C, Decomp> StlHomology<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    pub fn new(stl_paths: StlPathContainer<Ref, NodeId>, decomposition: Decomp) -> Self {
        let homology_idxs = homology_idxs(&decomposition);
        Self {
            stl_paths,
            decomposition,
            homology_idxs,
            c: PhantomData,
        }
    }

    pub fn ranks(&self) -> HashMap<usize, usize> {
        self.homology_idxs
            .iter()
            .map(|(&dim, idxs)| (dim, idxs.len()))
            .collect()
    }

    pub fn representatives(&self) -> Option<HashMap<usize, Vec<Vec<Path<NodeId>>>>> {
        if !self.decomposition.has_v() {
            return None;
        }

        let collect_rep = |k, rep_idx| {
            let dim_offset: usize = self.stl_paths.chain_group_sizes(k - 1).sum();

            self.decomposition
                // Get the V column
                .get_v_col(rep_idx)
                .unwrap()
                .entries()
                // Move each chain complex index back to (s, t, k, l) indexes
                // and lookup these paths in the path container
                .map(move |cc_idx| {
                    self.stl_paths
                        .path_at_index(k, cc_idx - dim_offset)
                        .unwrap()
                })
                .into_iter()
                .collect()
        };

        Some(
            self.homology_idxs
                .iter()
                .map(|(&k, rep_idxs)| {
                    (
                        k,
                        rep_idxs
                            .iter()
                            .map(|&rep_idx| collect_rep(k, rep_idx))
                            .collect(),
                    )
                })
                .collect(),
        )
    }
}

type StlKey<NodeId> = ((NodeId, NodeId), usize);

// TODO: Add direct sum class
pub struct DirectSum<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    summands: FxHashMap<StlKey<NodeId>, StlHomology<Ref, NodeId, C, Decomp>>,
}

impl<Ref, NodeId, C, Decomp> DirectSum<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    // TODO: Allow Parallel Iterator
    pub fn new(
        summands: impl Iterator<Item = (StlKey<NodeId>, StlHomology<Ref, NodeId, C, Decomp>)>,
    ) -> Self {
        Self {
            summands: summands.collect(),
        }
    }

    pub fn ranks(&self) -> HashMap<usize, usize> {
        let mut ranks = HashMap::new();
        for stl_hom in self.summands.values() {
            for (k, rk_k) in stl_hom.ranks() {
                *ranks.entry(k).or_default() += rk_k;
            }
        }
        ranks
    }

    // Returns None if any of the summands does not have reps
    pub fn representatives(&self) -> Option<HashMap<usize, Vec<Vec<Path<NodeId>>>>> {
        let mut reps: HashMap<usize, Vec<Vec<Path<NodeId>>>> = HashMap::new();
        for stl_hom in self.summands.values() {
            let stl_reps = stl_hom.representatives()?;
            for (k, reps_k) in stl_reps {
                reps.entry(k).or_default().extend(reps_k.into_iter())
            }
        }
        Some(reps)
    }

    pub fn get(&self, key: &StlKey<NodeId>) -> Option<&StlHomology<Ref, NodeId, C, Decomp>> {
        self.summands.get(key)
    }

    pub fn add(
        &mut self,
        key: StlKey<NodeId>,
        hom: StlHomology<Ref, NodeId, C, Decomp>,
    ) -> Option<StlHomology<Ref, NodeId, C, Decomp>> {
        self.summands.insert(key, hom)
    }
}
