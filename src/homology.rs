use std::{collections::HashMap, marker::PhantomData, ops::Deref, sync::Arc};

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::Column,
};

use rustc_hash::FxHashMap;

use rayon::prelude::*;

use crate::{
    path_search::{PathContainer, SensibleNode, StlPathContainer},
    utils::rank_map_to_rank_vec,
    MagError, Path, Representative,
};

/// Returns a map indexed by dimension k.
/// `map[k]` is a list of column-indices wherein non-dead homology is created.
/// `k_max` is maximum homology to compute dimension up to
pub fn homology_idxs<C, Algo>(decomposition: &Algo, k_max: usize) -> FxHashMap<usize, Vec<usize>>
where
    C: Column,
    Algo: Decomposition<C>,
{
    let mut map: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let diagram = decomposition.diagram();
    for idx in diagram.unpaired {
        let dim = decomposition.get_r_col(idx).dimension();
        if dim > k_max {
            continue;
        }
        map.entry(dim).or_default().push(idx);
    }
    map
}

pub fn all_homology_ranks_default<NodeId>(
    path_container: &PathContainer<NodeId>,
    node_pairs: &[(NodeId, NodeId)],
) -> Vec<Vec<usize>>
where
    NodeId: SensibleNode + Send + Sync,
{
    let l_max = path_container
        .l_max
        .unwrap_or_else(|| path_container.max_found_l());
    (0..=l_max)
        .map(|l| {
            // Compute MH^{(s, t)} for each (s, t) in arbitrary order
            let node_pair_wise: Vec<_> = node_pairs
                .iter()
                .par_bridge()
                .map(|node_pair| {
                    let homology = path_container.stl(*node_pair, l).serial_homology(false);

                    ((*node_pair, l), Arc::new(homology))
                })
                .collect();

            let ds = DirectSum::new(node_pair_wise.into_iter());
            rank_map_to_rank_vec(&ds.ranks())
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

// TODO: Add option to anti-transpose?
// TODO: Any way to associate this as a method of StlHomology?
pub(crate) fn build_stl_homology<Ref, NodeId, C, Algo>(
    stl_paths: StlPathContainer<Ref, NodeId>,
    options: Option<Algo::Options>,
) -> StlHomology<Ref, NodeId, C, Algo::Decomposition>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    // Setup algorithm to receive entries
    let mut algo = Algo::init(options);
    let k_max = stl_paths.parent_container.k_max;
    let sizes: Vec<_> = stl_paths.chain_group_sizes(k_max).collect();
    let empty_cols =
        (0..=k_max).flat_map(|k| (0..sizes[k]).map(move |_i| C::new_with_dimension(k)));
    algo = algo.add_cols(empty_cols);

    // Loop through each homological dimension (k)
    for k in 0..=k_max {
        // k= 0 => no boundary
        if k == 0 {
            continue;
        }

        // Setup offsets for k-paths and (k-1)-paths
        let k_offset: usize = sizes[0..k].iter().sum();
        let k_minus_1_offset: usize = sizes[0..(k - 1)].iter().sum();
        let k_paths = stl_paths
            .parent_container
            .paths
            .get(&stl_paths.key_from_k(k));
        let Some(k_paths) = k_paths else { continue };

        // Loop through all k-paths
        for entry in k_paths.value() {
            // Get path and its index in the chain complex
            let path = entry.key();
            let idx = entry.value();

            // Only test removing interior vertices
            let entries = (1..k)
                .filter(|&i| {
                    // Path without vertex i appears in boundary
                    // iff removing doesn't change length
                    let a = path[i - 1];
                    let mid = path[i];
                    let b = path[i + 1];
                    let d = &stl_paths.parent_container.d;
                    d.distance(&a, &mid) + d.distance(&mid, &b) == d.distance(&a, &b)
                })
                .map(|i| {
                    // Get index of path with i^th vertex removed
                    let mut bdry_path = path.clone();
                    bdry_path.remove(i);
                    // Could be slow
                    let bdry_path_idx = stl_paths
                        .index_of(&bdry_path)
                        .expect("Should have found this boundary and inserted with correct key");
                    // Add entry to boundary matrix, using appropriate offsets
                    (bdry_path_idx + k_minus_1_offset, idx + k_offset)
                });
            algo = algo.add_entries(entries);
        }
    }

    // Run the algorithm
    let decomposition = algo.decompose();

    StlHomology::new(stl_paths, decomposition)
}

impl<Ref, NodeId, C, Decomp> StlHomology<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    pub fn new(stl_paths: StlPathContainer<Ref, NodeId>, decomposition: Decomp) -> Self {
        // If k_max = l then we can compute up to l because there are no chains in higher degrees
        // Else we can only compute homology up to k_max -1
        let homology_idxs = homology_idxs(&decomposition, stl_paths.max_homology_dim());
        Self {
            stl_paths,
            decomposition,
            homology_idxs,
            c: PhantomData,
        }
    }

    pub fn ranks(&self) -> HashMap<usize, usize> {
        let mut rank_map: HashMap<_, _> = self
            .homology_idxs
            .iter()
            .map(|(&dim, idxs)| (dim, idxs.len()))
            .collect();

        let max_homology_dim = self.stl_paths.max_homology_dim();
        for dim in 0..=max_homology_dim {
            // Dimension must be 0 otherwise
            rank_map.entry(dim).or_insert(0);
        }
        rank_map
    }

    /// This might be quite slow because [`path_at_index`](StlPathContainer::path_at_index) can be slow if there are lots of paths with the same key
    pub fn representatives(&self) -> Result<HashMap<usize, Vec<Representative<NodeId>>>, MagError> {
        if !self.decomposition.has_v() {
            return Err(MagError::NoRepresentatives);
        }

        let collect_rep = |k, rep_idx| {
            let dim_offset: usize = if k == 0 {
                0
            } else {
                self.stl_paths.chain_group_sizes(k - 1).sum()
            };

            self.decomposition
                // Get the V column
                .get_v_col(rep_idx)
                .expect("Should have v_col because decomposition has_v")
                .entries()
                // Move each chain complex index back to (s, t, k, l) indexes
                // and lookup these paths in the path container
                .map(move |cc_idx| {
                    self.stl_paths
                        .path_at_index(k, cc_idx - dim_offset)
                        .expect("v_col should be a sum of (s,t,k,l) paths which should all be in the StlPathContainer")
                })
                .collect()
        };

        let mut reps_map: HashMap<_, _> = self
            .homology_idxs
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
            .collect();

        for dim in 0..self.stl_paths.max_homology_dim() {
            // Dimension 0 => no reps
            reps_map.entry(dim).or_insert_with(Vec::new);
        }
        Ok(reps_map)
    }
}

type StlKey<NodeId> = ((NodeId, NodeId), usize);

// TODO: Allow direct sum to just hold a reference? Maybe enfore Arc?
pub struct DirectSum<Ref, NodeId, C, Decomp>
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Decomp: Decomposition<C>,
{
    summands: FxHashMap<StlKey<NodeId>, Arc<StlHomology<Ref, NodeId, C, Decomp>>>,
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
        summands: impl Iterator<Item = (StlKey<NodeId>, Arc<StlHomology<Ref, NodeId, C, Decomp>>)>,
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
    pub fn representatives(&self) -> Result<HashMap<usize, Vec<Representative<NodeId>>>, MagError> {
        let mut reps: HashMap<usize, Vec<Vec<Path<NodeId>>>> = HashMap::new();
        for stl_hom in self.summands.values() {
            let stl_reps = stl_hom.representatives()?;
            for (k, reps_k) in stl_reps {
                reps.entry(k).or_default().extend(reps_k.into_iter())
            }
        }
        Ok(reps)
    }

    pub fn get(&self, key: &StlKey<NodeId>) -> Option<&StlHomology<Ref, NodeId, C, Decomp>> {
        self.summands.get(key).map(|arc_hom| arc_hom.deref())
    }

    pub fn add(
        &mut self,
        key: StlKey<NodeId>,
        hom: Arc<StlHomology<Ref, NodeId, C, Decomp>>,
    ) -> Option<Arc<StlHomology<Ref, NodeId, C, Decomp>>> {
        self.summands.insert(key, hom)
    }
}
