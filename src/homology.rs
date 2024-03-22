// TODO: Add a way to compute path homology

use std::{collections::HashMap, marker::PhantomData, ops::Deref, sync::Arc};

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo, SerialAlgorithm},
    columns::{Column, VecColumn},
    options::LoPhatOptions,
};

use petgraph::visit::{GraphRef, IntoNodeIdentifiers};
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

// Takes the k paths from the provided stl container
// and adds columns corresponding to each k_path
fn append_k_paths<Ref, NodeId, C, Algo>(
    stl_paths: &StlPathContainer<Ref, NodeId>,
    mut algo: Algo,
    k: usize,
    k_offset: usize,
    k_minus_1_offset: usize,
) -> Algo
where
    NodeId: SensibleNode,
    Ref: Deref<Target = PathContainer<NodeId>>,
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    // Add empty columns for each k path
    let number_of_k_paths = stl_paths.num_paths(k);
    algo = algo.add_cols((0..number_of_k_paths).map(|_i| C::new_with_dimension(k)));

    // Extract all the k paths from the container
    let k_paths = stl_paths
        .parent_container
        .paths
        .get(&stl_paths.key_from_k(k));
    let Some(k_paths) = k_paths else { return algo };

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
                let bdry_path_idx = stl_paths
                    .index_of(&bdry_path)
                    .expect("Should have found this boundary and inserted with correct key");
                // Add entry to boundary matrix, using appropriate offsets
                (bdry_path_idx + k_minus_1_offset, idx + k_offset)
            });
        algo = algo.add_entries(entries);
    }
    algo
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

    // Loop through each homological dimension (k)
    for k in 0..=k_max {
        // Setup offsets for k-paths and (k-1)-paths
        // For k=0, we have to manually override k_minus_1_offset
        let k_offset: usize = sizes[0..k].iter().sum();
        let k_minus_1_offset: usize = if k == 0 {
            0
        } else {
            sizes[0..(k - 1)].iter().sum()
        };
        algo = append_k_paths(&stl_paths, algo, k, k_offset, k_minus_1_offset)
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

    fn collect_rep(&self, k: usize, rep_idx: usize) -> Representative<NodeId> {
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
    }

    /// This might be quite slow because [`path_at_index`](StlPathContainer::path_at_index) can be slow if there are lots of paths with the same key
    pub fn representatives(&self) -> Result<HashMap<usize, Vec<Representative<NodeId>>>, MagError> {
        if !self.decomposition.has_v() {
            return Err(MagError::NoRepresentatives);
        }

        let mut reps_map: HashMap<_, _> = self
            .homology_idxs
            .iter()
            .map(|(&k, rep_idxs)| {
                (
                    k,
                    rep_idxs
                        .iter()
                        .map(|&rep_idx| self.collect_rep(k, rep_idx))
                        .collect(),
                )
            })
            .collect();

        for dim in 0..=self.stl_paths.max_homology_dim() {
            // Dimension 0 => no reps
            reps_map.entry(dim).or_insert_with(Vec::new);
        }
        Ok(reps_map)
    }
}

type StlKey<NodeId> = ((NodeId, NodeId), usize);

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

        let mut n_summands_by_dim: HashMap<_, usize> = HashMap::new();
        let n_summands = self.summands.len();

        // Add up all the ranks
        for stl_hom in self.summands.values() {
            for (k, rk_k) in stl_hom.ranks() {
                *ranks.entry(k).or_default() += rk_k;
                *n_summands_by_dim.entry(k).or_default() += 1;
            }
        }

        // Remove any dims that didn't appear in every summand
        for (k, n) in n_summands_by_dim {
            if n != n_summands {
                ranks.remove(&k);
            }
        }

        ranks
    }

    // Returns Err if any of the summands does not have reps
    pub fn representatives(&self) -> Result<HashMap<usize, Vec<Representative<NodeId>>>, MagError> {
        let mut reps: HashMap<usize, Vec<Vec<Path<NodeId>>>> = HashMap::new();
        let mut n_summands_by_dim: HashMap<_, usize> = HashMap::new();
        let n_summands = self.summands.len();

        // Collect all the reps
        for stl_hom in self.summands.values() {
            let stl_reps = stl_hom.representatives()?;
            for (k, reps_k) in stl_reps {
                reps.entry(k).or_default().extend(reps_k.into_iter());
                *n_summands_by_dim.entry(k).or_default() += 1;
            }
        }

        // Remove any dims that didn't appear in every summand
        for (k, n) in n_summands_by_dim {
            if n != n_summands {
                reps.remove(&k);
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

// Compute a basis for MH_{k, k}^{(s, t)}
// Since the boundary space is empty, each homology element has a unique representative!
// We just return a list of representatives (each rep is a set of k-paths)
pub fn mhkk_st_basis<NodeId>(
    path_container: &PathContainer<NodeId>,
    k: usize,
    node_pair: (NodeId, NodeId),
) -> Vec<Representative<NodeId>>
where
    NodeId: SensibleNode,
{
    let mut options: LoPhatOptions = Default::default();
    options.maintain_v = true;

    // Setup algorithm to receive entries
    let mut algo = SerialAlgorithm::<VecColumn>::init(Some(options));

    let stk_paths = path_container.stl(node_pair, k);
    algo = append_k_paths(&stk_paths, algo, k, 0, 0);

    let decomposition = algo.decompose();

    // Note: Can't use homology_idxs because the matrix represented in algo is NOT a chain complex
    // Instead we just look at cycle columns

    let get_rep = |col_idx| {
        decomposition
            .get_v_col(col_idx)
            .expect("Should have v_col because decomposition has_v")
            .entries()
            .map( |cc_idx| {
                stk_paths
                    .path_at_index(k, cc_idx)
                    .expect("v_col should be a sum of (s,t,k,l) paths which should all be in the StlPathContainer")
            })
            .collect::<Vec<_>>()
    };

    (0..decomposition.n_cols())
        .filter_map(|col_idx| {
            if decomposition.get_r_col(col_idx).is_cycle() {
                Some(get_rep(col_idx))
            } else {
                None
            }
        })
        .collect()
}

pub fn path_homology_basis<G>(
    graph: G,
    path_container: &PathContainer<G::NodeId>,
    k: usize,
) -> Vec<Representative<G::NodeId>>
where
    G: GraphRef + IntoNodeIdentifiers,
    G::NodeId: SensibleNode + Sync + Send,
{
    let node_pairs: Vec<_> = graph
        .node_identifiers()
        .flat_map(|s| graph.node_identifiers().map(move |t| (s, t)))
        .collect();

    node_pairs
        .into_par_iter()
        .map(|node_pair| mhkk_st_basis(&path_container, k, node_pair))
        .reduce(
            || vec![],
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

pub fn path_homology<NodeId>(_path_container: &PathContainer<NodeId>)
where
    NodeId: SensibleNode,
{
    // Step 1: Compute a basis for each MH_{k, k} (Maybe do in parallel over k, s, t?)
    // Step 2: Setup the index lookup hashmaps
    // Step 3: Construct the path chain complex
    // Step 4: Decompose
    // Step 5: Compute homology idxs
    // Step 6: Provide representatives?
}
