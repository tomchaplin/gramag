use std::{cmp::Reverse, collections::HashMap, ops::Deref, sync::Arc};

use petgraph::graph::NodeIndex;
use phlite::{
    fields::{Invertible, NonZeroCoefficient, Z2},
    reduction::Diagram,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    homology::StlKey,
    phlite_types::{
        MagnitudeReductionMatrix, PathIndex, PhlitePathContainer, PhliteStlPathContainer,
    },
    utils::rank_map_to_rank_vec,
    MagError, Path, Representative,
};

pub struct PhliteStlHomology<Ref, CF, R>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
    CF: NonZeroCoefficient + Invertible + 'static,
    R: IntoIterator<Item = usize> + Clone,
{
    pub stl_paths: PhliteStlPathContainer<Ref>,
    pub reduction_matrix: MagnitudeReductionMatrix<CF, R>,
    pub diagram: Diagram<Reverse<PathIndex>>,
}

impl<Ref, CF, R> PhliteStlHomology<Ref, CF, R>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
    CF: NonZeroCoefficient + Invertible + 'static,
    R: IntoIterator<Item = usize> + Clone,
{
    pub fn new(
        stl_paths: PhliteStlPathContainer<Ref>,
        reduction_matrix: MagnitudeReductionMatrix<CF, R>,
        diagram: Diagram<Reverse<PathIndex>>,
    ) -> Self {
        Self {
            stl_paths,
            reduction_matrix,
            diagram,
        }
    }

    pub fn ranks(&self) -> HashMap<usize, usize> {
        let n_nodes = self.stl_paths.parent_container.n_nodes;
        // Iterates over the k-dimension of unpaired paths
        let unpaired_dimension_iterator = self
            .diagram
            .essential
            .iter()
            .map(|idx| idx.0.dimension(n_nodes));

        let mut rank_map = HashMap::new();
        let max_homology_dim = self.stl_paths.max_homology_dim();
        for dim in 0..=max_homology_dim {
            rank_map.insert(dim, 0);
        }

        for dim in unpaired_dimension_iterator {
            if let Some(current_rank) = rank_map.get_mut(&dim) {
                *current_rank += 1
            }
        }

        rank_map
    }

    // TODO: Is it now possible to make this infallible?
    pub fn representatives(
        &self,
    ) -> Result<HashMap<usize, Vec<Representative<NodeIndex>>>, MagError> {
        // Do involuted representatives here?
        // Maybe store hashmap in a cache?
        todo!()
    }
}

pub struct PhliteDirectSum<Ref, CF, R>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
    CF: NonZeroCoefficient + Invertible + 'static,
    R: IntoIterator<Item = usize> + Clone,
{
    summands: FxHashMap<StlKey<NodeIndex>, Arc<PhliteStlHomology<Ref, CF, R>>>,
}

impl<Ref, CF, R> PhliteDirectSum<Ref, CF, R>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
    CF: NonZeroCoefficient + Invertible + 'static,
    R: IntoIterator<Item = usize> + Clone,
{
    pub fn new(
        summands: impl Iterator<Item = (StlKey<NodeIndex>, Arc<PhliteStlHomology<Ref, CF, R>>)>,
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
    pub fn representatives(
        &self,
    ) -> Result<HashMap<usize, Vec<Representative<NodeIndex>>>, MagError> {
        let mut reps: HashMap<usize, Vec<Vec<Path<NodeIndex>>>> = HashMap::new();
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

    pub fn get(&self, key: &StlKey<NodeIndex>) -> Option<&PhliteStlHomology<Ref, CF, R>> {
        self.summands.get(key).map(|arc_hom| arc_hom.deref())
    }

    pub fn add(
        &mut self,
        key: StlKey<NodeIndex>,
        hom: Arc<PhliteStlHomology<Ref, CF, R>>,
    ) -> Option<Arc<PhliteStlHomology<Ref, CF, R>>> {
        self.summands.insert(key, hom)
    }
}

pub fn all_homology_ranks_default(
    path_container: &PhlitePathContainer<NodeIndex>,
    node_pairs: &[(NodeIndex, NodeIndex)],
) -> Vec<Vec<usize>> {
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
                    let stl_paths = path_container.stl(*node_pair, l);
                    let k_max = stl_paths.max_homology_dim();
                    let homology = path_container
                        .stl(*node_pair, l)
                        .homology::<Z2, _>(0..=k_max);

                    ((*node_pair, l), Arc::new(homology))
                })
                .collect();

            let ds = PhliteDirectSum::new(node_pair_wise.into_iter());
            rank_map_to_rank_vec(&ds.ranks())
        })
        .collect()
}
