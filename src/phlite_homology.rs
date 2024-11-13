use std::{cmp::Reverse, collections::HashMap, ops::Deref, sync::Arc};

use log::info;

use petgraph::graph::NodeIndex;
use phlite::{
    fields::{Invertible, NonZeroCoefficient, Z2},
    matrices::MatrixOracle,
    reduction::{standard_algo_with_diagram, Diagram},
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    homology::StlKey,
    phlite_types::{
        MagnitudeReductionMatrix, PathIndex, PhlitePathContainer, PhliteStlPathContainer,
    },
    utils::rank_map_to_rank_vec,
    MagError, Representative,
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
    pub fn representatives(&self, k: usize) -> Result<Vec<Representative<usize>>, MagError> {
        // Do involuted representatives here?
        // Maybe store hashmap in a cache?
        let max_homology_dim = self.stl_paths.max_homology_dim();
        let homology_range = 0..=max_homology_dim;
        if !homology_range.contains(&k) {
            return Err(MagError::InsufficientKMax(k, max_homology_dim));
        }

        let n_nodes = self.stl_paths.parent_container.n_nodes;

        // Build up the involution basis
        // This is all of the killing columns and all the essential columns
        let mut involution_basis = vec![];
        for (death_cell, _birth_cell) in &self.diagram.pairings {
            let dim = death_cell.0.dimension(n_nodes);
            if dim == k {
                involution_basis.push(death_cell.0);
            }
        }
        for essential_cell in &self.diagram.essential {
            let dim = essential_cell.0.dimension(n_nodes);
            if dim == k {
                involution_basis.push(essential_cell.0);
            }
        }
        involution_basis.sort_unstable();

        //info!(
        //    "Reducing boundary {:?},{:?},{},{} ::: basis size = {}",
        //    self.stl_paths.node_pair.0,
        //    self.stl_paths.node_pair.1,
        //    k,
        //    self.stl_paths.l,
        //    involution_basis.len()
        //);

        // Get boundary and restrict columns to involution basis
        let boundary_matrix = self.stl_paths.parent_container.magnitude_bounday::<Z2>();
        let boundary_matrix = boundary_matrix
            .with_basis(involution_basis)
            .with_trivial_filtration();
        // Reduce
        let (v_matrix, _boundary_diagram) = standard_algo_with_diagram(&boundary_matrix, false);

        let mut representatives = vec![];
        for essential_cell in &self.diagram.essential {
            let v_col: Vec<_> = v_matrix
                .column(essential_cell.0)
                .map(|(_coeff, path_idx)| path_idx.to_vec(n_nodes))
                .collect();
            representatives.push(v_col);
        }

        Ok(representatives)
    }
}

pub struct PhliteDirectSum<Ref, CF, R>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
    CF: NonZeroCoefficient + Invertible + 'static,
    R: IntoIterator<Item = usize> + Clone,
{
    pub summands: FxHashMap<StlKey<NodeIndex>, Arc<PhliteStlHomology<Ref, CF, R>>>,
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
    pub fn representatives(&self, k: usize) -> Result<Vec<Representative<usize>>, MagError> {
        let mut reps = vec![];

        for summand in self.summands.values() {
            let mut summand_reps = summand.representatives(k)?;
            reps.append(&mut summand_reps);
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
