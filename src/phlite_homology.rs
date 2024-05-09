use petgraph::graph::NodeIndex;
use phlite::{fields::Z2, matrices::MatrixRef, reduction::ClearedReductionMatrix};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::phlite_types::PhlitePathContainer;

// TODO: Implement DirectSum types for phlite path containers
// Alternative: Make DirectSum generic over a homology type?

pub fn all_homology_ranks_default(
    path_container: &PhlitePathContainer<NodeIndex>,
    node_pairs: &[(NodeIndex, NodeIndex)],
) -> Vec<Vec<usize>> {
    let l_max = path_container
        .l_max
        .unwrap_or_else(|| path_container.max_found_l());
    (0..=l_max)
        .map(|l| {
            let max_k = path_container.max_ranks_dim(l);
            // Compute MH^{(s, t)} for each (s, t) in arbitrary order
            let node_pair_wise: Vec<_> = node_pairs
                .iter()
                .par_bridge()
                .map(|node_pair| {
                    let coboundary =
                        path_container.stl_magnitude_coboundary::<Z2, _>(*node_pair, l, 0..=max_k);
                    let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(
                        coboundary.with_trivial_filtration(),
                        0..=max_k,
                    );

                    let mut rank_map = FxHashMap::default();

                    for idx in diagram.essential {
                        let dim = idx.dimension(path_container.n_nodes);
                        if dim > max_k {
                            continue;
                        }
                        *rank_map.entry(dim).or_default() += 1;
                    }

                    rank_map
                })
                .collect();

            let mut rank_vec = vec![0; max_k + 1];
            for rank_map in node_pair_wise {
                for (k, rank_k) in rank_map.iter() {
                    rank_vec[*k] += rank_k;
                }
            }

            rank_vec
        })
        .collect()
}
