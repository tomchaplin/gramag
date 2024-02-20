use std::iter;

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo, SerialAlgorithm},
    columns::{Column, VecColumn},
};
use petgraph::visit::{GraphRef, IntoNodeIdentifiers};

use rayon::prelude::*;

use crate::path_search::{PathContainer, PathKey, PathQuery, SensibleNode};

pub fn compute_homology<G, C, Algo>(
    path_container: &PathContainer<G>,
    query: PathQuery<G>,
    l: usize,
    node_pair: (G::NodeId, G::NodeId),
) -> Algo::Decomposition
where
    G: GraphRef,
    G::NodeId: SensibleNode,
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    // Setup algorithm to receive entries
    let (s, t) = node_pair;
    let mut algo = Algo::init(None);
    let mut sizes = vec![];
    let empty_cols = (0..=query.l_max).flat_map(|k| {
        let key = PathKey { s, t, k, l };
        let num_k_cols = path_container.num_paths(&key);
        sizes.push(num_k_cols.clone());
        (0..num_k_cols).map(move |_i| C::new_with_dimension(k))
    });
    algo = algo.add_cols(empty_cols);

    // Compute offsets as cumulative sum of sizes of each dimension

    for k in 0..=query.l_max {
        let k_offset: usize = sizes[0..k].iter().sum();
        let k_minus_1_offset: usize = if k > 0 {
            sizes[0..(k - 1)].iter().sum()
        } else {
            0
        };
        let key = PathKey { s, t, k, l };
        let lower_key = PathKey { s, t, k: k - 1, l }; // This will overflow for k=0 but then the boundary is empty
        let paths_with_key = path_container.paths.get(&key);
        if let Some(paths_with_key) = paths_with_key {
            for entry in paths_with_key.value() {
                let path = entry.key();
                let idx = entry.value();
                // TODO:
                // 1. Compute boundary of path
                // 2. Add boundary to column total_idx
                // Q: We should be able to do this in parallel, is it worth writing our own wrapper?
                for i in 1..(path.len() - 1) {
                    let a = path[i - 1];
                    let mid = path[i];
                    let b = path[i + 1];

                    let d1 = query.d.distance(&a, &mid) + query.d.distance(&mid, &b);
                    let d2 = query.d.distance(&a, &b);
                    if d1 != d2 {
                        continue;
                    }

                    let mut bdry_path = path.clone();
                    bdry_path.remove(i);

                    let bdry_path_idx = path_container.index_of(&lower_key, &bdry_path);

                    algo = algo.add_entries(iter::once((
                        bdry_path_idx + k_minus_1_offset,
                        idx + k_offset,
                    )));
                }
            }
        }
    }
    algo.decompose()
}

pub fn homology_ranks<C, Algo>(decomp: &Algo::Decomposition) -> Vec<isize>
where
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    let mut ranks = vec![];

    let mut inc = |degree: usize, amount: isize| {
        let n_ranks = ranks.len();
        let needed_ranks = degree + 1;
        if needed_ranks > n_ranks {
            let new_ranks = needed_ranks - n_ranks;
            for _ in 0..new_ranks {
                ranks.push(0);
            }
        }
        ranks[degree] += amount
    };

    for idx in 0..decomp.n_cols() {
        let r_col = decomp.get_r_col(idx);
        let dim = r_col.dimension();
        let is_bdry = r_col.pivot().is_some();
        if is_bdry {
            inc(dim - 1, -1);
        } else {
            inc(dim, 1);
        }
    }

    ranks
}

// TODO: Convert this into an actual reduce operation
//       Ideally one that works with parallel iterators

pub fn reduce_homology_ranks(hom_ranks: impl Iterator<Item = Vec<isize>>) -> Vec<usize> {
    let mut ranks = vec![];

    let mut inc = |degree: usize, amount: usize| {
        let n_ranks = ranks.len();
        let needed_ranks = degree + 1;
        if needed_ranks > n_ranks {
            let new_ranks = needed_ranks - n_ranks;
            for _ in 0..new_ranks {
                ranks.push(0);
            }
        }
        ranks[degree] += amount
    };

    for rank_list in hom_ranks.into_iter() {
        for (degree, rank) in rank_list.into_iter().enumerate() {
            inc(degree, rank as usize);
        }
    }

    ranks
}

pub fn all_homology_ranks_default<G>(
    path_container: &PathContainer<G>,
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
                    let homology = compute_homology::<G, VecColumn, SerialAlgorithm<VecColumn>>(
                        path_container,
                        query.clone(),
                        l,
                        node_pair.clone(),
                    );
                    homology_ranks::<VecColumn, SerialAlgorithm<VecColumn>>(&homology)
                })
                .collect();
            // Sum across (s, t)
            reduce_homology_ranks(node_pair_wise.into_iter())
        })
        .collect()
}
