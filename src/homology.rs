use std::{collections::HashMap, iter};

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo, SerialAlgorithm},
    columns::{Column, VecColumn},
};
use petgraph::visit::{GraphRef, IntoNodeIdentifiers};

use rayon::prelude::*;

use crate::path_search::{PathContainer, PathKey, PathQuery, SensibleNode};

// TODO: Add options for
// 1. Anti-transpose
// 2. Representatives

pub fn chain_group_sizes<NodeId: SensibleNode>(
    path_container: &PathContainer<NodeId>,
    node_pair: (NodeId, NodeId),
    l: usize,
    k_max: usize,
) -> Vec<usize> {
    let (s, t) = node_pair;
    (0..=k_max)
        .map(|k| {
            let key = PathKey { s, t, k, l };
            path_container.num_paths(&key)
        })
        .collect()
}

pub fn compute_homology<G, C, Algo>(
    path_container: &PathContainer<G::NodeId>,
    query: PathQuery<G>,
    l: usize,
    node_pair: (G::NodeId, G::NodeId),
    options: Option<Algo::Options>,
) -> Algo::Decomposition
where
    G: GraphRef,
    G::NodeId: SensibleNode,
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    let (s, t) = node_pair;

    // Setup algorithm to receive entries

    let mut algo = Algo::init(options);
    let sizes = chain_group_sizes(&path_container, node_pair, l, query.l_max);
    let empty_cols =
        (0..=query.l_max).flat_map(|k| (0..sizes[k]).map(move |_i| C::new_with_dimension(k)));
    algo = algo.add_cols(empty_cols);

    // Compute offsets as cumulative sum of sizes of each dimension

    for k in 0..=query.l_max {
        if k == 0 {
            // k= 0 => no boundary
            continue;
        }

        let k_offset: usize = sizes[0..k].iter().sum();
        let k_minus_1_offset: usize = sizes[0..(k - 1)].iter().sum();
        let key = PathKey { s, t, k, l };
        let lower_key = PathKey { s, t, k: k - 1, l };

        let paths_with_key = path_container.paths.get(&key);

        if let Some(paths_with_key) = paths_with_key {
            for entry in paths_with_key.value() {
                // Get path and its index in the chain complex
                let path = entry.key();
                let idx = entry.value();
                for i in 1..(path.len() - 1) {
                    // For each interior vertex, check if removing changes length
                    let a = path[i - 1];
                    let mid = path[i];
                    let b = path[i + 1];

                    let d1 = query.d.distance(&a, &mid) + query.d.distance(&mid, &b);
                    let d2 = query.d.distance(&a, &b);
                    if d1 != d2 {
                        continue;
                    }

                    // Length is same so its in the boundary
                    // Just need to figure out which idx the path is in the chain complex

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

    // Run the algorithm
    algo.decompose()
}

pub fn homology_ranks<C, Algo>(decomp: &Algo::Decomposition) -> Vec<usize>
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
                ranks.push(0 as usize);
            }
        }
        // Should never overflow to 0 because all of kernels appears first
        // then all of the boundaries appear
        ranks[degree] = ((ranks[degree] as isize) + amount) as usize;
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

// Returns a map indexed by dimension i
// map[i] is a list of column-indices wherein non-dead homology is created

pub fn homology_idxs<C, Algo>(decomposition: &Algo) -> HashMap<usize, Vec<usize>>
where
    C: Column,
    Algo: Decomposition<C>,
{
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
    let diagram = decomposition.diagram();
    for idx in diagram.unpaired {
        let dim = decomposition.get_r_col(idx).dimension();
        map.entry(dim).or_default().push(idx);
    }
    map
}

// TODO: Convert this into an actual reduce operation
//       Ideally one that works with parallel iterators

pub fn reduce_homology_ranks(hom_ranks: impl Iterator<Item = Vec<usize>>) -> Vec<usize> {
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
                    let homology = compute_homology::<G, VecColumn, SerialAlgorithm<VecColumn>>(
                        path_container,
                        query.clone(),
                        l,
                        node_pair.clone(),
                        None,
                    );
                    homology_ranks::<VecColumn, SerialAlgorithm<VecColumn>>(&homology)
                })
                .collect();
            // Sum across (s, t)
            reduce_homology_ranks(node_pair_wise.into_iter())
        })
        .collect()
}
