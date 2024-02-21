use std::{collections::HashMap, sync::Arc};

use lophat::{
    algorithms::{Decomposition, SerialAlgorithm, SerialDecomposition},
    columns::{Column, VecColumn},
    options::LoPhatOptions,
};
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, chain_group_sizes, compute_homology, homology_idxs},
    path_search::{PathContainer, PathKey, PathQuery},
    utils::rank_table,
    Path,
};

type PyDigraph = Graph<(), (), Directed, u32>;

// TODO:
// 1. Add option for computing homology ranks at a given l and for a given (s, t)
// 2. Implement decomposition cache
// 3. Given (s, t, l) produce homology object which can
//    1. Report rank
//    2. Report representatives
// 4. List found paths

#[pyclass]
struct MagGraph {
    digraph: PyDigraph,
    distance_matrix: Arc<DistanceMatrix<NodeIndex<u32>>>,
    path_container: PathContainer<NodeIndex<u32>>,
    l_max: Option<usize>,
}

#[pyclass]
struct StlHomology {
    decomposition: SerialDecomposition<VecColumn>,
    mag_graph: Py<MagGraph>,
    homology_idxs: HashMap<usize, Vec<usize>>,
    #[pyo3(get)]
    has_reps: bool,
    #[pyo3(get)]
    node_pair: (u32, u32),
    #[pyo3(get)]
    l: usize,
}

// TODO:
// 1. Compute and store pairings upon init
// 2. Support representative lookup

impl StlHomology {
    fn new(
        decomposition: SerialDecomposition<VecColumn>,
        mag_graph: Py<MagGraph>,
        has_reps: bool,
        node_pair: (u32, u32),
        l: usize,
    ) -> Self {
        let homology_idxs = homology_idxs(&decomposition);
        Self {
            decomposition,
            mag_graph,
            homology_idxs,
            has_reps,
            node_pair,
            l,
        }
    }

    // idx is the chain complex index of a homology class (unpaired column)
    fn collect_representative(&self, k: usize, idx: usize, py: Python<'_>) -> Vec<Path<u32>> {
        let mg = self.mag_graph.borrow(py);
        let path_container = &mg.path_container;
        let (s, t) = self.node_pair;
        let sizes = chain_group_sizes(
            path_container,
            (NodeIndex::from(s), NodeIndex::from(t)),
            self.l,
            self.l,
        );

        let dim_offset: usize = sizes[0..k].iter().sum();

        let v_col: Vec<_> = self
            .decomposition
            .get_v_col(idx)
            .unwrap()
            .entries()
            .collect();

        let key = PathKey {
            s: NodeIndex::from(s),
            t: NodeIndex::from(t),
            k,
            l: self.l,
        };

        v_col
            .into_iter()
            .map(|cc_idx| {
                path_container
                    .path_at_index(&key, cc_idx - dim_offset)
                    .unwrap()
            })
            .map(|path| path.into_iter().map(|ix| ix.index() as u32).collect())
            .collect()
    }
}

#[pymethods]
impl StlHomology {
    #[getter]
    fn ranks(&self) -> HashMap<usize, usize> {
        self.homology_idxs
            .iter()
            .map(|(&dim, idxs)| (dim, idxs.len()))
            .collect()
    }

    // This might be quite slow because idx -> Path is slow in HashMap
    #[getter]
    fn representatives(&self, py: Python<'_>) -> Option<HashMap<usize, Vec<Vec<Path<u32>>>>> {
        // TODO: Check that we actually have representatives
        if !self.has_reps {
            return None;
        }

        Some(
            self.homology_idxs
                .iter()
                .map(|(&dim, idxs)| {
                    (
                        dim,
                        idxs.iter()
                            .map(|&i| self.collect_representative(dim, i, py))
                            .collect(),
                    )
                })
                .collect(),
        )
    }
}

impl MagGraph {
    fn build_query<'a>(&'a self, l_max: usize) -> PathQuery<&PyDigraph> {
        PathQuery::build(&self.digraph, self.distance_matrix.clone(), l_max)
    }
}

#[pymethods]
impl MagGraph {
    #[new]
    fn new(edges: Vec<(u32, u32)>) -> Self {
        let digraph = Graph::<(), ()>::from_edges(edges.iter());
        let distance_matrix = parallel_all_pairs_distance(&digraph);
        let distance_matrix = Arc::new(distance_matrix);
        let path_container = PathContainer::new();

        MagGraph {
            digraph,
            distance_matrix,
            path_container,
            l_max: None,
        }
    }

    fn produce_ref(slf: PyRef<'_, Self>) -> Py<Self> {
        slf.into()
    }

    fn populate_paths(&mut self, l_max: usize) {
        if let Some(lm) = self.l_max {
            if lm >= l_max {
                // Nothing to do
                return;
            }
        }
        let query = self.build_query(l_max);
        let path_container = query.run();
        self.path_container = path_container;
        self.l_max = Some(l_max);
    }

    fn rank_generators(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        if self.l_max.is_none() {
            return vec![];
        }

        if let Some(node_pairs) = node_pairs {
            self.path_container.rank_matrix(
                || {
                    node_pairs
                        .iter()
                        .map(|(s, t)| (NodeIndex::from(*s), NodeIndex::from(*t)))
                },
                self.l_max.unwrap(),
            )
        } else {
            self.path_container.rank_matrix(
                || {
                    self.digraph
                        .node_identifiers()
                        .flat_map(|s| self.digraph.node_identifiers().map(move |t| (s, t)))
                },
                self.l_max.unwrap(),
            )
        }
    }

    fn rank_homology(&self) -> Vec<Vec<usize>> {
        if self.l_max.is_none() {
            return vec![];
        }
        all_homology_ranks_default(&self.path_container, self.build_query(self.l_max.unwrap()))
    }

    // TODO: Add flag for representatives
    fn stl_homology(
        slf: PyRef<'_, Self>,
        node_pair: (u32, u32),
        l: usize,
        representatives: Option<bool>,
    ) -> Option<StlHomology> {
        let (s, t) = node_pair;
        let l_max = slf.l_max?;
        if l_max < l {
            return None;
        }
        let query = slf.build_query(l_max);

        let representatives = representatives.unwrap_or(false);

        let mut options = LoPhatOptions::default();
        options.maintain_v = representatives;

        let decomposition = compute_homology::<&PyDigraph, VecColumn, SerialAlgorithm<VecColumn>>(
            &slf.path_container,
            query,
            l,
            (NodeIndex::from(s), NodeIndex::from(t)),
            Some(options),
        );

        Some(StlHomology::new(
            decomposition,
            slf.into(),
            representatives,
            node_pair,
            l,
        ))
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn format_table(table: Vec<Vec<usize>>) -> PyResult<String> {
    Ok(rank_table(table))
}

/// A Python module implemented in Rust.
#[pymodule]
fn gramag(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(format_table, m)?)?;
    m.add_class::<MagGraph>()?;
    Ok(())
}
