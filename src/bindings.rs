use std::sync::Arc;

use lophat::{
    algorithms::{SerialAlgorithm, SerialDecomposition},
    columns::VecColumn,
};
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, compute_homology, homology_ranks},
    path_search::{PathContainer, PathQuery},
    utils::rank_table,
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
struct MagnitudeHomology {
    decomposition: SerialDecomposition<VecColumn>,
    mag_graph: Py<MagGraph>,
    #[pyo3(get)]
    ranks: Vec<usize>,
}

// TODO:
// 1. Compute and store pairings upon init
// 2. Support representative lookup

impl MagnitudeHomology {
    fn new(decomposition: SerialDecomposition<VecColumn>, mag_graph: Py<MagGraph>) -> Self {
        let ranks = homology_ranks::<VecColumn, SerialAlgorithm<VecColumn>>(&decomposition);
        Self {
            decomposition,
            mag_graph,
            ranks,
        }
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

    fn stl_homology(
        slf: PyRef<'_, Self>,
        node_pair: (u32, u32),
        l: usize,
    ) -> Option<MagnitudeHomology> {
        let (s, t) = node_pair;
        let l_max = slf.l_max?;
        if l_max < l {
            return None;
        }
        let query = slf.build_query(l_max);

        let decomposition = compute_homology::<&PyDigraph, VecColumn, SerialAlgorithm<VecColumn>>(
            &slf.path_container,
            query,
            l,
            (NodeIndex::from(s), NodeIndex::from(t)),
        );

        Some(MagnitudeHomology::new(decomposition, slf.into()))
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
