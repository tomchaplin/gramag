#![allow(dead_code)]
use std::sync::Arc;

use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::all_homology_ranks_default,
    path_search::{PathContainer, PathQuery},
    utils::rank_table,
};

type PyDigraph = Graph<(), (), Directed, u32>;

#[pyclass]
struct MagnitudeHomology {
    digraph: PyDigraph,
    distance_matrix: Arc<DistanceMatrix<NodeIndex<u32>>>,
    path_container: PathContainer<NodeIndex<u32>>,
    l_max: usize,
}

impl MagnitudeHomology {
    fn build_query<'a>(&'a self) -> PathQuery<&PyDigraph> {
        PathQuery::build(&self.digraph, self.distance_matrix.clone(), self.l_max)
    }
}

#[pymethods]
impl MagnitudeHomology {
    #[new]
    fn new(edges: Vec<(u32, u32)>, l_max: usize) -> Self {
        let digraph = Graph::<(), ()>::from_edges(edges.iter());
        let distance_matrix = parallel_all_pairs_distance(&digraph);
        let distance_matrix = Arc::new(distance_matrix);

        let query = PathQuery::build(&digraph, distance_matrix.clone(), l_max);
        let path_container = query.run();

        MagnitudeHomology {
            digraph,
            distance_matrix,
            path_container,
            l_max,
        }
    }

    fn rank_generators(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        if let Some(node_pairs) = node_pairs {
            self.path_container.rank_matrix(
                || {
                    node_pairs
                        .iter()
                        .map(|(s, t)| (NodeIndex::from(*s), NodeIndex::from(*t)))
                },
                self.l_max,
            )
        } else {
            self.path_container.rank_matrix(
                || {
                    self.digraph
                        .node_identifiers()
                        .flat_map(|s| self.digraph.node_identifiers().map(move |t| (s, t)))
                },
                self.l_max,
            )
        }
    }

    fn rank_homology(&self) -> Vec<Vec<usize>> {
        all_homology_ranks_default(&self.path_container, self.build_query())
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
    m.add_class::<MagnitudeHomology>()?;
    Ok(())
}
