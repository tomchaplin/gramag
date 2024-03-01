use std::{collections::HashMap, sync::Arc};

use rayon::prelude::*;

use lophat::{algorithms::SerialDecomposition, columns::VecColumn};
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, DirectSum, StlHomology},
    path_search::{PathContainer, PathQuery, StlPathContainer},
    utils::rank_table,
    Path,
};

type PyDigraph = Graph<(), (), Directed, u32>;

// TODO:
// 1. Add API to list found paths
// 2. Cache homology in the MagGraph?

#[pyclass]
struct MagGraph {
    digraph: PyDigraph,
    distance_matrix: Arc<DistanceMatrix<NodeIndex<u32>>>,
    path_container: Arc<PathContainer<NodeIndex<u32>>>,
    l_max: Option<usize>,
}

impl MagGraph {
    fn build_query<'a>(&'a self, l_max: usize) -> PathQuery<&PyDigraph> {
        PathQuery::build(&self.digraph, self.distance_matrix.clone(), l_max)
    }

    fn inner_compute_stl_homology(
        &self,
        node_pair: (NodeIndex<u32>, NodeIndex<u32>),
        l: usize,
        representatives: bool,
    ) -> StlHomology<
        Arc<PathContainer<NodeIndex>>,
        NodeIndex,
        VecColumn,
        SerialDecomposition<VecColumn>,
    > {
        StlPathContainer::new(self.path_container.clone(), node_pair, l)
            .serial_homology(representatives)
    }
}

#[pymethods]
impl MagGraph {
    #[new]
    fn new(edges: Vec<(u32, u32)>) -> Self {
        let digraph = Graph::<(), ()>::from_edges(edges.iter());
        let distance_matrix = parallel_all_pairs_distance(&digraph);
        let distance_matrix = Arc::new(distance_matrix);
        let path_container = Arc::new(PathContainer::new(distance_matrix.clone()));

        MagGraph {
            digraph,
            distance_matrix,
            path_container,
            l_max: None,
        }
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
        self.path_container = Arc::new(path_container);
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

    // TODO: Return an error instead
    fn stl_homology(
        &self,
        node_pair: (u32, u32),
        l: usize,
        representatives: Option<bool>,
    ) -> Option<PyStlHomology> {
        let (s, t) = node_pair;
        if self.l_max? < l {
            return None;
        }

        let homology = self.inner_compute_stl_homology(
            (NodeIndex::from(s), NodeIndex::from(t)),
            l,
            representatives.unwrap_or(false),
        );

        Some(PyStlHomology(Arc::new(homology)))
    }

    // TODO: New method - allow arbitrary (s, t) list
    fn l_homology(&self, l: usize, representatives: Option<bool>) -> Option<PyDirectSum> {
        if self.l_max? < l {
            return None;
        }
        let representatives = representatives.unwrap_or(false);
        let stl_homologies: Vec<_> = self
            .digraph
            .node_identifiers()
            .flat_map(|s| self.digraph.node_identifiers().map(move |t| (s, t)))
            .par_bridge()
            .map(|node_pair| {
                (
                    (node_pair, l),
                    Arc::new(self.inner_compute_stl_homology(node_pair, l, representatives)),
                )
            })
            .collect();
        Some(PyDirectSum(DirectSum::new(stl_homologies.into_iter())))
    }
}

#[pyclass]
struct PyStlHomology(
    Arc<
        StlHomology<
            Arc<PathContainer<NodeIndex<u32>>>,
            NodeIndex<u32>,
            VecColumn,
            SerialDecomposition<VecColumn>,
        >,
    >,
);

// This is awful and I hate it
fn convert_representatives(
    reps: HashMap<usize, Vec<Vec<Path<NodeIndex<u32>>>>>,
) -> HashMap<usize, Vec<Vec<Path<u32>>>> {
    let convert_path_to_u32 =
        |path: Path<NodeIndex<u32>>| path.into_iter().map(|node| node.index() as u32).collect();

    let convert_all_reps = |reps: Vec<Vec<Path<NodeIndex<u32>>>>| {
        reps.into_iter()
            .map(|rep| rep.into_iter().map(convert_path_to_u32).collect())
            .collect()
    };
    reps.into_iter()
        .map(|(dim, reps)| (dim, convert_all_reps(reps)))
        .collect()
}

#[pymethods]
impl PyStlHomology {
    #[getter]
    fn get_ranks(&self) -> HashMap<usize, usize> {
        self.0.ranks()
    }

    #[getter]
    fn get_representatives(&self) -> Option<HashMap<usize, Vec<Vec<Path<u32>>>>> {
        Some(convert_representatives(self.0.representatives()?))
    }
}

#[pyclass]
struct PyDirectSum(
    DirectSum<
        Arc<PathContainer<NodeIndex<u32>>>,
        NodeIndex<u32>,
        VecColumn,
        SerialDecomposition<VecColumn>,
    >,
);

#[pymethods]
impl PyDirectSum {
    #[getter]
    fn get_ranks(&self) -> HashMap<usize, usize> {
        self.0.ranks()
    }

    #[getter]
    fn get_representatives(&self) -> Option<HashMap<usize, Vec<Vec<Path<u32>>>>> {
        Some(convert_representatives(self.0.representatives()?))
    }
}

// TODO:
// 1. add_summand
// 2. Init from a vector of Py<StlHomology>

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
