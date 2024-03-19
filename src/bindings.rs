use std::{borrow::Borrow, collections::HashMap, iter, sync::Arc};

use rayon::prelude::*;

use lophat::{algorithms::SerialDecomposition, columns::VecColumn};
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, DirectSum, StlHomology},
    path_search::{PathContainer, PathQuery, StlPathContainer, StoppingCondition},
    utils::rank_table,
    MagError, Path, Representative,
};

type PyDigraph = Graph<(), (), Directed, u32>;

// TODO:
// 1. Add API to list found paths
// 2. Cache homology in the MagGraph?
// 3. Add API to convert map of ranks to vec of ranks (in order to format)

#[pyclass]
struct MagGraph {
    digraph: PyDigraph,
    distance_matrix: Arc<DistanceMatrix<NodeIndex<u32>>>,
    path_container: Option<Arc<PathContainer<NodeIndex<u32>>>>,
}

fn xor(a: bool, b: bool) -> bool {
    (a || b) && (!(a && b))
}

impl MagGraph {
    fn build_query(
        &self,
        k_max: Option<usize>,
        l_max: Option<usize>,
    ) -> Result<PathQuery<&PyDigraph>, MagError> {
        if !xor(k_max.is_some(), l_max.is_some()) {
            return Err(MagError::BadArguments(
                "Provide exactly one of the arguments k_max and l_max as a stopping condition."
                    .to_string(),
            ));
        }
        let stopping_condition = if let Some(k_max) = k_max {
            StoppingCondition::KMax(k_max)
        } else if let Some(l_max) = l_max {
            StoppingCondition::LMax(l_max)
        } else {
            unreachable!()
        };
        Ok(PathQuery::build(
            &self.digraph,
            self.distance_matrix.clone(),
            stopping_condition,
        ))
    }

    // Assumption: check_l has already been called!
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
        StlPathContainer::new(self.path_container.as_ref().unwrap().clone(), node_pair, l)
            .serial_homology(representatives)
    }

    fn check_l(&self, l: usize) -> Result<(), MagError> {
        let path_container = self
            .path_container
            .as_ref()
            .ok_or(MagError::InsufficientLMax(l, None))?;

        let l_max = path_container
            .l_max
            .unwrap_or_else(|| path_container.max_found_l());

        if l_max >= l {
            Ok(())
        } else {
            Err(MagError::InsufficientLMax(l, Some(l_max)))
        }
    }

    fn process_node_pairs_options(
        &self,
        node_pairs: Option<Vec<(u32, u32)>>,
    ) -> Vec<(NodeIndex<u32>, NodeIndex<u32>)> {
        node_pairs
            .map(|u32_node_pairs| {
                u32_node_pairs
                    .into_iter()
                    .map(|(s, t)| (NodeIndex::from(s), NodeIndex::from(t)))
                    .collect()
            })
            .unwrap_or_else(|| {
                self.digraph
                    .node_identifiers()
                    .flat_map(|s| self.digraph.node_identifiers().map(move |t| (s, t)))
                    .collect()
            })
    }
}

#[pymethods]
impl MagGraph {
    #[new]
    fn new(edges: Vec<(u32, u32)>) -> Self {
        let digraph = Graph::<(), ()>::from_edges(edges.iter());
        let distance_matrix = parallel_all_pairs_distance(&digraph);
        let distance_matrix = Arc::new(distance_matrix);

        MagGraph {
            digraph,
            distance_matrix,
            path_container: None,
        }
    }

    #[pyo3(signature=(*,k_max=None, l_max=None))]
    fn populate_paths(
        &mut self,
        k_max: Option<usize>,
        l_max: Option<usize>,
    ) -> Result<(), MagError> {
        let query = self.build_query(k_max, l_max)?;
        let path_container = query.run();
        self.path_container = Some(Arc::new(path_container));
        Ok(())
    }

    fn rank_generators(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        let node_pairs = self.process_node_pairs_options(node_pairs);
        if let Some(container) = self.path_container.as_ref() {
            container.rank_matrix(|| node_pairs.iter().copied())
        } else {
            vec![]
        }
    }

    fn rank_homology(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        if let Some(container) = self.path_container.as_ref() {
            all_homology_ranks_default(container, &self.process_node_pairs_options(node_pairs))
        } else {
            vec![]
        }
    }

    fn stl_homology(
        &self,
        node_pair: (u32, u32),
        l: usize,
        representatives: Option<bool>,
    ) -> Result<PyStlHomology, MagError> {
        let (s, t) = node_pair;
        self.check_l(l)?;

        let homology = self.inner_compute_stl_homology(
            (NodeIndex::from(s), NodeIndex::from(t)),
            l,
            representatives.unwrap_or(false),
        );

        Ok(PyStlHomology(Arc::new(homology)))
    }

    fn l_homology(
        &self,
        l: usize,
        representatives: Option<bool>,
        node_pairs: Option<Vec<(u32, u32)>>,
    ) -> Result<PyDirectSum, MagError> {
        self.check_l(l)?;
        let representatives = representatives.unwrap_or(false);
        let stl_homologies = self
            .process_node_pairs_options(node_pairs)
            .into_par_iter()
            .map(|node_pair| {
                (
                    (node_pair, l),
                    Arc::new(self.inner_compute_stl_homology(node_pair, l, representatives)),
                )
            })
            .collect::<Vec<_>>()
            .into_iter();

        Ok(PyDirectSum(DirectSum::new(stl_homologies)))
    }
}

#[pyclass(name = "StlHomology")]
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
    fn get_representatives(&self) -> Result<HashMap<usize, Vec<Representative<u32>>>, MagError> {
        Ok(convert_representatives(self.0.representatives()?))
    }
}

#[pyclass(name = "DirectSum")]
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
    #[new]
    fn new(summands: Option<Vec<PyRef<PyStlHomology>>>) -> Self {
        let mut ds = Self(DirectSum::new(iter::empty()));
        for hom in summands.unwrap_or_default() {
            ds.add(hom.borrow());
        }
        ds
    }

    #[getter]
    fn get_ranks(&self) -> HashMap<usize, usize> {
        self.0.ranks()
    }

    #[getter]
    fn get_representatives(&self) -> Result<HashMap<usize, Vec<Representative<u32>>>, MagError> {
        Ok(convert_representatives(self.0.representatives()?))
    }

    // Returns whether it replaced an old summand
    fn add(&mut self, summand: &PyStlHomology) -> bool {
        let stl_key = (summand.0.stl_paths.node_pair, summand.0.stl_paths.l);
        let hom = Arc::clone(&summand.0);
        self.0.add(stl_key, hom).is_some()
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
    m.add_class::<PyStlHomology>()?;
    m.add_class::<PyDirectSum>()?;
    Ok(())
}
