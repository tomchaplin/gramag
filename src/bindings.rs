use std::{borrow::Borrow, collections::HashMap, iter, sync::Arc};

use dashmap::DashMap;
use rayon::prelude::*;

use lophat::{algorithms::SerialDecomposition, columns::VecColumn};
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use pyo3::prelude::*;

use crate::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, DirectSum, StlHomology},
    path_search::{PathContainer, PathQuery, StlPathContainer, StoppingCondition},
    utils, MagError, Path, Representative,
};

type PyDigraph = Graph<(), (), Directed, u32>;

// TODO:
// 1. Add API to list found paths
// 2. Cache homology in the MagGraph?
// 3. Add API to convert map of ranks to vec of ranks (in order to format)
// 4. Add API to invoke max_found_l

/// The main container for a (directed, unweighted) graph from which magnitude homology can be computed.
/// Upon construction, all pairwise distances will be computed via Dijkstra's algorithm (in parallel starting from each node)
/// Before computing homology, you should first call the member function |populate_paths|_.
///
/// The graph container is constructed by provided a list of directed edges.
/// A few rules:
///
/// 1. Nodes are labelled by integers
/// 2. Edges are provided as a list of tuples of vertices
/// 3. Isolated vertices are not supported at the moment
///
/// :param edges: The list of directed edges in the graph
/// :type edges: list[tuple[int, int]]
///
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

    // TODO: if path_container.l_max is None then stopping condition was k_max and hence any l query should be fine?
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

    /// Call this to compute the magnitude homology of a finite quasimetric space.
    /// Simply pass in your distance matrix as the first parameter, using ``-1`` to denote an infinite distance.
    /// You will get back a |MagGraph|_ (this is a bit of a hack) from which you can compute magnitude homology.
    ///
    /// :param distance_matrix: The distance matrix of your finite quasimetric space.
    /// :type distance_matrix: list[list[int]]
    /// :return: A |MagGraph|_ object representing the apce.
    /// :rtype: MagGraph
    ///
    #[staticmethod]
    fn from_distance_matrix(distance_matrix: Vec<Vec<isize>>) -> Self {
        // Bit of a hack, we'll set up a line graph on the number of nodes
        // and then input our own distance matrix in lieu of running dijkstra

        let n_nodes = distance_matrix.len();
        for row in &distance_matrix {
            if row.len() != n_nodes {
                panic!("Not given a square matrix");
            }
        }

        let mut digraph = Graph::<(), ()>::new();
        for i in 0..n_nodes {
            let new_index = digraph.add_node(());
            assert!(new_index.index() == i as usize)
        }

        let new_distance_matrix = DashMap::new();
        for i in 0..n_nodes {
            let mut row = HashMap::new();
            for j in 0..n_nodes {
                if distance_matrix[i][j] != -1 {
                    let key = NodeIndex::from(j as u32);
                    row.insert(key, distance_matrix[i][j] as usize);
                }
            }
            let key = NodeIndex::from(i as u32);
            new_distance_matrix.insert(key, row);
        }

        let distance_matrix = Arc::new(DistanceMatrix(new_distance_matrix));

        MagGraph {
            digraph,
            distance_matrix,
            path_container: None,
        }
    }

    #[pyo3(signature=(*,k_max=None, l_max=None))]
    /// You must call this method before attempting to compute any homology.
    ///
    /// This method performs a parallelised breadth-first search to find all paths in the graph, subject to a stopping condition.
    /// You must provide exactly one of ``k_max`` or ``l_max``; typically ``l_max`` is faster but allows you to compute fewer homology groups:
    ///
    /// * If you provide ``l_max`` then you will be able to compute :math:`\mathrm{MH}_{k, l}` whenever :math:`l\leq` ``l_max`` (and hence :math:`k \leq` ``l_max``).
    /// * If you provide ``k_max`` then you will be able to compute :math:`\mathrm{MH}_{k, l}` whenever :math:`k <` ``k_max`` (in which case :math:`l` can be arbirarily large) or :math:`l \leq` ``k_max`` (in which case :math:`k\leq` ``k_max``).
    ///
    /// :param k_max: If provided, finds all :math:`(k, l)`-paths where :math:`k \leq` ``k_max``.
    /// :type k_max: init, optional
    /// :param l_max: If provided, finds all :math:`(k, l)`-paths where :math:`l \leq` ``l_max``.
    /// :type l_max: init, optional
    /// :raise TypeError: Raises an exception unless exactly one of ``k_max`` and ``l_max`` is provided.
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

    /// Computes all possible magnitude chain group ranks :math:`\operatorname{rank}(\mathrm{MC}_{k, l}^\mathcal{P})`, summed over a list of node pairs :math:`\mathcal{P}`.
    ///
    /// :param node_pairs: The list of node pairs :math:`\mathcal{P}` over which to compute and sum the ranks. Defaults to all possible node pairs.
    /// :type node_pairs: list[tuple[int, int]], optional
    /// :return: The ranks of the known chain groups groups, where :math:`\operatorname{rank}(\mathrm{MC}_{k,l}^\mathcal{P})` is stored in ``output[l][k]``.
    /// :rtype: list[list[int]]
    fn rank_generators(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        let node_pairs = self.process_node_pairs_options(node_pairs);
        if let Some(container) = self.path_container.as_ref() {
            container.rank_matrix(|| node_pairs.iter().copied())
        } else {
            vec![]
        }
    }

    /// Computes all possible magnitude homology ranks :math:`\operatorname{rank}(\mathrm{MH}_{k, l}^\mathcal{P})`, summed over a list of node pairs :math:`\mathcal{P}`.
    /// The homology for each node pair :math:`(s, t)\in\mathcal{P}` is computed in parallel.
    ///
    /// :param node_pairs: The list of node pairs :math:`\mathcal{P}` over which to compute and sum magnitude homology. Defaults to all possible node pairs.
    /// :type node_pairs: list[tuple[int, int]], optional
    /// :return: The ranks of the known homology groups, where :math:`\operatorname{rank}(\mathrm{MH}_{k,l}^\mathcal{P})` is stored in ``output[l][k]``.
    /// :rtype: list[List[int]]
    fn rank_homology(&self, node_pairs: Option<Vec<(u32, u32)>>) -> Vec<Vec<usize>> {
        if let Some(container) = self.path_container.as_ref() {
            all_homology_ranks_default(container, &self.process_node_pairs_options(node_pairs))
        } else {
            vec![]
        }
    }

    /// Computes magnitude homology :math:`\mathrm{MH}_{k, l}^{(s, t)}` for a fixed node pair :math:`(s, t)`, a fixed length :math:`l` and all possible homological degrees :math:`k`.
    ///
    /// :param node_pair: The node pair :math:`(s, t)`, as described above.
    /// :type node_pair: tuple[int, int]
    /// :param l: The fixed length :math:`l`, as described above.
    /// :type l: int
    /// :param representatives: Whether to compute representatives of each homology group. Defaults to ``False``.
    /// :type representatives: bool, optional
    /// :return: The requested homology groups.
    /// :rtype: StlHomology
    /// :raises TypeError: If no paths have been found with length :math:`\geq l`, an error is raised.
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

    /// Computes magnitude homology :math:`\mathrm{MH}_{k, l}^{\mathcal{P}}` for a fixed length :math:`l` and all possible homological degrees :math:`k`, summed over a list of node pairs :math:`\mathcal{P}`.
    /// The homology for each node pair :math:`(s, t)\in\mathcal{P}` is computed in parallel.
    ///
    /// :param l: The fixed length :math:`l`, as described above.
    /// :type l: int
    /// :param representatives: Whether to compute representatives of each homology group. Defaults to ``False``.
    /// :type representatives: bool, optional
    /// :param node_pairs: The list of node pairs :math:`\mathcal{P}` over which to compute and sum magnitude homology. Defaults to all possible node pairs.
    /// :type node_pairs: list[tuple[int, int]], optional
    /// :return: The requested homology groups.
    /// :rtype: DirectSum
    /// :raises TypeError: If no paths have been found with length :math:`\geq l`, an error is raised.
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

/// ``StlHomology`` objects represent the homology groups :math:`\mathrm{MH}_{k, l}^{(s, t)}` for some fixed node pair :math:`(s, t)\in V \times V` and a fixed length :math:`l`.
/// The range of the homological degree :math:`k` varies depending on the last call to |populate_paths|.
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
    /// Retrieves the ranks of the homology groups.
    ///
    /// :return: A dictionary where ``output[k]`` stores the rank :math:`\operatorname{rank}(\mathrm{MH}_{k, l}^{(s, t)})`.
    /// :rtype: dict[usize, usize]
    #[getter]
    fn get_ranks(&self) -> HashMap<usize, usize> {
        self.0.ranks()
    }

    /// Retrieves a (non-unique) set of representatives for the homology groups.
    /// Each representative is a :math:`\mathbb{Z}_2`-linear combination of paths (i.e. a set of paths) and is thus represented by a ``list[list[usize]]`` where each inner ``list[usize]`` represents a path.
    ///
    /// :return: A dictionary where ``output[k]`` stores a list of representatives for :math:`\mathrm{MH}_{k, l}^{(s, t)}` with ``len(output[k])`` :math:`=\operatorname{rank}(\mathrm{MH}_{k, l}^{(s, t)})`.
    /// :rtype: dict[usize, list[list[list[usize]]]]
    /// :raise TypeError: If this homology was computed with ``representatives = False`` then an error is raised.
    #[getter]
    fn get_representatives(&self) -> Result<HashMap<usize, Vec<Representative<u32>>>, MagError> {
        Ok(convert_representatives(self.0.representatives()?))
    }
}

/// ``DirectSum`` objects represent a direct sum of homology groups :math:`\oplus_{((s, t), l)\in I}\mathrm{MH}_{k, l}^{(s, t)}` for some indexing set :math:`I\subseteq (V\times V) \times \mathbb{N}`.
/// The range of the homological degree :math:`k` varies depending on the last call to |populate_paths| before each summand was created.
/// If the range varies amongst the summands, we restrict to the smallest common sub-range.
///
/// :param edges: The list of summands to enter into the direct sum. Defaults to the empty sum.
/// :type edges: list[StlHomology], optional
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

    /// Retrieves the ranks of the homology groups.
    ///
    /// :return: A dictionary where ``output[k]`` stores the rank :math:`\operatorname{rank}(\oplus_{((s, t), l)\in I}\mathrm{MH}_{k, l}^{(s, t)})`.
    /// :rtype: dict[usize, usize]
    #[getter]
    fn get_ranks(&self) -> HashMap<usize, usize> {
        self.0.ranks()
    }

    /// Retrieves a (non-unique) set of representatives for the homology groups.
    /// Each representative is a :math:`\mathbb{Z}_2`-linear combination of paths (i.e. a set of paths) and is thus represented by a ``list[list[usize]]`` where each inner ``list[usize]`` represents a path.
    ///
    /// :return: A dictionary where ``output[k]`` stores a list of representatives for :math:`\oplus_{((s, t), l)\in I}\mathrm{MH}_{k, l}^{\mathcal{P}}` with ``len(output[k])`` :math:`=\operatorname{rank}(\oplus_{((s, t), l)\in I}\mathrm{MH}_{k, l}^{\mathcal{P}})`.
    /// :rtype: dict[usize, list[list[list[usize]]]]
    /// :raise TypeError: If this homology was computed with ``representatives = False`` then an error is raised.
    #[getter]
    fn get_representatives(&self) -> Result<HashMap<usize, Vec<Representative<u32>>>, MagError> {
        Ok(convert_representatives(self.0.representatives()?))
    }

    /// Adds another summand to the direct sum, mutating the original sum.
    ///
    /// :param summand: The summand to be added to the sum
    /// :type summand: StlHomology
    /// :returns: Whether a summand with the same :math:`((s, t), l)`-index already existed in the sum and was just replaced by this summand.
    /// :rtype: bool
    fn add(&mut self, summand: &PyStlHomology) -> bool {
        let stl_key = (summand.0.stl_paths.node_pair, summand.0.stl_paths.l);
        let hom = Arc::clone(&summand.0);
        self.0.add(stl_key, hom).is_some()
    }
}

/// Formats a table of numbers, such as those outputted by |rank_homology|_ or |rank_generators|_.
///
/// :param table: The table to format.
/// :type table: list[list[Int]]
/// :param above_diagonal: The string to display for the ranks of all groups above the diagonal (which are necessarily 0). Defaults to the empty string.
/// :type above_diagonal: str, optional
/// :param unknown: The string to disply when the rank of a group is unknown. Defaults to ``"?"``.
/// :type unknown: str, optional
/// :param zero: The string to disply when the rank of a group is zero. Defaults to ``"."``.
/// :type zero: str, optional
/// :return: The formatted table.
/// :rtype: str
#[pyfunction]
fn format_rank_table(
    table: Vec<Vec<usize>>,
    above_diagonal: Option<String>,
    unknown: Option<String>,
    zero: Option<String>,
) -> PyResult<String> {
    let options = (above_diagonal, unknown, zero).into();
    Ok(utils::format_rank_table(table, options))
}

/// :return: The current version of ``gramag``.
/// :rtype: str
#[pyfunction]
fn version() -> String {
    format!("{}", env!("CARGO_PKG_VERSION"))
}

/// This Python package provides bindings to the Rust library ``gramag``, which computes magnitude homology of finite, directed graphs.
/// Usage of the package usually follows the following pattern:
///
/// 1. Construct a |MagGraph|_ from your graph.
/// 2. Search for paths to populate the magnitude chain groups, by calling |populate_paths|_ with an appropriate stopping condition.
/// 3. Report the number of paths found via |rank_generators|_.
/// 4. Compute the ranks of all of the possible homology groups via |rank_homology|_.
/// 5. Investigate a particular homology group, by calling |l_homology|_ with ``representatives=True``.
///
/// A simple example script illustrating this workflow is show below.
///
/// .. |MagGraph| replace:: ``MagGraph``
/// .. _MagGraph: #gramag.MagGraph
/// .. |populate_paths| replace:: ``populate_paths``
/// .. _populate_paths: #gramag.MagGraph.populate_paths
/// .. |rank_generators| replace:: ``rank_generators``
/// .. _rank_generators: #gramag.MagGraph.rank_generators
/// .. |rank_homology| replace:: ``rank_homology``
/// .. _rank_homology: #gramag.MagGraph.rank_homology
/// .. |l_homology| replace:: ``l_homology``
/// .. _l_homology: #gramag.MagGraph.l_homology
#[pymodule]
fn gramag(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(format_rank_table, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<MagGraph>()?;
    m.add_class::<PyStlHomology>()?;
    m.add_class::<PyDirectSum>()?;
    Ok(())
}
