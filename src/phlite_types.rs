use std::{iter, marker::PhantomData, ops::Deref, sync::Arc};

use petgraph::graph::NodeIndex;
use phlite::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{
        adaptors::{MatrixWithBasis, ReverseMatrix, WithTrivialFiltration},
        BasisElement, ColBasis, MatrixOracle, SplitByDimension,
    },
    reduction::ClearedReductionMatrix,
};
use rustc_hash::FxHashMap;

use crate::{
    distances::DistanceMatrix,
    path_search::{PathKey, SensibleNode},
    phlite_homology::PhliteStlHomology,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathIndex(u64);

impl BasisElement for PathIndex {}

// TODO: Decide whether we should just stick to u64 everywher?

impl PathIndex {
    /// Produces from least significant to most significant
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        let mut vec = vec![];
        let mut working = self.0;
        while working > 0 {
            let remainder = working.rem_euclid(n_points as u64 + 1) as usize;
            vec.push(remainder - 1);
            working = working.div_euclid(n_points as u64 + 1);
        }
        vec
    }

    /// NOTE: The indicies should be provided in ascending order!
    pub fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
        let inner = indices
            .enumerate()
            .map(|(i, coeff)| (n_points as u64 + 1).pow(i as u32) * (coeff as u64 + 1))
            .sum();
        if inner > 0 {
            Some(Self(inner))
        } else {
            None
        }
    }

    pub fn dimension(self, n_points: usize) -> usize {
        self.to_vec(n_points).len() - 1
    }

    /// Produces the path obtained by appending vertex to self
    /// You must provide `current_length` which is the current length of the path (#vertices - 1)
    /// No guarantees if `current_length` is incorrect
    pub fn push_unchecked(self, vertex: usize, n_points: usize, current_length: u32) -> Self {
        Self(self.0 + (vertex as u64 + 1) * (n_points as u64 + 1).pow(current_length + 1))
    }

    pub fn initial_vertex(self, n_points: usize) -> usize {
        self.0.rem_euclid(n_points as u64 + 1) as usize
    }
}

pub struct MagnitudeBoundary<'a, CF>
where
    CF: NonZeroCoefficient,
{
    d: &'a DistanceMatrix<NodeIndex>,
    n_nodes: usize,
    phantom: PhantomData<CF>,
}

impl<'a, CF> MagnitudeBoundary<'a, CF>
where
    CF: NonZeroCoefficient,
{
    pub fn new(n_nodes: usize, d: &'a DistanceMatrix<NodeIndex>) -> Self {
        Self {
            d,
            n_nodes,
            phantom: PhantomData,
        }
    }
}

impl<'a, CF> MatrixOracle for MagnitudeBoundary<'a, CF>
where
    CF: NonZeroCoefficient,
{
    type CoefficientField = CF;

    type ColT = PathIndex;

    type RowT = PathIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let path = col.to_vec(self.n_nodes);

        let k = path.len() - 1;

        (1..k).filter_map(move |i| {
            // Path without vertex i appears in boundary
            // iff removing doesn't change length
            let a = NodeIndex::new((&path)[i - 1]);
            let mid = NodeIndex::new(path[i]);
            let b = NodeIndex::new(path[i + 1]);
            if self.d.distance(&a, &mid) + self.d.distance(&mid, &b) != self.d.distance(&a, &b) {
                return None;
            };
            let bdry_path_idx = PathIndex::from_indices(
                path.iter()
                    .enumerate()
                    .filter(|(j, _n)| *j != i)
                    .map(|(_j, n)| *n),
                self.n_nodes,
            )
            .unwrap();
            let parity = if i % 2 == 0 {
                CF::one()
            } else {
                CF::one().additive_inverse()
            };
            Some((parity, bdry_path_idx))
        })
    }
}

// TODO: Implement MagnitudeCoboundary

#[derive(Clone)]
pub struct MagnitudeCoboundary<CF>
where
    CF: NonZeroCoefficient,
{
    d: Arc<DistanceMatrix<NodeIndex>>,
    n_nodes: usize,
    phantom: PhantomData<CF>,
}

impl<CF> MagnitudeCoboundary<CF>
where
    CF: NonZeroCoefficient,
{
    pub fn new(n_nodes: usize, d: Arc<DistanceMatrix<NodeIndex>>) -> Self {
        Self {
            d,
            n_nodes,
            phantom: PhantomData,
        }
    }
}
impl<CF> MatrixOracle for MagnitudeCoboundary<CF>
where
    CF: NonZeroCoefficient,
{
    type CoefficientField = CF;

    type ColT = PathIndex;

    type RowT = PathIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // TODO: Add check - if k == l then we know coboundary is 0!
        let path = col.to_vec(self.n_nodes);
        let k = path.len() - 1;
        (1..=k)
            .flat_map(|i| (0..self.n_nodes).map(move |v| (i, v)))
            .filter_map(move |(insertion_position, v)| {
                // Check that vertex is distinct from surrounding
                let a = path[insertion_position - 1];
                let b = path[insertion_position];
                if a == v {
                    return None;
                }
                if b == v {
                    return None;
                }

                // Check whether inserting v at insertion_position changes length
                // TODO: Figure out how to get rid of (as u32)
                let node_a = NodeIndex::from(a as u32);
                let node_b = NodeIndex::from(b as u32);
                let node_v = NodeIndex::from(v as u32);
                if self.d.distance(&node_a, &node_v) + self.d.distance(&node_v, &node_b)
                    != self.d.distance(&node_a, &node_b)
                {
                    return None;
                }

                // Determine parity
                let parity = if insertion_position % 2 == 0 {
                    CF::one()
                } else {
                    CF::one().additive_inverse()
                };

                // Construct path index of new path
                let (vertices_prior, vertices_after) = path.split_at(insertion_position);
                let new_path = vertices_prior
                    .iter()
                    .chain(iter::once(&v))
                    .chain(vertices_after.iter())
                    .copied();
                let new_path_index = PathIndex::from_indices(new_path, self.n_nodes).unwrap();

                Some((parity, new_path_index))
            })
    }
}

#[derive(Clone)]
pub struct MagnitudeBasis<R>
where
    R: IntoIterator<Item = usize> + Clone,
{
    paths: Arc<FxHashMap<PathKey<NodeIndex>, Vec<PathIndex>>>,
    node_pair: (NodeIndex, NodeIndex),
    l: usize,
    k_range: R,
    // How to avoid this?
    empty_basis: Vec<PathIndex>,
}

impl<R: IntoIterator<Item = usize> + Clone> MagnitudeBasis<R> {
    fn stkl(&self, k: usize) -> PathKey<NodeIndex> {
        PathKey {
            s: self.node_pair.0,
            t: self.node_pair.1,
            k,
            l: self.l,
        }
    }
}

impl<R: IntoIterator<Item = usize> + Clone> ColBasis for MagnitudeBasis<R> {
    type ElemT = PathIndex;

    fn element(&self, index: usize) -> Self::ElemT {
        let mut working = index;
        let mut k_iter = self.k_range.clone().into_iter();
        loop {
            let dim = k_iter.next().expect("Index is within range of k_range");
            let stkl = self.stkl(dim);
            let stkl_len = self.paths.get(&stkl).map(|paths| paths.len()).unwrap_or(0);
            if working >= stkl_len {
                working -= stkl_len;
            } else {
                return self
                    .paths
                    .get(&stkl)
                    .expect("Can only reach here if there is some stkl path")
                    .element(working);
            }
        }
    }

    fn size(&self) -> usize {
        self.k_range
            .clone()
            .into_iter()
            .map(|k| {
                let stkl = self.stkl(k);
                self.paths.get(&stkl).unwrap().len()
            })
            .sum()
    }
}

impl<R: IntoIterator<Item = usize> + Clone> SplitByDimension for MagnitudeBasis<R> {
    type SubBasisT = Vec<PathIndex>;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        let stkl = self.stkl(dimension);
        self.paths.get(&stkl).unwrap_or(&self.empty_basis)
    }
}

#[derive(Debug)]
pub struct PhlitePathContainer<NodeId>
where
    NodeId: SensibleNode,
{
    pub paths: Arc<FxHashMap<PathKey<NodeId>, Vec<PathIndex>>>,
    pub d: Arc<DistanceMatrix<NodeId>>,
    pub k_max: usize,
    pub l_max: Option<usize>,
    pub n_nodes: usize,
}

impl PhlitePathContainer<NodeIndex> {
    pub fn max_found_l(&self) -> usize {
        self.paths
            .iter()
            .map(|(key, _value)| key.l)
            .max()
            .unwrap_or(0)
    }

    pub fn max_reps_dim(&self, l: usize) -> usize {
        if self.k_max == l {
            l
        } else {
            usize::min(self.k_max - 1, l)
        }
    }

    pub fn max_ranks_dim(&self, l: usize) -> usize {
        usize::min(self.k_max, l)
    }

    pub fn num_paths(&self, key: &PathKey<NodeIndex>) -> usize {
        if let Some(sub_paths) = self.paths.get(key) {
            sub_paths.len()
        } else {
            0
        }
    }
    pub fn aggregated_rank<I: Iterator<Item = (NodeIndex, NodeIndex)>>(
        &self,
        node_identifiers: impl Fn() -> I,
        k: usize,
        l: usize,
    ) -> usize {
        node_identifiers()
            .map(|(s, t)| self.num_paths(&PathKey { s, t, k, l }))
            .sum()
    }
    pub fn rank_matrix<I: Iterator<Item = (NodeIndex, NodeIndex)>>(
        &self,
        node_identifiers: impl Fn() -> I + Copy,
    ) -> Vec<Vec<usize>> {
        let k_max = self.k_max;
        let l_max = self.l_max.unwrap_or_else(|| self.max_found_l());
        (0..=l_max)
            .map(|l| {
                (0..=k_max)
                    .map(|k| self.aggregated_rank(node_identifiers, k, l))
                    .collect()
            })
            .collect()
    }

    pub fn magnitude_bounday<CF: NonZeroCoefficient>(&self) -> MagnitudeBoundary<'_, CF> {
        MagnitudeBoundary::new(self.n_nodes, self.d.deref())
    }

    pub fn stkl_magnitude_boundary<CF: NonZeroCoefficient>(
        &self,
        stkl: &PathKey<NodeIndex>,
    ) -> MatrixWithBasis<MagnitudeBoundary<'_, CF>, &Vec<PathIndex>> {
        MatrixWithBasis {
            matrix: self.magnitude_bounday(),
            basis: self.paths.get(stkl).unwrap(),
        }
    }

    pub fn stl_magnitude_boundary<CF: NonZeroCoefficient, R: IntoIterator<Item = usize> + Clone>(
        &self,
        node_pair: (NodeIndex, NodeIndex),
        l: usize,
        k_range: R,
    ) -> MatrixWithBasis<MagnitudeBoundary<'_, CF>, MagnitudeBasis<R>> {
        MatrixWithBasis {
            matrix: self.magnitude_bounday(),
            basis: MagnitudeBasis {
                paths: self.paths.clone(),
                node_pair,
                l,
                k_range,
                empty_basis: vec![],
            },
        }
    }

    pub fn magnitude_cobounday<CF: NonZeroCoefficient>(&self) -> MagnitudeCoboundary<CF> {
        MagnitudeCoboundary::new(self.n_nodes, self.d.clone())
    }

    pub fn stkl_magnitude_coboundary<CF: NonZeroCoefficient>(
        &self,
        stkl: &PathKey<NodeIndex>,
    ) -> MatrixWithBasis<MagnitudeCoboundary<CF>, &Vec<PathIndex>> {
        MatrixWithBasis {
            matrix: self.magnitude_cobounday(),
            basis: self.paths.get(stkl).unwrap(),
        }
    }

    pub fn stl_magnitude_coboundary<
        CF: NonZeroCoefficient,
        R: IntoIterator<Item = usize> + Clone,
    >(
        &self,
        node_pair: (NodeIndex, NodeIndex),
        l: usize,
        k_range: R,
    ) -> MatrixWithBasis<MagnitudeCoboundary<CF>, MagnitudeBasis<R>> {
        MatrixWithBasis {
            matrix: self.magnitude_cobounday(),
            basis: MagnitudeBasis {
                paths: self.paths.clone(),
                node_pair,
                l,
                k_range,
                empty_basis: vec![],
            },
        }
    }

    pub fn stl(
        &self,
        node_pair: (NodeIndex, NodeIndex),
        l: usize,
    ) -> PhliteStlPathContainer<&Self> {
        PhliteStlPathContainer {
            parent_container: self,
            node_pair,
            l,
        }
    }
}

#[derive(Debug)]
pub struct PhliteStlPathContainer<Ref>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
{
    pub parent_container: Ref,
    pub node_pair: (NodeIndex, NodeIndex),
    pub l: usize,
}

pub type MagnitudeReductionMatrix<CF, R> = ClearedReductionMatrix<
    'static,
    WithTrivialFiltration<
        ReverseMatrix<MatrixWithBasis<MagnitudeCoboundary<CF>, MagnitudeBasis<R>>>,
    >,
>;

impl PhliteStlPathContainer<Arc<PhlitePathContainer<NodeIndex>>> {
    pub fn new(
        parent_container: Arc<PhlitePathContainer<NodeIndex>>,
        node_pair: (NodeIndex, NodeIndex),
        l: usize,
    ) -> Self {
        Self {
            parent_container,
            node_pair,
            l,
        }
    }
}

impl<Ref> PhliteStlPathContainer<Ref>
where
    Ref: Deref<Target = PhlitePathContainer<NodeIndex>>,
{
    pub fn max_homology_dim(&self) -> usize {
        usize::min(self.parent_container.k_max, self.l)
    }
    pub fn magnitude_coboundary<CF, R>(
        &self,
        k_range: R,
    ) -> MatrixWithBasis<MagnitudeCoboundary<CF>, MagnitudeBasis<R>>
    where
        CF: NonZeroCoefficient,
        R: IntoIterator<Item = usize> + Clone,
    {
        self.parent_container
            .stl_magnitude_coboundary(self.node_pair, self.l, k_range)
    }

    pub fn homology<CF, R>(self, k_range: R) -> PhliteStlHomology<Ref, CF, R>
    where
        CF: NonZeroCoefficient + Invertible,
        R: IntoIterator<Item = usize> + Clone,
    {
        let coboundary = self.magnitude_coboundary::<CF, R>(k_range.clone());
        let coboundary = coboundary.reverse();
        let (v, diagram) = ClearedReductionMatrix::build_with_diagram(
            coboundary.clone().with_trivial_filtration(),
            k_range.into_iter(),
        );
        PhliteStlHomology {
            stl_paths: self,
            reduction_matrix: v,
            diagram,
        }
    }
}
