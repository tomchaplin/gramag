use std::{collections::HashMap, iter, marker::PhantomData, ops::Deref, sync::Arc};

use petgraph::graph::NodeIndex;
use phlite::{
    fields::NonZeroCoefficient,
    matrices::{adaptors::MatrixWithBasis, BasisElement, ColBasis, MatrixOracle, SplitByDimension},
};

use crate::{
    distances::DistanceMatrix,
    path_search::{PathKey, SensibleNode},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathIndex(usize);

impl BasisElement for PathIndex {}

impl PathIndex {
    /// Produces from least significant to most significant
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        let mut vec = vec![];
        let mut working = self.0;
        while working > 0 {
            let remainder = working.rem_euclid(n_points + 1);
            vec.push(remainder - 1);
            working = working.div_euclid(n_points + 1);
        }
        vec
    }

    /// NOTE: The indicies should be provided in ascending order!
    pub fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
        let inner = indices
            .enumerate()
            .map(|(i, coeff)| (n_points + 1).pow(i as u32) * (coeff + 1))
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
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
        let path = col.to_vec(self.n_nodes);

        let k = path.len() - 1;

        Ok((1..k).filter_map(move |i| {
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
        }))
    }
}

// TODO: Implement MagnitudeCoboundary

pub struct MagnitudeCoboundary<'a, CF>
where
    CF: NonZeroCoefficient,
{
    d: &'a DistanceMatrix<NodeIndex>,
    n_nodes: usize,
    phantom: PhantomData<CF>,
}

impl<'a, CF> MagnitudeCoboundary<'a, CF>
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
impl<'a, CF> MatrixOracle for MagnitudeCoboundary<'a, CF>
where
    CF: NonZeroCoefficient,
{
    type CoefficientField = CF;

    type ColT = PathIndex;

    type RowT = PathIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
        // TODO: Add check - if k == l then we know coboundary is 0!
        let path = col.to_vec(self.n_nodes);
        let k = path.len() - 1;
        Ok((1..=k)
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
            }))
    }
}

pub struct MagnitudeBasis<'a, R>
where
    R: IntoIterator<Item = usize> + Clone,
{
    paths: &'a HashMap<PathKey<NodeIndex>, Vec<PathIndex>>,
    node_pair: (NodeIndex, NodeIndex),
    l: usize,
    k_range: R,
    // How to avoid this?
    empty_basis: Vec<PathIndex>,
}

impl<'a, R: IntoIterator<Item = usize> + Clone> MagnitudeBasis<'a, R> {
    fn stkl(&self, k: usize) -> PathKey<NodeIndex> {
        PathKey {
            s: self.node_pair.0,
            t: self.node_pair.1,
            k,
            l: self.l,
        }
    }
}

impl<'a, R: IntoIterator<Item = usize> + Clone> ColBasis for MagnitudeBasis<'a, R> {
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

impl<'a, R: IntoIterator<Item = usize> + Clone> SplitByDimension for MagnitudeBasis<'a, R> {
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
    pub paths: HashMap<PathKey<NodeId>, Vec<PathIndex>>,
    pub d: Arc<DistanceMatrix<NodeId>>,
    pub k_max: usize,
    pub l_max: Option<usize>,
    pub n_nodes: usize,
}

impl PhlitePathContainer<NodeIndex> {
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
    ) -> MatrixWithBasis<MagnitudeBoundary<'_, CF>, MagnitudeBasis<'_, R>> {
        MatrixWithBasis {
            matrix: self.magnitude_bounday(),
            basis: MagnitudeBasis {
                paths: &self.paths,
                node_pair,
                l,
                k_range,
                empty_basis: vec![],
            },
        }
    }

    pub fn magnitude_cobounday<CF: NonZeroCoefficient>(&self) -> MagnitudeCoboundary<'_, CF> {
        MagnitudeCoboundary::new(self.n_nodes, self.d.deref())
    }

    pub fn stkl_magnitude_coboundary<CF: NonZeroCoefficient>(
        &self,
        stkl: &PathKey<NodeIndex>,
    ) -> MatrixWithBasis<MagnitudeCoboundary<'_, CF>, &Vec<PathIndex>> {
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
    ) -> MatrixWithBasis<MagnitudeCoboundary<'_, CF>, MagnitudeBasis<'_, R>> {
        MatrixWithBasis {
            matrix: self.magnitude_cobounday(),
            basis: MagnitudeBasis {
                paths: &self.paths,
                node_pair,
                l,
                k_range,
                empty_basis: vec![],
            },
        }
    }
}
