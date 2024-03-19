#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod distances;
pub mod homology;
pub mod path_search;
pub mod utils;

// TODO:
// 1. Test
// 2. Document
// 3. Benchmark

#[cfg_attr(feautre = "python", pyclass)]
#[derive(Debug)]
pub enum MagError {
    NoRepresentatives,
    InsufficientLMax(usize, Option<usize>), // (l, l_max)
    BadArguments(String),
}

impl std::fmt::Display for MagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MagError::NoRepresentatives => write!(f, "Homology not computed with representatives"),
            MagError::InsufficientLMax(l, l_max) => write!(
                f,
                "Asked for l={}, but only have paths up to l={}",
                l,
                l_max.map(|i| i.to_string()).unwrap_or("None".to_owned())
            ),
            MagError::BadArguments(s) => s.fmt(f),
        }
    }
}

#[cfg(feature = "python")]
impl From<MagError> for PyErr {
    fn from(value: MagError) -> Self {
        pyo3::exceptions::PyTypeError::new_err(value.to_string())
    }
}

pub type Path<NodeId> = Vec<NodeId>;
pub type Representative<NodeId> = Vec<Path<NodeId>>;

#[cfg(feature = "python")]
mod bindings;
