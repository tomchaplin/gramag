pub mod distances;
pub mod homology;
pub mod path_search;
pub mod utils;

pub type Path<NodeId> = Vec<NodeId>;

#[cfg(feature = "python")]
mod bindings;
