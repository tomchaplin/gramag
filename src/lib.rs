pub mod distances;
pub mod homology;
pub mod path_search;
pub mod utils;

// TODO:
// 1. Test
// 2. Document
// 3. Benchmark
// 4. Switch to fxhash
// 5. Direct sum class + bindings

pub type Path<NodeId> = Vec<NodeId>;

#[cfg(feature = "python")]
mod bindings;
