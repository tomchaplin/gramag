pub mod distances;
pub mod homology;
pub mod path_search;
pub mod utils;

// TODO:
// 1. Test
// 2. Document
// 3. Benchmark
// 4. Direct sum class + bindings
// 5. Add an Arc<DistanceMatrix> to PathContainer to avoid copying around?

pub type Path<NodeId> = Vec<NodeId>;

#[cfg(feature = "python")]
mod bindings;
