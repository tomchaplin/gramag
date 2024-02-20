use petgraph::Graph;
use std::sync::Arc;

use crate::{distances::parallel_all_pairs_distance, path_search::PathQuery};

pub mod distances;
pub mod path_search;

type Path<NodeId> = Vec<NodeId>;

fn main() {
    // let graph = Graph::<(), ()>::from_edges((0..10).flat_map(|i| {
    //     let j1 = (i as i32 + 1).rem_euclid(100) as u32;
    //     let j2 = (i as i32 - 1).rem_euclid(100) as u32;
    //     [(i, j1)]
    // }));

    let graph = Graph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);

    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);

    let path_query = PathQuery::build(&graph, distance_matrix, 10);
    let container = path_query.run();

    //println!("{:#?}", container.paths);
    println!("{}", container.rank_table());
}
