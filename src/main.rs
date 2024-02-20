use petgraph::Graph;
use std::sync::Arc;

use crate::{distances::parallel_all_pairs_distance, path_search::PathQuery};

pub mod distances;
pub mod homology;
pub mod path_search;
pub mod utils;

type Path<NodeId> = Vec<NodeId>;

fn main() {
    let n = 100;
    let graph = Graph::<(), ()>::from_edges((0..n).flat_map(|i| {
        let j1 = (i as i32 + 1).rem_euclid(n) as u32;
        let j2 = (i as i32 - 1).rem_euclid(n) as u32;
        [(i as u32, j1), (i as u32, j2)]
    }));

    //let graph = Graph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);

    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);

    let l_max = 10;
    let path_query = PathQuery::build(&graph, distance_matrix, l_max);
    let container = path_query.run();

    //println!("{:#?}", container.paths);
    println!("Generators");
    println!("{}", utils::rank_table(container.rank_matrix()));

    println!("Homology");
    println!(
        "{}",
        utils::rank_table(homology::all_homology_ranks_default(
            &container,
            path_query.clone()
        ))
    );
}
