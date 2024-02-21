use petgraph::{visit::IntoNodeIdentifiers, Graph};
use std::sync::Arc;

use gramag::{
    distances::parallel_all_pairs_distance, homology::all_homology_ranks_default,
    path_search::PathQuery, utils::rank_table,
};

fn main() {
    //let n = 100;
    //let graph = Graph::<(), ()>::from_edges((0..n).flat_map(|i| {
    //    let j1 = (i as i32 + 1).rem_euclid(n) as u32;
    //    let j2 = (i as i32 - 1).rem_euclid(n) as u32;
    //    [(i as u32, j1), (i as u32, j2)]
    //}));

    let graph = Graph::<(), ()>::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);

    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);

    let l_max = 10;
    let path_query = PathQuery::build(&graph, distance_matrix, l_max);
    let container = path_query.run();

    println!("Generators");
    println!(
        "{}",
        rank_table(container.rank_matrix(
            || {
                graph
                    .node_identifiers()
                    .flat_map(|s| graph.node_identifiers().map(move |t| (s, t)))
            },
            l_max
        ))
    );

    println!("Homology");
    println!(
        "{}",
        rank_table(all_homology_ranks_default(&container, path_query.clone()))
    );
}
