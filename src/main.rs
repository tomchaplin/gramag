use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Graph};
use std::sync::Arc;

use gramag::{
    distances::parallel_all_pairs_distance, homology::all_homology_ranks_default,
    path_search::PathQuery, utils::rank_table,
};

fn main() {
    let graph =
        Graph::<(), ()>::from_edges([(0, 1), (0, 2), (0, 3), (1, 5), (2, 5), (3, 4), (4, 5)]);

    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);

    let l_max = 10;
    let path_query = PathQuery::build(&graph, distance_matrix.clone(), l_max);
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

    let reps = container
        .stl((NodeIndex::from(0), NodeIndex::from(5)), 3)
        .serial_homology(true)
        .representatives()
        .expect("Shoudl have reps because we passed true to serial_homology");
    println!("{:#?}", reps);
}
