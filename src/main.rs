use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Graph};
use phlite::{
    fields::Z2,
    matrices::{MatrixOracle, MatrixRef},
    reduction::ClearedReductionMatrix,
};
use std::sync::Arc;

use gramag::{
    distances::parallel_all_pairs_distance,
    homology::all_homology_ranks_default,
    path_search::{PathQuery, StoppingCondition},
    utils::format_rank_table,
};

fn main() {
    let graph =
        Graph::<(), ()>::from_edges([(0, 1), (0, 2), (0, 3), (1, 5), (2, 5), (3, 4), (4, 5)]);

    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);

    let l_max = 10;
    let path_query = PathQuery::build(
        &graph,
        distance_matrix.clone(),
        StoppingCondition::LMax(l_max),
    );
    let container = path_query.run();

    let all_node_pairs: Vec<_> = graph
        .node_identifiers()
        .flat_map(|s| graph.node_identifiers().map(move |t| (s, t)))
        .collect();

    println!("Generators");
    println!(
        "{}",
        format_rank_table(
            container.rank_matrix(|| all_node_pairs.iter().copied()),
            Default::default()
        )
    );

    println!("Homology");
    println!(
        "{}",
        format_rank_table(
            all_homology_ranks_default(&container, &all_node_pairs),
            Default::default()
        )
    );

    let reps = container
        .stl((NodeIndex::from(0), NodeIndex::from(5)), 3)
        .serial_homology(true)
        .representatives()
        .expect("Should have reps because we passed true to serial_homology");

    println!("{:#?}", reps);

    let phlite_container = path_query.run_phlite();
    for (key, value) in phlite_container.paths.iter() {
        let phlite_count = value.len();
        let old_count = container.paths.get(key).unwrap().len();
        assert_eq!(phlite_count, old_count);
    }

    let stl_phlite_bdry = phlite_container.stl_magnitude_coboundary::<Z2, _>(
        (NodeIndex::from(0), NodeIndex::from(5)),
        3,
        0..=3,
    );

    let (v, diagram) = ClearedReductionMatrix::build_with_diagram(
        stl_phlite_bdry.with_trivial_filtration(),
        0..=3,
    );

    println!("{:?}", diagram);

    println!("Essential");
    let n_nodes = graph.node_count();
    for idx in diagram.essential.iter() {
        let rep: Vec<_> = v
            .column(*idx)
            .unwrap()
            .map(|(coeff, path)| (coeff, path.to_vec(n_nodes)))
            .collect();
        println!("{:?}", rep);
    }
}
