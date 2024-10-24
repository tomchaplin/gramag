use dashmap::DashMap;
use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Directed, Graph};
use phlite::{fields::Z2, matrices::MatrixOracle, reduction::ClearedReductionMatrix};
use std::{collections::HashMap, sync::Arc, time::SystemTime};

use gramag::{
    distances::{parallel_all_pairs_distance, DistanceMatrix},
    homology::{all_homology_ranks_default, DirectSum},
    path_search::{PathQuery, StoppingCondition},
    phlite_homology::PhliteDirectSum,
    utils::format_rank_table,
};

fn _old_main() {
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
            .map(|(coeff, path)| (coeff, path.to_vec(n_nodes)))
            .collect();
        println!("{:?}", rep);
    }
}

fn _non_metric_main() {
    let mut row0 = HashMap::new();
    row0.insert(NodeIndex::new(0), 0);
    row0.insert(NodeIndex::new(1), 1);
    row0.insert(NodeIndex::new(2), 3);
    let mut row1 = HashMap::new();
    row1.insert(NodeIndex::new(0), 1);
    row1.insert(NodeIndex::new(1), 0);
    row1.insert(NodeIndex::new(2), 1);
    let mut row2 = HashMap::new();
    row2.insert(NodeIndex::new(0), 3);
    row2.insert(NodeIndex::new(1), 1);
    row2.insert(NodeIndex::new(2), 0);

    let d = DashMap::new();
    d.insert(NodeIndex::new(0), row0);
    d.insert(NodeIndex::new(1), row1);
    d.insert(NodeIndex::new(2), row2);

    let distance_matrix = DistanceMatrix(d);
    let distance_matrix = Arc::new(distance_matrix);

    let mut false_graph = Graph::<(), (), Directed, u32>::new();
    false_graph.add_node(());
    false_graph.add_node(());
    false_graph.add_node(());
    let path_query = PathQuery::build(
        &false_graph,
        distance_matrix.clone(),
        StoppingCondition::KMax(12),
    );

    let container = path_query.run();
    let all_node_pairs: Vec<_> = (0..=2)
        .flat_map(|i| (0..=2).map(move |j| (NodeIndex::new(i), NodeIndex::new(j))))
        .collect();
    let ranks = gramag::homology::all_homology_ranks_default(&container, &all_node_pairs);
    println!("Homology");
    println!("{}", format_rank_table(ranks, Default::default()));

    let ds = DirectSum::new(all_node_pairs.iter().map(|node_pair| {
        (
            (*node_pair, 3),
            Arc::new(container.stl(*node_pair, 3).serial_homology(true)),
        )
    }));

    println!("Reps");
    println!("{:?}", ds.representatives().unwrap().get(&1).unwrap())
}

fn main() {
    let n = 10;
    let graph = Graph::<(), ()>::from_edges((0..n).map(|i| (i, (i + 1) % n)));
    let distance_matrix = parallel_all_pairs_distance(&graph);
    let distance_matrix = Arc::new(distance_matrix);
    let k_max = 2;
    let all_node_pairs: Vec<_> = graph
        .node_identifiers()
        .flat_map(|s| graph.node_identifiers().map(move |t| (s, t)))
        .collect();
    // Phlite
    let tic = SystemTime::now();
    let path_query = PathQuery::build(
        &graph,
        distance_matrix.clone(),
        StoppingCondition::KMax(k_max),
    );
    let container = path_query.run_phlite();
    // Testing phlite StlHomology
    let ranks = container
        .stl((NodeIndex::new(0), NodeIndex::new(0)), n as usize)
        .homology::<Z2, _>(0..=k_max)
        .ranks();
    println!("{:?}", ranks);
    // Testing DirectSum
    let ds = PhliteDirectSum::new((0..n).map(|i| {
        let node_pair = (NodeIndex::new(i as usize), NodeIndex::new(i as usize));
        let length = n as usize;
        let homology = container
            .stl(node_pair, length)
            .homology::<Z2, _>(0..=k_max);
        let homology = Arc::new(homology);
        ((node_pair, length), homology)
    }));
    let ranks = ds.ranks();
    println!("{:?}", ranks);
    // Testing all ranks
    let ranks = gramag::phlite_homology::all_homology_ranks_default(&container, &all_node_pairs);
    let toc = SystemTime::now();
    let phlite_duration = toc.duration_since(tic).unwrap();
    println!("Homology");
    println!("{}", format_rank_table(ranks, Default::default()));
    // LoPhat
    let tic = SystemTime::now();
    let path_query = PathQuery::build(
        &graph,
        distance_matrix.clone(),
        StoppingCondition::KMax(k_max + 1),
    );
    let container = path_query.run();
    let ranks = all_homology_ranks_default(&container, &all_node_pairs);
    let toc = SystemTime::now();
    let lophat_duration = toc.duration_since(tic).unwrap();
    println!("Homology");
    println!("{}", format_rank_table(ranks, Default::default()));
    // Report time
    println!("Phlite: {}μs", phlite_duration.as_micros());
    println!("Lophat: {}μs", lophat_duration.as_micros());
}
