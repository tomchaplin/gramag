use std::collections::HashMap;
use std::iter;

use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::style::{HorizontalLine, VerticalLine};
use tabled::settings::{Alignment, Style};

pub fn rank_table(ranks: Vec<Vec<usize>>) -> String {
    let mut builder = Builder::new();

    let l_max = ranks.len() - 1;
    let k_max = ranks[0].len() - 1;

    let get_rank = |l: usize, k: usize| {
        ranks
            .get(l)
            .and_then(|l_ranks| l_ranks.get(k).copied())
            .unwrap_or(0)
    };

    let header =
        iter::once("k=".to_string()).chain((0..(k_max + 1)).map(move |k| format!("{}", k)));
    builder.push_record(header);

    for l in 0..(l_max + 1) {
        let ranks = (0..(k_max + 1)).map(|k| format!("{}", get_rank(l, k)));
        let row = iter::once(format!("l={}", l)).chain(ranks);

        builder.push_record(row)
    }

    let theme = Style::modern_rounded()
        .horizontals([(1, HorizontalLine::inherit(Style::modern_rounded()))])
        .verticals([(1, VerticalLine::inherit(Style::modern_rounded()))])
        .remove_horizontal()
        .remove_vertical();

    builder
        .build()
        .with(theme)
        .modify(Columns::new(1..), Alignment::right())
        .to_string()
}

pub fn rank_map_to_rank_vec(rank_map: &HashMap<usize, usize>) -> Vec<usize> {
    let k_max = rank_map.keys().max().copied().unwrap();

    let mut out = vec![0; k_max + 1];
    for (dim, out_dim) in out.iter_mut().enumerate() {
        *out_dim += rank_map
            .get(&dim)
            .copied()
            .expect("Should have computed all degrees");
    }
    out
}
