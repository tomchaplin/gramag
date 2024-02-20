use std::iter;

use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::style::{HorizontalLine, VerticalLine};
use tabled::settings::{Alignment, Style};

pub fn rank_table(ranks: Vec<Vec<usize>>) -> String {
    let mut builder = Builder::new();

    let l_max = ranks.len() - 1;

    let get_rank = |l: usize, k: usize| {
        ranks
            .get(l)
            .and_then(|l_ranks| l_ranks.get(k).copied())
            .unwrap_or(0)
    };

    // Header

    let header = iter::once(format!("k=")).chain((0..(l_max + 1)).map(move |k| format!("{}", k)));
    builder.push_record(header);

    for l in 0..(l_max + 1) {
        let ranks = (0..(l_max + 1)).map(|k| format!("{}", get_rank(l, k)));
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
