use std::collections::HashMap;
use std::iter;

use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::style::{HorizontalLine, VerticalLine};
use tabled::settings::{Alignment, Style};

pub struct RankTableOptions {
    pub above_diagonal: String,
    pub unknown: String,
    pub zero: String,
}

impl From<(Option<String>, Option<String>, Option<String>)> for RankTableOptions {
    fn from(value: (Option<String>, Option<String>, Option<String>)) -> Self {
        let mut options: RankTableOptions = Default::default();
        if let Some(above_diagonal) = value.0 {
            options.above_diagonal = above_diagonal
        };
        if let Some(unknown) = value.1 {
            options.unknown = unknown
        };
        if let Some(zero) = value.2 {
            options.zero = zero
        };
        options
    }
}

impl Default for RankTableOptions {
    fn default() -> Self {
        Self {
            above_diagonal: "".to_string(),
            unknown: "?".to_string(),
            zero: ".".to_string(),
        }
    }
}

pub fn format_rank_table(ranks: Vec<Vec<usize>>, options: RankTableOptions) -> String {
    let mut builder = Builder::new();

    let n_rows = ranks.len();
    let n_cols = ranks.iter().map(|row| row.len()).max().unwrap();

    let get_rank = |l: usize, k: usize| {
        let rank = ranks.get(l).and_then(|l_ranks| l_ranks.get(k).copied());
        if k <= l {
            match rank {
                Some(rank) => {
                    if rank == 0 {
                        options.zero.clone()
                    } else {
                        format!("{}", rank)
                    }
                }
                None => options.unknown.clone(),
            }
        } else {
            options.above_diagonal.clone()
        }
    };

    let header = iter::once("k=".to_string()).chain((0..n_cols).map(move |k| format!("{}", k)));
    builder.push_record(header);

    for l in 0..n_rows {
        let ranks = (0..n_cols).map(|k| get_rank(l, k));
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
