//! Trace shapes: the Cairo leg's components for a real transaction (`data/`), and the columns a
//! stripe of their commitment hashes.

/// A component of the Cairo trace: its log number of rows and its columns.
#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    pub log_rows: u32,
    pub base_columns: u32,
    pub interaction_columns: u32,
}

/// The Cairo leg's components for the pay capture (`data/cairo_trace_shapes_pay.txt`).
pub fn cairo_pay_components() -> Vec<Component> {
    include_str!("../data/cairo_trace_shapes_pay.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 4, "bad line: {line}");
            Component {
                name: fields[0].to_string(),
                log_rows: fields[1].parse().unwrap(),
                base_columns: fields[2].parse().unwrap(),
                interaction_columns: fields[3].parse().unwrap(),
            }
        })
        .collect()
}

/// The Cairo leg's log blowup factor and its trees' lifting log size
/// (`privacy_circuit_verify/src/consts.rs`): 8 stripes of height 20.
pub const CAIRO_LOG_BLOWUP: u32 = 3;
pub const CAIRO_LIFTING_LOG_SIZE: u32 = 23;
/// The circuit leg's: 4 stripes of height 21.
pub const CIRCUIT_LOG_BLOWUP: u32 = 2;
pub const CIRCUIT_LIFTING_LOG_SIZE: u32 = 23;

/// The base trace's columns' log sizes, in polynomial order: one per base column, of its
/// component's log number of rows less `scale` (but at least `min_log_size`). A column's stripe
/// has as many values as its coefficients, `2^log_rows`.
pub fn base_trace_log_sizes(components: &[Component], scale: u32, min_log_size: u32) -> Vec<u32> {
    components
        .iter()
        .flat_map(|c| {
            let log_size = c.log_rows.saturating_sub(scale).max(min_log_size);
            std::iter::repeat_n(log_size, c.base_columns as usize)
        })
        .collect()
}

/// The interaction trace's columns' log sizes, in polynomial order: one per interaction column
/// (M31 columns, 4 per secure column), of its component's log number of rows less `scale` (but
/// at least `min_log_size`).
pub fn interaction_trace_log_sizes(
    components: &[Component],
    scale: u32,
    min_log_size: u32,
) -> Vec<u32> {
    components
        .iter()
        .flat_map(|c| {
            let log_size = c.log_rows.saturating_sub(scale).max(min_log_size);
            std::iter::repeat_n(log_size, c.interaction_columns as usize)
        })
        .collect()
}

/// The order in which a tree's leaves hash columns of these log sizes (given in polynomial
/// order): by increasing size, columns of one size in polynomial order (a stable sort, as
/// `new_striped`'s `order` and `MerkleProverLifted::commit`'s).
pub fn tree_order(log_sizes: &[u32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..log_sizes.len()).collect();
    order.sort_by_key(|&i| log_sizes[i]);
    order
}

/// The `(log_size, n_columns)` runs of equal log sizes in nondecreasing `sorted_log_sizes`: the
/// per-size batches of a stripe.
pub fn size_batches(sorted_log_sizes: &[u32]) -> Vec<(u32, u32)> {
    let mut batches: Vec<(u32, u32)> = vec![];
    for &log_size in sorted_log_sizes {
        match batches.last_mut() {
            Some((last, n)) if *last == log_size => *n += 1,
            _ => {
                if let Some(&(last, _)) = batches.last() {
                    assert!(last < log_size, "log sizes not sorted");
                }
                batches.push((log_size, 1));
            }
        }
    }
    batches
}
