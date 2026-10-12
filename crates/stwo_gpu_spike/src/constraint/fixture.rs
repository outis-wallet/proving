//! A component's constraint evaluation captured from a proof (see [`super::capture`]), on disk:
//! one directory per component.
//!
//! - `meta.txt`: `key value...` lines: `component`, `trace_log_size`, `eval_log_size`,
//!   `n_constraints`, `n_columns` (per interaction), `preprocessed_column_indices`.
//! - `program.txt`: the lowered constraints ([`Program::to_text`]).
//! - `params.txt`: `name a b c d` per formal parameter value.
//! - `coeffs.bin` (4 words per constraint), `denom_inv.bin`: little-endian `u32` words.
//! - `columns.bin`: every column on the whole evaluation domain (bit-reversed), interaction by
//!   interaction, `2^eval_log_size` words each, as stored by the prover (raw words).
//! - `accum_before.bin`, `accum_after.bin`: the accumulator column of the evaluation domain's size
//!   before and after the SIMD evaluation, coordinate by coordinate (raw words).

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use stwo::core::fields::qm31::QM31;

use super::ir::{Program, qm31, words};

pub struct Fixture {
    pub component: String,
    pub trace_log_size: u32,
    pub eval_log_size: u32,
    pub n_constraints: usize,
    pub n_columns: Vec<usize>,
    pub preprocessed_column_indices: Vec<usize>,
    pub program: Program,
    pub params: HashMap<String, QM31>,
    pub coeffs: Vec<QM31>,
    pub denom_inv: Vec<u32>,
    /// `columns[interaction][index]`: `2^eval_log_size` words.
    pub columns: Vec<Vec<Vec<u32>>>,
    pub before: [Vec<u32>; 4],
    pub after: [Vec<u32>; 4],
}

pub fn write_words(path: &Path, chunks: &[&[u32]]) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::with_capacity(1 << 22, fs::File::create(path)?);
    for chunk in chunks {
        file.write_all(bytemuck::cast_slice(chunk))?;
    }
    file.flush()
}

pub fn read_words(path: &Path) -> std::io::Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    let mut words = vec![0u32; bytes.len() / 4];
    bytemuck::cast_slice_mut::<u32, u8>(&mut words).copy_from_slice(&bytes);
    Ok(words)
}

pub fn write_coeffs(path: &Path, coeffs: &[QM31]) -> std::io::Result<()> {
    let flat: Vec<u32> = coeffs.iter().flat_map(|c| words(*c)).collect();
    write_words(path, &[&flat])
}

pub fn write_params(path: &Path, params: &HashMap<String, QM31>) -> std::io::Result<()> {
    let mut names: Vec<&String> = params.keys().collect();
    names.sort();
    let mut s = String::new();
    for name in names {
        let w = words(params[name]);
        s += &format!("{name} {} {} {} {}\n", w[0], w[1], w[2], w[3]);
    }
    fs::write(path, s)
}

fn split4(words: Vec<u32>) -> [Vec<u32>; 4] {
    let n = words.len() / 4;
    std::array::from_fn(|c| words[c * n..(c + 1) * n].to_vec())
}

impl Fixture {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{}: {e}", dir.display());
        let meta = fs::read_to_string(dir.join("meta.txt")).map_err(err)?;
        let mut fields: HashMap<&str, Vec<&str>> = HashMap::new();
        for line in meta.lines() {
            let mut words = line.split_whitespace();
            if let Some(key) = words.next() {
                fields.insert(key, words.collect());
            }
        }
        let field = |key: &str| fields.get(key).cloned().unwrap_or_default();
        let nums =
            |key: &str| -> Vec<usize> { field(key).iter().map(|w| w.parse().unwrap()).collect() };
        let program =
            Program::from_text(&fs::read_to_string(dir.join("program.txt")).map_err(err)?)?;
        let mut params = HashMap::new();
        for line in fs::read_to_string(dir.join("params.txt")).map_err(err)?.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            let v: Vec<u32> = w[1..5].iter().map(|x| x.parse().unwrap()).collect();
            params.insert(w[0].to_owned(), qm31([v[0], v[1], v[2], v[3]]));
        }
        let coeffs = read_words(&dir.join("coeffs.bin"))
            .map_err(err)?
            .chunks(4)
            .map(|c| qm31([c[0], c[1], c[2], c[3]]))
            .collect();
        let trace_log_size = nums("trace_log_size")[0] as u32;
        let eval_log_size = nums("eval_log_size")[0] as u32;
        let n_columns = nums("n_columns");
        let size = 1usize << eval_log_size;
        let flat = read_words(&dir.join("columns.bin")).map_err(err)?;
        assert_eq!(flat.len(), n_columns.iter().sum::<usize>() * size, "columns.bin's size");
        let mut columns = vec![];
        let mut at = 0;
        for n in &n_columns {
            columns.push(
                (0..*n)
                    .map(|_| {
                        at += size;
                        flat[at - size..at].to_vec()
                    })
                    .collect(),
            );
        }
        Ok(Self {
            component: field("component").join(" "),
            trace_log_size,
            eval_log_size,
            n_constraints: nums("n_constraints")[0],
            n_columns,
            preprocessed_column_indices: nums("preprocessed_column_indices"),
            program,
            params,
            coeffs,
            denom_inv: read_words(&dir.join("denom_inv.bin")).map_err(err)?,
            columns,
            before: split4(read_words(&dir.join("accum_before.bin")).map_err(err)?),
            after: split4(read_words(&dir.join("accum_after.bin")).map_err(err)?),
        })
    }

    /// The columns as slices, `[interaction][index]`.
    pub fn column_slices(&self) -> Vec<Vec<&[u32]>> {
        self.columns.iter().map(|cs| cs.iter().map(|c| c.as_slice()).collect()).collect()
    }
}
