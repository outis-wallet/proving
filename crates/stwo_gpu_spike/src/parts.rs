//! Stripe 1 of a commitment's columns for constraint evaluation's part extension.
//!
//! A striped commitment keeps of every column only its first stripe (`new_striped`). Constraint
//! evaluation reads a column of `2^n` rows on the first `2^(n+1)` points of its committed domain:
//! stripes 0 and 1. For stripe 1 it interpolates the column from stripe 0 and evaluates it there,
//! on the CPU ("Constraint part extension", constraint-framework's `trace_part`), while the GPU
//! commitment computed stripe 1 already (one of its `2^log_blowup` stripes) and dropped it.
//! constraint-framework's `TracePartHook` asks for those columns first; [`PartStore`] answers
//! from what the commitments kept, as [`PartsMode`] says:
//! - [`PartsMode::Host`] (A1): the commitment reads stripe 1 back with stripe 0, and the store
//!   holds it in host memory until constraint evaluation takes it;
//! - [`PartsMode::Device`] (A3): the commitment copies stripe 1, on the GPU, into buffers it keeps
//!   ([`DeviceStripes`]), and each request reads its columns back;
//! - [`PartsMode::Recompute`] (A2): nothing is kept; each request's columns go up (stripe 0), are
//!   interpolated and evaluated on stripe 1 on the GPU, and come back (see [`crate::recompute`]).
//!
//! A column is found by its stripe 0's allocation (the address and length of the values the tree
//! holds, which stay put from the commitment to constraint evaluation), checked against a
//! fingerprint of its values, the committed domain's size and the blowup. A preprocessed column
//! (tree 0), which several components may read, is copied out; any other is taken (its memory
//! freed). Everything kept goes at the first commitment after a lookup (the composition tree's,
//! which follows constraint evaluation), when a commitment fails, and on [`PartStore::clear`].

use std::collections::HashMap;
use std::time::Duration;

use stwo::core::fields::m31::BaseField;
use stwo::prover::backend::simd::column::BaseColumn;

use crate::stream::DeviceStripes;

/// Where stripe 1 of the commitments' columns comes from, for constraint evaluation (see the
/// module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartsMode {
    /// Not provided: constraint evaluation computes it on the CPU.
    Off,
    /// A1: read back by the commitment and held in host memory.
    Host,
    /// A3: kept on the GPU by the commitment, read back on request.
    Device,
    /// A2: recomputed on the GPU from stripe 0 on request.
    Recompute,
}

impl PartsMode {
    /// `STWO_GPU_PARTS`: `host` (or `a1`), `device` (`a3`), `recompute` (`a2`); `Off` otherwise.
    pub fn from_env() -> Self {
        match std::env::var("STWO_GPU_PARTS").unwrap_or_default().to_ascii_lowercase().as_str() {
            "host" | "a1" => Self::Host,
            "device" | "a3" => Self::Device,
            "recompute" | "a2" => Self::Recompute,
            _ => Self::Off,
        }
    }
}

/// A sampled fingerprint of a column's values: its length and 16 words spread over it.
pub fn fingerprint(values: &[BaseField]) -> u64 {
    let len = values.len();
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ len as u64;
    if len == 0 {
        return hash;
    }
    for j in 0..16 {
        let word = values[j * (len - 1) / 15].0 as u64;
        hash = (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    hash
}

/// The key of a column the tree holds: the address and length of its values.
pub fn key(values: &BaseColumn) -> (usize, usize) {
    (values.data.as_ptr() as usize, values.length)
}

/// Where a kept stripe 1 is.
enum Kept {
    Host(BaseColumn),
    /// In `device[set]`'s buffer `buffer`, at byte `offset`.
    Device {
        set: usize,
        buffer: usize,
        offset: u64,
    },
}

struct Entry {
    fingerprint: u64,
    /// The column's committed domain's log size and the commitment's log blowup.
    domain_log_size: u32,
    log_blowup: u32,
    kept: Kept,
}

/// What a lookup found for a column: stripe 1 on the host, or its place on the GPU (see
/// [`PartStore::device_buffer`]).
pub enum Found {
    Host(BaseColumn),
    Device { set: usize, buffer: usize, offset: u64 },
}

/// Counts of the store's work, since they were last taken.
#[derive(Clone, Copy, Debug, Default)]
pub struct PartStats {
    /// Requests (one per component and part), and their columns.
    pub requests: u64,
    pub columns: u64,
    /// Columns answered, and their bytes.
    pub hits: u64,
    pub hit_bytes: u64,
    /// Columns not found, or found with another fingerprint, domain or blowup.
    pub misses: u64,
    pub mismatches: u64,
    /// Preprocessed columns copied out (rather than taken).
    pub copies: u64,
    /// Bytes kept: now, and at most since the stats were last taken (host memory in Host mode,
    /// GPU memory in Device mode).
    pub held_bytes: u64,
    pub max_held_bytes: u64,
    /// Bytes registered by commitments.
    pub kept_bytes: u64,
    /// Time in lookups (wall clock), and of it the readbacks (Device) or the GPU's round trips
    /// (Recompute).
    pub lookup: Duration,
    pub transfer: Duration,
    /// Device: the readback ring's waits (for the GPU's copies) and the CPU's copies out.
    pub readback_wait: Duration,
    pub readback_copy: Duration,
    /// Recompute: the requests' split (see [`crate::recompute::RecomputeTimes`]).
    pub recompute: crate::recompute::RecomputeTimes,
    /// Clears with something kept.
    pub clears: u64,
}

/// Stripe 1 of the columns of the trees committed since the last clear (see the module docs).
#[derive(Default)]
pub struct PartStore {
    entries: HashMap<(usize, usize), Entry>,
    device: Vec<Option<DeviceStripes>>,
    /// A lookup was made since the last commitment.
    looked_up: bool,
    stats: PartStats,
}

impl PartStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called at the start of every commitment: the first after a lookup clears the store.
    pub fn commitment_starts(&mut self) {
        if self.looked_up {
            self.clear();
        }
    }

    /// Drops everything kept (and frees the GPU buffers).
    pub fn clear(&mut self) {
        if !self.entries.is_empty() || self.device.iter().any(Option::is_some) {
            self.stats.clears += 1;
        }
        self.entries.clear();
        for set in self.device.drain(..).flatten() {
            set.destroy();
        }
        self.stats.held_bytes = 0;
        self.looked_up = false;
    }

    fn hold(&mut self, bytes: u64) {
        self.stats.held_bytes += bytes;
        self.stats.kept_bytes += bytes;
        self.stats.max_held_bytes = self.stats.max_held_bytes.max(self.stats.held_bytes);
    }

    /// Keeps `stripe1[i]` (host memory) for the column the tree holds as `stripe0[i]`, of
    /// committed domain log size `log_sizes[i] + log_blowup`.
    pub fn keep_host(
        &mut self,
        stripe0: &[&BaseColumn],
        stripe1: Vec<Option<BaseColumn>>,
        log_sizes: &[u32],
        log_blowup: u32,
    ) {
        for ((values, kept), &log_size) in stripe0.iter().zip(stripe1).zip(log_sizes) {
            let Some(kept) = kept else { continue };
            let bytes = 4 * kept.length as u64;
            self.entries.insert(
                key(values),
                Entry {
                    fingerprint: fingerprint(values.as_slice()),
                    domain_log_size: log_size + log_blowup,
                    log_blowup,
                    kept: Kept::Host(kept),
                },
            );
            self.hold(bytes);
        }
    }

    /// Keeps `stripes` (GPU memory), stripe 1 of the columns the tree holds as `stripe0`.
    pub fn keep_device(
        &mut self,
        stripe0: &[&BaseColumn],
        stripes: DeviceStripes,
        log_sizes: &[u32],
        log_blowup: u32,
    ) {
        let set = self.device.len();
        for ((values, place), &log_size) in stripe0.iter().zip(&stripes.places).zip(log_sizes) {
            let Some((buffer, offset)) = *place else { continue };
            self.entries.insert(
                key(values),
                Entry {
                    fingerprint: fingerprint(values.as_slice()),
                    domain_log_size: log_size + log_blowup,
                    log_blowup,
                    kept: Kept::Device { set, buffer, offset },
                },
            );
        }
        self.hold(stripes.bytes());
        self.device.push(Some(stripes));
    }

    /// Notes a request of `n` columns.
    pub fn request(&mut self, n: usize) {
        self.looked_up = true;
        self.stats.requests += 1;
        self.stats.columns += n as u64;
    }

    /// Stripe 1 of the column the tree holds as `stripe0`, of committed domain log size
    /// `domain_log_size` and blowup `log_blowup`, from tree `tree`: taken (or copied out, for a
    /// preprocessed column), or its place on the GPU. `None` if not kept, or kept for other
    /// values.
    pub fn find(
        &mut self,
        tree: usize,
        stripe0: &BaseColumn,
        domain_log_size: u32,
        log_blowup: u32,
    ) -> Option<Found> {
        let key = key(stripe0);
        let Some(entry) = self.entries.get(&key) else {
            self.stats.misses += 1;
            return None;
        };
        if entry.domain_log_size != domain_log_size
            || entry.log_blowup != log_blowup
            || entry.fingerprint != fingerprint(stripe0.as_slice())
        {
            self.stats.mismatches += 1;
            return None;
        }
        let place = match entry.kept {
            Kept::Host(_) => None,
            Kept::Device { set, buffer, offset } => Some((set, buffer, offset)),
        };
        self.stats.hits += 1;
        self.stats.hit_bytes += 4 * stripe0.length as u64;
        let shared = tree == 0;
        if shared {
            self.stats.copies += 1;
        }
        match place {
            None if shared => match &self.entries[&key].kept {
                Kept::Host(values) => Some(Found::Host(values.clone())),
                Kept::Device { .. } => unreachable!(),
            },
            None => match self.entries.remove(&key).map(|entry| entry.kept) {
                Some(Kept::Host(values)) => {
                    self.stats.held_bytes -= 4 * values.length as u64;
                    Some(Found::Host(values))
                }
                _ => unreachable!(),
            },
            Some((set, buffer, offset)) => {
                if !shared {
                    self.entries.remove(&key);
                }
                Some(Found::Device { set, buffer, offset })
            }
        }
    }

    /// Buffer `buffer` of the kept set `set` (see [`Found::Device`]).
    pub fn device_buffer(&self, set: usize, buffer: usize) -> &wgpu::Buffer {
        &self.device[set].as_ref().expect("a kept set").buffers[buffer]
    }

    /// Notes `n` columns recomputed (`bytes` of them), in `times`.
    pub fn recomputed(&mut self, n: usize, bytes: u64, times: &crate::recompute::RecomputeTimes) {
        self.stats.hits += n as u64;
        self.stats.hit_bytes += bytes;
        self.stats.recompute.add(times);
    }

    pub fn add_readback_times(&mut self, wait: Duration, copy: Duration) {
        self.stats.readback_wait += wait;
        self.stats.readback_copy += copy;
    }

    pub fn add_lookup_time(&mut self, lookup: Duration, transfer: Duration) {
        self.stats.lookup += lookup;
        self.stats.transfer += transfer;
    }

    /// The stats since the last call (`held_bytes` stays).
    pub fn take_stats(&mut self) -> PartStats {
        let held = self.stats.held_bytes;
        let stats = std::mem::take(&mut self.stats);
        self.stats.held_bytes = held;
        self.stats.max_held_bytes = held;
        stats
    }
}
