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

use crate::stream::{DeviceStripes, StripeContent};

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
    /// Each column's coefficients, kept until its tree's decommitment: constraint evaluation on
    /// the GPU evaluates the stripes it reads from them, the decommitment the rows it opens (see
    /// [`crate::decommit`]); the CPU evaluates the parts of the components it evaluates.
    Coefficients,
    /// A3 with stripe 0 too, each column's stripes 0 and 1 side by side in storage buffers: the
    /// columns resident for constraint evaluation on the GPU (see [`crate::constraint::composer`]),
    /// stripe 1 read back on request for the components evaluated on the CPU.
    Resident,
    /// A2: recomputed on the GPU from stripe 0 on request.
    Recompute,
}

impl PartsMode {
    /// `STWO_GPU_PARTS`: `host` (or `a1`), `device` (`a3`), `recompute` (`a2`), `resident`;
    /// `Off` otherwise.
    pub fn from_env() -> Self {
        match std::env::var("STWO_GPU_PARTS").unwrap_or_default().to_ascii_lowercase().as_str() {
            "host" | "a1" => Self::Host,
            "device" | "a3" => Self::Device,
            "recompute" | "a2" => Self::Recompute,
            "resident" => Self::Resident,
            "coefficients" => Self::Coefficients,
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
    /// In `device[set]`'s buffer `buffer`, at byte `offset`; and, if the set holds stripe 0
    /// too, the column's stripes 0 and 1 at byte `resident` of that buffer.
    Device {
        set: usize,
        buffer: usize,
        offset: u64,
        resident: Option<u64>,
    },
    /// The column's coefficients, at byte `offset` of `device[set]`'s buffer `buffer`.
    Coefficients { set: usize, buffer: usize, offset: u64 },
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
#[derive(Clone, Debug, Default)]
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
    /// Recompute: the requests' split (see [`crate::recompute::RecomputeTimes`]), and the GPU
    /// memory it keeps at most (working buffers, twiddles).
    pub recompute: crate::recompute::RecomputeTimes,
    pub recompute_bytes: u64,
    /// Clears with something kept.
    pub clears: u64,
    /// Trees not kept, as they would have exceeded the video memory policy (see
    /// [`crate::memory::VramPolicy`]): its explicit cap, the process's share of the device, the
    /// driver's budget.
    pub over_budget: u64,
    pub refused_cap: u64,
    pub refused_share: u64,
    pub refused_driver_budget: u64,
    /// Per tree the commitments offered to keep (in order), each way tried: its bytes, what it
    /// would hold, whether kept, and the process's reserved bytes at the time.
    pub trees: Vec<(u64, StripeContent, bool, u64)>,
    /// Resident columns found for constraint evaluation on the GPU, and not found.
    pub resident_hits: u64,
    pub resident_misses: u64,
}

/// Where a kept column is on the GPU (see [`PartStore::placement`]).
#[derive(Clone, Debug)]
pub enum Placement {
    /// Its stripes 0 and 1, from byte `byte_offset` of `buffer`.
    Both { buffer: wgpu::Buffer, byte_offset: u64 },
    /// Its stripe 1 only, from byte `byte_offset` of `buffer`.
    Stripe1 { buffer: wgpu::Buffer, byte_offset: u64 },
    /// Its coefficients, from byte `byte_offset` of `buffer`.
    Coefficients { buffer: wgpu::Buffer, byte_offset: u64 },
}

/// A column resident on the GPU: its stripes 0 and 1, `2^(log_size + 1)` words from word
/// `word_offset` of `buffer`.
#[derive(Clone, Debug)]
pub struct ResidentColumn {
    pub buffer: wgpu::Buffer,
    pub word_offset: u64,
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

    /// Called at the start of every commitment: the first after a lookup clears the store (and
    /// returns true: in a proof, the composition's commitment, whose columns constraint
    /// evaluation never reads); coefficients (`keep_coefficients`) stay until their tree's
    /// decommitment (see [`Self::decommitted`]).
    pub fn commitment_starts(&mut self, keep_coefficients: bool) -> bool {
        let after_lookup = self.looked_up;
        if after_lookup && !keep_coefficients {
            self.clear();
        }
        self.looked_up = false;
        after_lookup
    }

    /// Drops the coefficients of the columns the tree holds as `stripe0` (its decommitment is
    /// done), and frees each set none of whose columns is left.
    pub fn decommitted(&mut self, stripe0: &[&BaseColumn]) {
        let mut sets = vec![];
        for values in stripe0 {
            if let Some(Entry { kept: Kept::Coefficients { set, .. }, .. }) =
                self.entries.get(&key(values))
            {
                sets.push(*set);
                self.entries.remove(&key(values));
            }
        }
        sets.sort();
        sets.dedup();
        for set in sets {
            let in_use = self.entries.values().any(|entry| {
                matches!(entry.kept, Kept::Coefficients { set: s, .. } | Kept::Device { set: s, .. } if s == set)
            });
            if !in_use && let Some(stripes) = self.device[set].take() {
                self.stats.held_bytes -= stripes.bytes();
                stripes.destroy();
            }
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

    /// The GPU memory the kept stripes hold.
    pub fn held_bytes(&self) -> u64 {
        self.stats.held_bytes
    }

    /// Notes a tree offered to keep.
    pub fn offered(&mut self, bytes: u64, content: StripeContent, kept: bool, reserved: u64) {
        self.stats.trees.push((bytes, content, kept, reserved));
    }

    /// Counts a tree not kept, for `reason`.
    pub fn refused(&mut self, reason: crate::memory::VramRefusal) {
        use crate::memory::VramRefusal;
        self.stats.over_budget += 1;
        match reason {
            VramRefusal::Cap => self.stats.refused_cap += 1,
            VramRefusal::ProcessShare => self.stats.refused_share += 1,
            VramRefusal::DriverBudget => self.stats.refused_driver_budget += 1,
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
        for (i, (values, &log_size)) in stripe0.iter().zip(log_sizes).enumerate() {
            let kept = match stripes.content {
                StripeContent::Coefficients => {
                    let Some((buffer, offset)) = stripes.places[i] else { continue };
                    Kept::Coefficients { set, buffer, offset }
                }
                _ => {
                    let Some((buffer, offset)) = stripes.stripe1_place(i) else { continue };
                    let resident = stripes.with_stripe0().then(|| stripes.places[i].unwrap().1);
                    Kept::Device { set, buffer, offset, resident }
                }
            };
            self.entries.insert(
                key(values),
                Entry {
                    fingerprint: fingerprint(values.as_slice()),
                    domain_log_size: log_size + log_blowup,
                    log_blowup,
                    kept,
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
            Kept::Device { set, buffer, offset, .. } => Some((set, buffer, offset)),
            // Coefficients only: the CPU evaluates the part.
            Kept::Coefficients { .. } => {
                self.stats.misses += 1;
                return None;
            }
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
                _ => unreachable!(),
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

    /// The column the tree holds as `stripe0` (of committed domain log size `domain_log_size`
    /// and blowup `log_blowup`), resident on the GPU with its stripes 0 and 1 (in
    /// [`PartsMode::Resident`]); `None` if not. The column stays (unlike [`Self::find`]'s).
    pub fn resident(
        &mut self,
        stripe0: &BaseColumn,
        domain_log_size: u32,
        log_blowup: u32,
    ) -> Option<ResidentColumn> {
        self.looked_up = true;
        let found = self.entries.get(&key(stripe0)).and_then(|entry| {
            let Kept::Device { set, buffer, resident: Some(offset), .. } = entry.kept else {
                return None;
            };
            (entry.domain_log_size == domain_log_size
                && entry.log_blowup == log_blowup
                && entry.fingerprint == fingerprint(stripe0.as_slice()))
            .then(|| ResidentColumn {
                buffer: self.device[set].as_ref().expect("a kept set").buffers[buffer].clone(),
                word_offset: offset / 4,
            })
        });
        match found {
            Some(_) => self.stats.resident_hits += 1,
            None => self.stats.resident_misses += 1,
        }
        found
    }

    /// Where the column the tree holds as `stripe0` is on the GPU, without taking it: both its
    /// stripes (see [`Self::resident`]), or its stripe 1 only; `None` if not kept.
    pub fn placement(
        &mut self,
        stripe0: &BaseColumn,
        domain_log_size: u32,
        log_blowup: u32,
    ) -> Option<Placement> {
        self.looked_up = true;
        let entry = self.entries.get(&key(stripe0))?;
        if entry.domain_log_size != domain_log_size
            || entry.log_blowup != log_blowup
            || entry.fingerprint != fingerprint(stripe0.as_slice())
        {
            return None;
        }
        let device = |set: usize, buffer: usize| {
            self.device[set].as_ref().expect("a kept set").buffers[buffer].clone()
        };
        Some(match entry.kept {
            Kept::Host(_) => return None,
            Kept::Device { set, buffer, resident: Some(resident), .. } => {
                Placement::Both { buffer: device(set, buffer), byte_offset: resident }
            }
            Kept::Device { set, buffer, offset, resident: None } => {
                Placement::Stripe1 { buffer: device(set, buffer), byte_offset: offset }
            }
            Kept::Coefficients { set, buffer, offset } => {
                Placement::Coefficients { buffer: device(set, buffer), byte_offset: offset }
            }
        })
    }

    /// Buffer `buffer` of the kept set `set` (see [`Found::Device`]).
    pub fn device_buffer(&self, set: usize, buffer: usize) -> &wgpu::Buffer {
        &self.device[set].as_ref().expect("a kept set").buffers[buffer]
    }

    /// Notes `n` columns recomputed (`bytes` of them), in `times`.
    pub fn recomputed(
        &mut self,
        n: usize,
        bytes: u64,
        times: &crate::recompute::RecomputeTimes,
        kept_bytes: u64,
    ) {
        self.stats.hits += n as u64;
        self.stats.hit_bytes += bytes;
        self.stats.recompute.add(times);
        self.stats.recompute_bytes = self.stats.recompute_bytes.max(kept_bytes);
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
