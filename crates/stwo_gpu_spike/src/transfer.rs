//! Moving columns between the host and the GPU through persistent staging rings.
//!
//! `queue.write_buffer` (and `write_buffer_with`) allocates a fresh staging buffer per call: on
//! the RTX 3080 (Vulkan) the 516 MiB of the Cairo base trace's coefficients took 225 ms that
//! way (2.3 GB/s), the allocations of mapped memory most of it. An [`UploadRing`] keeps a few
//! mapped chunks (`MAP_WRITE`) for the life of the process instead, fills each straight from the
//! columns with all the CPU's threads, and copies it on the GPU into place: 54 ms (10 GB/s,
//! PCIe 3.0's practical bound). Where the driver offers device-local mapped memory (the BAR,
//! 256 MiB on that card, which wgpu prefers for `MAP_WRITE` buffers) the CPU's writes cross PCIe
//! themselves and the GPU's copy is local; otherwise the chunks are in host memory and the copy
//! crosses PCIe: the same speed (see `bench_transfer`).
//!
//! A [`ReadbackRing`] does the reverse with `MAP_READ` chunks (host memory): each chunk's copy is
//! its own submission, so that the CPU copies a chunk out (in parallel) while the GPU copies the
//! next ones in: 516 MiB in about 48 ms, against 62 ms through one staging buffer of the whole
//! (and that buffer's allocation, 275 ms, every time).
//!
//! On wgpu's one queue, copies and compute run one after the other (`bench_transfer` measures
//! it): a copy to or from the GPU never overlaps its compute. But the CPU's writes into a mapped
//! chunk do, as does the CPU's copying out of one.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::GpuContext;

/// The default size of a chunk of a ring.
pub const DEFAULT_CHUNK_BYTES: u64 = 32 << 20;
/// The default number of chunks of an upload ring (in the BAR when there is room: a few).
pub const DEFAULT_UPLOAD_DEPTH: usize = 2;
/// The default number of chunks of a readback ring.
pub const DEFAULT_READBACK_DEPTH: usize = 3;
/// The largest copy of one rayon task, in bytes.
const MAX_PART_BYTES: usize = 1 << 18;

/// Bytes of the host to write at `dst_offset` of `dst`.
#[derive(Clone, Copy, Debug)]
pub struct Upload<'a> {
    pub dst: &'a wgpu::Buffer,
    pub dst_offset: u64,
    pub src: &'a [u8],
}

/// Bytes at `src_offset` of `src` to read into `dst`.
#[derive(Debug)]
pub struct Readback<'a, 'b> {
    pub src: &'a wgpu::Buffer,
    pub src_offset: u64,
    pub dst: &'b mut [u8],
}

/// The outcome of a slot's `map_async`, set by its callback.
type MapResult = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// A chunk of a ring: its buffer, and the submission that last used it if it is being mapped
/// again.
struct Slot {
    buffer: wgpu::Buffer,
    /// Set while the slot's map is pending: the submission to wait for, and the map's outcome.
    pending: Option<(wgpu::SubmissionIndex, MapResult)>,
}

impl Slot {
    fn new(gpu: &GpuContext, label: &str, bytes: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: usage.contains(wgpu::BufferUsages::MAP_WRITE),
        });
        Self { buffer, pending: None }
    }

    /// Maps the slot again (for `mode`) once `index` has run.
    fn remap(&mut self, index: wgpu::SubmissionIndex, mode: wgpu::MapMode, bytes: u64) {
        let result: MapResult = Arc::default();
        let set = result.clone();
        self.buffer.slice(..bytes).map_async(mode, move |outcome| {
            *set.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(outcome);
        });
        self.pending = Some((index, result));
    }

    /// Waits until the slot is mapped.
    fn wait(&mut self, gpu: &GpuContext) {
        if let Some((index, result)) = self.pending.take() {
            gpu.device
                .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
                .expect("poll");
            let outcome = result.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
            match outcome {
                Some(Ok(())) => {}
                Some(Err(error)) => panic!("mapping a staging chunk: {error}"),
                None => panic!("mapping a staging chunk: not done after its submission"),
            }
        }
    }
}

/// A run of bytes of one chunk: `len` bytes at `chunk_offset` of the chunk, from or to
/// `offset` of `buffer` (the GPU's side), and the host's side, piece `piece` from its byte
/// `piece_offset`.
#[derive(Clone, Copy, Debug)]
struct Run {
    chunk_offset: u64,
    buffer: usize,
    offset: u64,
    len: u64,
    piece: usize,
    piece_offset: usize,
}

/// Packs pieces of `lens` bytes, at `(buffer, offset)` on the GPU, one after the other into
/// chunks of `chunk` bytes: the runs of each chunk.
fn pack(places: &[(usize, u64)], lens: &[usize], chunk: u64) -> Vec<Vec<Run>> {
    let mut chunks: Vec<Vec<Run>> = vec![];
    let mut fill = chunk;
    for (piece, (&(buffer, offset), &len)) in places.iter().zip(lens).enumerate() {
        let mut done = 0usize;
        while done < len {
            if fill == chunk {
                chunks.push(vec![]);
                fill = 0;
            }
            let n = ((len - done) as u64).min(chunk - fill);
            chunks.last_mut().unwrap().push(Run {
                chunk_offset: fill,
                buffer,
                offset: offset + done as u64,
                len: n,
                piece,
                piece_offset: done,
            });
            fill += n;
            done += n as usize;
        }
    }
    chunks
}

/// The copies of a chunk's runs: runs adjacent on both sides in one copy, `(chunk offset,
/// buffer, offset, len)`.
fn copies(runs: &[Run]) -> Vec<(u64, usize, u64, u64)> {
    let mut copies: Vec<(u64, usize, u64, u64)> = vec![];
    for run in runs {
        match copies.last_mut() {
            Some((chunk_offset, buffer, offset, len))
                if *buffer == run.buffer
                    && *offset + *len == run.offset
                    && *chunk_offset + *len == run.chunk_offset =>
            {
                *len += run.len
            }
            _ => copies.push((run.chunk_offset, run.buffer, run.offset, run.len)),
        }
    }
    copies
}

/// The distinct buffers of `buffers`, and each one's index among them.
fn buffer_indices<'a>(
    buffers: impl Iterator<Item = &'a wgpu::Buffer>,
) -> (Vec<&'a wgpu::Buffer>, Vec<usize>) {
    let mut distinct: Vec<&wgpu::Buffer> = vec![];
    let indices = buffers
        .map(|buffer| match distinct.iter().position(|&known| known == buffer) {
            Some(i) => i,
            None => {
                distinct.push(buffer);
                distinct.len() - 1
            }
        })
        .collect();
    (distinct, indices)
}

/// A pointer to a mapped chunk, for rayon's threads, which write disjoint ranges of it.
#[derive(Clone, Copy)]
struct ChunkPtr(*mut u8);
// Safety: the threads write disjoint ranges, within the mapping, while it lives.
unsafe impl Send for ChunkPtr {}
unsafe impl Sync for ChunkPtr {}

impl ChunkPtr {
    /// The pointer (a method, so that closures capture the whole `ChunkPtr`, `Sync`).
    fn at(&self, offset: usize) -> *mut u8 {
        self.0.wrapping_add(offset)
    }
}

/// Persistent mapped chunks (`MAP_WRITE`) the host's data goes to the GPU through (see the
/// module docs).
pub struct UploadRing {
    slots: Vec<Slot>,
    chunk: u64,
    next: usize,
}

impl UploadRing {
    /// `depth` chunks of `chunk` bytes (a multiple of 4).
    pub fn new(gpu: &GpuContext, chunk: u64, depth: usize) -> Self {
        assert!(depth > 0 && chunk > 0 && chunk.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
        let usage = wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC;
        let slots = (0..depth).map(|_| Slot::new(gpu, "upload chunk", chunk, usage)).collect();
        Self { slots, chunk, next: 0 }
    }

    /// With [`DEFAULT_CHUNK_BYTES`] and [`DEFAULT_UPLOAD_DEPTH`].
    pub fn with_defaults(gpu: &GpuContext) -> Self {
        Self::new(gpu, DEFAULT_CHUNK_BYTES, DEFAULT_UPLOAD_DEPTH)
    }

    /// The bytes of its chunks (host-visible memory: the BAR, or the host's).
    pub fn bytes(&self) -> u64 {
        self.chunk * self.slots.len() as u64
    }

    /// The bytes of a chunk.
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk
    }

    /// Writes every piece into its buffer: chunk by chunk, each filled by all the CPU's threads
    /// and then copied into place by its own submission. The writes are submitted when this
    /// returns (not necessarily done): later submissions see them.
    pub fn upload(&mut self, gpu: &GpuContext, pieces: &[Upload<'_>]) {
        let (buffers, indices) = buffer_indices(pieces.iter().map(|piece| piece.dst));
        let places = pieces
            .iter()
            .zip(&indices)
            .map(|(piece, &i)| (i, piece.dst_offset))
            .collect::<Vec<_>>();
        let lens = pieces.iter().map(|piece| piece.src.len()).collect::<Vec<_>>();
        for runs in pack(&places, &lens, self.chunk) {
            let bytes = runs.iter().map(|run| run.len).sum::<u64>();
            let depth = self.slots.len();
            let slot = &mut self.slots[self.next];
            self.next = (self.next + 1) % depth;
            slot.wait(gpu);
            {
                let mut view = slot.buffer.slice(..bytes).get_mapped_range_mut().expect("mapped");
                let mut whole = view.slice(..);
                assert_eq!(whole.len() as u64, bytes);
                let base = ChunkPtr(whole.as_raw_ptr().as_ptr() as *mut u8);
                // Every run, in parts of at most MAX_PART_BYTES, by all the threads.
                let parts = runs
                    .iter()
                    .flat_map(|run| {
                        let src = &pieces[run.piece].src
                            [run.piece_offset..run.piece_offset + run.len as usize];
                        src.chunks(MAX_PART_BYTES).enumerate().map(move |(k, part)| {
                            (run.chunk_offset as usize + k * MAX_PART_BYTES, part)
                        })
                    })
                    .collect::<Vec<_>>();
                parts.par_iter().for_each(|&(offset, src)| {
                    // Safety: the runs are disjoint ranges of the chunk, within `bytes`, the
                    // mapped range, which `view` keeps mapped until after this.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), base.at(offset), src.len())
                    };
                });
            }
            slot.buffer.unmap();
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("upload chunk"),
            });
            for (chunk_offset, buffer, offset, len) in copies(&runs) {
                encoder.copy_buffer_to_buffer(
                    &slot.buffer,
                    chunk_offset,
                    buffers[buffer],
                    offset,
                    len,
                );
            }
            let index = gpu.queue.submit([encoder.finish()]);
            slot.remap(index, wgpu::MapMode::Write, self.chunk);
        }
    }
}

/// Persistent chunks (`MAP_READ`) the GPU's data comes back to the host through (see the module
/// docs).
pub struct ReadbackRing {
    slots: Vec<Slot>,
    /// Per slot: whether a readback holds it (started, not finished).
    taken: Vec<bool>,
    chunk: u64,
}

/// A chunk being read back (see [`ReadbackRing::start`]).
#[must_use = "a chunk read back stays taken until it is finished"]
pub struct Ticket {
    slot: usize,
    /// The bytes of the chunk read back.
    bytes: u64,
    /// The chunk offsets of the ranges read, in order.
    offsets: Vec<u64>,
}

impl ReadbackRing {
    /// `depth` chunks of `chunk` bytes (a multiple of 4).
    pub fn new(gpu: &GpuContext, chunk: u64, depth: usize) -> Self {
        assert!(depth > 0 && chunk > 0 && chunk.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let slots = (0..depth).map(|_| Slot::new(gpu, "readback chunk", chunk, usage)).collect();
        Self { slots, taken: vec![false; depth], chunk }
    }

    /// With [`DEFAULT_CHUNK_BYTES`] and [`DEFAULT_READBACK_DEPTH`].
    pub fn with_defaults(gpu: &GpuContext) -> Self {
        Self::new(gpu, DEFAULT_CHUNK_BYTES, DEFAULT_READBACK_DEPTH)
    }

    /// The bytes of its chunks (host memory).
    pub fn bytes(&self) -> u64 {
        self.chunk * self.slots.len() as u64
    }

    /// The bytes of a chunk.
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk
    }

    /// Whether a chunk is free (see [`Self::start`]).
    pub fn has_free_chunk(&self) -> bool {
        self.taken.contains(&false)
    }

    /// Starts the copy of `ranges` (`(buffer, offset, bytes)`, at most a chunk in all) into a free
    /// chunk, after all the work submitted before, in a submission of its own. Panics if no
    /// chunk is free: [`Self::finish`] one first.
    pub fn start(&mut self, gpu: &GpuContext, ranges: &[(&wgpu::Buffer, u64, u64)]) -> Ticket {
        let (buffers, indices) = buffer_indices(ranges.iter().map(|range| range.0));
        let mut fill = 0;
        let runs = ranges
            .iter()
            .zip(indices)
            .enumerate()
            .map(|(piece, (&(_, offset, len), buffer))| {
                let run = Run { chunk_offset: fill, buffer, offset, len, piece, piece_offset: 0 };
                fill += len;
                run
            })
            .collect::<Vec<_>>();
        assert!(fill <= self.chunk, "{fill} bytes to read back exceed a chunk");
        let offsets = runs.iter().map(|run| run.chunk_offset).collect();
        let slot = self.start_runs(gpu, &buffers, &runs);
        Ticket { slot, bytes: fill, offsets }
    }

    /// Waits for `ticket`'s chunk and copies it out, by every thread: `dsts[i]` takes the bytes
    /// of the ticket's `i`-th range (as long). The chunk is free again.
    pub fn finish(&mut self, gpu: &GpuContext, ticket: Ticket, dsts: Vec<&mut [u8]>) {
        assert_eq!(dsts.len(), ticket.offsets.len());
        let parts = ticket.offsets.iter().map(|&offset| offset as usize).zip(dsts).collect();
        self.finish_runs(gpu, ticket.slot, ticket.bytes, parts);
    }

    /// Starts copying `runs` (from `buffers`) into a free chunk; returns the chunk's slot.
    fn start_runs(&mut self, gpu: &GpuContext, buffers: &[&wgpu::Buffer], runs: &[Run]) -> usize {
        let k = self.taken.iter().position(|&taken| !taken).expect("no free readback chunk");
        self.taken[k] = true;
        let slot = &mut self.slots[k];
        let bytes = runs.iter().map(|run| run.len).sum::<u64>();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback chunk"),
        });
        for (chunk_offset, buffer, offset, len) in copies(runs) {
            encoder.copy_buffer_to_buffer(buffers[buffer], offset, &slot.buffer, chunk_offset, len);
        }
        let index = gpu.queue.submit([encoder.finish()]);
        slot.remap(index, wgpu::MapMode::Read, bytes.max(wgpu::COPY_BUFFER_ALIGNMENT));
        k
    }

    /// Waits for chunk `k` (`bytes` read back) and copies `parts` (`(chunk offset,
    /// destination)`) out of it; frees it.
    fn finish_runs(
        &mut self,
        gpu: &GpuContext,
        k: usize,
        bytes: u64,
        parts: Vec<(usize, &mut [u8])>,
    ) {
        assert!(self.taken[k], "chunk {k} is not being read back");
        let slot = &mut self.slots[k];
        slot.wait(gpu);
        {
            let view = slot
                .buffer
                .slice(..bytes.max(wgpu::COPY_BUFFER_ALIGNMENT))
                .get_mapped_range()
                .expect("mapped");
            let chunk: &[u8] = &view;
            let parts = parts
                .into_iter()
                .flat_map(|(offset, dst)| {
                    dst.chunks_mut(MAX_PART_BYTES)
                        .enumerate()
                        .map(move |(j, part)| (offset + j * MAX_PART_BYTES, part))
                })
                .collect::<Vec<_>>();
            parts.into_par_iter().for_each(|(offset, part)| {
                part.copy_from_slice(&chunk[offset..offset + part.len()])
            });
        }
        slot.buffer.unmap();
        self.taken[k] = false;
    }

    /// Reads every piece from its buffer, after all the work submitted before: chunk by chunk,
    /// each copied by the GPU in its own submission, then out of its chunk by all the CPU's
    /// threads while the GPU copies the next ones. Every chunk must be free.
    pub fn read(&mut self, gpu: &GpuContext, pieces: Vec<Readback<'_, '_>>) {
        assert!(!self.taken.contains(&true), "chunks still being read back");
        let (buffers, indices) = buffer_indices(pieces.iter().map(|piece| piece.src));
        let places = pieces
            .iter()
            .zip(&indices)
            .map(|(piece, &i)| (i, piece.src_offset))
            .collect::<Vec<_>>();
        let lens = pieces.iter().map(|piece| piece.dst.len()).collect::<Vec<_>>();
        let chunks = pack(&places, &lens, self.chunk);
        // Each chunk's destinations: the runs' parts of the pieces, split off in order.
        let mut rests = pieces.into_iter().map(|piece| piece.dst).collect::<Vec<_>>();
        let mut destinations = chunks
            .iter()
            .map(|runs| {
                runs.iter()
                    .map(|run| {
                        let rest = std::mem::take(&mut rests[run.piece]);
                        let (dst, rest) = rest.split_at_mut(run.len as usize);
                        rests[run.piece] = rest;
                        (run.chunk_offset as usize, dst)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter();
        let depth = self.slots.len();
        let mut started = std::collections::VecDeque::new();
        for runs in chunks.iter().take(depth) {
            started.push_back(self.start_runs(gpu, &buffers, runs));
        }
        for (k, runs) in chunks.iter().enumerate() {
            let slot = started.pop_front().unwrap();
            let bytes = runs.iter().map(|run| run.len).sum::<u64>();
            self.finish_runs(gpu, slot, bytes, destinations.next().unwrap());
            if let Some(runs) = chunks.get(k + depth) {
                started.push_back(self.start_runs(gpu, &buffers, runs));
            }
        }
    }
}
