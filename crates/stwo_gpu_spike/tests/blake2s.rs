//! The WGSL Blake2s against stwo's hasher (`Blake2sHasher`, the `blake2` crate's Blake2s256) on
//! messages of whole words: the empty message, partial and exact blocks, several blocks.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use stwo_gpu_spike::GpuContext;
use stwo_gpu_spike::blake2s::{blake2s_wgsl, block_expr};

fn kernel() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
// Message i: (word offset, word count) at 2i, 2i + 1.
@group(0) @binding(1) var<storage, read> messages: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn word(offset: u32, len: u32, k: u32) -> u32 {{
    if (k < len) {{ return words[offset + k]; }}
    return 0u;
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&messages) / 2u) {{ return; }}
    let offset = messages[2u * i];
    let len = messages[2u * i + 1u];
    var h = blake2s_init();
    let n_blocks = select(0u, (len - 1u) / 16u, len > 0u);
    for (var b = 0u; b < n_blocks; b++) {{
        let base = 16u * b;
        h = blake2s_compress(h, {block}, 64u * (b + 1u), 0u, false);
    }}
    let base = 16u * n_blocks;
    h = blake2s_compress(h, {block}, 4u * len, 0u, true);
    for (var w = 0u; w < 8u; w++) {{ out[8u * i + w] = h[w]; }}
}}

"#,
        block = block_expr("word", "offset, len", "base"),
    )
}

const PAIRS_KERNEL: &str = r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

// Node hashing: pairs of hashes from `words`, (2i, 2i + 1).
@compute @workgroup_size(64)
fn pairs(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&out) / 8u) { return; }
    var l: array<u32, 8>;
    var r: array<u32, 8>;
    for (var w = 0u; w < 8u; w++) {
        l[w] = words[16u * i + w];
        r[w] = words[16u * i + 8u + w];
    }
    let h = blake2s_hash_pair(l, r);
    for (var w = 0u; w < 8u; w++) { out[8u * i + w] = h[w]; }
}
"#;

fn hash_from_words(words: &[u32]) -> Blake2sHash {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Blake2sHash(bytes.try_into().unwrap())
}

#[test]
fn blake2s_matches_stwo() {
    let gpu = GpuContext::new();
    let mut rng = SmallRng::seed_from_u64(2);
    // Lengths in words: empty, partial block, exact block, one past, exact multi-block, longer.
    let mut lens: Vec<usize> = vec![0, 1, 2, 15, 16, 17, 31, 32, 33, 48, 64, 100, 163, 256, 1000];
    lens.extend((0..200).map(|_| rng.random_range(0..400)));
    let mut words = vec![];
    let mut messages = vec![];
    for &len in &lens {
        messages.extend([words.len() as u32, len as u32]);
        words.extend((0..len).map(|_| rng.random::<u32>()));
    }
    words.push(0); // No empty buffer.

    let source = format!("{}\n{}", blake2s_wgsl(), kernel());
    let pipeline = gpu.pipeline("blake2s-test", &source, "main");
    let (words_buffer, messages_buffer) = (gpu.upload("words", &words), gpu.upload("messages", &messages));
    let out = gpu.storage("out", 8 * lens.len());
    gpu.run(&pipeline, &[&words_buffer, &messages_buffer, &out], (lens.len().div_ceil(64) as u32, 1, 1));
    let out = gpu.read(&out, 8 * lens.len());

    for (i, &len) in lens.iter().enumerate() {
        let offset = messages[2 * i] as usize;
        let message: Vec<u8> = words[offset..offset + len].iter().flat_map(|w| w.to_le_bytes()).collect();
        let expected = Blake2sHasher::hash(&message);
        assert_eq!(hash_from_words(&out[8 * i..8 * i + 8]), expected, "message of {len} words");
    }
}

#[test]
fn blake2s_node_hash_matches_stwo() {
    let gpu = GpuContext::new();
    let mut rng = SmallRng::seed_from_u64(3);
    let n = 1000;
    let words: Vec<u32> = (0..16 * n).map(|_| rng.random()).collect();
    let source = format!("{}\n{}", blake2s_wgsl(), PAIRS_KERNEL);
    let pipeline = gpu.pipeline("blake2s-pairs", &source, "pairs");
    let words_buffer = gpu.upload("words", &words);
    let out = gpu.storage("out", 8 * n);
    gpu.run(&pipeline, &[&words_buffer, &out], (n.div_ceil(64) as u32, 1, 1));
    let out = gpu.read(&out, 8 * n);
    for i in 0..n {
        let left = hash_from_words(&words[16 * i..16 * i + 8]);
        let right = hash_from_words(&words[16 * i + 8..16 * i + 16]);
        assert_eq!(
            hash_from_words(&out[8 * i..8 * i + 8]),
            Blake2sMerkleHasher::hash_children((left, right)),
            "node {i}"
        );
    }
}
