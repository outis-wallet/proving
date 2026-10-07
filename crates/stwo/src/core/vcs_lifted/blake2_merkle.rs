use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Blake2sChannel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};

pub type Blake2sMerkleHasher = Blake2sHasher;

impl MerkleHasherLifted for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Self::default();
        let (left_child, right_child) = children_hashes;
        hasher.update(&left_child.0);
        hasher.update(&right_child.0);

        hasher.finalize()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        column_values.iter().for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

/// A [`MerkleChannel`] that commits with the **standard** Blake2s Merkle hasher and runs
/// Fiat-Shamir through the **standard** Blake2s channel.
#[derive(Default)]
pub struct Blake2sMerkleChannel;

impl MerkleChannel for Blake2sMerkleChannel {
    type C = Blake2sChannel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2sHash) {
        channel.update_digest(Blake2sHasher::concat_and_hash(&channel.digest(), &root));
    }
}

/// Dummy implementations of `Serialize` and `Deserialize` for `Blake2sMerkleHasher` (we
/// cannot simply derive them because its inner field doesn't implement these traits and is from an
/// external crate).
/// Note: remove this code when possible.
impl Serialize for Blake2sMerkleHasher {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = serializer.serialize_struct("Blake2sMerkleHasher", 1)?;
        serde::ser::SerializeStruct::end(ser)
    }
}

impl<'de> Deserialize<'de> for Blake2sMerkleHasher {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}
