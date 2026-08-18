//! Deterministic seed derivation.
//!
//! Every canonical number in this repository is reproducible from a single
//! master seed. Seeds are derived through a pure hash of the coordinates of the
//! work unit, never by incrementing a counter, so that two different work units
//! cannot accidentally share a stream (a defect in the Python prototype, where
//! additive seed arithmetic such as `SEED + K*100 + N` collides).
//!
//! The derivation tree is
//!
//! ```text
//! master seed
//!   -> experiment  (stable string label, e.g. "asymptote")
//!     -> cell      (index of the parameter cell within the experiment)
//!       -> block   (independent seed block used for block-level standard errors)
//!         -> batch (parallel work unit inside a block)
//! ```
//!
//! Parallelism therefore never changes results: each batch owns a stream fixed
//! before any thread starts, and batch results are reduced in index order.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// Canonical master seed for all publication-scale results.
pub const MASTER_SEED: u64 = 20_260_915;

/// Coordinates of one deterministic work unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamId<'a> {
    /// Stable experiment label.
    pub experiment: &'a str,
    /// Parameter-cell index within the experiment.
    pub cell: u64,
    /// Seed-block index within the cell.
    pub block: u64,
    /// Batch index within the block.
    pub batch: u64,
}

impl<'a> StreamId<'a> {
    /// Build a stream identifier.
    pub fn new(experiment: &'a str, cell: u64, block: u64, batch: u64) -> Self {
        Self {
            experiment,
            cell,
            block,
            batch,
        }
    }
}

/// SplitMix64 finalizer: mixes a 64-bit state into a well-distributed output.
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// FNV-1a over bytes, seeded by an existing state.
fn fnv1a(mut state: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        state ^= u64::from(b);
        state = state.wrapping_mul(0x0000_0100_0000_01B3);
    }
    state
}

/// Derive the 64-bit seed for a work unit under a master seed.
///
/// The mapping is a pure function of `(master, experiment, cell, block, batch)`.
pub fn derive_seed(master: u64, id: StreamId<'_>) -> u64 {
    let mut state = fnv1a(0xCBF2_9CE4_8422_2325, &master.to_le_bytes());
    state = fnv1a(state, id.experiment.as_bytes());
    state = fnv1a(state, &id.cell.to_le_bytes());
    state = fnv1a(state, &id.block.to_le_bytes());
    state = fnv1a(state, &id.batch.to_le_bytes());
    splitmix64(state)
}

/// Build the deterministic RNG for a work unit.
pub fn stream(master: u64, id: StreamId<'_>) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(derive_seed(master, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn derivation_is_pure() {
        let id = StreamId::new("asymptote", 3, 1, 7);
        assert_eq!(
            derive_seed(MASTER_SEED, id),
            derive_seed(MASTER_SEED, id),
            "seed derivation must be a pure function"
        );
    }

    #[test]
    fn neighbouring_coordinates_do_not_collide() {
        let mut seen = HashSet::new();
        for experiment in ["asymptote", "grid", "worlds"] {
            for cell in 0..64u64 {
                for block in 0..4u64 {
                    for batch in 0..8u64 {
                        let s =
                            derive_seed(MASTER_SEED, StreamId::new(experiment, cell, block, batch));
                        assert!(seen.insert(s), "seed collision at {experiment}/{cell}");
                    }
                }
            }
        }
    }

    #[test]
    fn master_seed_changes_every_stream() {
        let id = StreamId::new("grid", 0, 0, 0);
        assert_ne!(
            derive_seed(MASTER_SEED, id),
            derive_seed(MASTER_SEED + 1, id)
        );
    }
}
