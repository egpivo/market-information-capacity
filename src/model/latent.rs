//! The latent valuation `V` sitting at the top of the information hierarchy.
//!
//! `V` is the quantity every other layer is trying to learn. It is drawn as a
//! standard normal, which fixes the numeraire: all error variances elsewhere in
//! the model are expressed relative to one unit of latent-value standard
//! deviation.

use rand::Rng;
use rand_distr::StandardNormal;

/// One draw of the latent valuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Latent(pub f64);

impl Latent {
    /// The scalar latent value.
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Draw `V ~ N(0, 1)`.
#[inline]
pub fn draw_latent<R: Rng + ?Sized>(rng: &mut R) -> Latent {
    Latent(rng.sample(StandardNormal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::metrics::OnlineStats;
    use crate::rng::{MASTER_SEED, StreamId, stream};

    #[test]
    fn latent_is_standard_normal() {
        let mut rng = stream(MASTER_SEED, StreamId::new("test-latent", 0, 0, 0));
        let mut stats = OnlineStats::new();
        for _ in 0..200_000 {
            stats.push(draw_latent(&mut rng).value());
        }
        assert!(stats.mean().abs() < 0.02, "mean was {}", stats.mean());
        assert!(
            (stats.variance() - 1.0).abs() < 0.02,
            "variance was {}",
            stats.variance()
        );
    }
}
