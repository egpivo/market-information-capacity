//! The latent valuation `V`, at the top of the information hierarchy.
//!
//! `V` is the quantity every other layer is trying to learn. Nothing downstream
//! of the source layer is allowed to see it.

use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

use crate::model::types::LatentValue;

/// A process generating the latent economic state.
///
/// This is the seam for changing *what the market is trying to price*: a
/// heavier-tailed valuation, a regime-switching one, a dynamic one, an
/// empirically calibrated one. Nothing downstream depends on which it is.
pub trait LatentProcess {
    /// Draw one realization of the latent valuation.
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> LatentValue;
}

/// The canonical latent process: `V ~ N(0, sigma_v^2)`.
///
/// The canonical model uses `sigma_v = 1`, which fixes the numeraire: every
/// error variance elsewhere is expressed relative to one unit of latent-value
/// standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianLatent {
    sigma: f64,
}

impl GaussianLatent {
    /// A standard normal latent value, the canonical choice.
    pub const fn standard() -> Self {
        Self { sigma: 1.0 }
    }

    /// A normal latent value with the given standard deviation.
    pub const fn with_sigma(sigma: f64) -> Self {
        Self { sigma }
    }

    /// The latent standard deviation.
    #[inline]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Default for GaussianLatent {
    fn default() -> Self {
        Self::standard()
    }
}

impl LatentProcess for GaussianLatent {
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> LatentValue {
        let z: f64 = rng.sample(StandardNormal);
        LatentValue(self.sigma * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::metrics::OnlineStats;
    use crate::rng::{MASTER_SEED, StreamId, stream};

    #[test]
    fn canonical_latent_is_standard_normal() {
        let process = GaussianLatent::standard();
        let mut rng = stream(MASTER_SEED, StreamId::new("test-latent", 0, 0, 0));
        let mut stats = OnlineStats::new();
        for _ in 0..200_000 {
            stats.push(process.sample(&mut rng).value());
        }
        assert!(stats.mean().abs() < 0.02, "mean was {}", stats.mean());
        assert!(
            (stats.variance() - 1.0).abs() < 0.02,
            "variance was {}",
            stats.variance()
        );
    }

    #[test]
    fn sigma_scales_the_latent_value() {
        let process = GaussianLatent::with_sigma(2.0);
        let mut rng = stream(MASTER_SEED, StreamId::new("test-latent-sigma", 0, 0, 0));
        let mut stats = OnlineStats::new();
        for _ in 0..200_000 {
            stats.push(process.sample(&mut rng).value());
        }
        assert!((stats.variance() - 4.0).abs() < 0.1, "{}", stats.variance());
    }
}
