//! The fundamental source layer.
//!
//! This module owns the *only* place in the crate where new fundamental
//! information about `V` is created. Source `k` is
//!
//! ```text
//! s_k = V + sigma_s * ( sqrt(rho_s) * u + sqrt(1 - rho_s) * eta_k )
//! ```
//!
//! where `u` is a single common source error shared by every source in a
//! realization and `eta_k` is source specific. The number of source draws is
//! `K`, the *source budget*. It is a property of the information environment,
//! never of the trader population: [`SourceLayer::count`] does not take a
//! trader count and [`SourceDraw`] always has exactly `K` entries.

use rand::Rng;
use rand_distr::StandardNormal;

use crate::config::SourceConfig;
use crate::error::{Error, Result};
use crate::model::latent::Latent;

/// One realization of the fundamental source vector `(s_1, ..., s_K)`.
///
/// The buffer is reused across Monte Carlo realizations to keep the simulation
/// allocation free in its hot loop.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceDraw {
    values: Vec<f64>,
}

impl SourceDraw {
    /// Allocate a draw buffer for `k` sources.
    pub fn with_capacity(k: usize) -> Self {
        Self {
            values: Vec::with_capacity(k),
        }
    }

    /// Number of fundamental sources in this draw. Always equals `K`.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the draw is empty. A validated configuration never produces one.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The source values.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Equal-weighted mean of the sources, `(1/K) * sum_k s_k`.
    #[inline]
    pub fn equal_weighted_mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }
}

/// Generator for the fundamental source layer.
#[derive(Debug, Clone)]
pub struct SourceLayer {
    count: usize,
    sigma: f64,
    sqrt_rho: f64,
    sqrt_one_minus_rho: f64,
}

impl SourceLayer {
    /// Build a source layer from a validated configuration.
    pub fn new(cfg: &SourceConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            count: cfg.count,
            sigma: cfg.sigma,
            sqrt_rho: cfg.correlation.sqrt(),
            sqrt_one_minus_rho: (1.0 - cfg.correlation).sqrt(),
        })
    }

    /// The source budget `K`.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Draw `(s_1, ..., s_K)` for one realization of the latent value.
    ///
    /// The draw is written into `out`, which is cleared first. Exactly `K`
    /// values are produced regardless of how many traders will read them.
    pub fn draw_into<R: Rng + ?Sized>(&self, latent: Latent, rng: &mut R, out: &mut SourceDraw) {
        let common: f64 = rng.sample(StandardNormal);
        let shared = latent.value() + self.sigma * self.sqrt_rho * common;
        out.values.clear();
        out.values.reserve(self.count);
        for _ in 0..self.count {
            let specific: f64 = rng.sample(StandardNormal);
            out.values
                .push(shared + self.sigma * self.sqrt_one_minus_rho * specific);
        }
    }
}

impl SourceConfig {
    /// Check the modelling invariants of a source configuration.
    pub fn validate(&self) -> Result<()> {
        if self.count == 0 {
            return Err(Error::Config("source count K must be at least 1".into()));
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(Error::Config(format!(
                "source sigma must be finite and non-negative, got {}",
                self.sigma
            )));
        }
        if !(0.0..=1.0).contains(&self.correlation) {
            return Err(Error::Config(format!(
                "source correlation rho_s must lie in [0, 1], got {}",
                self.correlation
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::metrics::OnlineStats;
    use crate::rng::{MASTER_SEED, StreamId, stream};

    fn cfg(count: usize, sigma: f64, correlation: f64) -> SourceConfig {
        SourceConfig {
            count,
            sigma,
            correlation,
        }
    }

    #[test]
    fn draw_always_has_exactly_k_entries() {
        for k in [1usize, 2, 10, 100] {
            let layer = SourceLayer::new(&cfg(k, 1.0, 0.5)).expect("valid config");
            let mut rng = stream(MASTER_SEED, StreamId::new("test-sources", k as u64, 0, 0));
            let mut draw = SourceDraw::with_capacity(k);
            layer.draw_into(Latent(0.0), &mut rng, &mut draw);
            assert_eq!(draw.len(), k);
        }
    }

    #[test]
    fn source_error_has_requested_variance_and_correlation() {
        let (k, sigma, rho) = (2usize, 1.3, 0.4);
        let layer = SourceLayer::new(&cfg(k, sigma, rho)).expect("valid config");
        let mut rng = stream(MASTER_SEED, StreamId::new("test-source-moments", 0, 0, 0));
        let mut draw = SourceDraw::with_capacity(k);
        let mut var = OnlineStats::new();
        let mut cross = OnlineStats::new();
        for _ in 0..400_000 {
            layer.draw_into(Latent(0.0), &mut rng, &mut draw);
            let s = draw.as_slice();
            var.push(s[0] * s[0]);
            cross.push(s[0] * s[1]);
        }
        let expected_var = sigma * sigma;
        assert!(
            (var.mean() - expected_var).abs() < 0.02,
            "var {} vs {expected_var}",
            var.mean()
        );
        let expected_cov = sigma * sigma * rho;
        assert!(
            (cross.mean() - expected_cov).abs() < 0.02,
            "cov {} vs {expected_cov}",
            cross.mean()
        );
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(SourceLayer::new(&cfg(0, 1.0, 0.5)).is_err());
        assert!(SourceLayer::new(&cfg(2, -1.0, 0.5)).is_err());
        assert!(SourceLayer::new(&cfg(2, 1.0, 1.5)).is_err());
    }
}
