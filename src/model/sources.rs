//! The fundamental source layer.
//!
//! This module owns the **only** place in the crate where new information about
//! `V` is created. Every implementation of [`InformationSourceModel`] takes the
//! latent value and returns exactly `K` fundamental signals; `K` is the source
//! budget, a property of the information environment, and no signature here
//! mentions a trader count.
//!
//! This is the information-capacity seam. Changing how much a market can know
//! means writing another implementation of this trait; it never means changing
//! anything downstream.

use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

use crate::config::SourceConfig;
use crate::error::{Error, Result};
use crate::model::types::{FundamentalSignal, LatentValue, SourceSet};

/// A model of the fundamental information available about the latent value.
pub trait InformationSourceModel {
    /// The source budget `K`: how many independent fundamental draws exist.
    fn source_count(&self) -> usize;

    /// Generate the source vector into a reusable buffer.
    ///
    /// This is the required method because it is the one the Monte Carlo hot
    /// loop calls; `out` is cleared first and receives exactly
    /// [`source_count`](Self::source_count) signals.
    fn generate_into<R: RngCore + ?Sized>(
        &self,
        latent: LatentValue,
        rng: &mut R,
        out: &mut SourceSet,
    );

    /// Generate the source vector, allocating.
    ///
    /// Convenience for tests and one-off calls; experiments use
    /// [`generate_into`](Self::generate_into).
    fn generate<R: RngCore + ?Sized>(&self, latent: LatentValue, rng: &mut R) -> SourceSet {
        let mut out = SourceSet::with_capacity(self.source_count());
        self.generate_into(latent, rng, &mut out);
        out
    }
}

/// The canonical source model: finitely many sources with correlated errors.
///
/// ```text
/// s_k = V + sigma_s ( sqrt(rho_s) u + sqrt(1 - rho_s) eta_k )
/// ```
///
/// `u` is drawn once per realization and shared by every source; `eta_k` is
/// source specific. Each source error therefore has variance `sigma_s^2` and
/// pairwise correlation `rho_s`. Setting `rho_s = 0` gives the
/// independent-source special case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelatedFiniteSources {
    count: usize,
    sigma: f64,
    sqrt_rho: f64,
    sqrt_one_minus_rho: f64,
}

impl CorrelatedFiniteSources {
    /// Build a source model from a validated configuration.
    pub fn new(cfg: &SourceConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            count: cfg.count,
            sigma: cfg.sigma,
            sqrt_rho: cfg.correlation.sqrt(),
            sqrt_one_minus_rho: (1.0 - cfg.correlation).sqrt(),
        })
    }

    /// The source error scale `sigma_s`.
    #[inline]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl InformationSourceModel for CorrelatedFiniteSources {
    #[inline]
    fn source_count(&self) -> usize {
        self.count
    }

    fn generate_into<R: RngCore + ?Sized>(
        &self,
        latent: LatentValue,
        rng: &mut R,
        out: &mut SourceSet,
    ) {
        let common: f64 = rng.sample(StandardNormal);
        let shared = latent.value() + self.sigma * self.sqrt_rho * common;
        out.reset(self.count);
        for _ in 0..self.count {
            let specific: f64 = rng.sample(StandardNormal);
            out.push(FundamentalSignal(
                shared + self.sigma * self.sqrt_one_minus_rho * specific,
            ));
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
    fn generated_set_always_has_exactly_k_signals() {
        for k in [1usize, 2, 10, 100] {
            let model = CorrelatedFiniteSources::new(&cfg(k, 1.0, 0.5)).expect("valid config");
            let mut rng = stream(MASTER_SEED, StreamId::new("test-sources", k as u64, 0, 0));
            assert_eq!(model.source_count(), k);
            assert_eq!(model.generate(LatentValue(0.0), &mut rng).len(), k);
        }
    }

    #[test]
    fn source_error_has_requested_variance_and_correlation() {
        let (k, sigma, rho) = (2usize, 1.3, 0.4);
        let model = CorrelatedFiniteSources::new(&cfg(k, sigma, rho)).expect("valid config");
        let mut rng = stream(MASTER_SEED, StreamId::new("test-source-moments", 0, 0, 0));
        let mut set = SourceSet::with_capacity(k);
        let mut var = OnlineStats::new();
        let mut cross = OnlineStats::new();
        for _ in 0..400_000 {
            model.generate_into(LatentValue(0.0), &mut rng, &mut set);
            let s = set.signals();
            var.push(s[0].value() * s[0].value());
            cross.push(s[0].value() * s[1].value());
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
    fn allocating_and_in_place_generation_agree() {
        let model = CorrelatedFiniteSources::new(&cfg(5, 1.0, 0.3)).expect("valid config");
        let mut rng_a = stream(MASTER_SEED, StreamId::new("test-source-api", 0, 0, 0));
        let mut rng_b = stream(MASTER_SEED, StreamId::new("test-source-api", 0, 0, 0));
        let mut buffer = SourceSet::with_capacity(5);
        for _ in 0..100 {
            let allocated = model.generate(LatentValue(0.25), &mut rng_a);
            model.generate_into(LatentValue(0.25), &mut rng_b, &mut buffer);
            assert_eq!(allocated, buffer);
        }
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(CorrelatedFiniteSources::new(&cfg(0, 1.0, 0.5)).is_err());
        assert!(CorrelatedFiniteSources::new(&cfg(2, -1.0, 0.5)).is_err());
        assert!(CorrelatedFiniteSources::new(&cfg(2, 1.0, 1.5)).is_err());
    }
}
