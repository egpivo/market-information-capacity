//! External information and how the market uses it.
//!
//! Two separate questions live here, and the module keeps them separate because
//! conflating them is how a mechanical rule gets mistaken for an optimal one:
//!
//! ```text
//! ExternalSignalProcess   what new information does the outside world produce?
//!         |
//!         v  ExternalSignal
//! RevisionRule            how does the market assimilate it?
//! ```
//!
//! The canonical model pairs a genuinely new draw about `V` with a
//! **fixed-weight** revision. That revision is *not* a Bayesian update, so a
//! market whose pre-official price is already accurate can be made worse by it.
//! The canonical Informative world is exactly that case. When the post-boundary
//! MSE rises, the reading is that the assimilation rule is not optimal — not
//! that the external information was harmful.

use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

use crate::config::OfficialSignalConfig;
use crate::error::{Error, Result};
use crate::model::types::{ExternalSignal, LatentValue, MarketPrice};

/// A process generating information about the latent value from outside the
/// market.
pub trait ExternalSignalProcess {
    /// Draw one external signal.
    fn generate<R: RngCore + ?Sized>(&self, latent: LatentValue, rng: &mut R) -> ExternalSignal;
}

/// A rule for assimilating an external signal into the price.
pub trait RevisionRule {
    /// Revise the pre-boundary price given the external signal.
    fn revise(&self, pre_price: MarketPrice, external: ExternalSignal) -> MarketPrice;
}

/// The canonical external signal: `S = V + sigma_o * epsilon`.
///
/// This is *new* information — an independent draw about `V`, not a function of
/// anything the market already knew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfficialSignalProcess {
    sigma: f64,
}

impl OfficialSignalProcess {
    /// Build the process from a validated configuration.
    pub fn new(cfg: &OfficialSignalConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { sigma: cfg.sigma })
    }

    /// The signal error scale `sigma_o`.
    #[inline]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl ExternalSignalProcess for OfficialSignalProcess {
    #[inline]
    fn generate<R: RngCore + ?Sized>(&self, latent: LatentValue, rng: &mut R) -> ExternalSignal {
        let z: f64 = rng.sample(StandardNormal);
        ExternalSignal(latent.value() + self.sigma * z)
    }
}

/// The canonical revision rule: a fixed-weight move toward the external signal.
///
/// ```text
/// P_post = (1 - w) P_pre + w S.
/// ```
///
/// `w` is a configured parameter, deliberately not an optimal Bayesian weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedWeightRevision {
    weight: f64,
}

impl FixedWeightRevision {
    /// Build the rule from a validated configuration.
    pub fn new(cfg: &OfficialSignalConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { weight: cfg.weight })
    }

    /// The revision weight `w`.
    #[inline]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

impl RevisionRule for FixedWeightRevision {
    #[inline]
    fn revise(&self, pre_price: MarketPrice, external: ExternalSignal) -> MarketPrice {
        MarketPrice((1.0 - self.weight) * pre_price.value() + self.weight * external.value())
    }
}

impl OfficialSignalConfig {
    /// Check the modelling invariants of an official-signal configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(Error::Config(format!(
                "official signal sigma must be finite and non-negative, got {}",
                self.sigma
            )));
        }
        if !(0.0..=1.0).contains(&self.weight) {
            return Err(Error::Config(format!(
                "official signal weight must lie in [0, 1], got {}",
                self.weight
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(weight: f64) -> FixedWeightRevision {
        FixedWeightRevision::new(&OfficialSignalConfig {
            sigma: 0.35,
            weight,
        })
        .expect("valid")
    }

    #[test]
    fn revision_interpolates_between_price_and_signal() {
        let pre = MarketPrice(1.0);
        let signal = ExternalSignal(3.0);
        assert!((rule(0.0).revise(pre, signal).value() - 1.0).abs() < 1e-12);
        assert!((rule(1.0).revise(pre, signal).value() - 3.0).abs() < 1e-12);
        assert!((rule(0.5).revise(pre, signal).value() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        assert!(
            OfficialSignalConfig {
                sigma: -0.1,
                weight: 0.5
            }
            .validate()
            .is_err()
        );
        assert!(
            OfficialSignalConfig {
                sigma: 0.35,
                weight: 1.5
            }
            .validate()
            .is_err()
        );
    }
}
