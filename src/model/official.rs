//! The official signal that arrives after the pre-official trading window.
//!
//! The onchain market prices `V` before any external anchor exists. The
//! official signal is a *genuinely new* draw about `V`,
//!
//! ```text
//! official = V + sigma_o * epsilon,
//! ```
//!
//! and the post-boundary price is a fixed-weight revision of the pre-official
//! price toward it:
//!
//! ```text
//! P_post = (1 - w) * P_pre + w * official.
//! ```
//!
//! This is a mechanical revision rule, **not** an optimal Bayesian update. The
//! weight `w` is a configured parameter. Consequently a market whose
//! pre-official price is already accurate can be made *worse* by the revision
//! (the Informative world in the canonical run is exactly this case), and the
//! repository makes no claim that official information always improves price
//! quality.

use rand::Rng;
use rand_distr::StandardNormal;

use crate::config::OfficialSignalConfig;
use crate::error::{Error, Result};
use crate::model::latent::Latent;

/// Draw the official signal about the latent value.
#[inline]
pub fn draw_official<R: Rng + ?Sized>(
    latent: Latent,
    cfg: &OfficialSignalConfig,
    rng: &mut R,
) -> f64 {
    let z: f64 = rng.sample(StandardNormal);
    latent.value() + cfg.sigma * z
}

/// Apply the fixed-weight revision toward the official signal.
#[inline]
pub fn revise(pre_price: f64, official: f64, weight: f64) -> f64 {
    (1.0 - weight) * pre_price + weight * official
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

    #[test]
    fn revision_interpolates() {
        assert!((revise(1.0, 3.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((revise(1.0, 3.0, 1.0) - 3.0).abs() < 1e-12);
        assert!((revise(1.0, 3.0, 0.5) - 2.0).abs() < 1e-12);
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
