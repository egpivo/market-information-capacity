//! The trader layer.
//!
//! Trader `i` **follows an existing source** `k(i)` and holds the belief
//!
//! ```text
//! m_i = s_{k(i)} + b_i + nu_i
//! ```
//!
//! where `b_i` is clientele / participation distortion and `nu_i` is
//! idiosyncratic interpretation noise.
//!
//! # Scientific-integrity constraint
//!
//! This module is deliberately unable to create fundamental information. It
//! accepts a [`SourceDraw`] and may only index into it; it holds no
//! [`SourceLayer`](super::sources::SourceLayer), no latent value and no source
//! random stream. Adding traders therefore changes only how interpretation
//! noise averages out, never how many independent draws about `V` exist. This
//! encodes, in the type system, the failure of the superseded v2 model in which
//! each additional trader implicitly supplied another independent fundamental
//! signal.

use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::config::TraderConfig;
use crate::error::{Error, Result};
use crate::model::sources::SourceDraw;

/// How the trader population is spread across the available sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    /// Equal asymptotic representation: every source carries weight `1/K`.
    ///
    /// This is the canonical v4 aggregation. It is the limit of any balanced
    /// assignment and keeps the source layer's contribution to the price equal
    /// to `(1/K) * sum_k s_k` independently of the trader count.
    #[default]
    Equal,
    /// Explicit round-robin assignment `k(i) = i mod K`.
    ///
    /// Source `k` then carries weight `n_k / N_T` where `n_k` is the number of
    /// traders assigned to it. This coincides with [`Representation::Equal`]
    /// whenever `N_T` is divisible by `K`, and lets a small trader population
    /// leave some sources unread.
    RoundRobin,
}

/// How the idiosyncratic interpretation noise term is realized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationNoise {
    /// Draw the *average* interpretation error directly.
    ///
    /// Because each `nu_i` is i.i.d. `N(0, sigma_nu^2)`, the average over `N_T`
    /// traders is exactly `N(0, sigma_nu^2 / N_T)`. Drawing it in closed form
    /// is distributionally exact, not an approximation, and makes the cost of a
    /// realization independent of `N_T`.
    #[default]
    Aggregated,
    /// Draw every trader's `nu_i` and average them.
    ///
    /// Used to verify the closed form above; `O(N_T)` per realization.
    PerTrader,
}

/// Aggregator for the trader layer.
#[derive(Debug, Clone)]
pub struct TraderLayer {
    count: usize,
    interpretation_sigma: f64,
    clientele_bias: f64,
    representation: Representation,
    noise: InterpretationNoise,
}

impl TraderLayer {
    /// Build a trader layer from a validated configuration.
    pub fn new(cfg: &TraderConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            count: cfg.count,
            interpretation_sigma: cfg.interpretation_sigma,
            clientele_bias: cfg.clientele_bias,
            representation: cfg.representation,
            noise: cfg.noise,
        })
    }

    /// The trader count `N_T`.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The representation scheme in use.
    #[inline]
    pub fn representation(&self) -> Representation {
        self.representation
    }

    /// Weight placed on source `k` by the trader population.
    ///
    /// Weights sum to one, so the source layer can never be amplified by adding
    /// traders.
    pub fn source_weight(&self, k: usize, source_count: usize) -> f64 {
        debug_assert!(k < source_count, "trader assigned to a nonexistent source");
        match self.representation {
            Representation::Equal => 1.0 / source_count as f64,
            Representation::RoundRobin => {
                let base = self.count / source_count;
                let remainder = self.count % source_count;
                let followers = base + usize::from(k < remainder);
                followers as f64 / self.count as f64
            }
        }
    }

    /// The share of the price explained by the fundamental source layer,
    /// `sum_k w_k * s_k`.
    ///
    /// This is the only channel through which information about `V` reaches the
    /// price.
    pub fn weighted_source_component(&self, sources: &SourceDraw) -> f64 {
        let values = sources.as_slice();
        match self.representation {
            Representation::Equal => sources.equal_weighted_mean(),
            Representation::RoundRobin => {
                let k = values.len();
                if k == 0 {
                    return 0.0;
                }
                let base = self.count / k;
                let remainder = self.count % k;
                let mut acc = 0.0;
                for (idx, &s) in values.iter().enumerate() {
                    let followers = base + usize::from(idx < remainder);
                    acc += followers as f64 * s;
                }
                acc / self.count as f64
            }
        }
    }

    /// Mean interpretation error across the trader population.
    pub fn mean_interpretation_error<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.interpretation_sigma == 0.0 {
            return 0.0;
        }
        match self.noise {
            InterpretationNoise::Aggregated => {
                let z: f64 = rng.sample(StandardNormal);
                self.interpretation_sigma * z / (self.count as f64).sqrt()
            }
            InterpretationNoise::PerTrader => {
                let mut acc = 0.0;
                for _ in 0..self.count {
                    let z: f64 = rng.sample(StandardNormal);
                    acc += self.interpretation_sigma * z;
                }
                acc / self.count as f64
            }
        }
    }

    /// Mean clientele distortion across the trader population.
    ///
    /// The canonical model uses a common clientele tilt, so the population mean
    /// of `b_i` is the configured bias and does not average away as `N_T`
    /// grows.
    #[inline]
    pub fn mean_clientele_bias(&self) -> f64 {
        self.clientele_bias
    }

    /// Population-average trader belief, `(1/N_T) * sum_i m_i`.
    pub fn mean_belief<R: Rng + ?Sized>(&self, sources: &SourceDraw, rng: &mut R) -> f64 {
        self.weighted_source_component(sources)
            + self.mean_clientele_bias()
            + self.mean_interpretation_error(rng)
    }
}

impl TraderConfig {
    /// Check the modelling invariants of a trader configuration.
    pub fn validate(&self) -> Result<()> {
        if self.count == 0 {
            return Err(Error::Config("trader count N_T must be at least 1".into()));
        }
        if !self.interpretation_sigma.is_finite() || self.interpretation_sigma < 0.0 {
            return Err(Error::Config(format!(
                "interpretation sigma must be finite and non-negative, got {}",
                self.interpretation_sigma
            )));
        }
        if !self.clientele_bias.is_finite() {
            return Err(Error::Config("clientele bias must be finite".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::metrics::OnlineStats;
    use crate::model::latent::Latent;
    use crate::model::sources::SourceLayer;
    use crate::rng::{MASTER_SEED, StreamId, stream};

    fn trader_cfg(count: usize, representation: Representation) -> TraderConfig {
        TraderConfig {
            count,
            interpretation_sigma: 0.8,
            clientele_bias: 0.0,
            representation,
            noise: InterpretationNoise::Aggregated,
        }
    }

    #[test]
    fn source_weights_sum_to_one() {
        for representation in [Representation::Equal, Representation::RoundRobin] {
            for (n, k) in [(50usize, 100usize), (2500, 10), (7, 3), (1, 4)] {
                let layer = TraderLayer::new(&trader_cfg(n, representation)).expect("valid");
                let total: f64 = (0..k).map(|idx| layer.source_weight(idx, k)).sum();
                assert!(
                    (total - 1.0).abs() < 1e-12,
                    "weights summed to {total} for {representation:?} n={n} k={k}"
                );
            }
        }
    }

    #[test]
    fn round_robin_matches_equal_when_divisible() {
        let k = 10;
        let sources_cfg = crate::config::SourceConfig {
            count: k,
            sigma: 1.0,
            correlation: 0.5,
        };
        let sources = SourceLayer::new(&sources_cfg).expect("valid");
        let mut rng = stream(MASTER_SEED, StreamId::new("test-representation", 0, 0, 0));
        let mut draw = SourceDraw::with_capacity(k);
        sources.draw_into(Latent(0.3), &mut rng, &mut draw);

        let equal = TraderLayer::new(&trader_cfg(2500, Representation::Equal)).expect("valid");
        let robin = TraderLayer::new(&trader_cfg(2500, Representation::RoundRobin)).expect("valid");
        assert!(
            (equal.weighted_source_component(&draw) - robin.weighted_source_component(&draw)).abs()
                < 1e-12
        );
    }

    #[test]
    fn aggregated_and_per_trader_noise_agree_in_distribution() {
        let n = 64;
        let mut aggregated_cfg = trader_cfg(n, Representation::Equal);
        aggregated_cfg.noise = InterpretationNoise::Aggregated;
        let mut per_trader_cfg = trader_cfg(n, Representation::Equal);
        per_trader_cfg.noise = InterpretationNoise::PerTrader;

        let aggregated = TraderLayer::new(&aggregated_cfg).expect("valid");
        let per_trader = TraderLayer::new(&per_trader_cfg).expect("valid");

        let mut rng_a = stream(MASTER_SEED, StreamId::new("test-noise", 0, 0, 0));
        let mut rng_b = stream(MASTER_SEED, StreamId::new("test-noise", 0, 1, 0));
        let mut stats_a = OnlineStats::new();
        let mut stats_b = OnlineStats::new();
        for _ in 0..200_000 {
            stats_a.push(aggregated.mean_interpretation_error(&mut rng_a));
            stats_b.push(per_trader.mean_interpretation_error(&mut rng_b));
        }
        let expected = 0.8 * 0.8 / n as f64;
        assert!((stats_a.variance() - expected).abs() < 0.1 * expected);
        assert!((stats_b.variance() - expected).abs() < 0.1 * expected);
    }
}
