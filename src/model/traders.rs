//! The trader layer: turning existing information into beliefs.
//!
//! Trader `i` **follows an existing source** `k(i)` and holds
//!
//! ```text
//! m_i = s_{k(i)} + b_i + nu_i
//! ```
//!
//! where `b_i` is clientele / participation distortion and `nu_i` is
//! idiosyncratic interpretation noise.
//!
//! # The dependency rule
//!
//! [`BeliefFormation::form`] receives a [`SourceSet`] and **not** a
//! [`LatentValue`](crate::model::types::LatentValue). That omission is the point
//! of the trait. A belief model cannot reach the latent state, cannot reach the
//! source generator, and cannot draw a new fundamental signal of its own — so no
//! implementation, present or future, can make trader growth increase the
//! market's fundamental information. This is the failure of the superseded v2
//! model encoded in the type system rather than in a comment.
//!
//! What a belief model *can* do is anything about interpretation: heterogeneous
//! precision, selected clienteles, attention weighting, traders who read
//! several sources. All of those operate on information that already exists.

use rand::{Rng, RngCore};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::config::TraderConfig;
use crate::error::{Error, Result};
use crate::model::types::{BeliefSet, SourceSet, TraderBelief};

/// A model of how a trader population forms beliefs from existing sources.
pub trait BeliefFormation {
    /// The trader count `N_T`.
    fn trader_count(&self) -> usize;

    /// Form the population's beliefs from the available fundamental sources.
    ///
    /// The signature deliberately excludes the latent value; see the module
    /// documentation.
    fn form<R: RngCore + ?Sized>(&self, sources: &SourceSet, rng: &mut R) -> BeliefSet;
}

/// How the trader population is spread across the available sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    /// Equal asymptotic representation: every source carries weight `1/K`.
    ///
    /// This is the canonical v4 aggregation. It is the limit of any balanced
    /// assignment and keeps the source layer's contribution to the price equal
    /// to `(1/K) sum_k s_k` independently of the trader count.
    #[default]
    Equal,
    /// Explicit round-robin assignment `k(i) = i mod K`, so source `k` carries
    /// weight `n_k / N_T`.
    ///
    /// This coincides with [`Representation::Equal`] whenever `N_T` is
    /// divisible by `K`, and lets a small trader population leave some sources
    /// unread.
    RoundRobin,
}

/// The resolution at which the trader population is realized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpretationNoise {
    /// Summarise the population by its mean belief.
    ///
    /// Because each `nu_i` is i.i.d. `N(0, sigma_nu^2)`, the average over `N_T`
    /// traders is exactly `N(0, sigma_nu^2 / N_T)`. Drawing that average
    /// directly is distributionally exact — not an approximation — and makes the
    /// cost of a realization independent of `N_T`.
    #[default]
    Aggregated,
    /// Materialise every trader's belief.
    ///
    /// Produces a [`BeliefSet::CrossSection`], at `O(N_T)` per realization. Used
    /// to verify the closed form above and available to clearing rules that need
    /// the cross-section.
    PerTrader,
}

/// The canonical belief model: each trader attaches to one existing source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceAttachedBeliefs {
    count: usize,
    interpretation_sigma: f64,
    clientele_bias: f64,
    representation: Representation,
    noise: InterpretationNoise,
}

impl SourceAttachedBeliefs {
    /// Build a belief model from a validated configuration.
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

    /// The representation scheme in use.
    #[inline]
    pub const fn representation(&self) -> Representation {
        self.representation
    }

    /// Weight the trader population places on source `k`.
    ///
    /// Weights sum to one, so the source layer can never be amplified by adding
    /// traders.
    pub fn source_weight(&self, k: usize, source_count: usize) -> f64 {
        debug_assert!(k < source_count, "trader assigned to a nonexistent source");
        match self.representation {
            Representation::Equal => 1.0 / source_count as f64,
            Representation::RoundRobin => {
                self.followers_of(k, source_count) as f64 / self.count as f64
            }
        }
    }

    /// How many traders follow source `k` under round-robin assignment.
    #[inline]
    fn followers_of(&self, k: usize, source_count: usize) -> usize {
        let base = self.count / source_count;
        let remainder = self.count % source_count;
        base + usize::from(k < remainder)
    }

    /// The share of the mean belief explained by the fundamental source layer,
    /// `sum_k w_k s_k`.
    ///
    /// This is the only channel through which information about `V` reaches a
    /// belief.
    pub fn weighted_source_component(&self, sources: &SourceSet) -> f64 {
        let signals = sources.signals();
        match self.representation {
            Representation::Equal => sources.equal_weighted_mean(),
            Representation::RoundRobin => {
                let k = signals.len();
                if k == 0 {
                    return 0.0;
                }
                let mut acc = 0.0;
                for (idx, signal) in signals.iter().enumerate() {
                    acc += self.followers_of(idx, k) as f64 * signal.value();
                }
                acc / self.count as f64
            }
        }
    }

    /// Mean clientele distortion across the population.
    ///
    /// The canonical model uses a common tilt, so this does not average away as
    /// `N_T` grows.
    #[inline]
    pub const fn mean_clientele_bias(&self) -> f64 {
        self.clientele_bias
    }

    /// Mean interpretation error across the population, drawn in closed form.
    pub fn mean_interpretation_error<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.interpretation_sigma == 0.0 {
            return 0.0;
        }
        let z: f64 = rng.sample(StandardNormal);
        self.interpretation_sigma * z / (self.count as f64).sqrt()
    }
}

impl BeliefFormation for SourceAttachedBeliefs {
    #[inline]
    fn trader_count(&self) -> usize {
        self.count
    }

    fn form<R: RngCore + ?Sized>(&self, sources: &SourceSet, rng: &mut R) -> BeliefSet {
        match self.noise {
            InterpretationNoise::Aggregated => BeliefSet::PopulationMean {
                count: self.count,
                mean: TraderBelief(
                    self.weighted_source_component(sources)
                        + self.mean_clientele_bias()
                        + self.mean_interpretation_error(rng),
                ),
            },
            InterpretationNoise::PerTrader => {
                // Literal `m_i = s_{k(i)} + b_i + nu_i` under round-robin
                // assignment, which coincides with equal representation
                // whenever N_T is divisible by K.
                let signals = sources.signals();
                let k = signals.len().max(1);
                let mut beliefs = Vec::with_capacity(self.count);
                for i in 0..self.count {
                    let source = signals
                        .get(i % k)
                        .map(|s| s.value())
                        .unwrap_or_else(|| sources.equal_weighted_mean());
                    let nu: f64 = if self.interpretation_sigma == 0.0 {
                        0.0
                    } else {
                        self.interpretation_sigma * rng.sample::<f64, _>(StandardNormal)
                    };
                    beliefs.push(TraderBelief(source + self.clientele_bias + nu));
                }
                BeliefSet::CrossSection { beliefs }
            }
        }
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
    use crate::config::SourceConfig;
    use crate::model::sources::{CorrelatedFiniteSources, InformationSourceModel};
    use crate::model::types::LatentValue;
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

    fn sources(k: usize) -> CorrelatedFiniteSources {
        CorrelatedFiniteSources::new(&SourceConfig {
            count: k,
            sigma: 1.0,
            correlation: 0.5,
        })
        .expect("valid")
    }

    #[test]
    fn source_weights_sum_to_one() {
        for representation in [Representation::Equal, Representation::RoundRobin] {
            for (n, k) in [(50usize, 100usize), (2500, 10), (7, 3), (1, 4)] {
                let model =
                    SourceAttachedBeliefs::new(&trader_cfg(n, representation)).expect("valid");
                let total: f64 = (0..k).map(|idx| model.source_weight(idx, k)).sum();
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
        let mut rng = stream(MASTER_SEED, StreamId::new("test-representation", 0, 0, 0));
        let set = sources(k).generate(LatentValue(0.3), &mut rng);
        let equal =
            SourceAttachedBeliefs::new(&trader_cfg(2500, Representation::Equal)).expect("valid");
        let robin = SourceAttachedBeliefs::new(&trader_cfg(2500, Representation::RoundRobin))
            .expect("valid");
        assert!(
            (equal.weighted_source_component(&set) - robin.weighted_source_component(&set)).abs()
                < 1e-12
        );
    }

    /// The two belief representations must agree in distribution: the population
    /// summary is a closed form for the cross-section's mean, not a shortcut.
    #[test]
    fn aggregated_and_cross_section_agree_in_distribution() {
        let (n, k) = (64usize, 8usize);
        let source_model = sources(k);

        let mut aggregated_cfg = trader_cfg(n, Representation::Equal);
        aggregated_cfg.noise = InterpretationNoise::Aggregated;
        let mut explicit_cfg = trader_cfg(n, Representation::Equal);
        explicit_cfg.noise = InterpretationNoise::PerTrader;

        let aggregated = SourceAttachedBeliefs::new(&aggregated_cfg).expect("valid");
        let explicit = SourceAttachedBeliefs::new(&explicit_cfg).expect("valid");

        let mut rng_a = stream(MASTER_SEED, StreamId::new("test-beliefs", 0, 0, 0));
        let mut rng_b = stream(MASTER_SEED, StreamId::new("test-beliefs", 0, 1, 0));
        let mut set = SourceSet::with_capacity(k);
        let mut stats_a = OnlineStats::new();
        let mut stats_b = OnlineStats::new();
        for _ in 0..200_000 {
            source_model.generate_into(LatentValue(0.0), &mut rng_a, &mut set);
            let a = aggregated.form(&set, &mut rng_a);
            stats_a.push(a.mean().value());

            source_model.generate_into(LatentValue(0.0), &mut rng_b, &mut set);
            let b = explicit.form(&set, &mut rng_b);
            stats_b.push(b.mean().value());

            assert_eq!(a.count(), n);
            assert_eq!(b.count(), n);
        }
        assert!(a_close(stats_a.mean(), stats_b.mean(), 0.02));
        assert!(
            a_close(stats_a.variance(), stats_b.variance(), 0.02),
            "{} vs {}",
            stats_a.variance(),
            stats_b.variance()
        );
    }

    fn a_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// The cross-section representation must be literal: with no interpretation
    /// noise and no tilt, trader `i` holds exactly source `i mod K`.
    #[test]
    fn cross_section_beliefs_attach_to_individual_sources() {
        let (n, k) = (6usize, 3usize);
        let mut cfg = trader_cfg(n, Representation::RoundRobin);
        cfg.interpretation_sigma = 0.0;
        cfg.noise = InterpretationNoise::PerTrader;
        let model = SourceAttachedBeliefs::new(&cfg).expect("valid");

        let mut rng = stream(MASTER_SEED, StreamId::new("test-cross-section", 0, 0, 0));
        let set = sources(k).generate(LatentValue(0.0), &mut rng);
        let beliefs = model.form(&set, &mut rng);
        let cross = beliefs.cross_section().expect("cross section");
        for (i, belief) in cross.iter().enumerate() {
            assert!((belief.value() - set.signals()[i % k].value()).abs() < 1e-12);
        }
    }
}
