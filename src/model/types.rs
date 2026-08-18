//! Domain newtypes and the containers that carry them between layers.
//!
//! Every quantity in this model is an `f64` to the machine and something
//! different to an economist. `V`, `s_k`, `m_i` and `P_OC` are not
//! interchangeable, and neither is an external signal. Wrapping each one makes
//! the compiler enforce that distinction, so an API like
//! `fn revise(price: f64, signal: f64)` cannot silently be called with its
//! arguments swapped or with the latent value in place of the price.
//!
//! The containers do the same job one level up. [`SourceSet`] holds fundamental
//! signals and [`BeliefSet`] holds trader beliefs, so a
//! [`ClearingRule`](crate::model::clearing::ClearingRule) can only be handed
//! beliefs — never sources, and never the latent state.

use std::fmt;

/// The latent valuation `V` the market is trying to price.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct LatentValue(pub f64);

/// One fundamental source signal `s_k` about the latent value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct FundamentalSignal(pub f64);

/// One trader's belief `m_i`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct TraderBelief(pub f64);

/// The market-clearing onchain price `P_OC`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct MarketPrice(pub f64);

/// A signal about the latent value arriving from outside the market.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ExternalSignal(pub f64);

macro_rules! scalar_newtype {
    ($name:ident, $label:literal) => {
        impl $name {
            /// The underlying scalar.
            #[inline]
            pub const fn value(self) -> f64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "({})"), self.0)
            }
        }
    };
}

scalar_newtype!(LatentValue, "V");
scalar_newtype!(FundamentalSignal, "s");
scalar_newtype!(TraderBelief, "m");
scalar_newtype!(MarketPrice, "P_OC");
scalar_newtype!(ExternalSignal, "S");

impl MarketPrice {
    /// Squared pricing error against the latent value.
    #[inline]
    pub fn squared_error(self, latent: LatentValue) -> f64 {
        let e = self.0 - latent.0;
        e * e
    }

    /// Absolute distance to another price, e.g. a revision.
    #[inline]
    pub fn abs_change_from(self, other: MarketPrice) -> f64 {
        (self.0 - other.0).abs()
    }
}

/// One realization of the fundamental source vector `(s_1, ..., s_K)`.
///
/// The length of a `SourceSet` is the source budget `K`. It is a property of
/// the information environment and is never a function of the trader count. The
/// buffer is reused across Monte Carlo realizations so the hot loop does not
/// allocate.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceSet {
    values: Vec<FundamentalSignal>,
}

impl SourceSet {
    /// Allocate a set sized for `k` sources.
    pub fn with_capacity(k: usize) -> Self {
        Self {
            values: Vec::with_capacity(k),
        }
    }

    /// Build a set from signals.
    pub fn from_signals(values: Vec<FundamentalSignal>) -> Self {
        Self { values }
    }

    /// The source budget `K`.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the set is empty. A validated configuration never produces one.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The signals.
    #[inline]
    pub fn signals(&self) -> &[FundamentalSignal] {
        &self.values
    }

    /// Equal-weighted mean of the sources, `(1/K) sum_k s_k`.
    #[inline]
    pub fn equal_weighted_mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().map(|s| s.0).sum::<f64>() / self.values.len() as f64
    }

    /// Clear the set, keeping its allocation, and reserve room for `k` signals.
    #[inline]
    pub fn reset(&mut self, k: usize) {
        self.values.clear();
        self.values.reserve(k);
    }

    /// Append a signal. Only an
    /// [`InformationSourceModel`](crate::model::sources::InformationSourceModel)
    /// has reason to call this.
    #[inline]
    pub fn push(&mut self, signal: FundamentalSignal) {
        self.values.push(signal);
    }
}

/// What the trader population believes, at whatever resolution the belief model
/// produced it.
///
/// A model that treats traders as an exchangeable population can summarise them
/// by a population mean, which is exact for mean-based clearing and keeps the
/// cost of a realization independent of `N_T`. A model with heterogeneous
/// traders materialises the cross-section instead. Both carry the trader count,
/// so a clearing rule always knows how many participants it is aggregating.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefSet {
    /// Every trader's belief, materialised.
    CrossSection {
        /// The individual beliefs; its length is `N_T`.
        beliefs: Vec<TraderBelief>,
    },
    /// A population summary sufficient for clearing rules that depend only on
    /// the mean belief.
    PopulationMean {
        /// The trader count `N_T`.
        count: usize,
        /// The population-average belief.
        mean: TraderBelief,
    },
}

impl BeliefSet {
    /// The trader count `N_T`.
    #[inline]
    pub fn count(&self) -> usize {
        match self {
            BeliefSet::CrossSection { beliefs } => beliefs.len(),
            BeliefSet::PopulationMean { count, .. } => *count,
        }
    }

    /// Whether the population is empty. A validated configuration never
    /// produces one.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// The population-average belief, available in both representations.
    #[inline]
    pub fn mean(&self) -> TraderBelief {
        match self {
            BeliefSet::CrossSection { beliefs } => {
                if beliefs.is_empty() {
                    return TraderBelief(0.0);
                }
                TraderBelief(beliefs.iter().map(|b| b.0).sum::<f64>() / beliefs.len() as f64)
            }
            BeliefSet::PopulationMean { mean, .. } => *mean,
        }
    }

    /// The individual beliefs, when the model materialised them.
    ///
    /// Returns `None` for a population summary. A clearing rule that needs the
    /// cross-section — a risk-weighted or inventory-constrained rule, say —
    /// must be paired with a belief model that provides it.
    #[inline]
    pub fn cross_section(&self) -> Option<&[TraderBelief]> {
        match self {
            BeliefSet::CrossSection { beliefs } => Some(beliefs),
            BeliefSet::PopulationMean { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_set_length_is_the_source_budget() {
        let mut set = SourceSet::with_capacity(4);
        set.reset(4);
        for i in 0..4 {
            set.push(FundamentalSignal(i as f64));
        }
        assert_eq!(set.len(), 4);
        assert!((set.equal_weighted_mean() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn both_belief_representations_agree_on_the_mean() {
        let beliefs = vec![
            TraderBelief(1.0),
            TraderBelief(2.0),
            TraderBelief(3.0),
            TraderBelief(4.0),
        ];
        let cross = BeliefSet::CrossSection { beliefs };
        let summary = BeliefSet::PopulationMean {
            count: 4,
            mean: TraderBelief(2.5),
        };
        assert_eq!(cross.count(), summary.count());
        assert!((cross.mean().value() - summary.mean().value()).abs() < 1e-12);
        assert!(cross.cross_section().is_some());
        assert!(summary.cross_section().is_none());
    }

    #[test]
    fn price_errors_use_the_latent_value_explicitly() {
        let price = MarketPrice(1.5);
        assert!((price.squared_error(LatentValue(1.0)) - 0.25).abs() < 1e-12);
        assert!((price.abs_change_from(MarketPrice(2.0)) - 0.5).abs() < 1e-12);
    }
}
