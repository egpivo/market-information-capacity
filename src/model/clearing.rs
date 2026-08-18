//! Market clearing: turning trader beliefs into the observable onchain price.
//!
//! A [`ClearingRule`] receives a [`BeliefSet`] and nothing else. It cannot see
//! the sources those beliefs came from, and it cannot see the latent value, so
//! no clearing rule can inject information the trader population did not
//! already hold.
//!
//! This is the price-formation seam. Risk-weighted clearing, linear demand
//! schedules, inventory constraints and auction mechanisms are all additional
//! implementations of this trait; none of them require touching the source or
//! belief layers.

use crate::model::types::{BeliefSet, MarketPrice};

/// A rule aggregating trader beliefs into a market-clearing price.
pub trait ClearingRule {
    /// Clear the market.
    fn clear(&self, beliefs: &BeliefSet) -> MarketPrice;
}

/// The canonical clearing rule: equal clearing weight on every trader.
///
/// With identical risk tolerance across traders the clearing weights are
/// uniform, so
///
/// ```text
/// P_OC = (1 / N_T) sum_i m_i.
/// ```
///
/// Combined with the canonical belief model this expands to
/// `sum_k w_k s_k + b + nu_bar`, which is the identity that makes the
/// information floor visible: the only `V`-bearing term is the source
/// aggregate, whose precision is set by `K` and `rho_s` alone.
///
/// This rule depends only on the mean belief, so it works with either
/// [`BeliefSet`] representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EqualRepresentationClearing;

impl ClearingRule for EqualRepresentationClearing {
    #[inline]
    fn clear(&self, beliefs: &BeliefSet) -> MarketPrice {
        MarketPrice(beliefs.mean().value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::TraderBelief;

    #[test]
    fn clearing_is_the_mean_belief_in_both_representations() {
        let cross = BeliefSet::CrossSection {
            beliefs: vec![TraderBelief(1.0), TraderBelief(3.0)],
        };
        let summary = BeliefSet::PopulationMean {
            count: 2,
            mean: TraderBelief(2.0),
        };
        let rule = EqualRepresentationClearing;
        assert!((rule.clear(&cross).value() - 2.0).abs() < 1e-12);
        assert!((rule.clear(&summary).value() - 2.0).abs() < 1e-12);
    }
}
