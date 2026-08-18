//! The composed market: one engine wiring the layers together.
//!
//! [`MarketEngine`] is generic over all six components and dispatches
//! statically, so the compiler inlines the whole realization path and there is
//! no vtable in the Monte Carlo hot loop. The generic parameters are the
//! degrees of freedom; the type of the engine records which model was run.
//!
//! The wiring enforces the dependency direction. The latent value reaches the
//! source model and stops there — it is never passed to belief formation or to
//! clearing. Sources reach belief formation and stop there. Beliefs reach
//! clearing and stop there. No component can look further upstream than the
//! layer immediately above it.

use rand::RngCore;

use crate::config::MarketConfig;
use crate::error::Result;
use crate::model::clearing::{ClearingRule, EqualRepresentationClearing};
use crate::model::latent::{GaussianLatent, LatentProcess};
use crate::model::official::{
    ExternalSignalProcess, FixedWeightRevision, OfficialSignalProcess, RevisionRule,
};
use crate::model::sources::{CorrelatedFiniteSources, InformationSourceModel};
use crate::model::traders::{BeliefFormation, SourceAttachedBeliefs};
use crate::model::types::{ExternalSignal, LatentValue, MarketPrice, SourceSet};

/// One pre-official realization of the market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Realization {
    /// The latent valuation `V`.
    pub latent: LatentValue,
    /// The pre-official onchain price `P_OC`.
    pub pre_price: MarketPrice,
}

impl Realization {
    /// Squared pricing error of the pre-official price.
    #[inline]
    pub fn squared_error(&self) -> f64 {
        self.pre_price.squared_error(self.latent)
    }

    /// Squared error of an uninformed price of zero, the reference point for how
    /// much the market knew before the boundary.
    #[inline]
    pub fn uninformed_squared_error(&self) -> f64 {
        MarketPrice(0.0).squared_error(self.latent)
    }
}

/// One realization carried through the external-information boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevisedRealization {
    /// The pre-official realization.
    pub pre: Realization,
    /// The external signal about `V`.
    pub external: ExternalSignal,
    /// The post-boundary price.
    pub post_price: MarketPrice,
}

impl RevisedRealization {
    /// Squared post-boundary pricing error.
    #[inline]
    pub fn post_squared_error(&self) -> f64 {
        self.post_price.squared_error(self.pre.latent)
    }

    /// Absolute price revision across the boundary.
    #[inline]
    pub fn abs_revision(&self) -> f64 {
        self.post_price.abs_change_from(self.pre.pre_price)
    }
}

/// Reusable per-thread scratch space, so the hot loop does not allocate.
#[derive(Debug, Clone, Default)]
pub struct Workspace {
    sources: SourceSet,
}

/// A market composed of one implementation of each economic component.
#[derive(Debug, Clone)]
pub struct MarketEngine<L, S, B, C, E, R> {
    latent: L,
    sources: S,
    beliefs: B,
    clearing: C,
    external: E,
    revision: R,
}

impl<L, S, B, C, E, R> MarketEngine<L, S, B, C, E, R> {
    /// Compose an engine from its components.
    pub fn new(latent: L, sources: S, beliefs: B, clearing: C, external: E, revision: R) -> Self {
        Self {
            latent,
            sources,
            beliefs,
            clearing,
            external,
            revision,
        }
    }

    /// The latent process.
    #[inline]
    pub fn latent_process(&self) -> &L {
        &self.latent
    }

    /// The fundamental source model.
    #[inline]
    pub fn source_model(&self) -> &S {
        &self.sources
    }

    /// The belief model.
    #[inline]
    pub fn belief_model(&self) -> &B {
        &self.beliefs
    }

    /// The clearing rule.
    #[inline]
    pub fn clearing_rule(&self) -> &C {
        &self.clearing
    }

    /// The external signal process.
    #[inline]
    pub fn external_signal_process(&self) -> &E {
        &self.external
    }

    /// The revision rule.
    #[inline]
    pub fn revision_rule(&self) -> &R {
        &self.revision
    }
}

impl<L, S, B, C, E, R> MarketEngine<L, S, B, C, E, R>
where
    L: LatentProcess,
    S: InformationSourceModel,
    B: BeliefFormation,
    C: ClearingRule,
    E: ExternalSignalProcess,
    R: RevisionRule,
{
    /// Allocate scratch space sized for this market's source budget.
    pub fn workspace(&self) -> Workspace {
        Workspace {
            sources: SourceSet::with_capacity(self.sources.source_count()),
        }
    }

    /// The source budget `K`.
    #[inline]
    pub fn source_count(&self) -> usize {
        self.sources.source_count()
    }

    /// The trader count `N_T`.
    #[inline]
    pub fn trader_count(&self) -> usize {
        self.beliefs.trader_count()
    }

    /// Draw one pre-official realization.
    ///
    /// Draw order is fixed as latent value, source errors, interpretation noise,
    /// which keeps a stream's output stable across refactors of the calling
    /// code.
    pub fn realize<Rn: RngCore + ?Sized>(&self, ws: &mut Workspace, rng: &mut Rn) -> Realization {
        let latent = self.latent.sample(rng);
        self.sources.generate_into(latent, rng, &mut ws.sources);
        let beliefs = self.beliefs.form(&ws.sources, rng);
        Realization {
            latent,
            pre_price: self.clearing.clear(&beliefs),
        }
    }

    /// Carry a realization through the external-information boundary.
    pub fn revise_with_external<Rn: RngCore + ?Sized>(
        &self,
        pre: Realization,
        rng: &mut Rn,
    ) -> RevisedRealization {
        let external = self.external.generate(pre.latent, rng);
        RevisedRealization {
            pre,
            external,
            post_price: self.revision.revise(pre.pre_price, external),
        }
    }

    /// Draw a realization and immediately revise it.
    pub fn realize_revised<Rn: RngCore + ?Sized>(
        &self,
        ws: &mut Workspace,
        rng: &mut Rn,
    ) -> RevisedRealization {
        let pre = self.realize(ws, rng);
        self.revise_with_external(pre, rng)
    }
}

/// The canonical v4 market composition.
///
/// A Gaussian latent value, finitely many correlated sources, source-attached
/// trader beliefs, equal-weight clearing, an official signal and a fixed-weight
/// revision. Every canonical result in this repository is produced by this
/// composition; the type spells out which model that is.
///
/// [`CanonicalMarket::from_config`] is the **only** way canonical results are
/// built. [`MarketEngine::new`] stays available for experimental compositions,
/// but the simulation and validation code never calls it, so a published number
/// and an ad-hoc composition can never be confused. An architecture test
/// enforces that split.
pub type CanonicalMarket = MarketEngine<
    GaussianLatent,
    CorrelatedFiniteSources,
    SourceAttachedBeliefs,
    EqualRepresentationClearing,
    OfficialSignalProcess,
    FixedWeightRevision,
>;

impl CanonicalMarket {
    /// Build the canonical market from a validated configuration.
    pub fn from_config(cfg: &MarketConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(MarketEngine::new(
            GaussianLatent::standard(),
            CorrelatedFiniteSources::new(&cfg.sources)?,
            SourceAttachedBeliefs::new(&cfg.traders)?,
            EqualRepresentationClearing,
            OfficialSignalProcess::new(&cfg.official)?,
            FixedWeightRevision::new(&cfg.official)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OfficialSignalConfig, SourceConfig, TraderConfig};
    use crate::model::traders::{InterpretationNoise, Representation};
    use crate::rng::{MASTER_SEED, StreamId, stream};

    fn market(k: usize, n: usize, interpretation_sigma: f64) -> MarketConfig {
        MarketConfig {
            sources: SourceConfig {
                count: k,
                sigma: 1.0,
                correlation: 0.5,
            },
            traders: TraderConfig {
                count: n,
                interpretation_sigma,
                clientele_bias: 0.0,
                representation: Representation::Equal,
                noise: InterpretationNoise::Aggregated,
            },
            official: OfficialSignalConfig {
                sigma: 0.35,
                weight: 0.8,
            },
        }
    }

    /// With interpretation noise switched off, the trader count cannot move the
    /// price at all: the price is a deterministic function of the source layer.
    #[test]
    fn trader_count_cannot_change_a_noiseless_price() {
        let small = CanonicalMarket::from_config(&market(4, 100, 0.0)).expect("valid");
        let large = CanonicalMarket::from_config(&market(4, 1_000_000, 0.0)).expect("valid");
        let mut ws_s = small.workspace();
        let mut ws_l = large.workspace();
        let mut rng_s = stream(MASTER_SEED, StreamId::new("test-market", 0, 0, 0));
        let mut rng_l = stream(MASTER_SEED, StreamId::new("test-market", 0, 0, 0));
        for _ in 0..1000 {
            let a = small.realize(&mut ws_s, &mut rng_s);
            let b = large.realize(&mut ws_l, &mut rng_l);
            assert_eq!(a.latent, b.latent);
            assert_eq!(a.pre_price, b.pre_price);
        }
    }

    /// The canonical constructor wires every component from the configuration,
    /// so `CanonicalMarket::from_config` is a faithful reading of a
    /// `MarketConfig` and not a partly hard-coded market.
    #[test]
    fn canonical_constructor_reads_every_component_from_the_config() {
        let mut cfg = market(13, 321, 0.6);
        cfg.sources.correlation = 0.31;
        cfg.sources.sigma = 1.7;
        cfg.traders.clientele_bias = 0.22;
        cfg.official.sigma = 0.44;
        cfg.official.weight = 0.55;
        let engine = CanonicalMarket::from_config(&cfg).expect("valid");

        assert_eq!(engine.source_count(), cfg.sources.count);
        assert_eq!(engine.source_model().sigma(), cfg.sources.sigma);
        assert_eq!(engine.trader_count(), cfg.traders.count);
        assert_eq!(
            engine.belief_model().mean_clientele_bias(),
            cfg.traders.clientele_bias
        );
        assert_eq!(
            engine.belief_model().representation(),
            Representation::Equal
        );
        assert_eq!(engine.latent_process().sigma(), 1.0);
        assert_eq!(engine.external_signal_process().sigma(), cfg.official.sigma);
        assert_eq!(engine.revision_rule().weight(), cfg.official.weight);
    }

    #[test]
    fn engine_exposes_the_composed_dimensions() {
        let engine = CanonicalMarket::from_config(&market(7, 250, 0.8)).expect("valid");
        assert_eq!(engine.source_count(), 7);
        assert_eq!(engine.trader_count(), 250);
        assert_eq!(engine.revision_rule().weight(), 0.8);
        assert_eq!(engine.external_signal_process().sigma(), 0.35);
    }

    /// A market with a swapped-in component still runs: the engine is generic,
    /// not hard-wired to the canonical composition.
    #[test]
    fn components_are_swappable() {
        let cfg = market(4, 100, 0.5);
        let engine = MarketEngine::new(
            GaussianLatent::with_sigma(2.0),
            CorrelatedFiniteSources::new(&cfg.sources).expect("valid"),
            SourceAttachedBeliefs::new(&cfg.traders).expect("valid"),
            EqualRepresentationClearing,
            OfficialSignalProcess::new(&cfg.official).expect("valid"),
            FixedWeightRevision::new(&cfg.official).expect("valid"),
        );
        let mut ws = engine.workspace();
        let mut rng = stream(MASTER_SEED, StreamId::new("test-swap", 0, 0, 0));
        let revised = engine.realize_revised(&mut ws, &mut rng);
        assert!(revised.post_squared_error().is_finite());
        assert!(revised.abs_revision() >= 0.0);
    }
}
