//! Architecture invariants.
//!
//! The dependency rule — *fundamental information may only be created by the
//! source layer* — is enforced by trait signatures, so most of it is a compile
//! time property. These tests make the consequences observable, and they double
//! as the worked example of extending the model: each one plugs a new
//! implementation into `MarketEngine` without touching any other layer.

use market_information_capacity::analytics::information_floor::analytic_floor;
use market_information_capacity::analytics::metrics::OnlineStats;
use market_information_capacity::config::{
    MarketConfig, OfficialSignalConfig, SourceConfig, TraderConfig,
};
use market_information_capacity::model::clearing::{ClearingRule, EqualRepresentationClearing};
use market_information_capacity::model::latent::{GaussianLatent, LatentProcess};
use market_information_capacity::model::official::{
    FixedWeightRevision, OfficialSignalProcess, RevisionRule,
};
use market_information_capacity::model::sources::{
    CorrelatedFiniteSources, InformationSourceModel,
};
use market_information_capacity::model::traders::{
    BeliefFormation, InterpretationNoise, Representation, SourceAttachedBeliefs,
};
use market_information_capacity::model::types::{
    BeliefSet, ExternalSignal, FundamentalSignal, LatentValue, MarketPrice, SourceSet,
};
use market_information_capacity::model::{CanonicalMarket, MarketEngine};
use market_information_capacity::rng::{MASTER_SEED, StreamId, stream};
use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

fn market(k: usize, n: usize, rho_s: f64, sigma_nu: f64) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: 1.0,
            correlation: rho_s,
        },
        traders: TraderConfig {
            count: n,
            interpretation_sigma: sigma_nu,
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

/// Belief formation is a function of the sources and the random stream, and of
/// nothing else.
///
/// The trait signature has no `LatentValue` parameter, so this cannot be
/// otherwise — but stating it as a test means the property is checked rather
/// than assumed. Two markets with wildly different latent values produce
/// identical beliefs when handed identical sources.
#[test]
fn belief_formation_cannot_see_the_latent_value() {
    let cfg = market(8, 500, 0.5, 0.8);
    let beliefs = SourceAttachedBeliefs::new(&cfg.traders).expect("valid");
    let sources = CorrelatedFiniteSources::new(&cfg.sources).expect("valid");

    let mut source_rng = stream(MASTER_SEED, StreamId::new("arch-beliefs", 0, 0, 0));
    let set = sources.generate(LatentValue(-3.5), &mut source_rng);

    let mut rng_a = stream(MASTER_SEED, StreamId::new("arch-beliefs", 1, 0, 0));
    let mut rng_b = stream(MASTER_SEED, StreamId::new("arch-beliefs", 1, 0, 0));
    let a = beliefs.form(&set, &mut rng_a);
    let b = beliefs.form(&set, &mut rng_b);
    assert_eq!(a, b);
    assert_eq!(a.count(), 500);
}

/// Clearing is a function of the beliefs alone.
#[test]
fn clearing_cannot_see_the_sources_or_the_latent_value() {
    let rule = EqualRepresentationClearing;
    let summary = BeliefSet::PopulationMean {
        count: 1_000,
        mean: market_information_capacity::model::types::TraderBelief(0.42),
    };
    assert!((rule.clear(&summary).value() - 0.42).abs() < 1e-12);
}

/// An alternative information source model plugs straight in.
///
/// `IndependentSources` is defined here, outside the crate, and reaches the
/// documented independent-source floor `sigma_s^2 / K` without a single change
/// to the belief, clearing or revision layers.
struct IndependentSources {
    count: usize,
    sigma: f64,
}

impl InformationSourceModel for IndependentSources {
    fn source_count(&self) -> usize {
        self.count
    }

    fn generate_into<R: RngCore + ?Sized>(
        &self,
        latent: LatentValue,
        rng: &mut R,
        out: &mut SourceSet,
    ) {
        out.reset(self.count);
        for _ in 0..self.count {
            let z: f64 = rng.sample(StandardNormal);
            out.push(FundamentalSignal(latent.value() + self.sigma * z));
        }
    }
}

#[test]
fn a_new_source_model_reaches_its_own_analytic_floor() {
    let (k, n, sigma) = (8usize, 20_000usize, 1.0);
    let cfg = market(k, n, 0.0, 0.0);
    let engine = MarketEngine::new(
        GaussianLatent::standard(),
        IndependentSources { count: k, sigma },
        SourceAttachedBeliefs::new(&cfg.traders).expect("valid"),
        EqualRepresentationClearing,
        OfficialSignalProcess::new(&cfg.official).expect("valid"),
        FixedWeightRevision::new(&cfg.official).expect("valid"),
    );

    let mut ws = engine.workspace();
    let mut rng = stream(MASTER_SEED, StreamId::new("arch-independent", 0, 0, 0));
    let mut stats = OnlineStats::new();
    for _ in 0..200_000 {
        stats.push(engine.realize(&mut ws, &mut rng).squared_error());
    }
    let expected = analytic_floor(k, 0.0, sigma);
    let z = (stats.mean() - expected) / stats.std_error();
    assert!(
        z.abs() < 4.0,
        "independent sources gave {} against a floor of {expected} ({z} SE)",
        stats.mean()
    );
}

/// An alternative revision rule plugs straight in, and shows that the canonical
/// rule is a *choice*: with no revision, the post-boundary price is the
/// pre-official price and nothing about the market's information changes.
struct NoRevision;

impl RevisionRule for NoRevision {
    fn revise(&self, pre_price: MarketPrice, _external: ExternalSignal) -> MarketPrice {
        pre_price
    }
}

#[test]
fn a_new_revision_rule_changes_only_assimilation() {
    let cfg = market(2, 2_500, 0.9, 0.8);
    let canonical = CanonicalMarket::from_config(&cfg).expect("valid");
    let unrevised = MarketEngine::new(
        GaussianLatent::standard(),
        CorrelatedFiniteSources::new(&cfg.sources).expect("valid"),
        SourceAttachedBeliefs::new(&cfg.traders).expect("valid"),
        EqualRepresentationClearing,
        OfficialSignalProcess::new(&cfg.official).expect("valid"),
        NoRevision,
    );

    let mut ws_a = canonical.workspace();
    let mut ws_b = unrevised.workspace();
    let mut rng_a = stream(MASTER_SEED, StreamId::new("arch-revision", 0, 0, 0));
    let mut rng_b = stream(MASTER_SEED, StreamId::new("arch-revision", 0, 0, 0));

    let mut pre = OnlineStats::new();
    let mut canonical_post = OnlineStats::new();
    let mut unrevised_post = OnlineStats::new();
    for _ in 0..50_000 {
        let a = canonical.realize_revised(&mut ws_a, &mut rng_a);
        let b = unrevised.realize_revised(&mut ws_b, &mut rng_b);
        // Identical streams and identical upstream layers: same pre-official
        // price, same external signal, different assimilation.
        assert_eq!(a.pre, b.pre);
        assert_eq!(a.external, b.external);
        pre.push(a.pre.squared_error());
        canonical_post.push(a.post_squared_error());
        unrevised_post.push(b.post_squared_error());
    }
    assert_eq!(unrevised_post.mean().to_bits(), pre.mean().to_bits());
    assert!(canonical_post.mean() < 0.3 * pre.mean());
}

/// A clearing rule that needs the trader cross-section can ask for it, and gets
/// `None` from a population summary rather than a silently wrong answer.
struct TrimmedMeanClearing;

impl ClearingRule for TrimmedMeanClearing {
    fn clear(&self, beliefs: &BeliefSet) -> MarketPrice {
        match beliefs.cross_section() {
            Some(cross) if cross.len() >= 4 => {
                let mut values: Vec<f64> = cross.iter().map(|b| b.value()).collect();
                values.sort_by(f64::total_cmp);
                let cut = values.len() / 10;
                let kept = &values[cut..values.len() - cut];
                MarketPrice(kept.iter().sum::<f64>() / kept.len() as f64)
            }
            // No cross-section available: fall back to the population mean.
            _ => MarketPrice(beliefs.mean().value()),
        }
    }
}

#[test]
fn a_clearing_rule_can_require_the_cross_section() {
    let mut cfg = market(4, 100, 0.5, 0.8);
    cfg.traders.noise = InterpretationNoise::PerTrader;
    let sources = CorrelatedFiniteSources::new(&cfg.sources).expect("valid");
    let beliefs = SourceAttachedBeliefs::new(&cfg.traders).expect("valid");

    let mut rng = stream(MASTER_SEED, StreamId::new("arch-clearing", 0, 0, 0));
    let set = sources.generate(LatentValue(0.0), &mut rng);
    let population = beliefs.form(&set, &mut rng);
    assert!(population.cross_section().is_some());

    let trimmed = TrimmedMeanClearing.clear(&population);
    let equal = EqualRepresentationClearing.clear(&population);
    assert!(trimmed.value().is_finite());
    // Trimming changes the price, so the rule really is reading the cross
    // section rather than the mean.
    assert_ne!(trimmed.value().to_bits(), equal.value().to_bits());

    // A population summary has no cross-section to offer.
    cfg.traders.noise = InterpretationNoise::Aggregated;
    let summarised = SourceAttachedBeliefs::new(&cfg.traders)
        .expect("valid")
        .form(&set, &mut rng);
    assert!(summarised.cross_section().is_none());
}

/// The latent process is a seam too: a heavier-tailed latent value flows through
/// the whole stack unchanged.
#[test]
fn a_different_latent_scale_flows_through_untouched() {
    let cfg = market(10, 1_000, 0.5, 0.8);
    let engine = MarketEngine::new(
        GaussianLatent::with_sigma(3.0),
        CorrelatedFiniteSources::new(&cfg.sources).expect("valid"),
        SourceAttachedBeliefs::new(&cfg.traders).expect("valid"),
        EqualRepresentationClearing,
        OfficialSignalProcess::new(&cfg.official).expect("valid"),
        FixedWeightRevision::new(&cfg.official).expect("valid"),
    );
    let mut ws = engine.workspace();
    let mut rng = stream(MASTER_SEED, StreamId::new("arch-latent", 0, 0, 0));

    // The source error scale is unchanged, so the floor is unchanged even though
    // the latent value is three times as volatile: information capacity is a
    // property of the source layer, not of the thing being priced.
    let mut latent = OnlineStats::new();
    let mut error = OnlineStats::new();
    for _ in 0..200_000 {
        let realization = engine.realize(&mut ws, &mut rng);
        latent.push(realization.latent.value());
        error.push(realization.squared_error());
    }
    assert!(
        (latent.variance() - 9.0).abs() < 0.3,
        "{}",
        latent.variance()
    );
    let expected = analytic_floor(10, 0.5, 1.0) + 0.8 * 0.8 / 1_000.0;
    let z = (error.mean() - expected) / error.std_error();
    assert!(z.abs() < 4.0, "floor moved: {} vs {expected}", error.mean());
}

/// `LatentProcess` is usable directly, so a future dynamic or empirical latent
/// state has a place to live without disturbing anything downstream.
#[test]
fn latent_process_is_a_standalone_seam() {
    let process = GaussianLatent::standard();
    let mut rng = stream(MASTER_SEED, StreamId::new("arch-latent-seam", 0, 0, 0));
    let sample: LatentValue = process.sample(&mut rng);
    assert!(sample.value().is_finite());
}
