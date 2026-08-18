//! Gate: the information hierarchy is intact.
//!
//! Two structural facts must hold, and they are the two the superseded v2 model
//! got wrong:
//!
//! 1. the number of fundamental draws is `K`, never `N_T`;
//! 2. with interpretation noise switched off, the trader count cannot move the
//!    price at all.

use crate::analytics::information_floor::analytic_floor;
use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::market::CanonicalMarket;
use crate::model::sources::InformationSourceModel;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::model::types::{LatentValue, SourceSet};
use crate::rng::{StreamId, stream};
use crate::simulation::monte_carlo::{Cell, pre_price_mse};
use crate::validation::{Evidence, ValidationCheck, ValidationContext};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "gate-hierarchy";

fn market(k: usize, n_traders: usize, rho_s: f64, interpretation_sigma: f64) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: 1.0,
            correlation: rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma,
            clientele_bias: 0.0,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: Default::default(),
    }
}

/// Check that the source layer produces exactly `K` draws whatever `N_T` is,
/// and that a noiseless price is invariant to the trader count.
fn check_hierarchy(cfg: &Config) -> Result<Evidence> {
    let mut failures = Vec::new();

    for &k in &[1usize, 2, 10, 100] {
        for &n_traders in &[10usize, 100_000] {
            let sim = CanonicalMarket::from_config(&market(k, n_traders, 0.5, 0.8))?;
            let mut ws = sim.workspace();
            let mut rng = stream(cfg.master_seed, StreamId::new(EXPERIMENT, k as u64, 0, 0));
            let _ = sim.realize(&mut ws, &mut rng);
            let mut draw = SourceSet::with_capacity(k);
            sim.source_model()
                .generate_into(LatentValue(0.0), &mut rng, &mut draw);
            if draw.len() != k {
                failures.push(format!(
                    "K={k}, N_T={n_traders}: source layer produced {} draws",
                    draw.len()
                ));
            }
        }
    }

    // With sigma_nu = 0 the price is a deterministic function of the sources.
    let small = CanonicalMarket::from_config(&market(4, 100, 0.5, 0.0))?;
    let large = CanonicalMarket::from_config(&market(4, 1_000_000, 0.5, 0.0))?;
    let mut ws_s = small.workspace();
    let mut ws_l = large.workspace();
    let mut rng_s = stream(cfg.master_seed, StreamId::new(EXPERIMENT, 900, 0, 0));
    let mut rng_l = stream(cfg.master_seed, StreamId::new(EXPERIMENT, 900, 0, 0));
    let mut identical = true;
    for _ in 0..10_000 {
        let a = small.realize(&mut ws_s, &mut rng_s);
        let b = large.realize(&mut ws_l, &mut rng_l);
        if a.pre_price.value().to_bits() != b.pre_price.value().to_bits() {
            identical = false;
            break;
        }
    }
    if !identical {
        failures.push("noiseless price changed when the trader count changed".to_string());
    }

    let detail = if failures.is_empty() {
        "source draws == K for K in {1,2,10,100} at N_T in {10, 100000}; noiseless price identical for N_T = 100 vs 1,000,000".to_string()
    } else {
        failures.join("; ")
    };
    Ok(Evidence::new(failures.is_empty(), detail))
}

/// Check that duplicating traders under a fixed, thin source budget does not
/// drive the price error toward zero.
///
/// This is the regression test for the v2 error in which each extra trader
/// implicitly supplied another independent fundamental signal. Under that error
/// the measured MSE at `N_T = 100,000` would collapse toward zero; here it must
/// stay pinned to the `K = 2` floor.
fn check_trader_duplication(cfg: &Config) -> Result<Evidence> {
    let (k, rho_s) = (2usize, 0.9);
    let floor = analytic_floor(k, rho_s, 1.0);
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let mut measured = Vec::new();
    for (idx, &n_traders) in cfg.validation.plateau_trader_values.iter().enumerate() {
        let stats = pre_price_mse(
            Cell::new(cfg.master_seed, EXPERIMENT, 1_000 + idx as u64, &schedule),
            &market(k, n_traders, rho_s, 0.8),
        )?;
        measured.push((n_traders, stats.mean(), stats.std_error()));
    }
    let ok = measured
        .iter()
        .all(|(_, mse, se)| *mse > floor - cfg.validation.parity_z_tolerance * se);
    let summary: Vec<String> = measured
        .iter()
        .map(|(n, mse, _)| format!("N_T={n}: {mse:.4}"))
        .collect();
    Ok(Evidence::new(
        ok,
        format!("floor {floor:.4}; {}", summary.join(", ")),
    ))
}

/// Gate: the source layer produces exactly `K` draws whatever `N_T` is, and a
/// noiseless price is invariant to the trader count.
#[derive(Debug, Clone, Copy, Default)]
pub struct HierarchyIntact;

impl ValidationCheck for HierarchyIntact {
    fn name(&self) -> &'static str {
        "information hierarchy"
    }

    fn run(&self, ctx: &ValidationContext<'_>) -> Result<Evidence> {
        let _ = ctx;
        check_hierarchy(ctx.config)
    }
}

/// Gate: duplicating traders under a fixed, thin source budget does not drive
/// the price error toward zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct TraderDuplicationIsNotInformation;

impl ValidationCheck for TraderDuplicationIsNotInformation {
    fn name(&self) -> &'static str {
        "trader duplication is not information"
    }

    fn run(&self, ctx: &ValidationContext<'_>) -> Result<Evidence> {
        let _ = ctx;
        check_trader_duplication(ctx.config)
    }
}
