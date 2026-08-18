//! Gate: the same seed and configuration reproduce the same numbers.
//!
//! The check runs one Monte Carlo cell twice through the parallel engine and
//! compares the results bit for bit, then confirms that changing the master
//! seed does change them.

use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::simulation::monte_carlo::{Cell, pre_price_mse};
use crate::validation::Check;

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "gate-determinism";

fn market() -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: 10,
            sigma: 1.0,
            correlation: 0.5,
        },
        traders: TraderConfig {
            count: 2_500,
            interpretation_sigma: 0.8,
            clientele_bias: 0.35,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: Default::default(),
    }
}

/// Check bitwise reproduction under a fixed seed and sensitivity to the seed.
pub fn check_determinism(cfg: &Config) -> Result<Check> {
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let market = market();
    let cell = Cell::new(cfg.master_seed, EXPERIMENT, 0, &schedule);
    let first = pre_price_mse(cell, &market)?;
    let second = pre_price_mse(cell, &market)?;
    let other_seed = pre_price_mse(
        Cell::new(cfg.master_seed + 1, EXPERIMENT, 0, &schedule),
        &market,
    )?;

    let reproduces = first.mean().to_bits() == second.mean().to_bits()
        && first.variance().to_bits() == second.variance().to_bits()
        && first.count() == second.count();
    let seed_matters = other_seed.mean().to_bits() != first.mean().to_bits();

    Ok(Check::new(
        "deterministic reproduction",
        reproduces && seed_matters,
        format!(
            "seed {} reproduces MSE {:.12} bitwise over {} realizations in {} batches; seed {} gives {:.12}",
            cfg.master_seed,
            first.mean(),
            first.count(),
            schedule.len(),
            cfg.master_seed + 1,
            other_seed.mean()
        ),
    ))
}
