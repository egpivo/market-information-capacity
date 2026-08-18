//! The finite-source asymptote benchmark.
//!
//! For each `(K, rho_s, N_T)` cell this compares the simulated pre-official
//! price MSE against the analytic floor `sigma_s^2 [rho_s + (1-rho_s)/K]` and
//! reports the gap in Monte Carlo standard errors. It is the experiment that
//! makes the analytic result falsifiable rather than decorative.

use serde::Serialize;

use crate::analytics::information_floor::analytic_floor;
use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::rng::{StreamId, derive_seed};
use crate::simulation::experiment::{Experiment, SimulationContext};
use crate::simulation::monte_carlo::{Cell, pre_price_mse};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "asymptote";

/// One benchmark cell.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AsymptoteRow {
    /// Source budget `K`.
    pub k: usize,
    /// Source correlation `rho_s`.
    pub rho_s: f64,
    /// Trader count `N_T`.
    pub n_traders: usize,
    /// Source error scale `sigma_s`.
    pub sigma_s: f64,
    /// Analytic asymptotic floor.
    pub analytic_floor: f64,
    /// Simulated price MSE.
    pub mc_mse: f64,
    /// Standard error of the simulated MSE.
    pub mc_se: f64,
    /// Gap to the floor in standard errors.
    pub z_gap: f64,
    /// Realizations in the cell.
    pub realizations: u64,
    /// Seed of the cell's first batch, for auditability.
    pub seed: u64,
}

/// Market configuration for one benchmark cell.
fn cell_market(cfg: &Config, k: usize, rho_s: f64, n_traders: usize) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: cfg.asymptote.sigma_s,
            correlation: rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: cfg.asymptote.interpretation_sigma,
            clientele_bias: cfg.asymptote.clientele_bias,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: cfg.official,
    }
}

/// Run the benchmark over the configured grid.
pub fn run(cfg: &Config) -> Result<Vec<AsymptoteRow>> {
    let schedule = cfg.batch_schedule(cfg.asymptote.realizations);
    let mut rows = Vec::new();
    let mut cell: u64 = 0;
    for &k in &cfg.asymptote.k_values {
        for &rho_s in &cfg.asymptote.rho_values {
            for &n_traders in &cfg.asymptote.trader_values {
                let market = cell_market(cfg, k, rho_s, n_traders);
                let stats = pre_price_mse(
                    Cell::new(cfg.master_seed, EXPERIMENT, cell, &schedule),
                    &market,
                )?;
                let floor = analytic_floor(k, rho_s, cfg.asymptote.sigma_s);
                let se = stats.std_error();
                rows.push(AsymptoteRow {
                    k,
                    rho_s,
                    n_traders,
                    sigma_s: cfg.asymptote.sigma_s,
                    analytic_floor: floor,
                    mc_mse: stats.mean(),
                    mc_se: se,
                    z_gap: (stats.mean() - floor) / se,
                    realizations: stats.count(),
                    seed: derive_seed(cfg.master_seed, StreamId::new(EXPERIMENT, cell, 0, 0)),
                });
                cell += 1;
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulated_mse_tracks_the_analytic_floor() {
        let mut cfg = Config::quick();
        cfg.asymptote.realizations = 40_000;
        let rows = run(&cfg).expect("benchmark runs");
        assert_eq!(rows.len(), 18);
        for row in &rows {
            // Interpretation noise adds sigma_nu^2 / N_T on top of the floor.
            let expected = row.analytic_floor
                + cfg.asymptote.interpretation_sigma.powi(2) / row.n_traders as f64;
            let z = (row.mc_mse - expected) / row.mc_se;
            assert!(
                z.abs() < 5.0,
                "K={} rho={} N={}: mc {} vs {} ({z} se)",
                row.k,
                row.rho_s,
                row.n_traders,
                row.mc_mse,
                expected
            );
        }
    }
}

/// The finite-source asymptote benchmark as an [`Experiment`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AsymptoteExperiment;

impl Experiment for AsymptoteExperiment {
    type Output = Vec<AsymptoteRow>;

    fn name(&self) -> &'static str {
        "asymptote"
    }

    fn run(&self, ctx: &SimulationContext<'_>) -> Result<Self::Output> {
        run(ctx.config)
    }
}
