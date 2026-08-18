//! Traders versus sources: the grid, the doubling comparison, and the
//! convergence thresholds.
//!
//! The grid sweeps `N_T` against `K` and reports the analytic floor next to
//! every simulated point, so the plateau is visible in the data file and not
//! only in a figure. The doubling comparison asks the question the grid implies:
//! at a given participation level, is it better to double the traders or to
//! double the independent sources?

use serde::Serialize;

use crate::analytics::convergence::{clean_threshold, first_grid_threshold};
use crate::analytics::information_floor::{analytic_floor, analytic_pre_price_mse};
use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::simulation::experiment::{Experiment, SimulationContext};
use crate::simulation::monte_carlo::{Cell, blocked_pre_price_mse, pre_price_mse};

/// Stable experiment label for the grid.
pub const GRID_EXPERIMENT: &str = "traders-vs-sources";
/// Stable experiment label for the doubling comparison.
pub const DOUBLING_EXPERIMENT: &str = "trader-source-doubling";

/// One grid cell.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GridRow {
    /// Trader count `N_T`.
    pub n_traders: usize,
    /// Source budget `K`.
    pub k_sources: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Source error scale.
    pub sigma_s: f64,
    /// Interpretation noise scale.
    pub interpretation_sigma: f64,
    /// Simulated price MSE (mean of block means).
    pub mse: f64,
    /// Standard error across seed blocks, on `blocks - 1` degrees of freedom.
    pub block_se: f64,
    /// Realization-level Monte Carlo standard error pooled across blocks.
    pub mc_se: f64,
    /// Analytic asymptotic floor.
    pub analytic_floor: f64,
    /// Exact finite-`N_T` price MSE, `floor + sigma_nu^2 / N_T`.
    pub analytic_mse: f64,
    /// Relative distance of the simulated MSE above the floor.
    pub gap_to_floor: f64,
    /// Traders per source.
    pub traders_per_source: f64,
    /// Realizations behind the estimate.
    pub realizations: u64,
}

/// One density point of the doubling comparison.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DoublingRow {
    /// Density label.
    pub density: String,
    /// Base trader count.
    pub n_traders: usize,
    /// Base source budget.
    pub k_sources: usize,
    /// Simulated base MSE.
    pub mse_base: f64,
    /// Standard error of the base MSE.
    pub se_base: f64,
    /// Simulated MSE after doubling traders.
    pub mse_double_traders: f64,
    /// Standard error after doubling traders.
    pub se_double_traders: f64,
    /// Simulated MSE after doubling sources.
    pub mse_double_sources: f64,
    /// Standard error after doubling sources.
    pub se_double_sources: f64,
    /// Simulated improvement from doubling traders.
    pub improvement_double_traders: f64,
    /// Simulated improvement from doubling sources.
    pub improvement_double_sources: f64,
    /// Standard error of the trader-doubling improvement.
    pub se_improvement_double_traders: f64,
    /// Standard error of the source-doubling improvement.
    pub se_improvement_double_sources: f64,
    /// Exact improvement from doubling traders, `sigma_nu^2 / (2 N_T)`.
    pub analytic_improvement_double_traders: f64,
    /// Exact improvement from doubling sources.
    pub analytic_improvement_double_sources: f64,
}

/// One convergence-threshold row.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConvergenceRow {
    /// Source budget.
    pub k_sources: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Relative gap targeted.
    pub threshold_gap: f64,
    /// First trader count on the publication grid meeting the gap, if any.
    pub first_grid_n_traders: Option<usize>,
    /// Analytic trader count meeting the gap in the clean benchmark.
    pub analytic_clean_threshold_n_traders: Option<u64>,
}

/// The three outputs of this experiment.
#[derive(Debug, Clone, Serialize)]
pub struct TradersVsSources {
    /// The grid.
    pub grid: Vec<GridRow>,
    /// The doubling comparison.
    pub doubling: Vec<DoublingRow>,
    /// The convergence thresholds.
    pub convergence: Vec<ConvergenceRow>,
}

fn grid_market(cfg: &Config, k: usize, n_traders: usize) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: cfg.grid.sigma_s,
            correlation: cfg.grid.rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: cfg.grid.interpretation_sigma,
            clientele_bias: cfg.grid.clientele_bias,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: cfg.official,
    }
}

fn doubling_market(cfg: &Config, k: usize, n_traders: usize) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: cfg.doubling.sigma_s,
            correlation: cfg.doubling.rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: cfg.doubling.interpretation_sigma,
            clientele_bias: 0.0,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: cfg.official,
    }
}

/// Run the grid.
pub fn run_grid(cfg: &Config) -> Result<Vec<GridRow>> {
    let schedule = cfg.batch_schedule(cfg.grid.realizations_per_block);
    let mut rows = Vec::new();
    let mut cell: u64 = 0;
    for &k in &cfg.grid.k_values {
        for &n_traders in &cfg.grid.trader_values {
            let market = grid_market(cfg, k, n_traders);
            let blocked = blocked_pre_price_mse(
                Cell::new(cfg.master_seed, GRID_EXPERIMENT, cell, &schedule),
                cfg.grid.blocks,
                &market,
            )?;
            let mean = blocked.mean;
            let floor = analytic_floor(k, cfg.grid.rho_s, cfg.grid.sigma_s);
            rows.push(GridRow {
                n_traders,
                k_sources: k,
                rho_s: cfg.grid.rho_s,
                sigma_s: cfg.grid.sigma_s,
                interpretation_sigma: cfg.grid.interpretation_sigma,
                mse: mean,
                block_se: blocked.block_se,
                mc_se: blocked.pooled.std_error(),
                analytic_floor: floor,
                analytic_mse: analytic_pre_price_mse(&market),
                gap_to_floor: (mean - floor) / floor,
                traders_per_source: n_traders as f64 / k as f64,
                realizations: (cfg.grid.realizations_per_block * cfg.grid.blocks) as u64,
            });
            cell += 1;
        }
    }
    Ok(rows)
}

/// Run the doubling comparison.
pub fn run_doubling(cfg: &Config) -> Result<Vec<DoublingRow>> {
    let schedule = cfg.batch_schedule(cfg.doubling.realizations);
    let k = cfg.doubling.k_sources;
    let sigma_nu2 = cfg.doubling.interpretation_sigma.powi(2);
    let mut rows = Vec::new();
    for (idx, density) in cfg.doubling.densities.iter().enumerate() {
        let cell = idx as u64 * 3;
        let n = density.traders;
        let at = |index: u64| Cell::new(cfg.master_seed, DOUBLING_EXPERIMENT, index, &schedule);
        let base = pre_price_mse(at(cell), &doubling_market(cfg, k, n))?;
        let double_traders = pre_price_mse(at(cell + 1), &doubling_market(cfg, k, 2 * n))?;
        let double_sources = pre_price_mse(at(cell + 2), &doubling_market(cfg, 2 * k, n))?;

        let se = |a: f64, b: f64| (a * a + b * b).sqrt();
        rows.push(DoublingRow {
            density: density.label.clone(),
            n_traders: n,
            k_sources: k,
            mse_base: base.mean(),
            se_base: base.std_error(),
            mse_double_traders: double_traders.mean(),
            se_double_traders: double_traders.std_error(),
            mse_double_sources: double_sources.mean(),
            se_double_sources: double_sources.std_error(),
            improvement_double_traders: base.mean() - double_traders.mean(),
            improvement_double_sources: base.mean() - double_sources.mean(),
            se_improvement_double_traders: se(base.std_error(), double_traders.std_error()),
            se_improvement_double_sources: se(base.std_error(), double_sources.std_error()),
            analytic_improvement_double_traders: sigma_nu2 / (2.0 * n as f64),
            analytic_improvement_double_sources: analytic_floor(
                k,
                cfg.doubling.rho_s,
                cfg.doubling.sigma_s,
            ) - analytic_floor(
                2 * k,
                cfg.doubling.rho_s,
                cfg.doubling.sigma_s,
            ),
        });
    }
    Ok(rows)
}

/// Derive convergence thresholds from the grid and the clean analytic formula.
pub fn run_convergence(cfg: &Config, grid: &[GridRow]) -> Vec<ConvergenceRow> {
    let mut rows = Vec::new();
    for &k in &cfg.grid.convergence_k_values {
        let mut points: Vec<(usize, f64)> = grid
            .iter()
            .filter(|row| row.k_sources == k)
            .map(|row| (row.n_traders, row.gap_to_floor))
            .collect();
        points.sort_by_key(|(n, _)| *n);
        for &gap in &cfg.grid.convergence_gaps {
            rows.push(ConvergenceRow {
                k_sources: k,
                rho_s: cfg.grid.rho_s,
                threshold_gap: gap,
                first_grid_n_traders: first_grid_threshold(&points, gap),
                analytic_clean_threshold_n_traders: clean_threshold(
                    k,
                    cfg.grid.rho_s,
                    cfg.grid.sigma_s,
                    cfg.grid.interpretation_sigma,
                    gap,
                ),
            });
        }
    }
    rows
}

/// Run the grid, the doubling comparison and the thresholds together.
pub fn run(cfg: &Config) -> Result<TradersVsSources> {
    let grid = run_grid(cfg)?;
    let doubling = run_doubling(cfg)?;
    let convergence = run_convergence(cfg, &grid);
    Ok(TradersVsSources {
        grid,
        doubling,
        convergence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_never_falls_meaningfully_below_the_floor() {
        let mut cfg = Config::quick();
        cfg.grid.trader_values = vec![50, 500, 5000];
        cfg.grid.k_values = vec![2, 10];
        let grid = run_grid(&cfg).expect("grid runs");
        for row in &grid {
            assert!(
                row.mse > row.analytic_floor * 0.9,
                "K={} N={} fell to {} below floor {}",
                row.k_sources,
                row.n_traders,
                row.mse,
                row.analytic_floor
            );
        }
    }

    #[test]
    fn doubling_sources_beats_doubling_traders_at_high_density() {
        let mut cfg = Config::quick();
        cfg.doubling.realizations = 200_000;
        cfg.doubling.densities = vec![crate::config::DensitySpec {
            label: "high".into(),
            traders: 5000,
        }];
        let rows = run_doubling(&cfg).expect("doubling runs");
        let row = &rows[0];
        assert!(
            row.improvement_double_sources > row.improvement_double_traders,
            "sources {} vs traders {}",
            row.improvement_double_sources,
            row.improvement_double_traders
        );
    }
}

/// The grid, doubling comparison and convergence thresholds as an [`Experiment`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TradersVsSourcesExperiment;

impl Experiment for TradersVsSourcesExperiment {
    type Output = TradersVsSources;

    fn name(&self) -> &'static str {
        "traders-vs-sources"
    }

    fn run(&self, ctx: &SimulationContext<'_>) -> Result<Self::Output> {
        run(ctx.config)
    }
}
