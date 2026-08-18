//! Gate: the simulated market really sits on the analytic floor, stays there as
//! participation grows, and moves when the source budget moves.

use crate::analytics::information_floor::analytic_floor;
use crate::config::{Config, MarketConfig, SourceConfig, TraderConfig};
use crate::error::Result;
use crate::model::traders::{InterpretationNoise, Representation};
use crate::simulation::monte_carlo::{Cell, pre_price_mse};
use crate::validation::Check;

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "gate-finite-source";

fn market(k: usize, n_traders: usize, rho_s: f64, sigma_nu: f64) -> MarketConfig {
    MarketConfig {
        sources: SourceConfig {
            count: k,
            sigma: 1.0,
            correlation: rho_s,
        },
        traders: TraderConfig {
            count: n_traders,
            interpretation_sigma: sigma_nu,
            clientele_bias: 0.0,
            representation: Representation::Equal,
            noise: InterpretationNoise::Aggregated,
        },
        official: Default::default(),
    }
}

/// Check that Monte Carlo price MSE matches `floor + sigma_nu^2 / N_T`.
pub fn check_floor_parity(cfg: &Config) -> Result<Check> {
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let sigma_nu = 0.15;
    let mut worst: Option<(usize, f64, f64)> = None;
    let mut failures = 0usize;
    let mut cells = 0usize;
    for (idx, (k, rho_s)) in [(2usize, 0.0), (2, 0.9), (10, 0.5), (50, 0.5), (50, 0.9)]
        .into_iter()
        .enumerate()
    {
        let n_traders = 5_000;
        let stats = pre_price_mse(
            Cell::new(cfg.master_seed, EXPERIMENT, idx as u64, &schedule),
            &market(k, n_traders, rho_s, sigma_nu),
        )?;
        let expected = analytic_floor(k, rho_s, 1.0) + sigma_nu * sigma_nu / n_traders as f64;
        let z = (stats.mean() - expected) / stats.std_error();
        cells += 1;
        if z.abs() > cfg.validation.parity_z_tolerance {
            failures += 1;
        }
        if worst.is_none_or(|(_, _, w)| z.abs() > w) {
            worst = Some((k, rho_s, z.abs()));
        }
    }
    let (k, rho_s, z) = worst.unwrap_or((0, 0.0, f64::NAN));
    Ok(Check::new(
        "monte carlo floor parity",
        failures == 0,
        format!(
            "{cells} cells; largest deviation {z:.2} SE at K={k}, rho_s={rho_s} (tolerance {:.1} SE)",
            cfg.validation.parity_z_tolerance
        ),
    ))
}

/// Check that a fixed source budget produces a plateau: growing `N_T` by three
/// orders of magnitude leaves the price error pinned above the floor.
pub fn check_plateau(cfg: &Config) -> Result<Check> {
    let (k, rho_s) = (10usize, 0.5);
    let floor = analytic_floor(k, rho_s, 1.0);
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let mut points = Vec::new();
    for (idx, &n_traders) in cfg.validation.plateau_trader_values.iter().enumerate() {
        let stats = pre_price_mse(
            Cell::new(cfg.master_seed, EXPERIMENT, 100 + idx as u64, &schedule),
            &market(k, n_traders, rho_s, 0.8),
        )?;
        points.push((n_traders, stats.mean(), stats.std_error()));
    }
    let above_floor = points
        .iter()
        .all(|(_, mse, se)| *mse > floor - cfg.validation.parity_z_tolerance * se);
    // The largest trader count must be within a few percent of the floor: the
    // curve flattens rather than continuing to fall.
    let (largest_n, largest_mse, _) = *points.last().unwrap_or(&(0, f64::NAN, f64::NAN));
    let relative_gap = (largest_mse - floor) / floor;
    let flattened = relative_gap.abs() < 0.05;
    let summary: Vec<String> = points
        .iter()
        .map(|(n, mse, _)| format!("N_T={n}: {mse:.4}"))
        .collect();
    Ok(Check::new(
        "finite-source plateau",
        above_floor && flattened,
        format!(
            "floor {floor:.4}; {}; relative gap at N_T={largest_n} is {:.3}",
            summary.join(", "),
            relative_gap
        ),
    ))
}

/// Check the source-count comparative static, and that at high trader density
/// doubling sources beats doubling traders.
pub fn check_source_comparative_static(cfg: &Config) -> Result<Check> {
    let rho_s = 0.5;
    let n_traders = 5_000;
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let mut measured = Vec::new();
    for (idx, k) in [2usize, 10, 50].into_iter().enumerate() {
        let stats = pre_price_mse(
            Cell::new(cfg.master_seed, EXPERIMENT, 200 + idx as u64, &schedule),
            &market(k, n_traders, rho_s, 0.8),
        )?;
        measured.push((k, stats.mean(), stats.std_error()));
    }
    let separated = measured.windows(2).all(|w| {
        let (_, hi, se_hi) = w[0];
        let (_, lo, se_lo) = w[1];
        hi - lo > 3.0 * (se_hi * se_hi + se_lo * se_lo).sqrt()
    });

    // Doubling comparison at high density, K = 10.
    let at = |index: u64| Cell::new(cfg.master_seed, EXPERIMENT, index, &schedule);
    let base = pre_price_mse(at(300), &market(10, n_traders, rho_s, 0.8))?;
    let double_traders = pre_price_mse(at(301), &market(10, 2 * n_traders, rho_s, 0.8))?;
    let double_sources = pre_price_mse(at(302), &market(20, n_traders, rho_s, 0.8))?;
    let gain_traders = base.mean() - double_traders.mean();
    let gain_sources = base.mean() - double_sources.mean();
    let se_diff = (double_traders.std_error().powi(2) + double_sources.std_error().powi(2)).sqrt();
    let sources_win = gain_sources - gain_traders > 3.0 * se_diff;

    let summary: Vec<String> = measured
        .iter()
        .map(|(k, mse, _)| format!("K={k}: {mse:.4}"))
        .collect();
    Ok(Check::new(
        "source-count comparative static",
        separated && sources_win,
        format!(
            "{}; at N_T={n_traders}, doubling sources gains {gain_sources:.4} vs {gain_traders:.4} from doubling traders (diff SE {se_diff:.4})",
            summary.join(", ")
        ),
    ))
}
