//! The same-price conditional experiment.
//!
//! Every world is filtered to the same narrow price band `|P_OC| < band`. The
//! spread of the latent value inside that band measures how much the observed
//! price actually pins down. Identical prices in different information
//! environments carry different amounts of information.

use serde::Serialize;

use crate::analytics::information_floor::{analytic_floor, approx_conditional_sd};
use crate::analytics::metrics::{OnlineStats, quantile_sorted};
use crate::config::Config;
use crate::error::Result;
use crate::simulation::monte_carlo::{Cell, run_conditional};

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "same-price";

/// One world's conditional distribution.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SamePriceRow {
    /// World name.
    pub world: String,
    /// Trader count.
    pub n_traders: usize,
    /// Source budget.
    pub k_sources: usize,
    /// Source correlation.
    pub rho_s: f64,
    /// Source error scale.
    pub sigma_s: f64,
    /// Clientele tilt.
    pub bias: f64,
    /// Half-width of the accepted band.
    pub band: f64,
    /// Analytic asymptotic floor of the world.
    pub analytic_floor: f64,
    /// Realizations drawn.
    pub drawn: u64,
    /// Realizations accepted into the band.
    pub accepted_n: u64,
    /// Acceptance rate.
    pub acceptance_rate: f64,
    /// Conditional mean latent value.
    pub mean_v: f64,
    /// Conditional standard deviation of the latent value.
    pub sd_v: f64,
    /// Conditional 5th percentile.
    pub q05_v: f64,
    /// Conditional 95th percentile.
    pub q95_v: f64,
    /// Closed-form approximation of the conditional standard deviation.
    pub approx_analytic_sd_v: f64,
    /// Whether the retained sample was subsampled.
    pub sample_truncated: bool,
}

/// Run the conditional experiment for every configured world.
pub fn run(cfg: &Config) -> Result<Vec<SamePriceRow>> {
    let schedule = cfg.batch_schedule(cfg.same_price.realizations);
    let band = cfg.same_price.band;
    let mut rows = Vec::with_capacity(cfg.worlds.presets.len());
    for (idx, world) in cfg.worlds.presets.iter().enumerate() {
        let market = world.market(cfg.official);
        let acc = run_conditional(
            Cell::new(cfg.master_seed, EXPERIMENT, idx as u64, &schedule),
            &market,
            band,
            cfg.same_price.retained_samples,
        )?;
        let drawn = acc.drawn;
        let accepted_n = acc.accepted.seen();
        let truncated = acc.accepted.truncated();
        let sorted = acc.accepted.into_sorted();
        let mut stats = OnlineStats::new();
        for &v in &sorted {
            stats.push(v);
        }
        rows.push(SamePriceRow {
            world: world.name.clone(),
            n_traders: world.traders,
            k_sources: world.sources,
            rho_s: world.rho_s,
            sigma_s: world.sigma_s,
            bias: world.clientele_bias,
            band,
            analytic_floor: analytic_floor(world.sources, world.rho_s, world.sigma_s),
            drawn,
            accepted_n,
            acceptance_rate: accepted_n as f64 / drawn.max(1) as f64,
            mean_v: stats.mean(),
            sd_v: stats.std_dev(),
            q05_v: quantile_sorted(&sorted, 0.05),
            q95_v: quantile_sorted(&sorted, 0.95),
            approx_analytic_sd_v: approx_conditional_sd(&market, band),
            sample_truncated: truncated,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_spread_widens_as_information_thins() {
        let cfg = Config::quick();
        let rows = run(&cfg).expect("conditional experiment runs");
        assert_eq!(rows.len(), 3);
        assert!(
            rows[0].sd_v < rows[1].sd_v && rows[1].sd_v < rows[2].sd_v,
            "conditional SDs were {} {} {}",
            rows[0].sd_v,
            rows[1].sd_v,
            rows[2].sd_v
        );
        for row in &rows {
            assert!(
                row.accepted_n > 1000,
                "{} accepted only {}",
                row.world,
                row.accepted_n
            );
            let rel = (row.sd_v - row.approx_analytic_sd_v).abs() / row.approx_analytic_sd_v;
            assert!(
                rel < 0.05,
                "{}: sd {} vs approx {}",
                row.world,
                row.sd_v,
                row.approx_analytic_sd_v
            );
        }
    }
}
