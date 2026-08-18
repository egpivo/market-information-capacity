//! Gate: the Fast Follower mechanism, tested as a mechanism.
//!
//! No canonical number is hard-coded. What must hold is the economic signature:
//! many traders reading a thin, highly correlated pre-official information set
//! price `V` poorly, and a genuinely new official signal then moves the price a
//! long way.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::simulation::monte_carlo::{Cell, run_world};
use crate::validation::Check;

/// Stable experiment label used for seed derivation.
pub const EXPERIMENT: &str = "gate-fast-follower";

/// Check the Fast Follower signature against the Informative world.
pub fn check_fast_follower(cfg: &Config) -> Result<Check> {
    let schedule = cfg.batch_schedule(cfg.validation.realizations);
    let mut rows = Vec::new();
    for (idx, world) in cfg.worlds.presets.iter().enumerate() {
        let acc = run_world(
            Cell::new(cfg.master_seed, EXPERIMENT, idx as u64, &schedule),
            &world.market(cfg.official),
            0,
        )?;
        rows.push((
            world.clone(),
            acc.pre_mse.mean(),
            acc.post_mse.mean(),
            acc.abs_revision.mean(),
        ));
    }

    let fast_follower = rows
        .iter()
        .max_by(|a, b| {
            (a.0.rho_s / a.0.sources as f64).total_cmp(&(b.0.rho_s / b.0.sources as f64))
        })
        .ok_or_else(|| Error::Experiment("no worlds configured".into()))?;
    let richest = rows
        .iter()
        .min_by(|a, b| {
            (a.0.rho_s / a.0.sources as f64).total_cmp(&(b.0.rho_s / b.0.sources as f64))
        })
        .ok_or_else(|| Error::Experiment("no worlds configured".into()))?;

    let (world, pre, post, revision) = fast_follower;
    let many_traders = world.traders >= 1_000;
    let thin_information = world.sources <= 2 && world.rho_s >= 0.75;
    let weak_pre = *pre > 5.0 * richest.1;
    let strong_response = *post < 0.3 * *pre;
    let material_revision = *revision > 0.3;

    let ok = many_traders && thin_information && weak_pre && strong_response && material_revision;
    Ok(Check::new(
        "fast-follower mechanism",
        ok,
        format!(
            "{}: N_T={}, K={}, rho_s={}; pre MSE {pre:.3} -> post MSE {post:.3}, mean |revision| {revision:.3} (richest world pre MSE {:.3})",
            world.name, world.traders, world.sources, world.rho_s, richest.1
        ),
    ))
}
